from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.io as scio
import torch

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from omni_hc.constraints.utils.spectral import (
    crop_spatial_2d,
    pad_spatial_2d,
    spectral_divergence_2d,
    spectral_gradient_2d,
)


TRAIN_FILE = "piececonst_r421_N1024_smooth1.mat"
TEST_FILE = "piececonst_r421_N1024_smooth2.mat"

EDGES = {
    "top_i0": (0, slice(None)),
    "bottom_iN": (-1, slice(None)),
    "left_j0": (slice(None), 0),
    "right_jN": (slice(None), -1),
}


def resolve_darcy_mat(data_dir: Path, split: str) -> Path:
    if split == "train":
        path = data_dir / TRAIN_FILE
    elif split == "test":
        path = data_dir / TEST_FILE
    else:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    if not path.exists():
        raise FileNotFoundError(f"Missing Darcy MAT file: {path}")
    return path


def load_darcy_arrays(
    data_dir: Path,
    *,
    split: str,
    downsamplex: int,
    downsampley: int,
) -> tuple[np.ndarray, np.ndarray, Path]:
    mat_path = resolve_darcy_mat(data_dir, split)
    raw = scio.loadmat(str(mat_path))
    if "coeff" not in raw or "sol" not in raw:
        raise KeyError(f"Expected 'coeff' and 'sol' variables in {mat_path}")

    coeff = np.asarray(raw["coeff"], dtype=np.float64)
    sol = np.asarray(raw["sol"], dtype=np.float64)
    if coeff.shape != sol.shape or coeff.ndim != 3:
        raise ValueError(
            "Expected coeff and sol with matching shape (N, H, W); "
            f"got coeff={coeff.shape}, sol={sol.shape}"
        )

    r1 = int(downsamplex)
    r2 = int(downsampley)
    if r1 <= 0 or r2 <= 0:
        raise ValueError("downsamplex and downsampley must be positive")

    h_full, w_full = int(coeff.shape[1]), int(coeff.shape[2])
    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    coeff = coeff[:, ::r1, ::r2][:, :s1, :s2]
    sol = sol[:, ::r1, ::r2][:, :s1, :s2]
    return coeff, sol, mat_path


def validate_samples(samples: list[int], n_samples: int) -> None:
    for sample_idx in samples:
        if sample_idx < 0 or sample_idx >= n_samples:
            raise IndexError(f"Sample {sample_idx} is outside [0, {n_samples})")


def sample_count(requested: int, available: int) -> int:
    n = min(int(requested), int(available))
    if n <= 0:
        raise ValueError("sample count must be positive")
    return n


def grid_spacing(shape: tuple[int, int], *, lower: float = 0.0, upper: float = 1.0):
    height, width = int(shape[0]), int(shape[1])
    extent = float(upper) - float(lower)
    dy = extent / max(height - 1, 1)
    dx = extent / max(width - 1, 1)
    return dy, dx


def boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def edge_values(field: np.ndarray, edge: str) -> np.ndarray:
    if edge not in EDGES:
        raise ValueError(f"Unknown edge {edge!r}")
    return np.asarray(field[EDGES[edge]], dtype=np.float64)


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean_abs": float(np.abs(values).mean()),
        "max_abs": float(np.abs(values).max()),
        "l2": float(np.sqrt(np.mean(values**2))),
    }


def residual_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    abs_values = np.abs(values)
    return {
        "mean_abs": float(abs_values.mean()),
        "median_abs": float(np.median(abs_values)),
        "p95_abs": float(np.percentile(abs_values, 95)),
        "max_abs": float(abs_values.max()),
        "signed_mean": float(values.mean()),
        "signed_std": float(values.std()),
        "l2": float(np.sqrt(np.mean(values**2))),
    }


def _crop_residual_eval(residual: np.ndarray, interior_crop: int) -> np.ndarray:
    if interior_crop <= 0:
        return residual
    crop = int(interior_crop)
    if 2 * crop >= min(residual.shape):
        raise ValueError(
            f"interior_crop={crop} removes the whole residual grid with shape {residual.shape}"
        )
    return residual[crop:-crop, crop:-crop]


def darcy_residual_finite_difference(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    force_value: float = 1.0,
    interior_crop: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy, dx = grid_spacing(sol.shape)
    du_dy, du_dx = np.gradient(sol, dy, dx, edge_order=2)
    flux_x = -coeff * du_dx
    flux_y = -coeff * du_dy
    dflux_y_dy = np.gradient(flux_y, dy, axis=0, edge_order=2)
    dflux_x_dx = np.gradient(flux_x, dx, axis=1, edge_order=2)
    residual = dflux_x_dx + dflux_y_dy - float(force_value)
    residual_eval = _crop_residual_eval(residual, interior_crop)
    return residual, residual_eval, flux_x, flux_y


def darcy_residual_spectral(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    force_value: float = 1.0,
    interior_crop: int = 1,
    padding: int = 8,
    padding_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dy, dx = grid_spacing(sol.shape)
    device = torch.device("cpu")
    sol_tensor = torch.as_tensor(sol, dtype=torch.float64, device=device).view(
        1, 1, *sol.shape
    )
    coeff_tensor = torch.as_tensor(coeff, dtype=torch.float64, device=device).view(
        1, 1, *coeff.shape
    )

    with torch.no_grad():
        sol_padded = pad_spatial_2d(sol_tensor, padding, mode=padding_mode)
        coeff_padded = pad_spatial_2d(coeff_tensor, padding, mode=padding_mode)
        gradient_padded = spectral_gradient_2d(sol_padded, dy=dy, dx=dx)
        flux_padded = -coeff_padded * gradient_padded
        residual_padded = spectral_divergence_2d(flux_padded, dy=dy, dx=dx) - float(
            force_value
        )

        residual_tensor = crop_spatial_2d(residual_padded, padding)
        flux_tensor = crop_spatial_2d(flux_padded, padding)
    residual = residual_tensor.squeeze(0).squeeze(0).cpu().numpy()
    flux_x = flux_tensor[0, 0].cpu().numpy()
    flux_y = flux_tensor[0, 1].cpu().numpy()
    residual_eval = _crop_residual_eval(residual, interior_crop)
    return residual, residual_eval, flux_x, flux_y


def darcy_residual(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    force_value: float = 1.0,
    interior_crop: int = 1,
    method: str = "finite_difference",
    padding: int = 8,
    padding_mode: str = "reflect",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if coeff.shape != sol.shape or coeff.ndim != 2:
        raise ValueError(
            "Expected coeff and sol with matching shape (H, W); "
            f"got coeff={coeff.shape}, sol={sol.shape}"
        )
    method = str(method)
    if method == "finite_difference":
        return darcy_residual_finite_difference(
            coeff,
            sol,
            force_value=force_value,
            interior_crop=interior_crop,
        )
    if method == "spectral":
        return darcy_residual_spectral(
            coeff,
            sol,
            force_value=force_value,
            interior_crop=interior_crop,
            padding=padding,
            padding_mode=padding_mode,
        )
    raise ValueError(
        f"Unknown residual method {method!r}; expected 'finite_difference' or 'spectral'"
    )


def write_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_matplotlib(plt):
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )
