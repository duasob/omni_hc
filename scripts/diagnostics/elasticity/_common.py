from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


SIGMA_FILE = "Random_UnitCell_sigma_10.npy"
XY_FILE = "Random_UnitCell_XY_10.npy"


def resolve_elasticity_files(data_dir: Path) -> tuple[Path, Path]:
    candidates = [
        data_dir / "Meshes",
        data_dir,
        data_dir / "elasticity" / "Meshes",
    ]
    for root in candidates:
        sigma_path = root / SIGMA_FILE
        xy_path = root / XY_FILE
        if sigma_path.exists() and xy_path.exists():
            return sigma_path, xy_path

    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Missing elasticity files. Expected "
        f"{SIGMA_FILE} and {XY_FILE} in one of:\n  {searched}"
    )


def orient_elasticity_arrays(
    sigma_raw: np.ndarray,
    xy_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays in NSL loader order: coords=(N, P, 2), sigma=(N, P)."""
    sigma_raw = np.asarray(sigma_raw)
    xy_raw = np.asarray(xy_raw)

    if xy_raw.ndim != 3:
        raise ValueError(f"Expected XY with 3 dimensions, got shape {xy_raw.shape}")
    if xy_raw.shape[1] == 2:
        coords = np.transpose(xy_raw, (2, 0, 1))
    elif xy_raw.shape[2] == 2:
        coords = xy_raw
    elif xy_raw.shape[0] == 2:
        coords = np.transpose(xy_raw, (2, 1, 0))
    else:
        raise ValueError(
            "Expected XY in shape (P, 2, N), (N, P, 2), or (2, P, N); "
            f"got {xy_raw.shape}"
        )

    n_samples, n_points, _ = coords.shape

    if sigma_raw.ndim == 3 and sigma_raw.shape[-1] == 1:
        sigma_raw = sigma_raw[..., 0]
    if sigma_raw.ndim != 2:
        raise ValueError(
            "Expected scalar sigma with 2 dimensions, optionally with a trailing "
            f"singleton channel; got shape {sigma_raw.shape}"
        )

    if sigma_raw.shape == (n_points, n_samples):
        sigma = sigma_raw.T
    elif sigma_raw.shape == (n_samples, n_points):
        sigma = sigma_raw
    elif sigma_raw.shape[0] == n_samples:
        sigma = sigma_raw
    elif sigma_raw.shape[1] == n_samples:
        sigma = sigma_raw.T
    else:
        raise ValueError(
            "Could not align sigma with XY. Expected sigma to contain "
            f"{n_samples} samples and {n_points} points; got {sigma_raw.shape}"
        )

    if sigma.shape != (n_samples, n_points):
        raise ValueError(
            f"Aligned sigma shape {sigma.shape} does not match coords {(n_samples, n_points, 2)}"
        )
    return np.asarray(coords), np.asarray(sigma)


def load_elasticity_arrays(
    data_dir: Path,
    *,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray, Path, Path]:
    sigma_path, xy_path = resolve_elasticity_files(data_dir)
    sigma_raw = np.load(sigma_path, mmap_mode=mmap_mode)
    xy_raw = np.load(xy_path, mmap_mode=mmap_mode)
    coords, sigma = orient_elasticity_arrays(sigma_raw, xy_raw)
    return coords, sigma, sigma_path, xy_path


def select_split(
    coords: np.ndarray,
    sigma: np.ndarray,
    *,
    split: str,
    ntrain: int,
    ntest: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(sigma.shape[0])
    if split == "all":
        return coords, sigma
    if split == "train":
        if ntrain <= 0 or ntrain > n_samples:
            raise ValueError(f"ntrain must be in [1, {n_samples}], got {ntrain}")
        return coords[:ntrain], sigma[:ntrain]
    if split == "test":
        if ntest <= 0 or ntest > n_samples:
            raise ValueError(f"ntest must be in [1, {n_samples}], got {ntest}")
        return coords[-ntest:], sigma[-ntest:]
    raise ValueError(f"split must be 'all', 'train', or 'test', got {split!r}")


def validate_samples(samples: list[int], n_samples: int) -> None:
    for sample_idx in samples:
        if sample_idx < 0 or sample_idx >= n_samples:
            raise IndexError(f"Sample {sample_idx} is outside [0, {n_samples})")


def sample_count(requested: int, available: int) -> int:
    n = min(int(requested), int(available))
    if n <= 0:
        raise ValueError("sample count must be positive")
    return n


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
