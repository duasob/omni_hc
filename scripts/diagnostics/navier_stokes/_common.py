from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.io as scio


DEFAULT_MAT = "NavierStokes_V1e-5_N1200_T20.mat"


def resolve_ns_mat(data_dir: Path) -> Path:
    """Resolve the Navier-Stokes vorticity MAT file from a dir or direct path."""
    root = Path(data_dir)
    if root.is_file() and root.suffix == ".mat":
        return root
    candidate = root / DEFAULT_MAT
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Expected a Navier-Stokes .mat file or a directory containing '{DEFAULT_MAT}'; "
        f"got {data_dir}"
    )


def load_ns_vorticity(
    data_dir: Path,
    *,
    downsamplex: int = 1,
    downsampley: int = 1,
    verbose: bool = False,
) -> tuple[np.ndarray, Path]:
    """Load the vorticity field ``u`` with shape (N, H, W, T).

    The dataset target is the vorticity directly, so this is the field the NS
    constraint (zero global vorticity mean) acts on.
    """
    mat_path = resolve_ns_mat(data_dir)
    if verbose:
        print(f"loading Navier-Stokes variable 'u' from {mat_path}", flush=True)
    raw = scio.loadmat(str(mat_path), variable_names=("u",))
    if "u" not in raw:
        raise KeyError(f"Expected key 'u' in {mat_path}")
    u = np.asarray(raw["u"], dtype=np.float64)
    if u.ndim != 4:
        raise ValueError(f"Expected 'u' with shape (N, H, W, T), got {u.shape}")

    r1 = int(downsamplex)
    r2 = int(downsampley)
    if r1 <= 0 or r2 <= 0:
        raise ValueError("downsamplex and downsampley must be positive")
    h_full, w_full = int(u.shape[1]), int(u.shape[2])
    s1 = int(((h_full - 1) / r1) + 1)
    s2 = int(((w_full - 1) / r2) + 1)
    u = u[:, ::r1, ::r2][:, :s1, :s2]
    if verbose:
        print(f"loaded vorticity u={u.shape}", flush=True)
    return u, mat_path


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
