from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def load_pipe_arrays(data_dir: Path):
    """Load Pipe_X, Pipe_Y, Pipe_Q as (x, y, q) memmaps.

    x, y are the curvilinear mesh coordinates with shape (N, H, W); q is the
    velocity field with shape (N, C, H, W) and at least two channels (ux, uy).
    """
    required = ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy")
    missing = [name for name in required if not (Path(data_dir) / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required pipe files in {data_dir}: {', '.join(missing)}"
        )

    x = np.load(Path(data_dir) / "Pipe_X.npy", mmap_mode="r")
    y = np.load(Path(data_dir) / "Pipe_Y.npy", mmap_mode="r")
    q = np.load(Path(data_dir) / "Pipe_Q.npy", mmap_mode="r")
    if x.shape != y.shape:
        raise ValueError(f"Pipe_X and Pipe_Y shapes differ: {x.shape} vs {y.shape}")
    if q.ndim != 4 or q.shape[0] != x.shape[0] or q.shape[2:] != x.shape[1:]:
        raise ValueError(
            "Expected Pipe_Q shape (N, C, H, W) matching Pipe_X/Pipe_Y; "
            f"got Pipe_X={x.shape}, Pipe_Q={q.shape}"
        )
    if q.shape[1] < 2:
        raise ValueError(f"Expected ux and uy channels in Pipe_Q, got {q.shape[1]}")
    return x, y, q


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
