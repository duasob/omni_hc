from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import scipy.io as scio


PLASTICITY_FILE = "plas_N987_T20.mat"

EDGES = {
    "left_i0": (0, slice(None)),
    "right_iN": (-1, slice(None)),
    "upper_j0": (slice(None), 0),
    "lower_jN": (slice(None), -1),
}


def resolve_plasticity_mat(data_dir: Path) -> Path:
    if data_dir.is_file() and data_dir.suffix == ".mat":
        return data_dir
    candidates = [
        data_dir / PLASTICITY_FILE,
        data_dir / "plasticity" / PLASTICITY_FILE,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Missing plasticity MAT file. Expected {PLASTICITY_FILE} in one of:\n  {searched}"
    )


def load_plasticity_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, Path]:
    mat_path = resolve_plasticity_mat(data_dir)
    raw = scio.loadmat(str(mat_path), variable_names=("input", "output"))
    if "input" not in raw or "output" not in raw:
        raise KeyError(f"Expected variables 'input' and 'output' in {mat_path}")
    die = np.asarray(raw["input"])
    output = np.asarray(raw["output"])
    if die.ndim != 2:
        raise ValueError(f"Expected input shape (N, X), got {die.shape}")
    if output.ndim != 5:
        raise ValueError(f"Expected output shape (N, X, Y, T, C), got {output.shape}")
    if output.shape[0] != die.shape[0] or output.shape[1] != die.shape[1]:
        raise ValueError(
            "Plasticity input/output sample or X dimensions do not match: "
            f"input={die.shape}, output={output.shape}"
        )
    if output.shape[-1] < 4:
        raise ValueError(f"Expected at least four output channels, got {output.shape[-1]}")
    return die, output, mat_path


def select_split(
    die: np.ndarray,
    output: np.ndarray,
    *,
    split: str,
    ntrain: int,
    ntest: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(output.shape[0])
    if split == "all":
        return die, output
    if split == "train":
        if ntrain <= 0 or ntrain > n_samples:
            raise ValueError(f"ntrain must be in [1, {n_samples}], got {ntrain}")
        return die[:ntrain], output[:ntrain]
    if split == "test":
        if ntest <= 0 or ntest > n_samples:
            raise ValueError(f"ntest must be in [1, {n_samples}], got {ntest}")
        return die[-ntest:], output[-ntest:]
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


def reference_grid(shape: tuple[int, int]) -> np.ndarray:
    size_x, size_y = int(shape[0]), int(shape[1])
    x = np.linspace(0.0, 1.0, size_x, dtype=np.float64)
    y = np.linspace(0.0, 1.0, size_y, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return np.stack([xx, yy], axis=-1)


def edge_values(field: np.ndarray, edge: str) -> np.ndarray:
    if edge not in EDGES:
        raise ValueError(f"Unknown edge {edge!r}")
    field = np.asarray(field)
    selector = EDGES[edge]
    if field.ndim == 2:
        full_selector = selector + (slice(None),) * max(field.ndim - 2, 0)
    else:
        full_selector = (slice(None),) + selector + (slice(None),) * (field.ndim - 3)
    return np.asarray(field[full_selector], dtype=np.float64)


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
        "l2": float(np.sqrt(np.mean(values**2))),
    }


def safe_corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError(f"Cannot correlate arrays with sizes {a.size} and {b.size}")
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require_matplotlib(plt):
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )
