from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify whether the pipe dataset is divergence free on the curvilinear mesh."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pipe"),
        help="Directory containing Pipe_X.npy, Pipe_Y.npy, and Pipe_Q.npy.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 10, 100],
        help="Sample indices to print and plot.",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=1000,
        help="Number of leading samples to include in aggregate statistics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/pipe_divergence"),
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print and write divergence statistics; do not write figures.",
    )
    return parser.parse_args()


def load_pipe_arrays(data_dir: Path):
    required = ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required pipe files in {data_dir}: {', '.join(missing)}"
        )

    x = np.load(data_dir / "Pipe_X.npy", mmap_mode="r")
    y = np.load(data_dir / "Pipe_Y.npy", mmap_mode="r")
    q = np.load(data_dir / "Pipe_Q.npy", mmap_mode="r")
    if x.shape != y.shape:
        raise ValueError(f"Pipe_X and Pipe_Y shapes differ: {x.shape} vs {y.shape}")
    if q.ndim != 4 or q.shape[0] != x.shape[0] or q.shape[2:] != x.shape[1:]:
        raise ValueError(f"Expected Pipe_Q=(N,C,H,W) matching X/Y; got {q.shape}")
    if q.shape[1] < 2:
        raise ValueError(f"Expected ux and uy channels in Pipe_Q, got {q.shape[1]}")
    return x, y, q


def cell_signed_area(x_cell: np.ndarray, y_cell: np.ndarray) -> float:
    return 0.5 * float(
        np.dot(x_cell, np.roll(y_cell, -1)) - np.dot(y_cell, np.roll(x_cell, -1))
    )


def cell_divergence_field(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> np.ndarray:
    h, w = x.shape
    div = np.empty((h - 1, w - 1), dtype=np.float64)
    for i in range(h - 1):
        for j in range(w - 1):
            vertices = (
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1),
            )
            x_cell = np.array([x[ii, jj] for ii, jj in vertices], dtype=np.float64)
            y_cell = np.array([y[ii, jj] for ii, jj in vertices], dtype=np.float64)
            area = abs(cell_signed_area(x_cell, y_cell))
            flux = 0.0
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                i0, j0 = start
                i1, j1 = end
                dx = float(x[i1, j1] - x[i0, j0])
                dy = float(y[i1, j1] - y[i0, j0])
                ux_mid = 0.5 * float(ux[i0, j0] + ux[i1, j1])
                uy_mid = 0.5 * float(uy[i0, j0] + uy[i1, j1])
                flux += ux_mid * dy - uy_mid * dx
            div[i, j] = flux / max(area, 1e-12)
    return div


def sample_divergence_stats(div: np.ndarray) -> dict[str, float]:
    abs_div = np.abs(div)
    return {
        "cell_div_abs_mean": float(abs_div.mean()),
        "cell_div_abs_median": float(np.median(abs_div)),
        "cell_div_abs_p95": float(np.percentile(abs_div, 95)),
        "cell_div_abs_max": float(abs_div.max()),
        "cell_div_signed_mean": float(div.mean()),
        "cell_div_signed_std": float(div.std()),
        "cell_div_l2": float(np.sqrt(np.mean(div**2))),
    }


def summarize_dataset(
    x_all: np.ndarray,
    y_all: np.ndarray,
    q_all: np.ndarray,
    *,
    sample_count: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    n = min(int(sample_count), int(q_all.shape[0]))
    rows: list[dict[str, object]] = []
    metrics = {
        "sample": np.arange(n, dtype=np.int64),
        "cell_div_abs_mean": np.empty(n, dtype=np.float64),
        "cell_div_abs_median": np.empty(n, dtype=np.float64),
        "cell_div_abs_p95": np.empty(n, dtype=np.float64),
        "cell_div_abs_max": np.empty(n, dtype=np.float64),
        "cell_div_signed_mean": np.empty(n, dtype=np.float64),
        "cell_div_signed_std": np.empty(n, dtype=np.float64),
        "cell_div_l2": np.empty(n, dtype=np.float64),
    }

    for sample_idx in range(n):
        div = cell_divergence_field(
            np.asarray(x_all[sample_idx], dtype=np.float64),
            np.asarray(y_all[sample_idx], dtype=np.float64),
            np.asarray(q_all[sample_idx, 0], dtype=np.float64),
            np.asarray(q_all[sample_idx, 1], dtype=np.float64),
        )
        stats = sample_divergence_stats(div)
        row = {"sample": sample_idx, **stats}
        rows.append(row)
        for key in metrics:
            if key == "sample":
                continue
            metrics[key][sample_idx] = float(stats[key])
    return rows, metrics


def print_sample_stats(sample_idx: int, stats: dict[str, float]) -> None:
    print(f"\nSample {sample_idx} finite-volume divergence stats")
    print(
        f"  abs_mean={stats['cell_div_abs_mean']: .6e}, "
        f"abs_median={stats['cell_div_abs_median']: .6e}, "
        f"abs_p95={stats['cell_div_abs_p95']: .6e}, "
        f"abs_max={stats['cell_div_abs_max']: .6e}"
    )
    print(
        f"  signed_mean={stats['cell_div_signed_mean']: .6e}, "
        f"signed_std={stats['cell_div_signed_std']: .6e}, "
        f"l2={stats['cell_div_l2']: .6e}"
    )


def print_dataset_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nDataset divergence summary over {len(rows)} sample(s)")
    for key in (
        "cell_div_abs_mean",
        "cell_div_abs_median",
        "cell_div_abs_p95",
        "cell_div_abs_max",
        "cell_div_signed_mean",
        "cell_div_signed_std",
        "cell_div_l2",
    ):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<22} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "cell_div_abs_mean",
        "cell_div_abs_median",
        "cell_div_abs_p95",
        "cell_div_abs_max",
        "cell_div_signed_mean",
        "cell_div_signed_std",
        "cell_div_l2",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_sample(
    sample_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    div: np.ndarray,
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)
    fig.suptitle(f"Pipe sample {sample_idx}: finite-volume divergence")

    centers_x = 0.25 * (x[:-1, :-1] + x[1:, :-1] + x[1:, 1:] + x[:-1, 1:])
    centers_y = 0.25 * (y[:-1, :-1] + y[1:, :-1] + y[1:, 1:] + y[:-1, 1:])
    abs_div = np.abs(div)

    im = axes[0].pcolormesh(centers_x, centers_y, div, shading="nearest", cmap="coolwarm")
    axes[0].set_title("signed divergence")
    axes[0].set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=axes[0])

    im = axes[1].pcolormesh(centers_x, centers_y, abs_div, shading="nearest", cmap="magma")
    axes[1].set_title("absolute divergence")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=axes[1])

    fig.tight_layout()
    out_path = out_dir / f"pipe_divergence_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_dataset_summary(metrics: dict[str, np.ndarray], *, out_dir: Path, show: bool) -> Path:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    sample = metrics["sample"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    fig.suptitle(f"Pipe dataset divergence summary over first {len(sample)} sample(s)")

    abs_ax = axes[0, 0]
    abs_ax.plot(sample, metrics["cell_div_abs_mean"], color="tab:blue", linewidth=1.1, label="mean |div|")
    abs_ax.plot(sample, metrics["cell_div_abs_p95"], color="tab:orange", linewidth=1.1, label="p95 |div|")
    abs_ax.set_title("samplewise absolute divergence")
    abs_ax.set_xlabel("sample index")
    abs_ax.set_ylabel("mean / p95 |div|")
    abs_ax.grid(True, alpha=0.25)

    abs_max_ax = abs_ax.twinx()
    abs_max_ax.plot(
        sample,
        metrics["cell_div_abs_max"],
        color="tab:red",
        linewidth=1.0,
        alpha=0.85,
        label="max |div|",
    )
    abs_max_ax.set_ylabel("max |div|")
    abs_max_ax.set_yscale("log")

    abs_handles, abs_labels = abs_ax.get_legend_handles_labels()
    abs_max_handles, abs_max_labels = abs_max_ax.get_legend_handles_labels()
    abs_ax.legend(
        abs_handles + abs_max_handles,
        abs_labels + abs_max_labels,
        loc="best",
        fontsize=8,
    )

    axes[0, 1].hist(metrics["cell_div_abs_mean"], bins=40, color="tab:blue", alpha=0.8)
    axes[0, 1].set_title("distribution of per-sample mean |div|")
    axes[0, 1].set_xlabel("mean |div|")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].hist(metrics["cell_div_abs_max"], bins=40, color="tab:red", alpha=0.8)
    axes[1, 0].set_title("distribution of per-sample max |div|")
    axes[1, 0].set_xlabel("max |div|")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(True, alpha=0.25)

    signed_ax = axes[1, 1]
    signed_ax.plot(
        sample,
        metrics["cell_div_signed_mean"],
        color="black",
        linewidth=1.0,
        label="signed mean",
    )
    signed_ax.axhline(0.0, color="0.3", linewidth=0.8)
    signed_ax.set_title("signed divergence diagnostics")
    signed_ax.set_xlabel("sample index")
    signed_ax.set_ylabel("signed mean(div)")
    signed_ax.grid(True, alpha=0.25)

    signed_std_ax = signed_ax.twinx()
    signed_std_ax.plot(
        sample,
        metrics["cell_div_signed_std"],
        color="tab:green",
        linewidth=1.0,
        alpha=0.85,
        label="signed std",
    )
    signed_std_ax.set_ylabel("std(div)")
    signed_std_ax.set_yscale("log")

    signed_handles, signed_labels = signed_ax.get_legend_handles_labels()
    signed_std_handles, signed_std_labels = signed_std_ax.get_legend_handles_labels()
    signed_ax.legend(
        signed_handles + signed_std_handles,
        signed_labels + signed_std_labels,
        loc="best",
        fontsize=8,
    )

    fig.tight_layout()
    out_path = out_dir / "pipe_divergence_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    x_all, y_all, q_all = load_pipe_arrays(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n = min(int(args.summary_samples), int(q_all.shape[0]))
    if n <= 0:
        raise ValueError("--summary-samples must be positive")

    print(
        "Loaded pipe data: "
        f"Pipe_X={x_all.shape}, Pipe_Y={y_all.shape}, Pipe_Q={q_all.shape}"
    )

    rows, metrics = summarize_dataset(x_all, y_all, q_all, sample_count=n)
    print_dataset_summary(rows)

    csv_path = args.out_dir / "pipe_divergence_summary.csv"
    write_summary_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= x_all.shape[0]:
            raise IndexError(f"Sample {sample_idx} is outside [0, {x_all.shape[0]})")
        div = cell_divergence_field(
            np.asarray(x_all[sample_idx], dtype=np.float64),
            np.asarray(y_all[sample_idx], dtype=np.float64),
            np.asarray(q_all[sample_idx, 0], dtype=np.float64),
            np.asarray(q_all[sample_idx, 1], dtype=np.float64),
        )
        print_sample_stats(sample_idx, sample_divergence_stats(div))
        if args.no_plot or plt is None:
            continue
        out_path = plot_sample(
            sample_idx,
            np.asarray(x_all[sample_idx], dtype=np.float64),
            np.asarray(y_all[sample_idx], dtype=np.float64),
            div,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"  wrote {out_path}")

    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    if not args.no_plot and plt is None:
        print("matplotlib is not installed; continuing with divergence statistics only.")
        return
    if args.no_plot:
        return

    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
