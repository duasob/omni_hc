from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from _common import (
    load_elasticity_arrays,
    require_matplotlib,
    sample_count,
    scalar_stats,
    select_split,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Elasticity point-cloud stress summary plots and CSV diagnostics."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/elasticity"),
        help=(
            "Directory containing the elasticity files. Accepts data/elasticity, "
            "data/fno, or a directory containing the .npy files directly."
        ),
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/elasticity_dataset_summary"),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=28.0,
        help="Marker size for point-cloud sample plots.",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def summarize_dataset(
    coords: np.ndarray,
    sigma: np.ndarray,
    *,
    sample_count_: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metrics = {
        "sample": np.arange(sample_count_, dtype=np.int64),
        "sigma_mean": np.empty(sample_count_, dtype=np.float64),
        "sigma_std": np.empty(sample_count_, dtype=np.float64),
        "sigma_min": np.empty(sample_count_, dtype=np.float64),
        "sigma_max": np.empty(sample_count_, dtype=np.float64),
        "sigma_mean_abs": np.empty(sample_count_, dtype=np.float64),
        "sigma_l2": np.empty(sample_count_, dtype=np.float64),
        "x_min": np.empty(sample_count_, dtype=np.float64),
        "x_max": np.empty(sample_count_, dtype=np.float64),
        "y_min": np.empty(sample_count_, dtype=np.float64),
        "y_max": np.empty(sample_count_, dtype=np.float64),
    }
    rows: list[dict[str, object]] = []
    for sample_idx in range(sample_count_):
        stats = scalar_stats(sigma[sample_idx])
        xy = np.asarray(coords[sample_idx], dtype=np.float64)
        row = {
            "sample": sample_idx,
            "sigma_mean": stats["mean"],
            "sigma_std": stats["std"],
            "sigma_min": stats["min"],
            "sigma_max": stats["max"],
            "sigma_mean_abs": stats["mean_abs"],
            "sigma_l2": stats["l2"],
            "x_min": float(xy[:, 0].min()),
            "x_max": float(xy[:, 0].max()),
            "y_min": float(xy[:, 1].min()),
            "y_max": float(xy[:, 1].max()),
        }
        rows.append(row)
        for key in metrics:
            if key != "sample":
                metrics[key][sample_idx] = float(row[key])
    return rows, metrics


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nElasticity dataset summary over {len(rows)} sample(s)")
    for key in (
        "sigma_mean",
        "sigma_std",
        "sigma_min",
        "sigma_max",
        "sigma_mean_abs",
        "sigma_l2",
    ):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<15} "
            f"mean={values.mean(): .6e}, "
            f"p05={np.percentile(values, 5): .6e}, "
            f"p95={np.percentile(values, 95): .6e}"
        )


def plot_sample(
    sample_idx: int,
    coords: np.ndarray,
    sigma: np.ndarray,
    *,
    out_dir: Path,
    point_size: float,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6), dpi=150)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=sigma,
        s=float(point_size),
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title(f"Elasticity {sample_idx}: stress sigma")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    fig.colorbar(scatter, ax=ax, label="sigma")
    fig.tight_layout()
    out_path = out_dir / f"elasticity_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_dataset_summary(
    metrics: dict[str, np.ndarray],
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    sample = metrics["sample"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=150)
    fig.suptitle(f"Elasticity stress summary over first {len(sample)} sample(s)")

    axes[0, 0].plot(sample, metrics["sigma_mean"], linewidth=1.0, label="mean")
    axes[0, 0].fill_between(
        sample,
        metrics["sigma_min"],
        metrics["sigma_max"],
        alpha=0.2,
        label="min-max",
    )
    axes[0, 0].set_title("stress range by sample")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("sigma")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(metrics["sigma_mean"], bins=40, color="tab:blue", alpha=0.85)
    axes[0, 1].set_title("distribution of mean stress")
    axes[0, 1].set_xlabel("mean sigma")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(sample, metrics["sigma_l2"], color="tab:purple", linewidth=1.0)
    axes[1, 0].set_title("stress RMS by sample")
    axes[1, 0].set_xlabel("sample")
    axes[1, 0].set_ylabel("RMS sigma")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(metrics["sigma_l2"], bins=40, color="tab:purple", alpha=0.85)
    axes[1, 1].set_title("distribution of stress RMS")
    axes[1, 1].set_xlabel("RMS sigma")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "elasticity_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

    coords, sigma, sigma_path, xy_path = load_elasticity_arrays(args.data_dir)
    coords, sigma = select_split(
        coords,
        sigma,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    validate_samples(args.samples, int(sigma.shape[0]))
    n = sample_count(args.summary_samples, int(sigma.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded Elasticity data: "
        f"sigma_path={sigma_path}, xy_path={xy_path}, "
        f"coords={coords.shape}, sigma={sigma.shape}, split={args.split}"
    )
    rows, metrics = summarize_dataset(coords, sigma, sample_count_=n)
    print_summary(rows)
    csv_path = args.out_dir / "elasticity_dataset_summary.csv"
    write_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    if args.no_plot:
        return
    if plt is None:
        print("matplotlib is not installed; continuing with CSV/statistics only.")
        return
    for sample_idx in args.samples:
        out_path = plot_sample(
            sample_idx,
            coords[sample_idx],
            sigma[sample_idx],
            out_dir=args.out_dir,
            point_size=args.point_size,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
