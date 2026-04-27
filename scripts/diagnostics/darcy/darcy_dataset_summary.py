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
    load_darcy_arrays,
    require_matplotlib,
    sample_count,
    scalar_stats,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset-wide Darcy coefficient and solution summary plots."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=5)
    parser.add_argument("--downsampley", type=int, default=5)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/darcy_dataset_summary"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def summarize_dataset(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    sample_count_: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metrics = {
        "sample": np.arange(sample_count_, dtype=np.int64),
        "coeff_mean": np.empty(sample_count_, dtype=np.float64),
        "coeff_std": np.empty(sample_count_, dtype=np.float64),
        "coeff_min": np.empty(sample_count_, dtype=np.float64),
        "coeff_max": np.empty(sample_count_, dtype=np.float64),
        "sol_mean": np.empty(sample_count_, dtype=np.float64),
        "sol_std": np.empty(sample_count_, dtype=np.float64),
        "sol_min": np.empty(sample_count_, dtype=np.float64),
        "sol_max": np.empty(sample_count_, dtype=np.float64),
        "sol_l2": np.empty(sample_count_, dtype=np.float64),
    }
    rows: list[dict[str, object]] = []
    for sample_idx in range(sample_count_):
        coeff_stats = scalar_stats(coeff[sample_idx])
        sol_stats = scalar_stats(sol[sample_idx])
        row = {
            "sample": sample_idx,
            "coeff_mean": coeff_stats["mean"],
            "coeff_std": coeff_stats["std"],
            "coeff_min": coeff_stats["min"],
            "coeff_max": coeff_stats["max"],
            "sol_mean": sol_stats["mean"],
            "sol_std": sol_stats["std"],
            "sol_min": sol_stats["min"],
            "sol_max": sol_stats["max"],
            "sol_l2": sol_stats["l2"],
        }
        rows.append(row)
        for key in metrics:
            if key != "sample":
                metrics[key][sample_idx] = float(row[key])
    return rows, metrics


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nDarcy dataset summary over {len(rows)} sample(s)")
    for key in (
        "coeff_mean",
        "coeff_std",
        "coeff_min",
        "coeff_max",
        "sol_mean",
        "sol_std",
        "sol_min",
        "sol_max",
        "sol_l2",
    ):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<12} "
            f"mean={values.mean(): .6e}, "
            f"p05={np.percentile(values, 5): .6e}, "
            f"p95={np.percentile(values, 95): .6e}"
        )


def plot_sample(
    sample_idx: int,
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.suptitle(f"Darcy {sample_idx}: permeability and solution")

    im0 = axes[0].imshow(coeff, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    axes[0].set_title("permeability a(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(sol, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    axes[1].set_title("solution u(x,y)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    out_path = out_dir / f"darcy_sample_{sample_idx:04d}.png"
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
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=150)
    fig.suptitle(f"Darcy dataset summary over first {len(sample)} sample(s)")

    axes[0, 0].plot(sample, metrics["coeff_mean"], linewidth=1.0, label="mean")
    axes[0, 0].fill_between(
        sample,
        metrics["coeff_min"],
        metrics["coeff_max"],
        alpha=0.2,
        label="min-max",
    )
    axes[0, 0].set_title("permeability by sample")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("a")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(metrics["coeff_mean"], bins=40, color="tab:blue", alpha=0.85)
    axes[0, 1].set_title("distribution of mean permeability")
    axes[0, 1].set_xlabel("mean a")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(sample, metrics["sol_l2"], color="tab:purple", linewidth=1.0)
    axes[1, 0].set_title("solution RMS by sample")
    axes[1, 0].set_xlabel("sample")
    axes[1, 0].set_ylabel("RMS u")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(metrics["sol_l2"], bins=40, color="tab:purple", alpha=0.85)
    axes[1, 1].set_title("distribution of solution RMS")
    axes[1, 1].set_xlabel("RMS u")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "darcy_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    validate_samples(args.samples, int(coeff.shape[0]))
    n = sample_count(args.summary_samples, int(coeff.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded Darcy data: "
        f"path={mat_path}, coeff={coeff.shape}, sol={sol.shape}, split={args.split}"
    )
    rows, metrics = summarize_dataset(coeff, sol, sample_count_=n)
    print_summary(rows)
    csv_path = args.out_dir / "darcy_dataset_summary.csv"
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
            coeff[sample_idx],
            sol[sample_idx],
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
