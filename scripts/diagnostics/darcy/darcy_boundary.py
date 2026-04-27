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
    EDGES,
    boundary_mask,
    edge_values,
    load_darcy_arrays,
    require_matplotlib,
    residual_stats,
    sample_count,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Darcy solution boundary values for the zero Dirichlet condition."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=5)
    parser.add_argument("--downsampley", type=int, default=5)
    parser.add_argument("--boundary-value", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/darcy_boundary"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def summarize_boundaries(
    sol: np.ndarray,
    *,
    sample_count_: int,
    boundary_value: float,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    mask = boundary_mask(sol.shape[1:])
    metrics = {
        "sample": np.arange(sample_count_, dtype=np.int64),
        "boundary_mean_abs": np.empty(sample_count_, dtype=np.float64),
        "boundary_p95_abs": np.empty(sample_count_, dtype=np.float64),
        "boundary_max_abs": np.empty(sample_count_, dtype=np.float64),
        "boundary_signed_mean": np.empty(sample_count_, dtype=np.float64),
    }
    rows: list[dict[str, object]] = []
    for sample_idx in range(sample_count_):
        residual = sol[sample_idx] - float(boundary_value)
        boundary = residual[mask]
        stats = residual_stats(boundary)
        row = {
            "sample": sample_idx,
            "edge": "all",
            **stats,
        }
        rows.append(row)
        metrics["boundary_mean_abs"][sample_idx] = stats["mean_abs"]
        metrics["boundary_p95_abs"][sample_idx] = stats["p95_abs"]
        metrics["boundary_max_abs"][sample_idx] = stats["max_abs"]
        metrics["boundary_signed_mean"][sample_idx] = stats["signed_mean"]

        for edge in EDGES:
            edge_residual = edge_values(residual, edge)
            rows.append({"sample": sample_idx, "edge": edge, **residual_stats(edge_residual)})
    return rows, metrics


def print_summary(rows: list[dict[str, object]]) -> None:
    all_rows = [row for row in rows if row["edge"] == "all"]
    print(f"\nDarcy boundary summary over {len(all_rows)} sample(s)")
    for key in ("mean_abs", "median_abs", "p95_abs", "max_abs", "signed_mean", "l2"):
        values = np.array([float(row[key]) for row in all_rows], dtype=np.float64)
        print(
            f"  {key:<12} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )
    print("Per-edge max over summary samples")
    for edge in EDGES:
        values = np.array(
            [float(row["max_abs"]) for row in rows if row["edge"] == edge],
            dtype=np.float64,
        )
        print(f"  {edge:<10} max_abs={values.max(): .6e}, mean_abs={values.mean(): .6e}")


def plot_sample(
    sample_idx: int,
    sol: np.ndarray,
    *,
    boundary_value: float,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    residual = np.abs(sol - float(boundary_value))
    mask = boundary_mask(sol.shape)
    boundary_residual = np.full_like(residual, np.nan, dtype=np.float64)
    boundary_residual[mask] = residual[mask]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    fig.suptitle(f"Darcy sample {sample_idx}: boundary residual")

    im0 = axes[0].imshow(sol, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    axes[0].set_title("solution u(x,y)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        boundary_residual,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="viridis",
    )
    axes[1].set_title("|u - boundary value| on boundary")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    out_path = out_dir / f"darcy_boundary_sample_{sample_idx:04d}.png"
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
    fig.suptitle(f"Darcy boundary summary over first {len(sample)} sample(s)")

    axes[0, 0].plot(sample, metrics["boundary_mean_abs"], label="mean |res|")
    axes[0, 0].plot(sample, metrics["boundary_p95_abs"], label="p95 |res|")
    axes[0, 0].set_title("samplewise boundary residual")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("absolute residual")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(metrics["boundary_mean_abs"], bins=40, color="tab:blue", alpha=0.85)
    axes[0, 1].set_title("distribution of mean boundary residual")
    axes[0, 1].set_xlabel("mean |res|")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].hist(metrics["boundary_max_abs"], bins=40, color="tab:red", alpha=0.85)
    axes[1, 0].set_title("distribution of max boundary residual")
    axes[1, 0].set_xlabel("max |res|")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(sample, metrics["boundary_signed_mean"], color="black")
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].set_title("signed mean boundary residual")
    axes[1, 1].set_xlabel("sample")
    axes[1, 1].set_ylabel("signed mean")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "darcy_boundary_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    _coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    validate_samples(args.samples, int(sol.shape[0]))
    n = sample_count(args.summary_samples, int(sol.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded Darcy solutions: path={mat_path}, sol={sol.shape}, split={args.split}")
    rows, metrics = summarize_boundaries(
        sol,
        sample_count_=n,
        boundary_value=args.boundary_value,
    )
    print_summary(rows)
    csv_path = args.out_dir / "darcy_boundary_summary.csv"
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
            sol[sample_idx],
            boundary_value=args.boundary_value,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
