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
    darcy_residual,
    load_darcy_arrays,
    require_matplotlib,
    residual_stats,
    sample_count,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the discrete Darcy PDE residual div(-a grad u) - f."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=5)
    parser.add_argument("--downsampley", type=int, default=5)
    parser.add_argument("--force-value", type=float, default=1.0)
    parser.add_argument(
        "--method",
        choices=("finite_difference", "spectral"),
        default="finite_difference",
        help="Differentiate with NumPy finite differences or padded spectral derivatives.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Padding cells used by --method spectral before periodic differentiation.",
    )
    parser.add_argument(
        "--padding-mode",
        choices=("reflect", "replicate", "circular", "zeros"),
        default="reflect",
        help="Padding mode used by --method spectral.",
    )
    parser.add_argument(
        "--interior-crop",
        type=int,
        default=1,
        help="Ignore this many cells near each boundary when reporting residual stats.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/darcy_residual"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def summarize_residuals(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    sample_count_: int,
    force_value: float,
    interior_crop: int,
    method: str,
    padding: int,
    padding_mode: str,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metrics = {
        "sample": np.arange(sample_count_, dtype=np.int64),
        "res_mean_abs": np.empty(sample_count_, dtype=np.float64),
        "res_p95_abs": np.empty(sample_count_, dtype=np.float64),
        "res_max_abs": np.empty(sample_count_, dtype=np.float64),
        "res_signed_mean": np.empty(sample_count_, dtype=np.float64),
        "res_l2": np.empty(sample_count_, dtype=np.float64),
    }
    rows: list[dict[str, object]] = []
    for sample_idx in range(sample_count_):
        _residual_full, residual_eval, _flux_x, _flux_y = darcy_residual(
            coeff[sample_idx],
            sol[sample_idx],
            force_value=force_value,
            interior_crop=interior_crop,
            method=method,
            padding=padding,
            padding_mode=padding_mode,
        )
        stats = residual_stats(residual_eval)
        row = {"sample": sample_idx, **stats}
        rows.append(row)
        metrics["res_mean_abs"][sample_idx] = stats["mean_abs"]
        metrics["res_p95_abs"][sample_idx] = stats["p95_abs"]
        metrics["res_max_abs"][sample_idx] = stats["max_abs"]
        metrics["res_signed_mean"][sample_idx] = stats["signed_mean"]
        metrics["res_l2"][sample_idx] = stats["l2"]
    return rows, metrics


def print_summary(
    rows: list[dict[str, object]],
    *,
    interior_crop: int,
    method: str,
    padding: int,
    padding_mode: str,
) -> None:
    method_detail = f"method={method}"
    if method == "spectral":
        method_detail += f", padding={padding}, padding_mode={padding_mode}"
    print(
        f"\nDarcy PDE residual summary over {len(rows)} sample(s), "
        f"interior_crop={interior_crop}, {method_detail}"
    )
    for key in ("mean_abs", "median_abs", "p95_abs", "max_abs", "signed_mean", "l2"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<12} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )


def plot_sample(
    sample_idx: int,
    coeff: np.ndarray,
    sol: np.ndarray,
    residual: np.ndarray,
    flux_x: np.ndarray,
    flux_y: np.ndarray,
    *,
    method: str,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    flux_mag = np.sqrt(flux_x**2 + flux_y**2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle(f"Darcy sample {sample_idx}: PDE residual ({method})")

    im0 = axes[0, 0].imshow(coeff, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    axes[0, 0].set_title("permeability a")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(sol, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    axes[0, 1].set_title("solution u")
    fig.colorbar(im1, ax=axes[0, 1])

    im2 = axes[1, 0].imshow(residual, origin="lower", extent=(0, 1, 0, 1), cmap="coolwarm")
    axes[1, 0].set_title("div(-a grad u) - f")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(flux_mag, origin="lower", extent=(0, 1, 0, 1), cmap="plasma")
    axes[1, 1].set_title("|-a grad u|")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    out_path = out_dir / f"darcy_residual_{method}_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_dataset_summary(
    metrics: dict[str, np.ndarray],
    *,
    method: str,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    sample = metrics["sample"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), dpi=150)
    fig.suptitle(
        f"Darcy PDE residual summary over first {len(sample)} sample(s) ({method})"
    )

    axes[0, 0].plot(sample, metrics["res_mean_abs"], label="mean |res|")
    axes[0, 0].plot(sample, metrics["res_p95_abs"], label="p95 |res|")
    axes[0, 0].set_title("samplewise residual")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("absolute residual")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(metrics["res_mean_abs"], bins=40, color="tab:blue", alpha=0.85)
    axes[0, 1].set_title("distribution of mean |res|")
    axes[0, 1].set_xlabel("mean |res|")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].hist(metrics["res_max_abs"], bins=40, color="tab:red", alpha=0.85)
    axes[1, 0].set_title("distribution of max |res|")
    axes[1, 0].set_xlabel("max |res|")
    axes[1, 0].set_ylabel("count")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(sample, metrics["res_signed_mean"], color="black", label="signed mean")
    axes[1, 1].plot(sample, metrics["res_l2"], color="tab:green", label="L2")
    axes[1, 1].axhline(0.0, color="0.35", linewidth=0.8)
    axes[1, 1].set_title("signed mean and L2")
    axes[1, 1].set_xlabel("sample")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"darcy_residual_{method}_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    if args.padding < 0:
        raise ValueError("--padding must be non-negative")
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    validate_samples(args.samples, int(sol.shape[0]))
    n = sample_count(args.summary_samples, int(sol.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded Darcy data: "
        f"path={mat_path}, coeff={coeff.shape}, sol={sol.shape}, split={args.split}"
    )
    rows, metrics = summarize_residuals(
        coeff,
        sol,
        sample_count_=n,
        force_value=args.force_value,
        interior_crop=args.interior_crop,
        method=args.method,
        padding=args.padding,
        padding_mode=args.padding_mode,
    )
    print_summary(
        rows,
        interior_crop=args.interior_crop,
        method=args.method,
        padding=args.padding,
        padding_mode=args.padding_mode,
    )
    csv_path = args.out_dir / f"darcy_residual_{args.method}_summary.csv"
    write_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    if args.no_plot:
        return
    if plt is None:
        print("matplotlib is not installed; continuing with CSV/statistics only.")
        return
    for sample_idx in args.samples:
        residual_full, _residual_eval, flux_x, flux_y = darcy_residual(
            coeff[sample_idx],
            sol[sample_idx],
            force_value=args.force_value,
            interior_crop=args.interior_crop,
            method=args.method,
            padding=args.padding,
            padding_mode=args.padding_mode,
        )
        out_path = plot_sample(
            sample_idx,
            coeff[sample_idx],
            sol[sample_idx],
            residual_full,
            flux_x,
            flux_y,
            method=args.method,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(
        metrics,
        method=args.method,
        out_dir=args.out_dir,
        show=args.show,
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
