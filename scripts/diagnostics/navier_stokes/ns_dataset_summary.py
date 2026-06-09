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
    load_ns_vorticity,
    require_matplotlib,
    sample_count,
    scalar_stats,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset-wide Navier-Stokes vorticity summary plots and CSV diagnostics."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/NavierStokes_V1e-5_N1200_T20"))
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=1)
    parser.add_argument("--downsampley", type=int, default=1)
    parser.add_argument(
        "--snapshot-timesteps",
        type=int,
        nargs="+",
        default=[0, 9, 19],
        help="Timesteps shown in the per-sample vorticity snapshots.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/navier_stokes/ns_dataset_summary"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def summarize_dataset(
    u: np.ndarray,
    *,
    n_samples: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    metrics = {
        "sample": np.arange(n_samples, dtype=np.int64),
        "vort_mean": np.empty(n_samples, dtype=np.float64),
        "vort_std": np.empty(n_samples, dtype=np.float64),
        "vort_min": np.empty(n_samples, dtype=np.float64),
        "vort_max": np.empty(n_samples, dtype=np.float64),
        "vort_mean_abs": np.empty(n_samples, dtype=np.float64),
        "vort_l2": np.empty(n_samples, dtype=np.float64),
    }
    rows: list[dict[str, object]] = []
    for idx in range(n_samples):
        stats = scalar_stats(u[idx])
        row = {
            "sample": idx,
            "vort_mean": stats["mean"],
            "vort_std": stats["std"],
            "vort_min": stats["min"],
            "vort_max": stats["max"],
            "vort_mean_abs": stats["mean_abs"],
            "vort_l2": stats["l2"],
        }
        rows.append(row)
        for key in metrics:
            if key != "sample":
                metrics[key][idx] = float(row[key])
    return rows, metrics


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nNavier-Stokes vorticity summary over {len(rows)} sample(s)")
    for key in ("vort_mean", "vort_std", "vort_min", "vort_max", "vort_mean_abs", "vort_l2"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<14} "
            f"mean={values.mean(): .6e}, "
            f"p05={np.percentile(values, 5): .6e}, "
            f"p95={np.percentile(values, 95): .6e}"
        )


def plot_dataset_summary(metrics: dict[str, np.ndarray], *, out_dir: Path, show: bool) -> Path:
    require_matplotlib(plt)
    sample = metrics["sample"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), dpi=150)
    fig.suptitle(f"Navier-Stokes vorticity summary over first {len(sample)} sample(s)")

    axes[0, 0].plot(sample, metrics["vort_mean"], linewidth=1.0, label="mean")
    axes[0, 0].fill_between(sample, metrics["vort_min"], metrics["vort_max"], alpha=0.2, label="min-max")
    axes[0, 0].set_title("vorticity range by sample")
    axes[0, 0].set_xlabel("sample")
    axes[0, 0].set_ylabel("vorticity")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].hist(metrics["vort_std"], bins=40, color="tab:blue", alpha=0.85)
    axes[0, 1].set_title("distribution of vorticity std")
    axes[0, 1].set_xlabel("std")
    axes[0, 1].set_ylabel("count")
    axes[0, 1].grid(True, alpha=0.25)

    axes[1, 0].plot(sample, metrics["vort_l2"], color="tab:purple", linewidth=1.0)
    axes[1, 0].set_title("vorticity RMS by sample")
    axes[1, 0].set_xlabel("sample")
    axes[1, 0].set_ylabel("RMS")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].hist(metrics["vort_l2"], bins=40, color="tab:purple", alpha=0.85)
    axes[1, 1].set_title("distribution of vorticity RMS")
    axes[1, 1].set_xlabel("RMS")
    axes[1, 1].set_ylabel("count")
    axes[1, 1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "ns_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_sample(
    sample_idx: int,
    u_sample: np.ndarray,
    *,
    timesteps: list[int],
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    t_total = int(u_sample.shape[-1])
    steps = [t for t in timesteps if 0 <= t < t_total]
    if not steps:
        steps = [0, t_total // 2, t_total - 1]
    vmax = float(np.abs(u_sample).max())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4.0), dpi=150, squeeze=False)
    fig.suptitle(f"NS sample {sample_idx}: vorticity field")
    for ax, t in zip(axes[0], steps):
        im = ax.imshow(u_sample[:, :, t], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"t={t}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = out_dir / f"ns_vorticity_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    u, mat_path = load_ns_vorticity(
        args.data_dir,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    validate_samples(args.samples, int(u.shape[0]))
    n = sample_count(args.summary_samples, int(u.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded Navier-Stokes vorticity: path={mat_path}, u={u.shape}  # (N, H, W, T)")
    rows, metrics = summarize_dataset(u, n_samples=n)
    print_summary(rows)
    csv_path = args.out_dir / "ns_dataset_summary.csv"
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
            np.asarray(u[sample_idx], dtype=np.float64),
            timesteps=args.snapshot_timesteps,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
