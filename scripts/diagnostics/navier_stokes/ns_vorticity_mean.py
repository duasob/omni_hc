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
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the zero global vorticity-mean condition on the Navier-Stokes "
            "ground-truth data: the spatial mean of the vorticity field should "
            "vanish for every sample and timestep."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/NavierStokes_V1e-5_N1200_T20"))
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=1)
    parser.add_argument("--downsampley", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/navier_stokes/ns_vorticity_mean"))
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def spatial_mean_per_timestep(u_sample: np.ndarray) -> np.ndarray:
    """Spatial mean of vorticity for one sample, per timestep -> shape (T,)."""
    return u_sample.mean(axis=(0, 1))


def summarize_dataset(
    u: np.ndarray,
    *,
    n_samples: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    rows: list[dict[str, object]] = []
    metrics = {
        "sample": np.arange(n_samples, dtype=np.int64),
        "vort_abs_mean": np.empty(n_samples, dtype=np.float64),
        "vort_abs_mean_max_t": np.empty(n_samples, dtype=np.float64),
        "field_abs_scale": np.empty(n_samples, dtype=np.float64),
        "normalised_abs_mean": np.empty(n_samples, dtype=np.float64),
    }
    for idx in range(n_samples):
        u_sample = np.asarray(u[idx], dtype=np.float64)  # (H, W, T)
        per_t_mean = spatial_mean_per_timestep(u_sample)  # (T,)
        abs_mean = np.abs(per_t_mean)
        field_scale = float(np.abs(u_sample).mean())
        row = {
            "sample": idx,
            "vort_abs_mean": float(abs_mean.mean()),
            "vort_abs_mean_max_t": float(abs_mean.max()),
            "field_abs_scale": field_scale,
            "normalised_abs_mean": float(abs_mean.mean() / max(field_scale, 1e-12)),
        }
        rows.append(row)
        for key in metrics:
            if key != "sample":
                metrics[key][idx] = float(row[key])
    return rows, metrics


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nNS global vorticity-mean summary over {len(rows)} sample(s)")
    print("  (|w_bar| is the spatial mean of vorticity; field_abs_scale is mean |w| for context)")
    for key in ("vort_abs_mean", "vort_abs_mean_max_t", "field_abs_scale", "normalised_abs_mean"):
        values = np.array([float(row[key]) for row in rows], dtype=np.float64)
        print(
            f"  {key:<22} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )


def plot_dataset_summary(metrics: dict[str, np.ndarray], *, out_dir: Path, show: bool) -> Path:
    require_matplotlib(plt)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=150)
    fig.suptitle(f"NS ground-truth |vorticity mean| over {len(metrics['sample'])} sample(s)")

    axes[0].hist(metrics["vort_abs_mean"], bins=40, color="tab:blue", alpha=0.85)
    axes[0].set_title("per-sample mean |w_bar|")
    axes[0].set_xlabel("|w_bar|")
    axes[0].set_ylabel("count")
    axes[0].grid(True, alpha=0.25)

    axes[1].hist(metrics["normalised_abs_mean"], bins=40, color="tab:green", alpha=0.85)
    axes[1].set_title("|w_bar| relative to mean |w|")
    axes[1].set_xlabel("|w_bar| / mean|w|")
    axes[1].set_ylabel("count")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    out_path = out_dir / "ns_vorticity_mean_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_sample(sample_idx: int, u_sample: np.ndarray, *, out_dir: Path, show: bool) -> Path:
    require_matplotlib(plt)
    per_t_mean = spatial_mean_per_timestep(u_sample)
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.4), dpi=150)
    ax.plot(np.arange(per_t_mean.size), per_t_mean, marker="o", linewidth=1.0)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_title(f"NS sample {sample_idx}: spatial vorticity mean by timestep")
    ax.set_xlabel("timestep")
    ax.set_ylabel("w_bar")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"ns_vorticity_mean_sample_{sample_idx:04d}.png"
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
    csv_path = args.out_dir / "ns_vorticity_mean_summary.csv"
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
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")
    summary_path = plot_dataset_summary(metrics, out_dir=args.out_dir, show=args.show)
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
