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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check whether Pipe ux/uy are consistent with a 2D stream function."
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
        default=[0, 10, 100, 110],
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
        default=Path("artifacts/pipe_stream_function"),
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
        help="Only print and write statistics; do not write figures.",
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
    if q.ndim != 4 or q.shape[1] < 2 or q.shape[0] != x.shape[0] or q.shape[2:] != x.shape[1:]:
        raise ValueError(f"Expected Pipe_Q=(N,C,H,W) matching X/Y; got {q.shape}")
    return x, y, q


def reconstruct_stream_from_ux(y: np.ndarray, ux: np.ndarray) -> np.ndarray:
    psi = np.zeros_like(ux, dtype=np.float64)
    dy = np.diff(y, axis=1)
    ux_mid = 0.5 * (ux[:, 1:] + ux[:, :-1])
    psi[:, 1:] = np.cumsum(ux_mid * dy, axis=1)
    return psi


def stream_velocity_curvilinear(
    x: np.ndarray,
    y: np.ndarray,
    psi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_s = np.gradient(x, axis=0, edge_order=2)
    x_t = np.gradient(x, axis=1, edge_order=2)
    y_s = np.gradient(y, axis=0, edge_order=2)
    y_t = np.gradient(y, axis=1, edge_order=2)
    psi_s = np.gradient(psi, axis=0, edge_order=2)
    psi_t = np.gradient(psi, axis=1, edge_order=2)
    jac = x_s * y_t - x_t * y_s
    safe_jac = np.where(np.abs(jac) < 1e-12, np.sign(jac) * 1e-12 + (jac == 0) * 1e-12, jac)
    ux = (-x_t * psi_s + x_s * psi_t) / safe_jac
    uy = (y_s * psi_t - y_t * psi_s) / safe_jac
    return ux, uy, jac


def relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(target.reshape(-1))), 1e-12)
    return float(np.linalg.norm((pred - target).reshape(-1)) / denom)


def abs_stats(error: np.ndarray) -> tuple[float, float]:
    abs_error = np.abs(error)
    return float(abs_error.mean()), float(abs_error.max())


def sample_stats(x, y, ux, uy) -> dict[str, object]:
    psi = reconstruct_stream_from_ux(y, ux)
    ux_rec, uy_rec, jac = stream_velocity_curvilinear(x, y, psi)
    ux_err = ux_rec - ux
    uy_err = uy_rec - uy
    ux_abs_mean, ux_abs_max = abs_stats(ux_err)
    uy_abs_mean, uy_abs_max = abs_stats(uy_err)
    flux = psi[:, -1] - psi[:, 0]
    return {
        "psi": psi,
        "ux_rec": ux_rec,
        "uy_rec": uy_rec,
        "jac": jac,
        "ux_rel_l2": relative_l2(ux_rec, ux),
        "uy_rel_l2": relative_l2(uy_rec, uy),
        "ux_abs_mean": ux_abs_mean,
        "ux_abs_max": ux_abs_max,
        "uy_abs_mean": uy_abs_mean,
        "uy_abs_max": uy_abs_max,
        "psi_wall_lower_abs_max": float(np.max(np.abs(psi[:, 0]))),
        "psi_wall_upper_drift_abs_max": float(np.max(np.abs(flux - flux[0]))),
        "psi_flux_inlet": float(flux[0]),
        "psi_flux_outlet": float(flux[-1]),
        "jac_min": float(jac.min()),
        "jac_max": float(jac.max()),
    }


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "ux_rel_l2",
        "uy_rel_l2",
        "ux_abs_mean",
        "ux_abs_max",
        "uy_abs_mean",
        "uy_abs_max",
        "psi_wall_lower_abs_max",
        "psi_wall_upper_drift_abs_max",
        "psi_flux_inlet",
        "psi_flux_outlet",
        "jac_min",
        "jac_max",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row[name]) for name in fieldnames) + "\n")


def summarize(rows: list[dict[str, object]]) -> None:
    print(f"Aggregate stream-function consistency over {len(rows)} sample(s)")
    for key in (
        "ux_rel_l2",
        "uy_rel_l2",
        "ux_abs_mean",
        "uy_abs_mean",
        "psi_wall_upper_drift_abs_max",
    ):
        values = np.array([float(row[key]) for row in rows])
        print(
            f"  {key:<30} "
            f"mean={values.mean(): .6e}, "
            f"p50={np.percentile(values, 50): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )
    worst = sorted(rows, key=lambda row: float(row["uy_rel_l2"]), reverse=True)[:5]
    print("  worst uy reconstruction samples:")
    for row in worst:
        print(
            f"    sample={int(row['sample']):<4} "
            f"uy_rel_l2={float(row['uy_rel_l2']):.3e} "
            f"ux_rel_l2={float(row['ux_rel_l2']):.3e} "
            f"flux_drift={float(row['psi_wall_upper_drift_abs_max']):.3e}"
        )


def print_sample(sample_idx: int, stats: dict[str, object]) -> None:
    print(f"\nSample {sample_idx} stream-function consistency")
    print(
        f"  ux_rec vs ux: rel_l2={stats['ux_rel_l2']: .6e}, "
        f"abs_mean={stats['ux_abs_mean']: .6e}, abs_max={stats['ux_abs_max']: .6e}"
    )
    print(
        f"  uy_rec vs uy: rel_l2={stats['uy_rel_l2']: .6e}, "
        f"abs_mean={stats['uy_abs_mean']: .6e}, abs_max={stats['uy_abs_max']: .6e}"
    )
    print(
        f"  psi upper-wall flux drift max={stats['psi_wall_upper_drift_abs_max']: .6e}, "
        f"inlet={stats['psi_flux_inlet']: .6e}, outlet={stats['psi_flux_outlet']: .6e}"
    )
    print(f"  jac min={stats['jac_min']: .6e}, jac max={stats['jac_max']: .6e}")


def plot_sample(out_dir: Path, sample_idx: int, x, y, ux, uy, stats, *, show: bool) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots. Rerun with --no-plot.")
    ux_rec = stats["ux_rec"]
    uy_rec = stats["uy_rec"]
    fields = [
        ("ux target", ux, "viridis"),
        ("ux from psi", ux_rec, "viridis"),
        ("ux error", ux_rec - ux, "coolwarm"),
        ("uy target", uy, "viridis"),
        ("uy from psi", uy_rec, "viridis"),
        ("uy error", uy_rec - uy, "coolwarm"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), dpi=130)
    for ax, (title, field, cmap) in zip(axes.reshape(-1), fields):
        vmax = None
        vmin = None
        if "error" in title:
            scale = max(float(np.abs(field).max()), 1e-12)
            vmin, vmax = -scale, scale
        mesh = ax.pcolormesh(x, y, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Pipe stream-function reconstruction, sample {sample_idx}")
    fig.tight_layout()
    out_path = out_dir / f"pipe_stream_function_sample_{sample_idx:04d}.png"
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    x_all, y_all, q_all = load_pipe_arrays(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_summary = min(max(int(args.summary_samples), 0), int(q_all.shape[0]))
    rows: list[dict[str, object]] = []
    for sample_idx in range(n_summary):
        stats = sample_stats(
            np.asarray(x_all[sample_idx], dtype=np.float64),
            np.asarray(y_all[sample_idx], dtype=np.float64),
            np.asarray(q_all[sample_idx, 0], dtype=np.float64),
            np.asarray(q_all[sample_idx, 1], dtype=np.float64),
        )
        rows.append(
            {
                key: value
                for key, value in ({"sample": sample_idx} | stats).items()
                if not isinstance(value, np.ndarray)
            }
        )

    if rows:
        summarize(rows)
        csv_path = args.out_dir / "pipe_stream_function_summary.csv"
        write_summary_csv(csv_path, rows)
        print(f"wrote {csv_path}")

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= q_all.shape[0]:
            raise IndexError(f"Sample {sample_idx} is outside [0, {q_all.shape[0]})")
        x = np.asarray(x_all[sample_idx], dtype=np.float64)
        y = np.asarray(y_all[sample_idx], dtype=np.float64)
        ux = np.asarray(q_all[sample_idx, 0], dtype=np.float64)
        uy = np.asarray(q_all[sample_idx, 1], dtype=np.float64)
        stats = sample_stats(x, y, ux, uy)
        print_sample(sample_idx, stats)
        if args.no_plot or plt is None:
            continue
        out_path = plot_sample(args.out_dir, sample_idx, x, y, ux, uy, stats, show=args.show)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
