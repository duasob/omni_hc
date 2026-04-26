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


CHANNELS = ("ux", "uy", "p")
EDGES = {
    "inlet_i0": (0, slice(None)),
    "outlet_iN": (-1, slice(None)),
    "lower_wall_j0": (slice(None), 0),
    "upper_wall_jN": (slice(None), -1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect Pipe benchmark boundary geometry and Q values."
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
        default=[0],
        help="Sample indices to visualize.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/pipe_boundary"),
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=8,
        help="Subsampling stride for the mesh overlay.",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=1000,
        help="Number of leading samples to summarize for boundary Q values.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print boundary statistics; do not write figures.",
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
        raise ValueError(
            "Expected Pipe_Q shape (N, C, H, W) matching Pipe_X/Pipe_Y; "
            f"got Pipe_X={x.shape}, Pipe_Q={q.shape}"
        )
    if q.shape[1] < len(CHANNELS):
        raise ValueError(f"Expected at least {len(CHANNELS)} Q channels, got {q.shape[1]}")
    return x, y, q


def boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def edge_xy(x: np.ndarray, y: np.ndarray, edge: str) -> tuple[np.ndarray, np.ndarray]:
    selector = EDGES[edge]
    return np.asarray(x[selector]), np.asarray(y[selector])


def edge_q(q_channel: np.ndarray, edge: str) -> np.ndarray:
    return np.asarray(q_channel[EDGES[edge]])


def print_sample_stats(sample_idx: int, q_sample: np.ndarray) -> None:
    mask = boundary_mask(q_sample.shape[1:])
    print(f"\nSample {sample_idx} boundary Q stats")
    for channel_idx, name in enumerate(CHANNELS):
        values = np.asarray(q_sample[channel_idx])
        boundary_values = values[mask]
        print(
            f"  {name:>2} boundary overall: "
            f"mean={boundary_values.mean(): .6e}, "
            f"mean_abs={np.abs(boundary_values).mean(): .6e}, "
            f"max_abs={np.abs(boundary_values).max(): .6e}"
        )
        for edge in EDGES:
            edge_values = edge_q(values, edge)
            print(
                f"    {edge:<14} "
                f"mean={edge_values.mean(): .6e}, "
                f"mean_abs={np.abs(edge_values).mean(): .6e}, "
                f"max_abs={np.abs(edge_values).max(): .6e}"
            )


def print_dataset_summary(q: np.ndarray, sample_count: int) -> None:
    n = min(int(sample_count), int(q.shape[0]))
    if n <= 0:
        return

    mask = boundary_mask(q.shape[2:])
    print(f"Boundary summary over first {n} sample(s)")
    for channel_idx, name in enumerate(CHANNELS):
        max_abs_by_sample = np.empty(n, dtype=np.float64)
        mean_abs_by_sample = np.empty(n, dtype=np.float64)
        for sample_idx in range(n):
            values = np.asarray(q[sample_idx, channel_idx])
            boundary_values = values[mask]
            max_abs_by_sample[sample_idx] = np.abs(boundary_values).max()
            mean_abs_by_sample[sample_idx] = np.abs(boundary_values).mean()
        print(
            f"  {name:>2}: "
            f"mean_abs={mean_abs_by_sample.mean(): .6e}, "
            f"median_max_abs={np.median(max_abs_by_sample): .6e}, "
            f"max_abs={max_abs_by_sample.max(): .6e}"
        )

    print(f"Per-edge summary over first {n} sample(s)")
    for channel_idx, name in enumerate(CHANNELS):
        print(f"  {name:>2}:")
        for edge in EDGES:
            edge_mean_abs = np.empty(n, dtype=np.float64)
            edge_max_abs = np.empty(n, dtype=np.float64)
            for sample_idx in range(n):
                edge_values = edge_q(np.asarray(q[sample_idx, channel_idx]), edge)
                edge_mean_abs[sample_idx] = np.abs(edge_values).mean()
                edge_max_abs[sample_idx] = np.abs(edge_values).max()
            print(
                f"    {edge:<14} "
                f"mean_abs={edge_mean_abs.mean(): .6e}, "
                f"max_abs={edge_max_abs.max(): .6e}"
            )


def plot_boundary_overlay(ax, x: np.ndarray, y: np.ndarray) -> None:
    colors = {
        "inlet_i0": "tab:green",
        "outlet_iN": "tab:red",
        "lower_wall_j0": "tab:blue",
        "upper_wall_jN": "tab:orange",
    }
    for edge, color in colors.items():
        xb, yb = edge_xy(x, y, edge)
        ax.plot(xb, yb, color=color, linewidth=2.0, label=edge)


def plot_mesh_lines(ax, x: np.ndarray, y: np.ndarray, step: int) -> None:
    row_indices = range(0, x.shape[0], step)
    col_indices = range(0, x.shape[1], step)
    for row_idx in row_indices:
        ax.plot(x[row_idx, :], y[row_idx, :], color="0.7", linewidth=0.5)
    for col_idx in col_indices:
        ax.plot(x[:, col_idx], y[:, col_idx], color="0.7", linewidth=0.5)


def plot_sample(
    sample_idx: int,
    x: np.ndarray,
    y: np.ndarray,
    q: np.ndarray,
    *,
    mesh_step: int,
    out_dir: Path,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    fig, axes = plt.subplots(3, 3, figsize=(16, 11), dpi=120)
    fig.suptitle(f"Pipe sample {sample_idx}: curvilinear boundary and Q channels")

    mesh_step = max(int(mesh_step), 1)
    ax = axes[0, 0]
    plot_mesh_lines(ax, x, y, mesh_step)
    plot_boundary_overlay(ax, x, y)
    ax.set_title("mesh boundary")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)

    speed = np.sqrt(q[0] ** 2 + q[1] ** 2)
    im = axes[0, 1].pcolormesh(x, y, speed, shading="gouraud")
    plot_boundary_overlay(axes[0, 1], x, y)
    axes[0, 1].set_title("speed magnitude")
    axes[0, 1].set_aspect("equal", adjustable="box")
    fig.colorbar(im, ax=axes[0, 1])

    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.0,
        0.9,
        "Boundary convention\n"
        "inlet_i0: first streamwise index\n"
        "outlet_iN: last streamwise index\n"
        "lower_wall_j0: first transverse index\n"
        "upper_wall_jN: last transverse index",
        va="top",
        family="monospace",
    )

    for channel_idx, name in enumerate(CHANNELS):
        ax_field = axes[1, channel_idx]
        im = ax_field.pcolormesh(x, y, q[channel_idx], shading="gouraud")
        plot_boundary_overlay(ax_field, x, y)
        ax_field.set_title(f"{name} on physical mesh")
        ax_field.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax_field)

        ax_profile = axes[2, channel_idx]
        for edge in EDGES:
            ax_profile.plot(edge_q(q[channel_idx], edge), label=edge)
        ax_profile.axhline(0.0, color="0.25", linewidth=0.8)
        ax_profile.set_title(f"{name} boundary traces")
        ax_profile.set_xlabel("edge index")
        ax_profile.grid(True, alpha=0.25)
        if channel_idx == 0:
            ax_profile.legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = out_dir / f"pipe_boundary_sample_{sample_idx:04d}.png"
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    x_all, y_all, q_all = load_pipe_arrays(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded pipe data: "
        f"Pipe_X={x_all.shape}, Pipe_Y={y_all.shape}, Pipe_Q={q_all.shape}"
    )
    print_dataset_summary(q_all, args.summary_samples)
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    if not args.no_plot and plt is None:
        print("matplotlib is not installed; continuing with boundary statistics only.")
        print("Install matplotlib, or rerun explicitly with --no-plot to suppress this note.")

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= x_all.shape[0]:
            raise IndexError(f"Sample {sample_idx} is outside [0, {x_all.shape[0]})")
        x = np.asarray(x_all[sample_idx])
        y = np.asarray(y_all[sample_idx])
        q = np.asarray(q_all[sample_idx, : len(CHANNELS)])
        print_sample_stats(sample_idx, q)
        if args.no_plot or plt is None:
            continue
        out_path = plot_sample(
            sample_idx,
            x,
            y,
            q,
            mesh_step=args.mesh_step,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
