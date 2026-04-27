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


def normalized_edge_coordinate(coord: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    coord_min = float(coord.min())
    coord_max = float(coord.max())
    if coord_max <= coord_min:
        raise ValueError("Cannot normalize degenerate edge coordinate range")
    return (coord - coord_min) / (coord_max - coord_min)


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


def dataset_ux_edge_profiles(
    y_all: np.ndarray,
    q_all: np.ndarray,
    *,
    sample_count: int,
    edge: str,
) -> dict[str, np.ndarray]:
    n = min(int(sample_count), int(q_all.shape[0]))
    if n <= 0:
        raise ValueError("sample_count must be positive")
    if edge not in {"inlet_i0", "outlet_iN"}:
        raise ValueError(f"Unsupported ux profile edge {edge!r}")

    t_profiles = []
    ux_profiles = []
    for sample_idx in range(n):
        y_edge = edge_q(np.asarray(y_all[sample_idx]), edge)
        ux_edge = edge_q(np.asarray(q_all[sample_idx, 0]), edge)
        order = np.argsort(y_edge)
        t_profiles.append(normalized_edge_coordinate(y_edge[order]))
        ux_profiles.append(np.asarray(ux_edge[order], dtype=np.float64))

    t = np.stack(t_profiles, axis=0)
    ux = np.stack(ux_profiles, axis=0)
    t_ref = t.mean(axis=0)
    return {
        "t": t_ref,
        "mean": ux.mean(axis=0),
        "std": ux.std(axis=0),
        "q05": np.quantile(ux, 0.05, axis=0),
        "q25": np.quantile(ux, 0.25, axis=0),
        "q75": np.quantile(ux, 0.75, axis=0),
        "q95": np.quantile(ux, 0.95, axis=0),
        "min": ux.min(axis=0),
        "max": ux.max(axis=0),
        "all": ux,
    }


def dataset_ux_edge_series(
    q_all: np.ndarray,
    *,
    sample_count: int,
    edge: str,
) -> dict[str, np.ndarray]:
    n = min(int(sample_count), int(q_all.shape[0]))
    if n <= 0:
        raise ValueError("sample_count must be positive")
    if edge not in {"inlet_i0", "outlet_iN"}:
        raise ValueError(f"Unsupported ux series edge {edge!r}")

    ux = np.asarray(q_all[:n, 0], dtype=np.float64)
    edge_values = np.empty((n, ux.shape[2]), dtype=np.float64)
    for sample_idx in range(n):
        edge_values[sample_idx] = edge_q(ux[sample_idx], edge)

    sample_index = np.arange(n, dtype=np.int64)
    return {
        "sample": sample_index,
        "mean": edge_values.mean(axis=1),
        "std": edge_values.std(axis=1),
        "min": edge_values.min(axis=1),
        "max": edge_values.max(axis=1),
        "l2": np.sqrt(np.mean(edge_values**2, axis=1)),
    }


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


def plot_dataset_summary(
    x: np.ndarray,
    y: np.ndarray,
    y_all: np.ndarray,
    q_all: np.ndarray,
    *,
    mesh_step: int,
    sample_count: int,
    out_dir: Path,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    n = min(int(sample_count), int(q_all.shape[0]))
    inlet = dataset_ux_edge_profiles(y_all, q_all, sample_count=n, edge="inlet_i0")
    outlet = dataset_ux_edge_profiles(y_all, q_all, sample_count=n, edge="outlet_iN")
    inlet_series = dataset_ux_edge_series(q_all, sample_count=n, edge="inlet_i0")
    outlet_series = dataset_ux_edge_series(q_all, sample_count=n, edge="outlet_iN")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), dpi=140)
    fig.suptitle(f"Pipe dataset boundary diagnostics over first {n} sample(s)")

    mesh_step = max(int(mesh_step), 1)
    ax = axes[0, 0]
    plot_mesh_lines(ax, x, y, mesh_step)
    plot_boundary_overlay(ax, x, y)
    ax.set_title("curvilinear pipe mesh and boundary convention")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)

    _plot_ux_profile_panel(axes[0, 1], inlet, title="inlet ux profile over samples")
    _plot_ux_profile_panel(axes[0, 2], outlet, title="outlet ux profile over samples")
    _plot_ux_series_panel(axes[1, 0], inlet_series, title="inlet ux variation across samples")
    _plot_ux_series_panel(
        axes[1, 1], outlet_series, title="outlet ux variation across samples"
    )

    axes[1, 2].axis("off")
    wall_ux = np.concatenate(
        [
            np.asarray(q_all[:n, 0, :, 0], dtype=np.float64).reshape(n, -1),
            np.asarray(q_all[:n, 0, :, -1], dtype=np.float64).reshape(n, -1),
        ],
        axis=1,
    )
    inlet_peak = inlet["all"].max(axis=1)
    outlet_peak = outlet["all"].max(axis=1)
    outlet_mean_abs = np.abs(outlet["all"]).mean(axis=1)
    axes[1, 2].text(
        0.0,
        0.98,
        "Dataset-level boundary observations\n\n"
        f"samples summarized: {n}\n"
        f"wall ux mean_abs: {np.abs(wall_ux).mean():.3e}\n"
        f"wall ux max_abs: {np.abs(wall_ux).max():.3e}\n"
        f"inlet ux peak mean +- std: {inlet_peak.mean():.4f} +- {inlet_peak.std():.4f}\n"
        f"inlet ux mean(range): {(inlet_series['max'] - inlet_series['min']).mean():.4f}\n"
        f"outlet ux peak mean +- std: {outlet_peak.mean():.4f} +- {outlet_peak.std():.4f}\n"
        f"outlet ux mean(range): {(outlet_series['max'] - outlet_series['min']).mean():.4f}\n"
        f"outlet ux mean_abs mean +- std: {outlet_mean_abs.mean():.4f} +- {outlet_mean_abs.std():.4f}\n\n"
        "Bands show 5-95% and 25-75% quantiles.\n"
        "Black curve is the dataset mean profile.\n"
        "Light traces are a subsample of individual profiles.\n"
        "Lower panels track mean ux with min-max spread per sample.",
        va="top",
        family="monospace",
    )

    fig.tight_layout()
    out_path = out_dir / "pipe_boundary_dataset_summary.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _plot_ux_profile_panel(ax, profile: dict[str, np.ndarray], *, title: str) -> None:
    t = profile["t"]
    all_profiles = profile["all"]
    line_step = max(all_profiles.shape[0] // 64, 1)
    for values in all_profiles[::line_step]:
        ax.plot(t, values, color="0.75", linewidth=0.8, alpha=0.35)

    ax.fill_between(t, profile["q05"], profile["q95"], color="tab:blue", alpha=0.18)
    ax.fill_between(t, profile["q25"], profile["q75"], color="tab:blue", alpha=0.30)
    ax.plot(t, profile["mean"], color="black", linewidth=2.0, label="mean")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("normalized transverse coordinate")
    ax.set_ylabel("ux")
    ax.grid(True, alpha=0.25)


def _plot_ux_series_panel(ax, series: dict[str, np.ndarray], *, title: str) -> None:
    sample = series["sample"]
    ax.fill_between(sample, series["min"], series["max"], color="tab:orange", alpha=0.18)
    ax.plot(sample, series["mean"], color="black", linewidth=1.4, label="mean")
    ax.plot(sample, series["max"], color="tab:red", linewidth=0.9, alpha=0.7, label="max")
    ax.plot(sample, series["min"], color="tab:blue", linewidth=0.9, alpha=0.7, label="min")
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel("ux")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


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
    if not args.no_plot and plt is not None:
        summary_path = plot_dataset_summary(
            np.asarray(x_all[0]),
            np.asarray(y_all[0]),
            y_all,
            q_all,
            mesh_step=args.mesh_step,
            sample_count=args.summary_samples,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {summary_path}")

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
