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


CHANNELS = ("ux", "uy", "p")
EDGES = {
    "inlet_i0": (0, slice(None)),
    "outlet_iN": (-1, slice(None)),
    "lower_wall_j0": (slice(None), 0),
    "upper_wall_jN": (slice(None), -1),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset-wide pipe boundary summary plots for documentation."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pipe"),
        help="Directory containing Pipe_X.npy, Pipe_Y.npy, and Pipe_Q.npy.",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=1000,
        help="Number of leading samples to include in the dataset summary.",
    )
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=8,
        help="Subsampling stride for the mesh overlay.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/pipe_dataset_summary"),
        help="Directory where figures and CSV summaries are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print and write summary statistics; do not write figures.",
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


def edge_q(q_channel: np.ndarray, edge: str) -> np.ndarray:
    return np.asarray(q_channel[EDGES[edge]], dtype=np.float64)


def edge_xy(x: np.ndarray, y: np.ndarray, edge: str) -> tuple[np.ndarray, np.ndarray]:
    selector = EDGES[edge]
    return np.asarray(x[selector], dtype=np.float64), np.asarray(y[selector], dtype=np.float64)


def normalized_coordinate(coord: np.ndarray) -> np.ndarray:
    coord = np.asarray(coord, dtype=np.float64)
    coord_min = float(coord.min())
    coord_max = float(coord.max())
    if coord_max <= coord_min:
        raise ValueError("Cannot normalize degenerate coordinate range")
    return (coord - coord_min) / (coord_max - coord_min)


def parabolic_profile(t: np.ndarray, amplitude: float = 0.25) -> np.ndarray:
    return amplitude * 4.0 * t * (1.0 - t)


def edge_profile_bundle(
    coord_all: np.ndarray,
    q_all: np.ndarray,
    *,
    sample_count: int,
    channel_idx: int,
    edge: str,
) -> dict[str, np.ndarray]:
    n = min(int(sample_count), int(q_all.shape[0]))
    if n <= 0:
        raise ValueError("sample_count must be positive")

    t_profiles = []
    values_profiles = []
    for sample_idx in range(n):
        coord_edge = edge_q(np.asarray(coord_all[sample_idx]), edge)
        values_edge = edge_q(np.asarray(q_all[sample_idx, channel_idx]), edge)
        t = normalized_coordinate(coord_edge)
        order = np.argsort(t)
        t_profiles.append(t[order])
        values_profiles.append(values_edge[order])

    t_arr = np.stack(t_profiles, axis=0)
    values = np.stack(values_profiles, axis=0)
    return {
        "t": t_arr.mean(axis=0),
        "mean": values.mean(axis=0),
        "std": values.std(axis=0),
        "q05": np.quantile(values, 0.05, axis=0),
        "q25": np.quantile(values, 0.25, axis=0),
        "q75": np.quantile(values, 0.75, axis=0),
        "q95": np.quantile(values, 0.95, axis=0),
        "min": values.min(axis=0),
        "max": values.max(axis=0),
        "all": values,
    }


def per_sample_edge_stats(
    q_all: np.ndarray,
    *,
    sample_count: int,
    channel_idx: int,
    edge: str,
) -> dict[str, np.ndarray]:
    n = min(int(sample_count), int(q_all.shape[0]))
    values = np.empty((n, q_all.shape[2 if edge in {"lower_wall_j0", "upper_wall_jN"} else 3]), dtype=np.float64)
    for sample_idx in range(n):
        values[sample_idx] = edge_q(np.asarray(q_all[sample_idx, channel_idx]), edge)

    return {
        "sample": np.arange(n, dtype=np.int64),
        "mean": values.mean(axis=1),
        "mean_abs": np.abs(values).mean(axis=1),
        "max_abs": np.abs(values).max(axis=1),
        "l2": np.sqrt(np.mean(values**2, axis=1)),
    }


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "group",
        "channel",
        "metric",
        "value",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_dataset(
    x_all: np.ndarray,
    y_all: np.ndarray,
    q_all: np.ndarray,
    *,
    sample_count: int,
) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]]]:
    n = min(int(sample_count), int(q_all.shape[0]))
    profiles: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict[str, object]] = []

    for edge in EDGES:
        for channel_idx, channel_name in enumerate(CHANNELS[:2]):
            coord_all = y_all if edge in {"inlet_i0", "outlet_iN"} else x_all
            bundle = edge_profile_bundle(
                coord_all,
                q_all,
                sample_count=n,
                channel_idx=channel_idx,
                edge=edge,
            )
            profiles[f"{edge}:{channel_name}"] = bundle

            per_sample = per_sample_edge_stats(
                q_all,
                sample_count=n,
                channel_idx=channel_idx,
                edge=edge,
            )
            rows.extend(
                [
                    {
                        "group": edge,
                        "channel": channel_name,
                        "metric": "mean_abs_mean",
                        "value": float(per_sample["mean_abs"].mean()),
                    },
                    {
                        "group": edge,
                        "channel": channel_name,
                        "metric": "mean_abs_p95",
                        "value": float(np.quantile(per_sample["mean_abs"], 0.95)),
                    },
                    {
                        "group": edge,
                        "channel": channel_name,
                        "metric": "max_abs_max",
                        "value": float(per_sample["max_abs"].max()),
                    },
                ]
            )

    inlet_ux = profiles["inlet_i0:ux"]
    inlet_t = inlet_ux["t"]
    inlet_target = parabolic_profile(inlet_t)
    inlet_residual = inlet_ux["all"] - inlet_target[None, :]
    rows.extend(
        [
            {
                "group": "inlet_i0",
                "channel": "ux",
                "metric": "parabola_residual_mean_abs",
                "value": float(np.abs(inlet_residual).mean()),
            },
            {
                "group": "inlet_i0",
                "channel": "ux",
                "metric": "parabola_residual_max_abs",
                "value": float(np.abs(inlet_residual).max()),
            },
            {
                "group": "inlet_i0",
                "channel": "ux",
                "metric": "peak_mean",
                "value": float(inlet_ux["all"].max(axis=1).mean()),
            },
            {
                "group": "inlet_i0",
                "channel": "ux",
                "metric": "peak_std",
                "value": float(inlet_ux["all"].max(axis=1).std()),
            },
        ]
    )

    return profiles, rows


def plot_mesh_lines(ax, x: np.ndarray, y: np.ndarray, step: int) -> None:
    row_indices = range(0, x.shape[0], step)
    col_indices = range(0, x.shape[1], step)
    for row_idx in row_indices:
        ax.plot(x[row_idx, :], y[row_idx, :], color="0.75", linewidth=0.5)
    for col_idx in col_indices:
        ax.plot(x[:, col_idx], y[:, col_idx], color="0.75", linewidth=0.5)


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


def plot_band_panel(
    ax,
    bundle: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    color: str,
    reference: np.ndarray | None = None,
    reference_label: str | None = None,
) -> None:
    t = bundle["t"]
    ax.fill_between(t, bundle["q05"], bundle["q95"], color=color, alpha=0.15)
    ax.fill_between(t, bundle["q25"], bundle["q75"], color=color, alpha=0.28)
    ax.plot(t, bundle["mean"], color="black", linewidth=1.8, label="dataset mean")
    if reference is not None:
        ax.plot(t, reference, color="tab:red", linewidth=1.4, linestyle="--", label=reference_label)
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("normalized edge coordinate")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if reference is not None:
        ax.legend(loc="best", fontsize=8)


def plot_two_wall_panel(
    ax,
    lower: dict[str, np.ndarray],
    upper: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
) -> None:
    t = lower["t"]
    ax.fill_between(t, lower["q05"], lower["q95"], color="tab:blue", alpha=0.12)
    ax.fill_between(t, upper["q05"], upper["q95"], color="tab:orange", alpha=0.12)
    ax.plot(t, lower["mean"], color="tab:blue", linewidth=1.6, label="lower wall mean")
    ax.plot(t, upper["mean"], color="tab:orange", linewidth=1.6, label="upper wall mean")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_title(title)
    ax.set_xlabel("normalized streamwise coordinate")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_sample_metric_panel(
    ax,
    series: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    sample = series["sample"]
    ax.plot(sample, series["mean_abs"], color=color, linewidth=1.2, label="mean abs")
    ax.plot(sample, series["max_abs"], color="black", linewidth=1.0, alpha=0.8, label="max abs")
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)


def plot_dataset_overview(
    x: np.ndarray,
    y: np.ndarray,
    profiles: dict[str, dict[str, np.ndarray]],
    q_all: np.ndarray,
    *,
    sample_count: int,
    mesh_step: int,
    out_dir: Path,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    n = min(int(sample_count), int(q_all.shape[0]))
    inlet_ux = profiles["inlet_i0:ux"]
    inlet_uy = profiles["inlet_i0:uy"]
    outlet_ux = profiles["outlet_iN:ux"]
    outlet_uy = profiles["outlet_iN:uy"]
    wall_lower_ux = profiles["lower_wall_j0:ux"]
    wall_upper_ux = profiles["upper_wall_jN:ux"]
    wall_lower_uy = profiles["lower_wall_j0:uy"]
    wall_upper_uy = profiles["upper_wall_jN:uy"]

    inlet_target = parabolic_profile(inlet_ux["t"])
    inlet_residual = inlet_ux["all"] - inlet_target[None, :]
    inlet_uy_series = per_sample_edge_stats(q_all, sample_count=n, channel_idx=1, edge="inlet_i0")
    outlet_uy_series = per_sample_edge_stats(q_all, sample_count=n, channel_idx=1, edge="outlet_iN")

    fig, axes = plt.subplots(3, 3, figsize=(18, 13), dpi=150)
    fig.suptitle(f"Pipe dataset boundary summary over first {n} sample(s)")

    ax = axes[0, 0]
    plot_mesh_lines(ax, x, y, max(int(mesh_step), 1))
    plot_boundary_overlay(ax, x, y)
    ax.set_title("mesh boundary convention")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)

    plot_two_wall_panel(
        axes[0, 1],
        wall_lower_ux,
        wall_upper_ux,
        title="wall ux traces collapse to zero",
        ylabel="ux",
    )
    plot_two_wall_panel(
        axes[0, 2],
        wall_lower_uy,
        wall_upper_uy,
        title="wall uy traces collapse to zero",
        ylabel="uy",
    )
    plot_band_panel(
        axes[1, 0],
        inlet_ux,
        title="inlet ux matches a fixed parabola",
        ylabel="ux",
        color="tab:green",
        reference=inlet_target,
        reference_label="0.25 * 4t(1-t)",
    )
    plot_band_panel(
        axes[1, 1],
        inlet_uy,
        title="inlet uy is identically zero",
        ylabel="uy",
        color="tab:purple",
    )
    axes[1, 2].fill_between(
        inlet_ux["t"],
        np.quantile(inlet_residual, 0.05, axis=0),
        np.quantile(inlet_residual, 0.95, axis=0),
        color="tab:red",
        alpha=0.18,
    )
    axes[1, 2].plot(
        inlet_ux["t"],
        inlet_residual.mean(axis=0),
        color="black",
        linewidth=1.6,
        label="mean residual",
    )
    axes[1, 2].axhline(0.0, color="0.3", linewidth=0.8)
    axes[1, 2].set_title("inlet ux residual relative to parabola")
    axes[1, 2].set_xlabel("normalized inlet coordinate")
    axes[1, 2].set_ylabel("ux - target")
    axes[1, 2].grid(True, alpha=0.25)
    axes[1, 2].legend(loc="best", fontsize=8)

    plot_band_panel(
        axes[2, 0],
        outlet_ux,
        title="outlet ux profile varies across samples",
        ylabel="ux",
        color="tab:orange",
    )
    plot_band_panel(
        axes[2, 1],
        outlet_uy,
        title="outlet uy profile varies across samples",
        ylabel="uy",
        color="tab:brown",
    )
    plot_sample_metric_panel(
        axes[2, 2],
        outlet_uy_series,
        title="outlet uy samplewise magnitude",
        ylabel="|uy| statistics",
        color="tab:brown",
    )

    fig.tight_layout()
    out_path = out_dir / "pipe_dataset_boundary_overview.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_doc_panels(
    profiles: dict[str, dict[str, np.ndarray]],
    q_all: np.ndarray,
    *,
    sample_count: int,
    out_dir: Path,
    show: bool,
) -> list[Path]:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plots. Install matplotlib or rerun with --no-plot."
        )

    n = min(int(sample_count), int(q_all.shape[0]))
    paths: list[Path] = []

    wall_fig, wall_axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    plot_two_wall_panel(
        wall_axes[0],
        profiles["lower_wall_j0:ux"],
        profiles["upper_wall_jN:ux"],
        title="wall no-slip for ux",
        ylabel="ux",
    )
    plot_two_wall_panel(
        wall_axes[1],
        profiles["lower_wall_j0:uy"],
        profiles["upper_wall_jN:uy"],
        title="wall no-slip for uy",
        ylabel="uy",
    )
    wall_fig.tight_layout()
    wall_path = out_dir / "pipe_dataset_wall_zero.png"
    wall_fig.savefig(wall_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(wall_fig)
    paths.append(wall_path)

    inlet_fig, inlet_axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
    inlet_ux = profiles["inlet_i0:ux"]
    inlet_target = parabolic_profile(inlet_ux["t"])
    plot_band_panel(
        inlet_axes[0],
        inlet_ux,
        title="inlet ux is a dataset-wide parabola",
        ylabel="ux",
        color="tab:green",
        reference=inlet_target,
        reference_label="target parabola",
    )
    plot_band_panel(
        inlet_axes[1],
        profiles["inlet_i0:uy"],
        title="inlet uy is identically zero",
        ylabel="uy",
        color="tab:purple",
    )
    inlet_fig.tight_layout()
    inlet_path = out_dir / "pipe_dataset_inlet_profiles.png"
    inlet_fig.savefig(inlet_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(inlet_fig)
    paths.append(inlet_path)

    outlet_fig, outlet_axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    plot_band_panel(
        outlet_axes[0],
        profiles["outlet_iN:ux"],
        title="outlet ux profile spread",
        ylabel="ux",
        color="tab:orange",
    )
    plot_band_panel(
        outlet_axes[1],
        profiles["outlet_iN:uy"],
        title="outlet uy profile spread",
        ylabel="uy",
        color="tab:brown",
    )
    plot_sample_metric_panel(
        outlet_axes[2],
        per_sample_edge_stats(q_all, sample_count=n, channel_idx=1, edge="outlet_iN"),
        title="outlet uy across samples",
        ylabel="|uy| statistics",
        color="tab:brown",
    )
    outlet_fig.tight_layout()
    outlet_path = out_dir / "pipe_dataset_outlet_profiles.png"
    outlet_fig.savefig(outlet_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(outlet_fig)
    paths.append(outlet_path)

    return paths


def print_summary(rows: list[dict[str, object]], sample_count: int) -> None:
    print(f"Dataset-wide pipe boundary summary over first {sample_count} sample(s)")
    for row in rows:
        print(
            f"  {row['group']:<14} {row['channel']:<2} "
            f"{row['metric']:<28} {float(row['value']): .6e}"
        )


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

    profiles, rows = summarize_dataset(x_all, y_all, q_all, sample_count=n)
    print_summary(rows, n)

    csv_path = args.out_dir / "pipe_dataset_boundary_summary.csv"
    write_summary_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")
    if not args.no_plot and plt is None:
        print("matplotlib is not installed; continuing with summary statistics only.")
        return
    if args.no_plot:
        return

    overview_path = plot_dataset_overview(
        np.asarray(x_all[0]),
        np.asarray(y_all[0]),
        profiles,
        q_all,
        sample_count=n,
        mesh_step=args.mesh_step,
        out_dir=args.out_dir,
        show=args.show,
    )
    print(f"wrote {overview_path}")
    for path in plot_doc_panels(
        profiles,
        q_all,
        sample_count=n,
        out_dir=args.out_dir,
        show=False,
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
