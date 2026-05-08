from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
except ImportError:
    animation = None
    plt = None

from _common import (
    load_plasticity_arrays,
    require_matplotlib,
    select_split,
    validate_samples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed vertical distance between a moving raw-coordinate "
            "input die and the j=0 upper boundary for all timesteps."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/plasticity"),
        help="Directory containing plas_N987_T20.mat, or the .mat file itself.",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--ntrain", type=int, default=900)
    parser.add_argument("--ntest", type=int, default=80)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--die-speed",
        type=float,
        default=6.0,
        help="Downward die speed in physical y units per physical time unit.",
    )
    parser.add_argument(
        "--time-duration",
        type=float,
        default=1.0,
        help=(
            "Physical duration corresponding to one full T_out-step sequence. "
            "Frame times are sampled at t/T_out, so the final displayed frame "
            "is at (T_out - 1) / T_out of this duration."
        ),
    )
    parser.add_argument(
        "--flip-die-x",
        action="store_true",
        help="Reverse the raw input die profile along x before computing distance.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Output PNG path. Defaults to "
            "artifacts/plasticity/plasticity_die_j0_distance/plasticity_die_j0_distance_sample_XXXX.png."
        ),
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help=(
            "Output GIF path. Defaults to "
            "artifacts/plasticity/plasticity_die_j0_distance/plasticity_die_j0_distance_sample_XXXX.gif."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional CSV path with per-index signed distances.",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=130)
    return parser.parse_args()


def infer_material_grid(sample: np.ndarray) -> np.ndarray:
    coords = sample[..., 0:2].astype(np.float64)
    disp = sample[..., 2:4].astype(np.float64)
    return coords - disp


def selected_timestep_count(sample: np.ndarray) -> int:
    return int(sample.shape[2])


def compute_distances(
    *,
    die_profile: np.ndarray,
    sample: np.ndarray,
    die_speed: float,
    time_duration: float,
) -> dict[str, np.ndarray]:
    coords = sample[..., 0:2].astype(np.float64)
    material = infer_material_grid(sample)
    t_count = selected_timestep_count(sample)

    material_x = material[:, 0, 0, 0]
    x_ref = np.linspace(
        float(np.nanmin(material_x)),
        float(np.nanmax(material_x)),
        int(die_profile.shape[0]),
    )
    time = np.arange(t_count, dtype=np.float64) * float(time_duration) / max(t_count, 1)

    die_y = die_profile[:, None] - float(die_speed) * time[None, :]

    j0 = coords[:, 0, :, :]
    j0_x = j0[..., 0]
    j0_y = j0[..., 1]

    j0_y_on_die = np.empty_like(die_y)
    for timestep in range(t_count):
        x_t = j0_x[:, timestep]
        y_t = j0_y[:, timestep]
        order = np.argsort(x_t)
        j0_y_on_die[:, timestep] = np.interp(x_ref, x_t[order], y_t[order])

    signed_distance = die_y - j0_y_on_die
    return {
        "x": x_ref,
        "time": time,
        "die_y": die_y,
        "j0_y": j0_y_on_die,
        "signed_distance": signed_distance,
    }


def write_distance_csv(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = payload["x"]
    time = payload["time"]
    die_y = payload["die_y"]
    j0_y = payload["j0_y"]
    signed_distance = payload["signed_distance"]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestep", "time", "i", "x", "die_y", "j0_y", "signed_distance"],
        )
        writer.writeheader()
        for timestep in range(time.shape[0]):
            for i in range(x.shape[0]):
                writer.writerow(
                    {
                        "timestep": timestep,
                        "time": float(time[timestep]),
                        "i": i,
                        "x": float(x[i]),
                        "die_y": float(die_y[i, timestep]),
                        "j0_y": float(j0_y[i, timestep]),
                        "signed_distance": float(signed_distance[i, timestep]),
                    }
                )


def plot_distances(
    *,
    payload: dict[str, np.ndarray],
    sample_idx: int,
    out_path: Path,
    die_speed: float,
    time_duration: float,
) -> Path:
    require_matplotlib(plt)
    x = payload["x"]
    time = payload["time"]
    signed_distance = payload["signed_distance"]
    die_y = payload["die_y"]
    j0_y = payload["j0_y"]

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 10.2), dpi=150, sharex=True)
    cmap = plt.get_cmap("viridis")

    ax = axes[0]
    for timestep in range(time.shape[0]):
        color = cmap(timestep / max(time.shape[0] - 1, 1))
        ax.plot(
            x,
            signed_distance[:, timestep],
            color=color,
            alpha=0.30 + 0.65 * timestep / max(time.shape[0] - 1, 1),
            linewidth=1.0,
        )
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title(
        f"Sample {sample_idx}: signed distance die - j=0 boundary, "
        f"die_speed={die_speed:g}, duration={time_duration:g}"
    )
    ax.set_ylabel("die_y - j0_y")
    ax.grid(True, alpha=0.25)

    summary_ax = axes[1]
    min_d = np.nanmin(signed_distance, axis=0)
    mean_d = np.nanmean(signed_distance, axis=0)
    max_d = np.nanmax(signed_distance, axis=0)
    p05 = np.nanpercentile(signed_distance, 5, axis=0)
    p95 = np.nanpercentile(signed_distance, 95, axis=0)
    summary_ax.fill_between(time, p05, p95, color="#2563eb", alpha=0.16, label="5-95% over x")
    summary_ax.plot(time, min_d, color="#dc2626", linewidth=1.5, label="min over x")
    summary_ax.plot(time, mean_d, color="#2563eb", linewidth=1.8, label="mean over x")
    summary_ax.plot(time, max_d, color="#111827", linewidth=1.2, label="max over x")
    summary_ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.8)
    summary_ax.set_title("Distance summary over x")
    summary_ax.set_xlabel("time")
    summary_ax.set_ylabel("signed distance")
    summary_ax.grid(True, alpha=0.25)
    summary_ax.legend(fontsize=8)

    curves_ax = axes[2]
    final_timestep = int(time.shape[0] - 1)
    curves_ax.plot(x, die_y[:, 0], color="#94a3b8", linewidth=1.6, label="die t=0")
    curves_ax.plot(x, die_y[:, final_timestep], color="#111827", linewidth=2.0, label="die final")
    curves_ax.plot(x, j0_y[:, 0], color="#fca5a5", linewidth=1.4, label="j=0 t=0")
    curves_ax.plot(x, j0_y[:, final_timestep], color="#dc2626", linewidth=2.0, label="j=0 final")
    curves_ax.set_title("Die and j=0 curves at start/final timesteps")
    curves_ax.set_xlabel("physical x coordinate")
    curves_ax.set_ylabel("physical y coordinate")
    curves_ax.grid(True, alpha=0.25)
    curves_ax.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(
        "Final signed distance: "
        f"min={float(np.nanmin(signed_distance[:, -1])):.8g}, "
        f"mean={float(np.nanmean(signed_distance[:, -1])):.8g}, "
        f"max={float(np.nanmax(signed_distance[:, -1])):.8g}"
    )
    print(
        "Closest approach over all timesteps/x: "
        f"{float(np.nanmin(np.abs(signed_distance))):.8g}"
    )
    return out_path


def save_distance_gif(
    *,
    payload: dict[str, np.ndarray],
    sample_idx: int,
    out_path: Path,
    die_speed: float,
    time_duration: float,
    fps: int,
    dpi: int,
) -> Path:
    require_matplotlib(plt)
    if animation is None:
        raise RuntimeError("matplotlib.animation is required to save the GIF.")

    x = payload["x"]
    time = payload["time"]
    signed_distance = payload["signed_distance"]
    die_y = payload["die_y"]
    j0_y = payload["j0_y"]

    distance_min = float(np.nanmin(signed_distance))
    distance_max = float(np.nanmax(signed_distance))
    y_min = float(min(np.nanmin(die_y), np.nanmin(j0_y)))
    y_max = float(max(np.nanmax(die_y), np.nanmax(j0_y)))
    distance_pad = 0.08 * max(distance_max - distance_min, 1e-6)
    y_pad = 0.08 * max(y_max - y_min, 1e-6)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.6), dpi=dpi, sharex=True)
    distance_ax, curves_ax = axes
    (distance_line,) = distance_ax.plot([], [], color="#2563eb", linewidth=2.0)
    distance_ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    distance_ax.set_xlim(float(np.nanmin(x)), float(np.nanmax(x)))
    distance_ax.set_ylim(distance_min - distance_pad, distance_max + distance_pad)
    distance_ax.set_ylabel("die_y - j0_y")
    distance_ax.grid(True, alpha=0.25)

    (die_line,) = curves_ax.plot([], [], color="#111827", linewidth=2.2, label="moving die")
    (j0_line,) = curves_ax.plot([], [], color="#dc2626", linewidth=2.0, label="j=0 boundary")
    curves_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    curves_ax.set_xlabel("physical x coordinate")
    curves_ax.set_ylabel("physical y coordinate")
    curves_ax.grid(True, alpha=0.25)
    curves_ax.legend(fontsize=8)
    title = fig.suptitle("")

    def update(timestep: int):
        distance_line.set_data(x, signed_distance[:, timestep])
        die_line.set_data(x, die_y[:, timestep])
        j0_line.set_data(x, j0_y[:, timestep])
        title.set_text(
            f"Sample {sample_idx}: die to j=0 signed distance | "
            f"timestep {timestep + 1}/{time.shape[0]}, "
            f"speed={die_speed:g}, duration={time_duration:g}"
        )
        return distance_line, die_line, j0_line, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=int(time.shape[0]),
        interval=int(1000 / max(fps, 1)),
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=max(int(fps), 1)), dpi=dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.time_duration <= 0.0:
        raise ValueError("--time-duration must be positive")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    die, output, mat_path = load_plasticity_arrays(args.data_dir)
    die, output = select_split(
        die,
        output,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    validate_samples([args.sample], int(output.shape[0]))
    die_profile = np.asarray(die[args.sample], dtype=np.float64)
    if args.flip_die_x:
        die_profile = die_profile[::-1]

    out_path = args.out_path
    if out_path is None:
        out_path = (
            Path("artifacts")
            / "plasticity"
            / "plasticity_die_j0_distance"
            / f"plasticity_die_j0_distance_sample_{args.sample:04d}.png"
        )
    gif_path = args.gif_path
    if gif_path is None:
        gif_path = (
            Path("artifacts")
            / "plasticity"
            / "plasticity_die_j0_distance"
            / f"plasticity_die_j0_distance_sample_{args.sample:04d}.gif"
        )

    payload = compute_distances(
        die_profile=die_profile,
        sample=output[args.sample],
        die_speed=float(args.die_speed),
        time_duration=float(args.time_duration),
    )
    saved_path = plot_distances(
        payload=payload,
        sample_idx=args.sample,
        out_path=out_path,
        die_speed=float(args.die_speed),
        time_duration=float(args.time_duration),
    )
    saved_gif = save_distance_gif(
        payload=payload,
        sample_idx=args.sample,
        out_path=gif_path,
        die_speed=float(args.die_speed),
        time_duration=float(args.time_duration),
        fps=int(args.fps),
        dpi=int(args.dpi),
    )
    if args.csv_path is not None:
        write_distance_csv(args.csv_path, payload)
        print(f"Wrote CSV: {args.csv_path}")

    print(f"Loaded: {mat_path}")
    print(f"Wrote plot: {saved_path}")
    print(f"Wrote GIF: {saved_gif}")


if __name__ == "__main__":
    main()
