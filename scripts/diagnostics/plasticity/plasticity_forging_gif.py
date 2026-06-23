from __future__ import annotations

import argparse
import os
import tempfile
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
            "Create an animated GIF for the plasticity forging benchmark: the "
            "input die profile moves down while the block deformation evolves."
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
        "--samples",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Sample indices to concatenate into one GIF. If omitted, --sample "
            "is used for backward-compatible single-sample output."
        ),
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Output GIF path. Defaults to "
            "artifacts/plasticity/plasticity_forging_gif/plasticity_forging_sample_XXXX.gif."
        ),
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--point-size",
        type=float,
        default=13.0,
        help="Scatter point size for material grid points.",
    )
    parser.add_argument(
        "--die-travel-fraction",
        type=float,
        default=0.18,
        help="Visual-only die travel as a fraction of deformed block height.",
    )
    parser.add_argument(
        "--die-fit-mode",
        choices=("raw-coordinate", "contact-envelope", "upper-boundary", "normalized"),
        default="raw-coordinate",
        help=(
            "How to draw the input die profile. 'raw-coordinate' maps the raw "
            "101 input values directly to physical y coordinates over the "
            "block x range; 'contact-envelope' uses the final upper envelope "
            "of the deformed block; 'upper-boundary' fits the input profile "
            "to the nominal j=0 edge; 'normalized' uses "
            "--die-amplitude-fraction."
        ),
    )
    parser.add_argument(
        "--die-speed",
        type=float,
        default=6.0,
        help=(
            "Downward die speed in physical y units per physical time unit for "
            "--die-fit-mode raw-coordinate."
        ),
    )
    parser.add_argument(
        "--time-duration",
        type=float,
        default=1.0,
        help=(
            "Physical duration corresponding to one full T_out-step sequence. "
            "The displayed frames are sampled at t/T_out, so the final frame "
            "occurs at (T_out - 1) / T_out of this duration."
        ),
    )
    parser.add_argument(
        "--flip-die-x",
        action="store_true",
        help=(
            "Reverse the input die profile along x before plotting. Disabled "
            "by default because the GIF treats the die as a schematic input "
            "profile, not an inferred contact boundary."
        ),
    )
    parser.add_argument(
        "--die-gap-fraction",
        type=float,
        default=0.09,
        help="Final visual gap between die and top boundary as a fraction of block height.",
    )
    parser.add_argument(
        "--die-amplitude-fraction",
        type=float,
        default=0.07,
        help="Die profile amplitude for --die-fit-mode normalized.",
    )
    parser.add_argument(
        "--boundary-debug-path",
        type=Path,
        default=None,
        help=(
            "Optional PNG comparing raw/fitted die profile, nominal j=0 upper "
            "edge, and full deformed upper envelope over time."
        ),
    )
    return parser.parse_args()


def normalize_profile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    centered = values - float(np.nanmean(values))
    scale = float(np.nanmax(np.abs(centered)))
    if scale <= 0.0 or not np.isfinite(scale):
        return np.zeros_like(centered)
    return centered / scale


def infer_material_grid(sample: np.ndarray) -> np.ndarray:
    coords = sample[..., 0:2].astype(np.float64)
    disp = sample[..., 2:4].astype(np.float64)
    return coords - disp


def fit_profile_to_upper_boundary(
    die_profile: np.ndarray,
    *,
    upper_y: np.ndarray,
    y_span: float,
    final_gap: float,
) -> np.ndarray:
    profile = normalize_profile(die_profile)
    target = np.asarray(upper_y, dtype=np.float64)
    if profile.shape[0] != target.shape[0]:
        source_x = np.linspace(0.0, 1.0, int(profile.shape[0]))
        target_x = np.linspace(0.0, 1.0, int(target.shape[0]))
        profile = np.interp(target_x, source_x, profile)

    design = np.stack([np.ones_like(profile), profile], axis=1)
    coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
    fitted = coeff[0] + coeff[1] * profile

    # Keep the orientation that best matches the final top edge. The input die
    # is a profile signal, while the output channels are physical coordinates.
    flipped = coeff[0] - coeff[1] * profile
    if np.mean((flipped - target) ** 2) < np.mean((fitted - target) ** 2):
        fitted = flipped

    fitted = fitted + float(final_gap) * float(y_span)
    return fitted


def smooth_profile(values: np.ndarray, *, window: int = 3) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    window = int(window)
    if window <= 1:
        return values
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def make_animation(
    *,
    die_profile: np.ndarray,
    sample: np.ndarray,
    sample_idx: int,
    out_path: Path,
    fps: int,
    dpi: int,
    point_size: float,
    die_travel_fraction: float,
    die_fit_mode: str,
    die_speed: float | None,
    time_duration: float,
    die_gap_fraction: float,
    die_amplitude_fraction: float,
) -> Path:
    require_matplotlib(plt)
    if animation is None:
        raise RuntimeError("matplotlib.animation is required to save the GIF.")

    die_profile = np.asarray(die_profile, dtype=np.float64)
    coords = sample[..., 0:2].astype(np.float64)
    disp = sample[..., 2:4].astype(np.float64)
    disp_mag = np.linalg.norm(disp, axis=-1)
    material = infer_material_grid(sample)

    t_count = int(sample.shape[2])
    block_x = coords[..., 0]
    block_y = coords[..., 1]
    material_x = material[:, 0, 0, 0]
    ref_x_min, ref_x_max = float(np.nanmin(material_x)), float(np.nanmax(material_x))
    x_min = min(float(np.nanmin(block_x)), ref_x_min)
    x_max = max(float(np.nanmax(block_x)), ref_x_max)
    y_min, y_max = float(np.nanmin(block_y)), float(np.nanmax(block_y))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    die_x = np.linspace(ref_x_min, ref_x_max, int(die_profile.shape[0]))
    raw_coordinate_mode = die_fit_mode == "raw-coordinate"
    die_speed_per_step = None
    if raw_coordinate_mode:
        die_y_initial = die_profile.copy()
        final_top = np.nanmax(coords[:, :, -1, 1], axis=1)
        final_top_at_die = np.interp(
            np.linspace(0.0, 1.0, die_profile.shape[0]),
            np.linspace(0.0, 1.0, final_top.shape[0]),
            final_top,
        )
        if die_speed is None:
            final_gap = float(die_gap_fraction) * y_span
            available_drop = die_y_initial - final_top_at_die - final_gap
            die_speed_per_step = max(
                float(np.nanmin(available_drop)) / max(t_count - 1, 1),
                0.0,
            )
        else:
            die_speed_per_step = (
                float(die_speed) * float(time_duration) / max(t_count, 1)
            )
        die_y_final = die_y_initial - die_speed_per_step * max(t_count - 1, 0)
    elif die_fit_mode == "contact-envelope":
        # The nominal upper edge is j=0, but the deformed cloud can contain
        # higher points near the sides. Use the actual final envelope so the
        # die remains visually above the material throughout the last frame.
        final_envelope = np.nanmax(coords[:, :, -1, 1], axis=1)
        smoothed_envelope = smooth_profile(final_envelope)
        die_y_final = np.maximum(final_envelope, smoothed_envelope)
        die_y_final = die_y_final + float(die_gap_fraction) * y_span
        die_x = np.linspace(x_min, x_max, int(die_y_final.shape[0]))
    elif die_fit_mode == "upper-boundary":
        final_upper_y = coords[:, 0, -1, 1]
        die_y_final = fit_profile_to_upper_boundary(
            die_profile,
            upper_y=final_upper_y,
            y_span=y_span,
            final_gap=float(die_gap_fraction),
        )
        die_x = np.linspace(x_min, x_max, int(die_y_final.shape[0]))
    else:
        die_y_final = (
            y_max
            + float(die_gap_fraction) * y_span
            + normalize_profile(die_profile) * die_amplitude_fraction * y_span
        )
    die_travel = float(die_travel_fraction) * y_span
    if not raw_coordinate_mode:
        die_y_initial = die_y_final + die_travel

    pad_x = 0.08 * x_span
    pad_bottom = 0.08 * y_span
    die_y_max = max(float(np.nanmax(die_y_initial)), float(np.nanmax(die_y_final)))
    pad_top = max(die_y_max - y_max, 0.0) + 0.16 * y_span

    vmin = float(np.nanmin(disp_mag))
    vmax = float(np.nanmax(disp_mag))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    reference = ax.scatter(
        material[:, :, 0, 0].reshape(-1),
        material[:, :, 0, 1].reshape(-1),
        s=max(point_size * 0.35, 1.0),
        c="#cbd5e1",
        linewidths=0,
        alpha=0.35,
        # label="material grid",
    )
    scatter = ax.scatter(
        coords[:, :, 0, 0].reshape(-1),
        coords[:, :, 0, 1].reshape(-1),
        c=disp_mag[:, :, 0].reshape(-1),
        s=point_size,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        label="deformed block",
    )
    die_x0 = die_x
    die_y0 = die_y_initial
    die_fill_x0 = np.concatenate([die_x0, die_x0[::-1]])
    die_fill_y0 = np.concatenate([die_y0, np.full_like(die_y0, y_max + pad_top)])
    (die_line,) = ax.plot(
        die_x0,
        die_y0,
        color="#111827",
        linewidth=2.4,
        label="input die",
    )
    (die_fill,) = ax.fill(
        die_fill_x0,
        die_fill_y0,
        color="#111827",
        alpha=0.10,
    )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.015, fraction=0.03, shrink=0.7, aspect=30)
    cbar.set_label("||u||")

    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(x_min - pad_x, x_max + pad_x)
    ax.set_ylim(y_min - pad_bottom, y_max + pad_top)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    def die_xy(timestep: int) -> tuple[np.ndarray, np.ndarray]:
        if raw_coordinate_mode:
            return die_x, die_y_initial - float(die_speed_per_step) * timestep
        progress = timestep / max(t_count - 1, 1)
        return die_x, die_y_final + (1.0 - progress) * die_travel

    def update(timestep: int):
        x = coords[:, :, timestep, 0].reshape(-1)
        y = coords[:, :, timestep, 1].reshape(-1)
        scatter.set_offsets(np.column_stack([x, y]))
        scatter.set_array(disp_mag[:, :, timestep].reshape(-1))

        dx, dy = die_xy(timestep)
        die_line.set_data(dx, dy)
        die_fill.set_xy(
            np.column_stack(
                [
                    np.concatenate([dx, dx[::-1]]),
                    np.concatenate([dy, np.full_like(dy, y_max + pad_top)]),
                ]
            )
        )
        title.set_text(
            f"Plasticity forging sample {sample_idx}  |  timestep {timestep + 1}/{t_count}"
        )
        return reference, scatter, die_line, die_fill, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=t_count,
        interval=int(1000 / max(fps, 1)),
        blit=False,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.PillowWriter(fps=max(int(fps), 1))
    anim.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)

    final_top = np.nanmax(coords[:, :, -1, 1], axis=1)
    final_die_at_top = np.interp(
        np.linspace(0.0, 1.0, final_top.shape[0]),
        np.linspace(0.0, 1.0, die_y_final.shape[0]),
        die_y_final,
    )
    crossing = final_top - final_die_at_top
    max_crossing = float(np.nanmax(crossing))
    if max_crossing > 1e-8:
        print(
            "WARNING: final block envelope crosses visual die by "
            f"{max_crossing:.6g} coordinate units."
        )
    if raw_coordinate_mode:
        print(
            "Raw-coordinate die speed: "
            f"{float(die_speed):.6g} y-units/time-unit, "
            f"{float(die_speed_per_step):.6g} y-units/frame, "
            f"final displayed drop={float(die_speed_per_step) * max(t_count - 1, 0):.6g}"
        )
    return out_path


def make_concatenated_animation(
    *,
    die_profiles: np.ndarray,
    samples: np.ndarray,
    sample_indices: list[int],
    out_path: Path,
    fps: int,
    dpi: int,
    point_size: float,
    die_travel_fraction: float,
    die_fit_mode: str,
    die_speed: float | None,
    time_duration: float,
    die_gap_fraction: float,
    die_amplitude_fraction: float,
    flip_die_x: bool = False,
) -> Path:
    """Render selected samples and concatenate their frames into one GIF."""
    if len(sample_indices) == 0:
        raise ValueError("sample_indices must contain at least one sample")
    validate_samples(sample_indices, int(samples.shape[0]))

    if len(sample_indices) == 1:
        sample_idx = int(sample_indices[0])
        die_profile = die_profiles[sample_idx]
        if flip_die_x:
            die_profile = die_profile[::-1]
        return make_animation(
            die_profile=die_profile,
            sample=samples[sample_idx],
            sample_idx=sample_idx,
            out_path=out_path,
            fps=fps,
            dpi=dpi,
            point_size=point_size,
            die_travel_fraction=die_travel_fraction,
            die_fit_mode=die_fit_mode,
            die_speed=die_speed,
            time_duration=time_duration,
            die_gap_fraction=die_gap_fraction,
            die_amplitude_fraction=die_amplitude_fraction,
        )

    try:
        from PIL import Image, ImageSequence
    except ImportError as exc:
        raise RuntimeError("Pillow is required to concatenate GIFs.") from exc

    frames = []
    with tempfile.TemporaryDirectory(prefix="plasticity_forging_gif_") as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        for sample_idx in sample_indices:
            sample_idx = int(sample_idx)
            die_profile = die_profiles[sample_idx]
            if flip_die_x:
                die_profile = die_profile[::-1]
            tmp_path = tmp_dir_path / f"sample_{sample_idx:04d}.gif"
            make_animation(
                die_profile=die_profile,
                sample=samples[sample_idx],
                sample_idx=sample_idx,
                out_path=tmp_path,
                fps=fps,
                dpi=dpi,
                point_size=point_size,
                die_travel_fraction=die_travel_fraction,
                die_fit_mode=die_fit_mode,
                die_speed=die_speed,
                time_duration=time_duration,
                die_gap_fraction=die_gap_fraction,
                die_amplitude_fraction=die_amplitude_fraction,
            )
            with Image.open(tmp_path) as gif:
                frames.extend(
                    frame.convert("RGB") for frame in ImageSequence.Iterator(gif)
                )

    if not frames:
        raise RuntimeError("No frames were rendered for concatenated GIF.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_duration_ms = int(round(1000 / max(int(fps), 1)))
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    return out_path


def plot_boundary_debug(
    *,
    die_profile: np.ndarray,
    sample: np.ndarray,
    sample_idx: int,
    out_path: Path,
) -> Path:
    require_matplotlib(plt)
    die_profile = np.asarray(die_profile, dtype=np.float64)
    coords = sample[..., 0:2].astype(np.float64)
    material = infer_material_grid(sample)
    t_count = int(sample.shape[2])
    x = material[:, 0, 0, 0]
    y_span = max(float(np.nanmax(coords[..., 1]) - np.nanmin(coords[..., 1])), 1e-6)

    die_norm = normalize_profile(die_profile)
    if die_norm.shape[0] != x.shape[0]:
        die_norm = np.interp(
            np.linspace(0.0, 1.0, x.shape[0]),
            np.linspace(0.0, 1.0, die_norm.shape[0]),
            die_norm,
        )

    top_y = np.nanmax(coords[..., 1], axis=1)
    top_j = np.argmax(coords[..., 1], axis=1)
    top_x = coords[
        np.arange(coords.shape[0])[:, None],
        top_j,
        np.arange(t_count)[None, :],
        0,
    ]
    nominal_upper_y = coords[:, 0, :, 1]
    initial_top_y = top_y[:, 0]
    indentation_depth = initial_top_y[:, None] - top_y

    fit_rows = []
    design = np.stack([np.ones_like(die_norm), die_norm], axis=1)
    for timestep in range(1, t_count):
        target = indentation_depth[:, timestep]
        if float(np.nanmax(np.abs(target))) <= 1e-10:
            continue
        coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = coeff[0] + coeff[1] * die_norm
        flipped_coeff, *_ = np.linalg.lstsq(
            np.stack([np.ones_like(die_norm), -die_norm], axis=1),
            target,
            rcond=None,
        )
        flipped_fitted = flipped_coeff[0] - flipped_coeff[1] * die_norm
        if np.nanmean((flipped_fitted - target) ** 2) < np.nanmean(
            (fitted - target) ** 2
        ):
            coeff = np.array([flipped_coeff[0], -flipped_coeff[1]], dtype=np.float64)
            fitted = flipped_fitted
        rmse = float(np.sqrt(np.nanmean((fitted - target) ** 2)))
        fit_rows.append((rmse, timestep, coeff, fitted))
    if not fit_rows:
        raise RuntimeError(
            "Could not find a non-flat timestep for die/deformation fitting."
        )
    best_rmse, best_timestep, best_coeff, best_die = min(
        fit_rows, key=lambda item: item[0]
    )

    fig, axes = plt.subplots(4, 1, figsize=(9.2, 11.0), dpi=150, sharex=True)
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    for timestep in range(t_count):
        alpha = 0.15 + 0.55 * timestep / max(t_count - 1, 1)
        color = cmap(timestep / max(t_count - 1, 1))
        ax.plot(
            x,
            top_y[:, timestep],
            color=color,
            alpha=alpha,
            linewidth=0.9,
        )
    ax.scatter(
        top_x[:, best_timestep],
        top_y[:, best_timestep],
        s=14,
        color="#dc2626",
        linewidths=0,
        label=f"top point per x at best fit timestep {best_timestep}",
        zorder=4,
    )
    ax.plot(
        x,
        initial_top_y - best_die,
        color="#2563eb",
        linewidth=2.3,
        label=f"initial top minus best-fit die depth, RMSE={best_rmse:.4g}",
        zorder=5,
    )
    ax.plot(
        x,
        nominal_upper_y[:, best_timestep],
        color="#111827",
        linewidth=1.4,
        linestyle="--",
        label=f"nominal j=0 edge at timestep {best_timestep}",
    )
    ax.set_title(
        f"Plasticity sample {sample_idx}: input die versus top material points"
    )
    ax.set_ylabel("physical y coordinate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    raw_ax = axes[1]
    raw_ax.plot(
        x,
        die_profile,
        color="#2563eb",
        linewidth=2.0,
        label="given input die, x-aligned",
    )
    raw_ax.set_title("Given input die profile, raw dataset values")
    raw_ax.set_ylabel("input value")
    raw_ax.grid(True, alpha=0.25)
    raw_ax.legend(fontsize=8)

    profile_ax = axes[2]
    profile_ax.plot(
        x, die_norm, color="#2563eb", linewidth=2.0, label="given input die, normalized"
    )
    profile_ax.plot(
        x,
        normalize_profile(indentation_depth[:, best_timestep]),
        color="#dc2626",
        linewidth=1.8,
        label=f"top indentation depth, normalized, timestep {best_timestep}",
    )
    profile_ax.plot(
        x,
        normalize_profile(nominal_upper_y[:, 0] - nominal_upper_y[:, best_timestep]),
        color="#111827",
        linewidth=1.2,
        linestyle="--",
        label=f"nominal j=0 indentation, normalized, timestep {best_timestep}",
    )
    profile_ax.set_title("Shape comparison after removing offset and scale")
    profile_ax.set_ylabel("normalized shape")
    profile_ax.grid(True, alpha=0.25)
    profile_ax.legend(fontsize=8)

    residual_ax = axes[3]
    residual_ax.plot(
        x,
        indentation_depth[:, best_timestep] - best_die,
        color="#dc2626",
        linewidth=1.8,
    )
    residual_ax.axhline(0.0, color="#111827", linewidth=0.8)
    residual_ax.set_title("Top indentation depth minus best-fit die depth")
    residual_ax.set_xlabel("physical x coordinate")
    residual_ax.set_ylabel("y difference")
    residual_ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(
        "Boundary debug: best input-die fit to top indentation occurs at "
        f"timestep={best_timestep} with RMSE={best_rmse:.6g}"
    )
    print(
        "Boundary debug: best affine fit "
        f"indentation_depth={float(best_coeff[0]):.6g} + "
        f"{float(best_coeff[1]):.6g} * normalized_die"
    )
    print(
        "Boundary debug: max(top_y - nominal_j0_y) at best timestep="
        f"{float(np.nanmax(top_y[:, best_timestep] - nominal_upper_y[:, best_timestep])):.6g}"
    )
    return out_path


def main() -> None:
    args = parse_args()
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
    sample_indices = (
        [int(sample_idx) for sample_idx in args.samples]
        if args.samples is not None
        else [int(args.sample)]
    )
    validate_samples(sample_indices, int(output.shape[0]))

    out_path = args.out_path
    if out_path is None:
        if len(sample_indices) == 1:
            sample_tag = f"sample_{sample_indices[0]:04d}"
        else:
            sample_tag = "samples_" + "_".join(f"{idx:04d}" for idx in sample_indices)
        out_path = (
            Path("artifacts")
            / "plasticity"
            / "plasticity_forging_gif"
            / f"plasticity_forging_{sample_tag}.gif"
        )

    if args.boundary_debug_path is not None:
        if len(sample_indices) != 1:
            raise ValueError("--boundary-debug-path can only be used with one sample")
        sample_idx = sample_indices[0]
        die_profile = die[sample_idx]
        if args.flip_die_x:
            die_profile = die_profile[::-1]
        debug_path = plot_boundary_debug(
            die_profile=die_profile,
            sample=output[sample_idx],
            sample_idx=sample_idx,
            out_path=args.boundary_debug_path,
        )
        print(f"Wrote boundary debug: {debug_path}")

    saved_path = make_concatenated_animation(
        die_profiles=die,
        samples=output,
        sample_indices=sample_indices,
        out_path=out_path,
        fps=args.fps,
        dpi=args.dpi,
        point_size=args.point_size,
        die_travel_fraction=args.die_travel_fraction,
        die_fit_mode=args.die_fit_mode,
        die_speed=args.die_speed,
        time_duration=args.time_duration,
        die_gap_fraction=args.die_gap_fraction,
        die_amplitude_fraction=args.die_amplitude_fraction,
        flip_die_x=args.flip_die_x,
    )
    print(f"Loaded: {mat_path}")
    print(f"Wrote: {saved_path}")


if __name__ == "__main__":
    main()
