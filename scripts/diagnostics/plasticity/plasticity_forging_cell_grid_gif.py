from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
except ImportError:
    animation = None
    plt = None
    PolyCollection = None

from _common import (
    load_plasticity_arrays,
    require_matplotlib,
    select_split,
    validate_samples,
)
from plasticity_forging_gif import (
    fit_profile_to_upper_boundary,
    infer_material_grid,
    normalize_profile,
    plot_boundary_debug,
    smooth_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an animated GIF for the plasticity forging benchmark using "
            "the deformed quadrilateral cell grid instead of individual points."
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
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Output GIF path. Defaults to artifacts/plasticity/"
            "plasticity_forging_cell_grid_gif/"
            "plasticity_forging_cell_grid_sample_XXXX.gif."
        ),
    )
    parser.add_argument(
        "--focus-window",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help=(
            "Optional physical-coordinate window for a second focused GIF. "
            "Bounds may be given in either order. Example: "
            "--focus-window -35 -15 15 5."
        ),
    )
    parser.add_argument(
        "--focus-out-path",
        type=Path,
        default=None,
        help=(
            "Output path for the focused GIF. Defaults to the full GIF path "
            "with '_focus' appended before the suffix."
        ),
    )
    parser.add_argument(
        "--focus-padding-fraction",
        type=float,
        default=0.03,
        help=(
            "Extra padding around --focus-window as a fraction of the larger "
            "window span."
        ),
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument(
        "--cell-alpha",
        type=float,
        default=0.78,
        help="Face alpha for deformed quadrilateral cells.",
    )
    parser.add_argument(
        "--edge-linewidth",
        type=float,
        default=0.5,
        help="Line width for deformed quadrilateral cell edges.",
    )
    parser.add_argument(
        "--reference-linewidth",
        type=float,
        default=0.12,
        help="Line width for the undeformed reference cell grid.",
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
        help=("How to draw the input die profile. Matches plasticity_forging_gif.py."),
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
            "The displayed frames are sampled at t/T_out."
        ),
    )
    parser.add_argument(
        "--flip-die-x",
        action="store_true",
        help="Reverse the input die profile along x before plotting.",
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


def quadrilateral_cells(coords_t: np.ndarray) -> np.ndarray:
    """Return cells with the same vertex ordering as signed_cell_areas."""
    p00 = coords_t[:-1, :-1, :]
    p10 = coords_t[1:, :-1, :]
    p11 = coords_t[1:, 1:, :]
    p01 = coords_t[:-1, 1:, :]
    return np.stack([p00, p10, p11, p01], axis=2).reshape(-1, 4, 2)


def cell_corner_average(values_t: np.ndarray) -> np.ndarray:
    p00 = values_t[:-1, :-1]
    p10 = values_t[1:, :-1]
    p11 = values_t[1:, 1:]
    p01 = values_t[:-1, 1:]
    return 0.25 * (p00 + p10 + p11 + p01).reshape(-1)


def make_animation(
    *,
    die_profile: np.ndarray,
    sample: np.ndarray,
    sample_idx: int,
    out_path: Path,
    fps: int,
    dpi: int,
    cell_alpha: float,
    edge_linewidth: float,
    reference_linewidth: float,
    die_travel_fraction: float,
    die_fit_mode: str,
    die_speed: float | None,
    time_duration: float,
    die_gap_fraction: float,
    die_amplitude_fraction: float,
    view_bounds: tuple[float, float, float, float] | None = None,
    title_suffix: str = "",
) -> Path:
    require_matplotlib(plt)
    if animation is None or PolyCollection is None:
        raise RuntimeError("matplotlib.animation and PolyCollection are required.")

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
    if view_bounds is None:
        xlim = (x_min - pad_x, x_max + pad_x)
        ylim = (y_min - pad_bottom, y_max + pad_top)
    else:
        focus_x_min, focus_x_max, focus_y_min, focus_y_max = view_bounds
        xlim = (float(focus_x_min), float(focus_x_max))
        ylim = (float(focus_y_min), float(focus_y_max))

    cell_values = [
        cell_corner_average(disp_mag[:, :, timestep]) for timestep in range(t_count)
    ]
    vmin = float(np.nanmin(cell_values))
    vmax = float(np.nanmax(cell_values))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8fafc")

    reference_cells = quadrilateral_cells(material[:, :, 0, :])
    reference_grid = PolyCollection(
        reference_cells,
        facecolors="none",
        edgecolors="#94a3b8",
        linewidths=max(float(reference_linewidth), 0.0),
        alpha=0.45,
        # label="reference cell grid",
    )
    ax.add_collection(reference_grid)

    cell_grid = PolyCollection(
        quadrilateral_cells(coords[:, :, 0, :]),
        cmap="viridis",
        edgecolors="#0f172a",
        linewidths=max(float(edge_linewidth), 0.0),
        alpha=float(np.clip(cell_alpha, 0.0, 1.0)),
        label="deformed quadrilateral cells",
    )
    cell_grid.set_array(cell_values[0])
    cell_grid.set_clim(vmin, vmax)
    ax.add_collection(cell_grid)

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

    cbar = fig.colorbar(cell_grid, ax=ax, pad=0.02)
    cbar.set_label("||u||")

    title = ax.set_title("")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#d1d5db", linewidth=0.7, alpha=0.55)
    ax.legend(loc="lower right", frameon=True, fontsize=8)

    def die_xy(timestep: int) -> tuple[np.ndarray, np.ndarray]:
        if raw_coordinate_mode:
            return die_x, die_y_initial - float(die_speed_per_step) * timestep
        progress = timestep / max(t_count - 1, 1)
        return die_x, die_y_final + (1.0 - progress) * die_travel

    def update(timestep: int):
        cell_grid.set_verts(quadrilateral_cells(coords[:, :, timestep, :]))
        cell_grid.set_array(cell_values[timestep])

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
            f"Plasticity forging cell grid sample {sample_idx}{title_suffix}  |  "
            f"timestep {timestep + 1}/{t_count}"
        )
        return reference_grid, cell_grid, die_line, die_fill, title

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


def padded_focus_window(
    values: list[float] | None,
    *,
    padding_fraction: float,
) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    x0, x1, y0, y1 = (float(value) for value in values)
    if not all(np.isfinite([x0, x1, y0, y1])):
        raise ValueError("--focus-window values must be finite")
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))
    if x_min >= x_max:
        raise ValueError("--focus-window requires distinct x bounds")
    if y_min >= y_max:
        raise ValueError("--focus-window requires distinct y bounds")

    padding = max(x_max - x_min, y_max - y_min) * max(float(padding_fraction), 0.0)
    return (
        x_min - padding,
        x_max + padding,
        y_min - padding,
        y_max + padding,
    )


def default_focus_out_path(out_path: Path) -> Path:
    suffix = out_path.suffix or ".gif"
    return out_path.with_name(f"{out_path.stem}_focus{suffix}")


def main() -> None:
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    focus_bounds = padded_focus_window(
        args.focus_window,
        padding_fraction=float(args.focus_padding_fraction),
    )

    die, output, mat_path = load_plasticity_arrays(args.data_dir)
    die, output = select_split(
        die,
        output,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    validate_samples([args.sample], int(output.shape[0]))

    out_path = args.out_path
    if out_path is None:
        out_path = (
            Path("artifacts")
            / "plasticity"
            / "plasticity_forging_cell_grid_gif"
            / f"plasticity_forging_cell_grid_sample_{args.sample:04d}.gif"
        )

    die_profile = die[args.sample]
    if args.flip_die_x:
        die_profile = die_profile[::-1]

    if args.boundary_debug_path is not None:
        debug_path = plot_boundary_debug(
            die_profile=die_profile,
            sample=output[args.sample],
            sample_idx=args.sample,
            out_path=args.boundary_debug_path,
        )
        print(f"Wrote boundary debug: {debug_path}")

    saved_path = make_animation(
        die_profile=die_profile,
        sample=output[args.sample],
        sample_idx=args.sample,
        out_path=out_path,
        fps=args.fps,
        dpi=args.dpi,
        cell_alpha=args.cell_alpha,
        edge_linewidth=args.edge_linewidth,
        reference_linewidth=args.reference_linewidth,
        die_travel_fraction=args.die_travel_fraction,
        die_fit_mode=args.die_fit_mode,
        die_speed=args.die_speed,
        time_duration=args.time_duration,
        die_gap_fraction=args.die_gap_fraction,
        die_amplitude_fraction=args.die_amplitude_fraction,
    )
    print(f"Loaded: {mat_path}")
    print(f"Wrote: {saved_path}")

    if focus_bounds is not None:
        focus_out_path = (
            args.focus_out_path
            if args.focus_out_path is not None
            else default_focus_out_path(saved_path)
        )
        focused_path = make_animation(
            die_profile=die_profile,
            sample=output[args.sample],
            sample_idx=args.sample,
            out_path=focus_out_path,
            fps=args.fps,
            dpi=args.dpi,
            cell_alpha=args.cell_alpha,
            edge_linewidth=args.edge_linewidth,
            reference_linewidth=args.reference_linewidth,
            die_travel_fraction=args.die_travel_fraction,
            die_fit_mode=args.die_fit_mode,
            die_speed=args.die_speed,
            time_duration=args.time_duration,
            die_gap_fraction=args.die_gap_fraction,
            die_amplitude_fraction=args.die_amplitude_fraction,
            view_bounds=focus_bounds,
            title_suffix=" focused",
        )
        print(f"Wrote focused: {focused_path}")


if __name__ == "__main__":
    main()
