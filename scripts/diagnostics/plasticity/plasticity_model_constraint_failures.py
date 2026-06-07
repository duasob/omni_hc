from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
except ImportError:
    plt = None
    LineCollection = None
    PolyCollection = None

try:
    from PIL import Image
except ImportError:
    Image = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
NSL_ROOT = PROJECT_ROOT / "external" / "Neural-Solver-Library"
for path in (SRC_ROOT, NSL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import require_matplotlib, write_csv

from omni_hc.benchmarks.plasticity.data import build_test_loader
from omni_hc.core import compose_run_config, parse_dotted_overrides
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import (
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
    resolve_output_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize where a trained plasticity model violates the geometric "
            "assumptions encoded by PlasticityMeshConsistencyConstraint."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Resolved or experiment YAML config. If omitted, component flags are used.",
    )
    parser.add_argument("--benchmark", type=str, default="plasticity")
    parser.add_argument("--backbone", type=str, default="FNO")
    parser.add_argument(
        "--constraint",
        type=str,
        default="none",
        help="Constraint used for model construction. Omit for unconstrained checkpoints.",
    )
    parser.add_argument("--budget", type=str, default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timestep indices to render and include in the GIF. Defaults to all.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <output_dir>/diagnostics/plasticity_constraint_failures.",
    )
    parser.add_argument(
        "--focus-window",
        type=float,
        nargs=4,
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
        default=None,
        help=(
            "Optional physical-coordinate window to zoom every frame. The "
            "script converts it to a stable i/j crop over the selected timesteps."
        ),
    )
    parser.add_argument(
        "--focus-padding-fraction",
        type=float,
        default=0.03,
        help="Extra padding around --focus-window as a fraction of the larger span.",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1.0e-3,
        help="Point residual tolerance for |xy - (material + u)|.",
    )
    parser.add_argument(
        "--cell-area-tol",
        type=float,
        default=1.0e-10,
        help="Minimum oriented cell area before a cell is marked failed.",
    )
    parser.add_argument(
        "--axis-order-tol",
        type=float,
        default=0.0,
        help="Minimum signed nearest-neighbor spacing before adjacent cells are marked failed.",
    )
    parser.add_argument(
        "--bottom-y-tol",
        type=float,
        default=1.0e-3,
        help="Bottom-edge y tolerance against the material grid bottom boundary.",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override. Repeatable.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_config(args: argparse.Namespace) -> dict:
    overrides = parse_dotted_overrides(args.override)
    if args.config is not None:
        return compose_run_config(
            experiment=args.config,
            mode="test",
            extra_overrides=overrides,
        )
    return compose_run_config(
        benchmark=args.benchmark,
        backbone=args.backbone,
        constraint=args.constraint,
        budget=args.budget,
        mode="test",
        extra_overrides=overrides,
    )


def runtime_overrides(meta: dict) -> dict:
    return {
        "shapelist": tuple(meta["shapelist"]),
        "task": "dynamic_conditional",
        "loader": "plas",
        "geotype": "structured_2D",
        "space_dim": int(meta["space_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "out_dim": int(meta["out_dim"]),
        "T_out": int(meta["t_out"]),
        "time_input": True,
    }


def decode_if_needed(normalizer, tensor: torch.Tensor) -> torch.Tensor:
    return normalizer.decode(tensor) if normalizer is not None else tensor


def signed_cell_areas(coords: np.ndarray) -> np.ndarray:
    p00 = coords[:-1, :-1]
    p10 = coords[1:, :-1]
    p11 = coords[1:, 1:]
    p01 = coords[:-1, 1:]
    vertices = np.stack((p00, p10, p11, p01), axis=-2)
    x = vertices[..., 0]
    y = vertices[..., 1]
    return 0.5 * (
        np.sum(x * np.roll(y, shift=-1, axis=-1), axis=-1)
        - np.sum(y * np.roll(x, shift=-1, axis=-1), axis=-1)
    )


def infer_material(target_seq: np.ndarray) -> np.ndarray:
    return target_seq[..., :2] - target_seq[..., 2:4]


def plot_coords(seq: np.ndarray, material: np.ndarray) -> np.ndarray:
    return 0.5 * (seq[..., :2] + material + seq[..., 2:4])


def mesh_segments(coords: np.ndarray) -> list[np.ndarray]:
    segments: list[np.ndarray] = []
    for i in range(coords.shape[0]):
        segments.append(coords[i, :, :])
    for j in range(coords.shape[1]):
        segments.append(coords[:, j, :])
    return segments


def cell_polygons(coords: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for i, j in zip(*np.nonzero(mask)):
        polygons.append(
            np.array(
                [
                    coords[i, j],
                    coords[i + 1, j],
                    coords[i + 1, j + 1],
                    coords[i, j + 1],
                ]
            )
        )
    return polygons


def cells_touching_points(point_mask: np.ndarray) -> np.ndarray:
    cells = np.zeros(
        (max(point_mask.shape[0] - 1, 0), max(point_mask.shape[1] - 1, 0)),
        dtype=bool,
    )
    if cells.size == 0:
        return cells
    cells |= point_mask[:-1, :-1]
    cells |= point_mask[1:, :-1]
    cells |= point_mask[1:, 1:]
    cells |= point_mask[:-1, 1:]
    return cells


def constraint_failure_fields(
    pred_seq: np.ndarray,
    target_seq: np.ndarray,
    *,
    timestep: int,
    coord_tol: float,
    cell_area_tol: float,
    axis_order_tol: float,
    bottom_y_tol: float,
) -> dict[str, np.ndarray | float | int]:
    material = infer_material(target_seq)
    pred_coords = plot_coords(pred_seq, material)
    target_coords = plot_coords(target_seq, material)
    pred_t = pred_seq[:, :, timestep]
    coords_t = pred_coords[:, :, timestep]
    target_t = target_coords[:, :, timestep]
    material_t = material[:, :, timestep]

    coord_residual = np.linalg.norm(
        pred_t[..., :2] - (material_t + pred_t[..., 2:4]),
        axis=-1,
    )
    channel_bad = coord_residual > float(coord_tol)
    bottom_y_residual = np.abs(coords_t[:, -1, 1] - material_t[:, -1, 1])
    bottom_y_bad = bottom_y_residual > float(bottom_y_tol)
    bottom_point_bad = np.zeros_like(channel_bad, dtype=bool)
    bottom_point_bad[:, -1] = bottom_y_bad

    target_area = signed_cell_areas(target_t)
    orientation = np.sign(np.nanmean(target_area))
    if orientation == 0.0 or not np.isfinite(orientation):
        orientation = 1.0
    oriented_area = signed_cell_areas(coords_t) * orientation
    area_bad = oriented_area <= float(cell_area_tol)

    dx_spacing = coords_t[:-1, :, 0] - coords_t[1:, :, 0]
    dy_spacing = coords_t[:, :-1, 1] - coords_t[:, 1:, 1]
    dx_bad = dx_spacing <= float(axis_order_tol)
    dy_bad = dy_spacing <= float(axis_order_tol)

    cell_bad = area_bad.copy()
    cell_bad |= dx_bad[:, :-1] | dx_bad[:, 1:]
    cell_bad |= dy_bad[:-1, :] | dy_bad[1:, :]
    affected_cell = cell_bad | cells_touching_points(channel_bad | bottom_point_bad)

    return {
        "coords": coords_t,
        "target_coords": target_t,
        "coord_residual": coord_residual,
        "channel_bad": channel_bad,
        "bottom_y_residual": bottom_y_residual,
        "bottom_y_bad": bottom_y_bad,
        "oriented_area": oriented_area,
        "area_bad": area_bad,
        "dx_spacing": dx_spacing,
        "dy_spacing": dy_spacing,
        "cell_bad": cell_bad,
        "affected_cell": affected_cell,
        "bad_cell_count": int(np.count_nonzero(cell_bad)),
        "affected_cell_count": int(np.count_nonzero(affected_cell)),
        "bad_channel_point_count": int(np.count_nonzero(channel_bad)),
        "bad_bottom_point_count": int(np.count_nonzero(bottom_y_bad)),
        "min_oriented_area": float(np.nanmin(oriented_area)),
        "min_dx_spacing": float(np.nanmin(dx_spacing)),
        "min_dy_spacing": float(np.nanmin(dy_spacing)),
        "max_bottom_y_residual": float(np.nanmax(bottom_y_residual)),
        "max_coord_residual": float(np.nanmax(coord_residual)),
        "mean_coord_residual": float(np.nanmean(coord_residual)),
    }


def set_limits(ax, *coord_sets: np.ndarray) -> None:
    finite = [
        coords.reshape(-1, 2)
        for coords in coord_sets
        if coords is not None and np.isfinite(coords).any()
    ]
    if not finite:
        return
    merged = np.concatenate(finite, axis=0)
    x_min, y_min = np.nanmin(merged, axis=0)
    x_max, y_max = np.nanmax(merged, axis=0)
    x_span = max(float(x_max - x_min), 1.0e-6)
    y_span = max(float(y_max - y_min), 1.0e-6)
    ax.set_xlim(float(x_min) - 0.05 * x_span, float(x_max) + 0.05 * x_span)
    ax.set_ylim(float(y_min) - 0.05 * y_span, float(y_max) + 0.05 * y_span)


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


def infer_focus_crop(
    coords_seq: np.ndarray,
    *,
    timesteps: list[int],
    window: tuple[float, float, float, float] | None,
) -> tuple[slice, slice, tuple[float, float, float, float] | None] | None:
    if window is None:
        return None
    x_min, x_max, y_min, y_max = window
    coords_selected = coords_seq[:, :, timesteps, :]
    point_mask = (
        (coords_selected[..., 0] >= x_min)
        & (coords_selected[..., 0] <= x_max)
        & (coords_selected[..., 1] >= y_min)
        & (coords_selected[..., 1] <= y_max)
    )
    ij_mask = np.any(point_mask, axis=2)
    if not np.any(ij_mask):
        raise ValueError(
            "--focus-window did not include any predicted mesh points over the "
            "selected timesteps."
        )

    i_idx, j_idx = np.nonzero(ij_mask)
    h, w = coords_seq.shape[:2]
    i0 = max(int(np.min(i_idx)) - 1, 0)
    i1 = min(int(np.max(i_idx)) + 2, h)
    j0 = max(int(np.min(j_idx)) - 1, 0)
    j1 = min(int(np.max(j_idx)) + 2, w)
    if i1 - i0 < 2:
        i0 = max(i0 - 1, 0)
        i1 = min(i1 + 1, h)
    if j1 - j0 < 2:
        j0 = max(j0 - 1, 0)
        j1 = min(j1 + 1, w)
    if i1 - i0 < 2 or j1 - j0 < 2:
        raise ValueError("--focus-window must cover at least one mesh cell.")
    return slice(i0, i1), slice(j0, j1), window


def apply_focus_crop(
    coords: np.ndarray,
    target_coords: np.ndarray,
    cell_mask: np.ndarray,
    area: np.ndarray,
    focus_crop,
):
    if focus_crop is None:
        return coords, target_coords, cell_mask, area, None
    i_slice, j_slice, window = focus_crop
    return (
        coords[i_slice, j_slice],
        target_coords[i_slice, j_slice],
        cell_mask[i_slice.start : i_slice.stop - 1, j_slice.start : j_slice.stop - 1],
        area[i_slice.start : i_slice.stop - 1, j_slice.start : j_slice.stop - 1],
        (i_slice, j_slice, window),
    )


def add_mesh(
    ax, coords: np.ndarray, *, color: str, linewidth: float, alpha: float
) -> None:
    collection = LineCollection(
        mesh_segments(coords),
        colors=color,
        linewidths=linewidth,
        alpha=alpha,
    )
    ax.add_collection(collection)


def draw_frame(
    *,
    pred_seq: np.ndarray,
    target_seq: np.ndarray,
    timestep: int,
    args: argparse.Namespace,
    sample: int,
    focus_crop=None,
    out_path: Path | None = None,
):
    fields = constraint_failure_fields(
        pred_seq,
        target_seq,
        timestep=timestep,
        coord_tol=args.coord_tol,
        cell_area_tol=args.cell_area_tol,
        axis_order_tol=args.axis_order_tol,
        bottom_y_tol=args.bottom_y_tol,
    )
    coords = fields["coords"]
    target_coords = fields["target_coords"]
    cell_bad = fields["cell_bad"]
    area = fields["oriented_area"]
    coords, target_coords, cell_bad, area, active_focus = apply_focus_crop(
        coords,
        target_coords,
        cell_bad,
        area,
        focus_crop,
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), dpi=args.dpi)
    fig.suptitle(
        f"Plasticity model constraint failures | sample {sample} | timestep {timestep}",
        fontsize=12,
    )

    ax = axes[0]
    add_mesh(ax, coords, color="#1f2937", linewidth=0.28, alpha=0.68)
    polygons = cell_polygons(coords, cell_bad)
    if polygons:
        ax.add_collection(
            PolyCollection(
                polygons,
                facecolors="#ef4444",
                edgecolors="#991b1b",
                linewidths=0.25,
                alpha=0.62,
            )
        )
        add_mesh(ax, coords, color="#111827", linewidth=0.2, alpha=0.42)
    ax.set_title("prediction mesh\nred cells have invalid area/order")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.35, alpha=0.16)
    if active_focus is None:
        set_limits(ax, coords, target_coords)
    else:
        _, _, window = active_focus
        ax.set_xlim(float(window[0]), float(window[1]))
        ax.set_ylim(float(window[2]), float(window[3]))

    ax = axes[1]
    max_area = max(
        float(np.nanpercentile(np.abs(area), 99)), args.cell_area_tol, 1.0e-12
    )
    extent = None
    if active_focus is not None:
        i_slice, j_slice, _ = active_focus
        extent = (
            float(i_slice.stop - 1),
            float(i_slice.start),
            float(j_slice.stop - 1),
            float(j_slice.start),
        )
    im = ax.imshow(
        area.T,
        origin="upper",
        aspect="auto",
        cmap="RdBu",
        vmin=-max_area,
        vmax=max_area,
        extent=extent,
    )
    if active_focus is None:
        ax.invert_xaxis()
    ax.set_title("oriented cell area\nblue: positive, red: negative")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("oriented area")

    summary = (
        f"bad_cells={fields['bad_cell_count']}  "
        f"bad_channel_pts={fields['bad_channel_point_count']}  "
        f"bad_bottom_pts={fields['bad_bottom_point_count']}  "
        f"min_area={fields['min_oriented_area']:.3e}  "
        f"max_channel_res={fields['max_coord_residual']:.3e}"
    )
    fig.text(0.5, 0.015, summary, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.94))

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
    return fig, fields


def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def save_gif(frames: list[np.ndarray], out_path: Path, *, fps: int) -> Path | None:
    if Image is None or not frames:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(int(1000 / max(int(fps), 1)), 1)
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path


def get_sample_batch(
    loader, sample_idx: int, *, device: torch.device
) -> dict[str, torch.Tensor]:
    dataset = loader.dataset
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample {sample_idx} is outside [0, {len(dataset)})")
    item = dataset[sample_idx]
    return {key: value.unsqueeze(0).to(device) for key, value in item.items()}


def predict_sequence(
    model, batch: dict[str, torch.Tensor], *, y_normalizer, t_out: int, out_dim: int
):
    preds = []
    aux_by_timestep = []
    with torch.no_grad():
        for timestep in range(t_out):
            input_t = batch["time"][:, timestep : timestep + 1].reshape(
                batch["coords"].shape[0],
                1,
            )
            out = forward_with_optional_aux(
                model,
                batch["coords"],
                batch["x"],
                T=input_t,
            )
            preds.append(decode_if_needed(y_normalizer, out["pred"]))
            aux_by_timestep.append(out["aux_tensors"])
    pred = torch.cat(preds, dim=-1)
    target = decode_if_needed(y_normalizer, batch["y"])
    return pred, target, aux_by_timestep


def main() -> None:
    args = parse_args()
    require_matplotlib(plt)
    if LineCollection is None or PolyCollection is None:
        raise RuntimeError("matplotlib collections are required for this diagnostic.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    focus_window = padded_focus_window(
        args.focus_window,
        padding_fraction=float(args.focus_padding_fraction),
    )

    cfg = load_config(args)
    device = resolve_device(args.device)
    output_dir = resolve_output_dir(cfg)
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    out_dir = (
        args.out_dir or output_dir / "diagnostics" / "plasticity_constraint_failures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_test_loader(cfg)
    meta = loader.plasticity_meta
    y_normalizer = getattr(loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    model, _, _ = create_model(
        cfg,
        device=device,
        runtime_overrides=runtime_overrides(meta),
    )
    checkpoint = load_checkpoint_state(checkpoint_path, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    batch = get_sample_batch(loader, args.sample, device=device)
    t_out = int(meta["t_out"])
    out_dim = int(meta["out_dim"])
    timesteps = list(range(t_out)) if args.timesteps is None else args.timesteps
    for timestep in timesteps:
        if timestep < 0 or timestep >= t_out:
            raise IndexError(f"timestep {timestep} is outside [0, {t_out})")

    pred, target, _ = predict_sequence(
        model,
        batch,
        y_normalizer=y_normalizer,
        t_out=t_out,
        out_dim=out_dim,
    )
    h, w = tuple(meta["shapelist"])
    pred_seq = pred[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    target_seq = target[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    focus_crop = infer_focus_crop(
        plot_coords(pred_seq, infer_material(target_seq)),
        timesteps=timesteps,
        window=focus_window,
    )
    if focus_crop is not None:
        i_slice, j_slice, window = focus_crop
        print(
            "focus crop: "
            f"i=[{i_slice.start}, {i_slice.stop}) "
            f"j=[{j_slice.start}, {j_slice.stop}) "
            f"window=({window[0]:.6g}, {window[1]:.6g}, "
            f"{window[2]:.6g}, {window[3]:.6g})"
        )

    frames = []
    rows: list[dict[str, object]] = []
    for timestep in timesteps:
        png_path = out_dir / f"sample_{args.sample:04d}_timestep_{timestep:03d}.png"
        fig, fields = draw_frame(
            pred_seq=pred_seq,
            target_seq=target_seq,
            timestep=timestep,
            args=args,
            sample=args.sample,
            focus_crop=focus_crop,
            out_path=png_path,
        )
        frames.append(fig_to_rgb(fig))
        if args.show:
            plt.show()
        plt.close(fig)
        rows.append(
            {
                "sample": args.sample,
                "timestep": timestep,
                "png": str(png_path),
                "affected_cell_count": fields["affected_cell_count"],
                "bad_cell_count": fields["bad_cell_count"],
                "bad_channel_point_count": fields["bad_channel_point_count"],
                "bad_bottom_point_count": fields["bad_bottom_point_count"],
                "min_oriented_area": fields["min_oriented_area"],
                "min_dx_spacing": fields["min_dx_spacing"],
                "min_dy_spacing": fields["min_dy_spacing"],
                "max_bottom_y_residual": fields["max_bottom_y_residual"],
                "mean_coord_residual": fields["mean_coord_residual"],
                "max_coord_residual": fields["max_coord_residual"],
                "coord_tol": args.coord_tol,
                "cell_area_tol": args.cell_area_tol,
                "axis_order_tol": args.axis_order_tol,
                "bottom_y_tol": args.bottom_y_tol,
            }
        )

    csv_path = out_dir / f"sample_{args.sample:04d}_constraint_failures.csv"
    write_csv(csv_path, rows)
    gif_path = save_gif(
        frames,
        out_dir / f"sample_{args.sample:04d}_constraint_failures.gif",
        fps=args.fps,
    )

    print(f"wrote summary: {csv_path}")
    if gif_path is not None:
        print(f"wrote gif: {gif_path}")
    elif Image is None:
        print("GIF skipped: Pillow is not installed.")
    for row in rows:
        print(
            f"t={row['timestep']} affected_cells={row['affected_cell_count']} "
            f"bad_cells={row['bad_cell_count']} "
            f"bad_channel_pts={row['bad_channel_point_count']} "
            f"bad_bottom_pts={row['bad_bottom_point_count']} "
            f"min_area={float(row['min_oriented_area']):.3e} "
            f"max_coord_res={float(row['max_coord_residual']):.3e} "
            f"png={row['png']}"
        )


if __name__ == "__main__":
    main()
