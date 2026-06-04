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
    from matplotlib.collections import LineCollection
except ImportError:
    plt = None
    LineCollection = None

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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render model-vs-ground-truth plasticity forging GIFs for selected "
            "test split samples."
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
        default=None,
        help="Constraint used for model construction. Omit for unconstrained checkpoints.",
    )
    parser.add_argument("--budget", type=str, default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0],
        help="Test split sample indices to render.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=None,
        help="Timestep indices to include. Defaults to all timesteps.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Defaults to <output_dir>/diagnostics/plasticity_model_gifs.",
    )
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument("--mesh-linewidth", type=float, default=0.22)
    parser.add_argument(
        "--mesh-step",
        type=int,
        default=1,
        help="Draw every Nth mesh line. Use >1 for faster/lighter GIFs.",
    )
    parser.add_argument(
        "--save-pngs",
        action="store_true",
        help="Also save every rendered frame as a PNG.",
    )
    parser.add_argument(
        "--show-envelope",
        action="store_true",
        help="Overlay the envelope constraint's moving die envelope when available.",
    )
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


def resolve_local_output_dir(args: argparse.Namespace, cfg: dict) -> Path:
    configured = Path(cfg["paths"]["output_dir"]).expanduser()
    if configured.exists() or not configured.is_absolute():
        return configured
    if args.checkpoint is not None:
        return Path(args.checkpoint).expanduser().parent
    if args.config is not None:
        return Path(args.config).expanduser().parent
    return configured


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


def infer_material(target_seq: np.ndarray) -> np.ndarray:
    return target_seq[..., :2] - target_seq[..., 2:4]


def plot_coords(seq: np.ndarray, material: np.ndarray) -> np.ndarray:
    return 0.5 * (seq[..., :2] + material + seq[..., 2:4])


def mesh_segments(coords: np.ndarray, *, step: int) -> list[np.ndarray]:
    step = max(int(step), 1)
    segments: list[np.ndarray] = []
    for i in range(0, coords.shape[0], step):
        segments.append(coords[i, :, :])
    for j in range(0, coords.shape[1], step):
        segments.append(coords[:, j, :])
    if (coords.shape[0] - 1) % step:
        segments.append(coords[-1, :, :])
    if (coords.shape[1] - 1) % step:
        segments.append(coords[:, -1, :])
    return segments


def add_mesh(
    ax,
    coords: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
    step: int,
) -> None:
    ax.add_collection(
        LineCollection(
            mesh_segments(coords, step=step),
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
        )
    )


def set_shared_limits(axes, *coord_sets: np.ndarray) -> None:
    merged = np.concatenate([coords.reshape(-1, 2) for coords in coord_sets], axis=0)
    x_min, y_min = np.nanmin(merged, axis=0)
    x_max, y_max = np.nanmax(merged, axis=0)
    x_span = max(float(x_max - x_min), 1.0e-6)
    y_span = max(float(y_max - y_min), 1.0e-6)
    for ax in axes:
        ax.set_xlim(float(x_min) - 0.04 * x_span, float(x_max) + 0.04 * x_span)
        ax.set_ylim(float(y_min) - 0.04 * y_span, float(y_max) + 0.04 * y_span)


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


def get_sample_batch(loader, sample_idx: int, *, device: torch.device) -> dict[str, torch.Tensor]:
    dataset = loader.dataset
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample {sample_idx} is outside [0, {len(dataset)})")
    item = dataset[sample_idx]
    return {key: value.unsqueeze(0).to(device) for key, value in item.items()}


def predict_sequence(model, batch: dict[str, torch.Tensor], *, y_normalizer, t_out: int):
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


def envelope_for_timestep(aux_by_timestep, timestep: int) -> tuple[np.ndarray, np.ndarray] | None:
    aux = aux_by_timestep[timestep] if timestep < len(aux_by_timestep) else {}
    if "envelope_x" not in aux or "envelope_y" not in aux:
        return None
    x = aux["envelope_x"][0].detach().cpu().numpy()
    y = aux["envelope_y"][0].detach().cpu().numpy()
    return x, y


def draw_frame(
    *,
    pred_coords: np.ndarray,
    target_coords: np.ndarray,
    timestep: int,
    sample: int,
    error_vmax: float,
    disp_vmax: float,
    args: argparse.Namespace,
    envelope: tuple[np.ndarray, np.ndarray] | None,
):
    pred_t = pred_coords[:, :, timestep]
    target_t = target_coords[:, :, timestep]
    error = np.linalg.norm(pred_t - target_t, axis=-1)
    disp_mag = np.linalg.norm(pred_t - pred_coords[:, :, 0], axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.3), dpi=args.dpi)
    fig.suptitle(f"Plasticity rollout | test sample {sample} | timestep {timestep}")

    ax = axes[0]
    add_mesh(
        ax,
        target_t,
        color="#1f2937",
        linewidth=float(args.mesh_linewidth),
        alpha=0.72,
        step=args.mesh_step,
    )
    ax.set_title("ground truth")

    ax = axes[1]
    im_disp = ax.scatter(
        pred_t[..., 0].reshape(-1),
        pred_t[..., 1].reshape(-1),
        c=disp_mag.reshape(-1),
        s=3.0,
        cmap="viridis",
        vmin=0.0,
        vmax=disp_vmax,
        linewidths=0.0,
    )
    add_mesh(
        ax,
        pred_t,
        color="#111827",
        linewidth=float(args.mesh_linewidth),
        alpha=0.52,
        step=args.mesh_step,
    )
    if envelope is not None:
        env_x, env_y = envelope
        ax.plot(env_x, env_y, color="#ef4444", linewidth=1.2, label="envelope")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("prediction")
    fig.colorbar(im_disp, ax=ax, fraction=0.046, pad=0.04).set_label("|motion|")

    ax = axes[2]
    add_mesh(
        ax,
        target_t,
        color="#64748b",
        linewidth=float(args.mesh_linewidth),
        alpha=0.55,
        step=args.mesh_step,
    )
    add_mesh(
        ax,
        pred_t,
        color="#ef4444",
        linewidth=max(float(args.mesh_linewidth) * 1.3, 0.25),
        alpha=0.62,
        step=args.mesh_step,
    )
    ax.plot(
        target_t[:, 0, 0],
        target_t[:, 0, 1],
        color="#334155",
        linewidth=1.0,
        label="gt top",
    )
    ax.plot(
        pred_t[:, 0, 0],
        pred_t[:, 0, 1],
        color="#dc2626",
        linewidth=1.0,
        label="pred top",
    )
    if envelope is not None:
        env_x, env_y = envelope
        ax.plot(env_x, env_y, color="#111827", linewidth=0.9, linestyle="--", label="envelope")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title("mesh overlay")

    ax = axes[3]
    im_err = ax.scatter(
        pred_t[..., 0].reshape(-1),
        pred_t[..., 1].reshape(-1),
        c=error.reshape(-1),
        s=4.0,
        cmap="magma",
        vmin=0.0,
        vmax=error_vmax,
        linewidths=0.0,
    )
    add_mesh(
        ax,
        pred_t,
        color="#111827",
        linewidth=float(args.mesh_linewidth),
        alpha=0.34,
        step=args.mesh_step,
    )
    ax.set_title("same-index coordinate error")
    fig.colorbar(im_err, ax=ax, fraction=0.046, pad=0.04).set_label("|pred - gt|")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linewidth=0.35, alpha=0.16)
    set_shared_limits(axes, pred_t, target_t)

    frame_rel_l2 = float(
        np.linalg.norm((pred_t - target_t).reshape(-1))
        / max(np.linalg.norm(target_t.reshape(-1)), 1.0e-12)
    )
    top_cols = slice(0, min(3, pred_t.shape[1]))
    top_error = error[:, top_cols]
    top_x_abs_error = np.abs(pred_t[:, 0, 0] - target_t[:, 0, 0])
    top_y_abs_error = np.abs(pred_t[:, 0, 1] - target_t[:, 0, 1])
    envelope_stats = {
        "envelope_pred_clearance_min": float("nan"),
        "envelope_pred_clearance_mean": float("nan"),
        "envelope_gt_top_y_abs_error_mean": float("nan"),
        "envelope_gt_top_y_abs_error_max": float("nan"),
    }
    if envelope is not None:
        _, env_y = envelope
        pred_clearance = env_y - pred_t[:, 0, 1]
        gt_top_y_error = np.abs(env_y - target_t[:, 0, 1])
        envelope_stats = {
            "envelope_pred_clearance_min": float(np.nanmin(pred_clearance)),
            "envelope_pred_clearance_mean": float(np.nanmean(pred_clearance)),
            "envelope_gt_top_y_abs_error_mean": float(np.nanmean(gt_top_y_error)),
            "envelope_gt_top_y_abs_error_max": float(np.nanmax(gt_top_y_error)),
        }
    fig.text(
        0.5,
        0.012,
        f"frame_rel_l2={frame_rel_l2:.4e}  "
        f"mean_coord_error={float(error.mean()):.4e}  "
        f"max_coord_error={float(error.max()):.4e}",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.045, 1.0, 0.93))
    return fig, {
        "frame_rel_l2": frame_rel_l2,
        "mean_coord_error": float(error.mean()),
        "max_coord_error": float(error.max()),
        "top3_mean_coord_error": float(top_error.mean()),
        "top3_max_coord_error": float(top_error.max()),
        "top_j0_mean_x_abs_error": float(top_x_abs_error.mean()),
        "top_j0_max_x_abs_error": float(top_x_abs_error.max()),
        "top_j0_mean_y_abs_error": float(top_y_abs_error.mean()),
        "top_j0_max_y_abs_error": float(top_y_abs_error.max()),
        **envelope_stats,
    }


def render_sample(
    *,
    model,
    loader,
    sample: int,
    timesteps: list[int],
    meta: dict,
    y_normalizer,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    batch = get_sample_batch(loader, sample, device=device)
    t_out = int(meta["t_out"])
    out_dim = int(meta["out_dim"])
    pred, target, aux_by_timestep = predict_sequence(
        model,
        batch,
        y_normalizer=y_normalizer,
        t_out=t_out,
    )
    h, w = tuple(meta["shapelist"])
    pred_seq = pred[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    target_seq = target[0].detach().cpu().reshape(h, w, t_out, out_dim).numpy()
    material = infer_material(target_seq)
    pred_coords = plot_coords(pred_seq, material)
    target_coords = plot_coords(target_seq, material)

    selected_error = np.linalg.norm(
        pred_coords[:, :, timesteps] - target_coords[:, :, timesteps],
        axis=-1,
    )
    selected_motion = np.linalg.norm(
        pred_coords[:, :, timesteps] - pred_coords[:, :, [0]],
        axis=-1,
    )
    error_vmax = float(np.nanpercentile(selected_error, 99.0)) or 1.0
    disp_vmax = float(np.nanpercentile(selected_motion, 99.0)) or 1.0

    frames: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    sample_dir = out_dir / f"sample_{sample:04d}"
    for timestep in timesteps:
        envelope = (
            envelope_for_timestep(aux_by_timestep, timestep)
            if args.show_envelope
            else None
        )
        fig, stats = draw_frame(
            pred_coords=pred_coords,
            target_coords=target_coords,
            timestep=timestep,
            sample=sample,
            error_vmax=error_vmax,
            disp_vmax=disp_vmax,
            args=args,
            envelope=envelope,
        )
        if args.save_pngs:
            sample_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(sample_dir / f"timestep_{timestep:03d}.png", bbox_inches="tight")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
        rows.append(
            {
                "sample": sample,
                "timestep": timestep,
                "frame_rel_l2": stats["frame_rel_l2"],
                "mean_coord_error": stats["mean_coord_error"],
                "max_coord_error": stats["max_coord_error"],
                "top3_mean_coord_error": stats["top3_mean_coord_error"],
                "top3_max_coord_error": stats["top3_max_coord_error"],
                "top_j0_mean_x_abs_error": stats["top_j0_mean_x_abs_error"],
                "top_j0_max_x_abs_error": stats["top_j0_max_x_abs_error"],
                "top_j0_mean_y_abs_error": stats["top_j0_mean_y_abs_error"],
                "top_j0_max_y_abs_error": stats["top_j0_max_y_abs_error"],
                "envelope_pred_clearance_min": stats["envelope_pred_clearance_min"],
                "envelope_pred_clearance_mean": stats["envelope_pred_clearance_mean"],
                "envelope_gt_top_y_abs_error_mean": stats[
                    "envelope_gt_top_y_abs_error_mean"
                ],
                "envelope_gt_top_y_abs_error_max": stats[
                    "envelope_gt_top_y_abs_error_max"
                ],
            }
        )

    gif_path = save_gif(
        frames,
        out_dir / f"sample_{sample:04d}_model_rollout.gif",
        fps=args.fps,
    )
    if gif_path is None:
        print("GIF skipped: Pillow is not installed.")
    else:
        print(f"wrote gif: {gif_path}")
    return rows


def main() -> None:
    args = parse_args()
    require_matplotlib(plt)
    if LineCollection is None:
        raise RuntimeError("matplotlib.collections.LineCollection is required.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.mesh_step <= 0:
        raise ValueError("--mesh-step must be positive")

    cfg = load_config(args)
    device = resolve_device(args.device)
    output_dir = resolve_local_output_dir(args, cfg)
    checkpoint_path = args.checkpoint or output_dir / "best.pt"
    out_dir = args.out_dir or output_dir / "diagnostics" / "plasticity_model_gifs"
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_test_loader(cfg)
    meta = loader.plasticity_meta
    x_normalizer = getattr(loader, "x_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
    y_normalizer = getattr(loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    model, _, _ = create_model(
        cfg,
        device=device,
        runtime_overrides=runtime_overrides(meta),
    )
    if (
        x_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_input_normalizer")
    ):
        model.constraint.set_input_normalizer(x_normalizer)
    checkpoint = load_checkpoint_state(checkpoint_path, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()

    t_out = int(meta["t_out"])
    timesteps = list(range(t_out)) if args.timesteps is None else args.timesteps
    for timestep in timesteps:
        if timestep < 0 or timestep >= t_out:
            raise IndexError(f"timestep {timestep} is outside [0, {t_out})")

    rows: list[dict[str, object]] = []
    for sample in args.samples:
        rows.extend(
            render_sample(
                model=model,
                loader=loader,
                sample=int(sample),
                timesteps=timesteps,
                meta=meta,
                y_normalizer=y_normalizer,
                out_dir=out_dir,
                args=args,
                device=device,
            )
        )

    csv_path = out_dir / "plasticity_model_gif_metrics.csv"
    write_csv(csv_path, rows)
    print(f"wrote metrics: {csv_path}")


if __name__ == "__main__":
    main()
