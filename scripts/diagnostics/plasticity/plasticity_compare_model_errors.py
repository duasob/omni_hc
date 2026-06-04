from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
NSL_ROOT = PROJECT_ROOT / "external" / "Neural-Solver-Library"
for path in (SRC_ROOT, NSL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _common import require_matplotlib, write_csv
from omni_hc.benchmarks.plasticity.data import build_test_loader
from omni_hc.core import compose_run_config
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import (
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
)
from plasticity_model_gif import decode_if_needed, runtime_overrides


@dataclass
class RunSpec:
    name: str
    config: Path
    checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare axis-wise plasticity prediction errors for two trained models."
        )
    )
    parser.add_argument(
        "--run-a-name",
        default="unconstrained_e50",
        help="Label for the first run.",
    )
    parser.add_argument(
        "--run-a-config",
        type=Path,
        default=Path(
            "outputs/plasticity/none/transolver/e50_t500/seed_42/resolved_config.yaml"
        ),
    )
    parser.add_argument(
        "--run-a-checkpoint",
        type=Path,
        default=Path("outputs/plasticity/none/transolver/e50_t500/seed_42/best.pt"),
    )
    parser.add_argument(
        "--run-b-name",
        default="envelope_e100",
        help="Label for the second run.",
    )
    parser.add_argument(
        "--run-b-config",
        type=Path,
        default=Path(
            "outputs/plasticity/plasticity_envelope_constraint/transolver/e100_t900/seed_42/resolved_config.yaml"
        ),
    )
    parser.add_argument(
        "--run-b-checkpoint",
        type=Path,
        default=Path(
            "outputs/plasticity/plasticity_envelope_constraint/transolver/e100_t900/seed_42/best.pt"
        ),
    )
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on test samples for quick probes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional evaluation batch size override.",
    )
    parser.add_argument(
        "--top-rows",
        type=int,
        default=3,
        help="Number of j rows nearest the die/top boundary to summarize.",
    )
    parser.add_argument(
        "--die-drop-threshold",
        type=float,
        default=0.05,
        help=(
            "Top-boundary points with GT y below the timestep max by at least "
            "this amount are treated as the die-contact/depression region."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/plasticity_error_comparison/unconstrained_e50_vs_envelope_e100"),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_run(spec: RunSpec, *, device: torch.device, batch_size: int | None):
    cfg = compose_run_config(experiment=str(spec.config), mode="test")
    if batch_size is not None:
        cfg.setdefault("evaluation", {})
        cfg["evaluation"]["batch_size"] = int(batch_size)
    loader = build_test_loader(cfg)
    meta = loader.plasticity_meta
    x_normalizer = getattr(loader, "x_normalizer", None)
    y_normalizer = getattr(loader, "y_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
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
    checkpoint = load_checkpoint_state(spec.checkpoint, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return cfg, loader, meta, model, y_normalizer


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device) for key, value in batch.items()}


def predict_batch(model, batch: dict[str, torch.Tensor], *, y_normalizer, t_out: int):
    preds = []
    with torch.no_grad():
        for timestep in range(t_out):
            input_t = batch["time"][:, timestep : timestep + 1].reshape(
                batch["coords"].shape[0],
                1,
            )
            out = forward_with_optional_aux(model, batch["coords"], batch["x"], T=input_t)
            preds.append(decode_if_needed(y_normalizer, out["pred"]))
    pred = torch.cat(preds, dim=-1)
    target = decode_if_needed(y_normalizer, batch["y"])
    return pred, target


def safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmean(values))


def safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.nanmax(values))


def summarize_error(
    *,
    pred_seq: np.ndarray,
    target_seq: np.ndarray,
    top_rows: int,
    die_drop_threshold: float,
) -> list[dict[str, object]]:
    # Shape: batch, i, j, timestep, channel. Channels 0:2 are deformed coordinates.
    err = pred_seq[..., :2] - target_seq[..., :2]
    abs_x = np.abs(err[..., 0])
    abs_y = np.abs(err[..., 1])
    coord = np.linalg.norm(err, axis=-1)
    material_from_target = target_seq[..., :2] - target_seq[..., 2:4]
    consistency = np.linalg.norm(
        pred_seq[..., :2] - (material_from_target + pred_seq[..., 2:4]),
        axis=-1,
    )
    rows = []
    n_time = int(pred_seq.shape[3])
    top_rows = min(max(int(top_rows), 1), int(pred_seq.shape[2]))
    for timestep in range(n_time):
        top_slice = (slice(None), slice(None), slice(0, top_rows), timestep)
        top_boundary = (slice(None), slice(None), 0, timestep)
        gt_top_y = target_seq[top_boundary + (1,)]
        max_gt_top_y = np.nanmax(gt_top_y, axis=1, keepdims=True)
        die_mask = gt_top_y <= (max_gt_top_y - float(die_drop_threshold))
        rows.append(
            {
                "timestep": timestep,
                "global_x_mae": float(abs_x[:, :, :, timestep].mean()),
                "global_y_mae": float(abs_y[:, :, :, timestep].mean()),
                "global_x_rmse": float(np.sqrt(np.mean(err[:, :, :, timestep, 0] ** 2))),
                "global_y_rmse": float(np.sqrt(np.mean(err[:, :, :, timestep, 1] ** 2))),
                "global_coord_mae": float(coord[:, :, :, timestep].mean()),
                "global_coord_max": float(coord[:, :, :, timestep].max()),
                "top_rows_x_mae": float(abs_x[top_slice].mean()),
                "top_rows_y_mae": float(abs_y[top_slice].mean()),
                "top_rows_coord_mae": float(coord[top_slice].mean()),
                "top_rows_coord_max": float(coord[top_slice].max()),
                "top_j0_x_mae": float(abs_x[top_boundary].mean()),
                "top_j0_y_mae": float(abs_y[top_boundary].mean()),
                "top_j0_x_max": float(abs_x[top_boundary].max()),
                "top_j0_y_max": float(abs_y[top_boundary].max()),
                "die_region_x_mae": safe_mean(abs_x[top_boundary][die_mask]),
                "die_region_y_mae": safe_mean(abs_y[top_boundary][die_mask]),
                "die_region_coord_mae": safe_mean(coord[top_boundary][die_mask]),
                "die_region_coord_max": safe_max(coord[top_boundary][die_mask]),
                "die_region_point_count": int(np.count_nonzero(die_mask)),
                "channel_consistency_mean": float(consistency[:, :, :, timestep].mean()),
                "channel_consistency_max": float(consistency[:, :, :, timestep].max()),
            }
        )
    return rows


def run_metrics(
    spec: RunSpec,
    *,
    device: torch.device,
    batch_size: int | None,
    max_samples: int | None,
    top_rows: int,
    die_drop_threshold: float,
) -> list[dict[str, object]]:
    _, loader, meta, model, y_normalizer = load_run(
        spec,
        device=device,
        batch_size=batch_size,
    )
    h, w = tuple(meta["shapelist"])
    t_out = int(meta["t_out"])
    out_dim = int(meta["out_dim"])
    remaining = None if max_samples is None else int(max_samples)
    accum: list[dict[str, object]] = []
    sample_offset = 0
    for batch in loader:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None:
            keep = min(remaining, int(batch["x"].shape[0]))
            batch = {key: value[:keep] for key, value in batch.items()}
            remaining -= keep
        batch = batch_to_device(batch, device)
        pred, target = predict_batch(model, batch, y_normalizer=y_normalizer, t_out=t_out)
        batch_size_actual = int(pred.shape[0])
        pred_seq = pred.detach().cpu().reshape(
            batch_size_actual,
            h,
            w,
            t_out,
            out_dim,
        ).numpy()
        target_seq = target.detach().cpu().reshape(
            batch_size_actual,
            h,
            w,
            t_out,
            out_dim,
        ).numpy()
        rows = summarize_error(
            pred_seq=pred_seq,
            target_seq=target_seq,
            top_rows=top_rows,
            die_drop_threshold=die_drop_threshold,
        )
        for row in rows:
            row["run"] = spec.name
            row["samples_in_batch"] = batch_size_actual
            row["sample_offset"] = sample_offset
            accum.append(row)
        sample_offset += batch_size_actual
    return aggregate_timestep_rows(accum)


def aggregate_timestep_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return []
    by_timestep: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        by_timestep.setdefault(int(row["timestep"]), []).append(row)
    aggregated = []
    for timestep, timestep_rows in sorted(by_timestep.items()):
        total_samples = sum(int(row["samples_in_batch"]) for row in timestep_rows)
        out = {
            "run": timestep_rows[0]["run"],
            "timestep": timestep,
            "n_samples": total_samples,
        }
        for key in timestep_rows[0]:
            if key in {"run", "timestep", "samples_in_batch", "sample_offset"}:
                continue
            values = np.asarray([float(row[key]) for row in timestep_rows], dtype=np.float64)
            weights = np.asarray(
                [float(row["samples_in_batch"]) for row in timestep_rows],
                dtype=np.float64,
            )
            if key.endswith("_max"):
                out[key] = float(np.nanmax(values))
            elif key == "die_region_point_count":
                out[key] = int(np.nansum(values))
            else:
                valid = np.isfinite(values)
                out[key] = (
                    float(np.average(values[valid], weights=weights[valid]))
                    if np.any(valid)
                    else float("nan")
                )
        aggregated.append(out)
    return aggregated


def summarize_over_time(rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"run": rows[0]["run"], "n_samples": rows[0]["n_samples"]}
    for key in rows[0]:
        if key in {"run", "timestep", "n_samples", "die_region_point_count"}:
            continue
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        valid = np.isfinite(values)
        out[f"{key}_time_mean"] = float(np.nanmean(values)) if np.any(valid) else float("nan")
        out[f"{key}_time_max"] = float(np.nanmax(values)) if np.any(valid) else float("nan")
    out["die_region_point_count_total"] = int(
        sum(int(row["die_region_point_count"]) for row in rows)
    )
    return out


def write_plot(rows_by_run: dict[str, list[dict[str, object]]], out_path: Path) -> None:
    require_matplotlib(plt)
    metrics = [
        ("global_x_mae", "global x MAE"),
        ("global_y_mae", "global y MAE"),
        ("top_j0_x_mae", "top edge x MAE"),
        ("top_j0_y_mae", "top edge y MAE"),
        ("die_region_x_mae", "die-region x MAE"),
        ("die_region_y_mae", "die-region y MAE"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), dpi=150)
    for ax, (key, title) in zip(axes.reshape(-1), metrics):
        for run_name, rows in rows_by_run.items():
            x = [int(row["timestep"]) for row in rows]
            y = [float(row[key]) for row in rows]
            ax.plot(x, y, marker="o", linewidth=1.4, markersize=3.0, label=run_name)
        ax.set_title(title)
        ax.set_xlabel("timestep")
        ax.grid(True, linewidth=0.4, alpha=0.25)
    axes[0, 0].set_ylabel("absolute error")
    axes[1, 0].set_ylabel("absolute error")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        RunSpec(args.run_a_name, args.run_a_config, args.run_a_checkpoint),
        RunSpec(args.run_b_name, args.run_b_config, args.run_b_checkpoint),
    ]
    rows_by_run = {}
    all_rows = []
    summary_rows = []
    for spec in specs:
        print(f"evaluating {spec.name}: {spec.checkpoint}", flush=True)
        rows = run_metrics(
            spec,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            top_rows=args.top_rows,
            die_drop_threshold=args.die_drop_threshold,
        )
        rows_by_run[spec.name] = rows
        all_rows.extend(rows)
        summary_rows.append(summarize_over_time(rows))

    timestep_csv = out_dir / "timestep_axis_errors.csv"
    summary_csv = out_dir / "summary_axis_errors.csv"
    write_csv(timestep_csv, all_rows)
    write_csv(summary_csv, summary_rows)
    print(f"wrote timestep metrics: {timestep_csv}")
    print(f"wrote summary metrics: {summary_csv}")
    if not args.no_plots:
        plot_path = out_dir / "axis_error_curves.png"
        write_plot(rows_by_run, plot_path)
        print(f"wrote plot: {plot_path}")


if __name__ == "__main__":
    main()
