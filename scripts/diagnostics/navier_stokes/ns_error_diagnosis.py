"""
Navier-Stokes rollout error diagnosis.

Produces two figures:
  1. A (1 + 2*N_models) × t_out grid: target | pred per model | rel-L2 map per model
  2. Per-timestep rel. L2 error curve over the full test set, one line per model

Usage:
  python ns_error_diagnosis.py \
    --data-dir data/NavierStokes_V1e-5_N1200_T20 \
    --outputs-dir outputs/navier_stokes/mean_constraint \
    --models ONO Factformer \
    --out-dir artifacts/navier_stokes/error_diagnosis
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None  # type: ignore[assignment]
    mticker = None  # type: ignore[assignment]

_SRC_DIR = Path(__file__).resolve().parents[3] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import torch

from omni_hc.benchmarks.navier_stokes.adapter import (
    _init_task_state,
    _prepare_batch,
    _runtime_overrides,
)
from omni_hc.benchmarks.navier_stokes.data import build_test_loader
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import load_checkpoint_state, load_model_state_dict
from omni_hc.training.tasks.autoregressive import rollout_autoregressive

CMAP = "viridis"
T_IN = 10

DEFAULT_MODEL_SUBDIRS: dict[str, str] = {
    "Factformer": "factformer/final/seed_42",
    "Transolver": "transolver/final/seed_42",
    "Transolver_exp": "transolver/experiment/latent_head_500e/seed_42",
    "Galerkin-T": "galerkin_transformer/validation/seed_42",
    "ONO": "ono/final/seed_42",
    "GNOT": "gnot/final/seed_42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NS rollout diagnosis: per-timestep rel. L2 grid and curve."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/NavierStokes_V1e-5_N1200_T20"),
        help="Directory containing the NS .mat file.",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/navier_stokes/mean_constraint"),
        help="Root directory for model run outputs.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["ONO", "Factformer"],
        metavar="MODEL",
        help=(
            "Models to include (space-separated). "
            "Available: " + ", ".join(DEFAULT_MODEL_SUBDIRS.keys())
        ),
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Test-set sample index to show in the rollout grid.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/navier_stokes/error_diagnosis"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def load_ns_model(
    run_dir: Path,
    data_dir: Path,
    meta: dict,
    device: torch.device,
) -> torch.nn.Module:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    cfg["paths"]["root_dir"] = str(data_dir)
    m, _, _ = create_model(
        cfg, device=device, runtime_overrides=_runtime_overrides(meta)
    )
    ckpt = load_checkpoint_state(run_dir / "best.pt", device=device)
    load_model_state_dict(m, ckpt["model_state_dict"])
    m.eval()
    return m


def get_sample_rollout(
    m: torch.nn.Module,
    loader,
    sample_idx: int,
    h: int,
    w: int,
    t_out: int,
    meta: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run rollout and return (pred_field, gt_field) for a single test-set sample."""
    ts = _init_task_state(meta, sample_dtype=torch.float32, device=device)
    cumulative = 0
    for batch in loader:
        coords_b, fx_b, target_b = _prepare_batch(batch, device=device, task_state=ts)
        bsz = target_b.shape[0]
        if cumulative + bsz > sample_idx:
            local_idx = sample_idx - cumulative
            with torch.no_grad():
                pred_b, _, _ = rollout_autoregressive(
                    m,
                    coords_b,
                    fx_b,
                    target_b,
                    t_out=meta["t_out"],
                    out_dim=meta["out_dim"],
                    teacher_forcing=False,
                )
            pred_f = pred_b[local_idx].view(h, w, t_out).cpu().numpy()
            gt_f = target_b[local_idx].view(h, w, t_out).cpu().numpy()
            return pred_f, gt_f
        cumulative += bsz
        ts = _init_task_state(meta, sample_dtype=torch.float32, device=device)
    raise IndexError(f"sample_idx={sample_idx} is out of range for the test set")


def compute_step_rel_l2(
    m: torch.nn.Module,
    loader,
    h: int,
    w: int,
    t_out: int,
    meta: dict,
    device: torch.device,
) -> np.ndarray:
    """Returns (N_test, t_out) per-sample per-timestep relative L2 over the full test set."""
    ts = _init_task_state(meta, sample_dtype=torch.float32, device=device)
    errs = []
    for batch in loader:
        coords_b, fx_b, target_b = _prepare_batch(batch, device=device, task_state=ts)
        with torch.no_grad():
            pred_b, _, _ = rollout_autoregressive(
                m,
                coords_b,
                fx_b,
                target_b,
                t_out=meta["t_out"],
                out_dim=meta["out_dim"],
                teacher_forcing=False,
            )
        bsz = pred_b.shape[0]
        p = pred_b.cpu().numpy().reshape(bsz, h, w, t_out)
        g = target_b.cpu().numpy().reshape(bsz, h, w, t_out)
        num = np.sqrt(((p - g) ** 2).sum(axis=(1, 2)))  # (bsz, t_out)
        den = np.sqrt((g**2).sum(axis=(1, 2))) + 1e-8  # (bsz, t_out)
        errs.append(num / den)
    return np.concatenate(errs, axis=0)


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def rel_l2_field(pred_f: np.ndarray, gt_f: np.ndarray) -> np.ndarray:
    """Spatial rel-L2 map: |pred - gt| / ||gt||_F per timestep. Shape: (h, w, t_out)."""
    out = np.empty_like(pred_f)
    for t in range(pred_f.shape[2]):
        gt_t = gt_f[:, :, t]
        out[:, :, t] = np.abs(pred_f[:, :, t] - gt_t) / (np.linalg.norm(gt_t) + 1e-8)
    return out


def _band(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        arr.mean(axis=0),
        np.percentile(arr, 5, axis=0),
        np.percentile(arr, 95, axis=0),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_rollout_grid(
    gt: np.ndarray,
    preds: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    model_names: list[str],
    t_out: int,
    *,
    sample_idx: int,
    out_dir: Path,
    show: bool,
) -> Path:
    n_m = len(model_names)
    n_rows = 1 + 2 * n_m
    _cell = 1.5

    vmin = min(gt.min(), *(preds[n].min() for n in model_names))
    vmax = max(gt.max(), *(preds[n].max() for n in model_names))
    err_max = max(errors[n].max() for n in model_names)

    fig, axes = plt.subplots(
        n_rows,
        t_out,
        figsize=(t_out * _cell + 1.5, n_rows * _cell + 0.5),
        constrained_layout=True,
    )

    for col in range(t_out):
        t_label = f"$t_{{{col + T_IN + 1}}}$"
        axes[0, col].imshow(
            gt[:, :, col], cmap=CMAP, origin="lower", vmin=vmin, vmax=vmax
        )
        axes[0, col].set_title(t_label, fontsize=9)
        axes[0, col].axis("off")
        for i, name in enumerate(model_names):
            axes[1 + i, col].imshow(
                preds[name][:, :, col], cmap=CMAP, origin="lower", vmin=vmin, vmax=vmax
            )
            axes[1 + i, col].axis("off")
            axes[1 + n_m + i, col].imshow(
                errors[name][:, :, col], cmap=CMAP, origin="lower", vmin=0, vmax=err_max
            )
            axes[1 + n_m + i, col].axis("off")

    axes[0, 0].set_ylabel("Ground truth", fontsize=9)
    for i, name in enumerate(model_names):
        axes[1 + i, 0].set_ylabel(name, fontsize=9)
        axes[1 + n_m + i, 0].set_ylabel(f"{name} rel. $L_2$", fontsize=9)

    fig.colorbar(
        plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=vmin, vmax=vmax)),
        ax=axes[: 1 + n_m],
        fraction=0.015,
        pad=0.02,
        label="Vorticity",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=0, vmax=err_max)),
        ax=axes[1 + n_m :],
        fraction=0.015,
        pad=0.02,
        label=r"Rel. $L_2$",
    )
    fig.suptitle(f"NS diagnosis — test sample {sample_idx}", fontsize=11)

    out_path = out_dir / "ns_diag_rollout.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_rel_l2_curve(
    step_errors: dict[str, np.ndarray],
    model_names: list[str],
    t_out: int,
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    step_idx = np.arange(1, t_out + 1)
    curve_colors = plt.get_cmap(CMAP)(np.linspace(0.15, 0.85, len(model_names)))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, color in zip(model_names, curve_colors):
        mean_e, p5_e, p95_e = _band(step_errors[name])
        ax.fill_between(step_idx, p5_e, p95_e, alpha=0.2, color=color)
        ax.plot(step_idx, mean_e, color=color, lw=1.8, label=name)

    ax.set_xlabel("Output timestep")
    ax.set_ylabel(r"Rel. $L_2$")
    ax.set_title(r"NS per-timestep rel. $L_2$ error")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.legend(frameon=False)
    fig.tight_layout()

    out_path = out_dir / "ns_diag_rel_l2_curve.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used with --no-plot")

    unknown = [n for n in args.models if n not in DEFAULT_MODEL_SUBDIRS]
    if unknown:
        raise ValueError(
            f"Unknown model(s): {unknown}. Available: {list(DEFAULT_MODEL_SUBDIRS.keys())}"
        )

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {n: args.outputs_dir / DEFAULT_MODEL_SUBDIRS[n] for n in args.models}
    for name, rd in run_dirs.items():
        if not rd.exists():
            raise FileNotFoundError(f"Run directory not found for {name}: {rd}")

    # Build test loader from the first model's config (data is shared across models)
    first_cfg = yaml.safe_load(
        (next(iter(run_dirs.values())) / "resolved_config.yaml").read_text()
    )
    first_cfg["paths"]["root_dir"] = str(args.data_dir)
    test_loader = build_test_loader(first_cfg)
    meta = test_loader.ns_meta
    h, w = meta["shapelist"]
    t_out = meta["t_out"]
    print(f"meta: h={h}, w={w}, t_out={t_out}, models={args.models}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    # ---- Grid: rollout on a single test sample ----------------------------
    print(f"Running rollout grid for test sample {args.sample}...")
    preds: dict[str, np.ndarray] = {}
    gt_field: np.ndarray | None = None
    for name in args.models:
        print(f"  {name}")
        m = load_ns_model(run_dirs[name], args.data_dir, meta, device)
        pred_f, gt_f = get_sample_rollout(
            m, test_loader, args.sample, h, w, t_out, meta, device
        )
        preds[name] = pred_f
        gt_field = gt_f

    errors = {n: rel_l2_field(preds[n], gt_field) for n in args.models}

    if not args.no_plot:
        if plt is None:
            print("matplotlib not available; skipping plots.")
        else:
            out = plot_rollout_grid(
                gt_field,
                preds,
                errors,
                args.models,
                t_out,
                sample_idx=args.sample,
                out_dir=args.out_dir,
                show=args.show,
            )
            print(f"wrote {out}")

    # ---- Curve: per-timestep rel. L2 over the full test set ---------------
    print("Computing per-timestep rel. L2 over the full test set...")
    step_errors: dict[str, np.ndarray] = {}
    for name in args.models:
        print(f"  {name}")
        m = load_ns_model(run_dirs[name], args.data_dir, meta, device)
        step_errors[name] = compute_step_rel_l2(
            m, test_loader, h, w, t_out, meta, device
        )
        mean_final = step_errors[name][:, -1].mean()
        print(f"    mean rel. L2 at t={t_out}: {mean_final:.4f}")

    if not args.no_plot:
        if plt is None:
            print("matplotlib not available; skipping plots.")
        else:
            out = plot_rel_l2_curve(
                step_errors,
                args.models,
                t_out,
                out_dir=args.out_dir,
                show=args.show,
            )
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
