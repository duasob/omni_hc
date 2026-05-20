from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from omni_hc.training.search import run_optuna_search
from omni_hc.training.tasks.steady import test_steady_task, train_steady_task

from .data import build_test_loader, build_train_val_loaders


def _get_meta(loader):
    return loader.darcy_meta


def _runtime_overrides(meta: dict):
    return {
        "shapelist": tuple(meta["shapelist"]),
        "task": str(meta["task"]),
        "loader": str(meta["loader"]),
        "geotype": str(meta["geotype"]),
        "space_dim": int(meta["space_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "out_dim": int(meta["out_dim"]),
    }


def _prepare_batch(batch, *, device):
    return batch["coords"].to(device), batch["x"].to(device), batch["y"].to(device)


def _log_boundary_profiles(ctx) -> dict[str, str]:
    import matplotlib.pyplot as plt
    import numpy as np

    H, W = ctx.meta["shapelist"]
    pred   = ctx.pred[0, :, 0].cpu().numpy()
    target = ctx.target[0, :, 0].cpu().numpy()
    pred_2d   = pred.reshape(H, W)
    target_2d = target.reshape(H, W)

    xs = np.linspace(0, 1, W)
    ys = np.linspace(0, 1, H)
    edges = {
        "Bottom ($y=0$)": (xs, pred_2d[0,  :], target_2d[0,  :]),
        "Top ($y=1$)":    (xs, pred_2d[-1, :], target_2d[-1, :]),
        "Left ($x=0$)":   (ys, pred_2d[:,  0], target_2d[:,  0]),
        "Right ($x=1$)":  (ys, pred_2d[:, -1], target_2d[:, -1]),
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    fig.suptitle("Boundary: prediction vs ground truth", fontsize=12)
    for ax, (label, (pos, pred_edge, gt_edge)) in zip(axes.flat, edges.items()):
        ax.plot(pos, gt_edge,   color="steelblue",  linewidth=1.5, label="Ground truth")
        ax.plot(pos, pred_edge, color="darkorange", linewidth=1.5, label="Prediction", linestyle="--")
        ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("Position")
        ax.set_ylabel("$u$")
    axes.flat[0].legend(fontsize=8, frameon=False)

    out = {}
    if ctx.out_dir is not None:
        stem = ctx.prefix if ctx.step is None else f"epoch{ctx.epoch:04d}"
        out_path = Path(ctx.out_dir) / f"boundary_profiles_{stem}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        out["boundary_profiles"] = str(out_path)
    else:
        try:
            import wandb
            wandb.log(
                {"constraint/boundary_profiles": wandb.Image(fig)},
                step=ctx.step,
            )
        except Exception:
            pass
        plt.close(fig)
    return out


def log_darcy(ctx) -> dict[str, str]:
    from omni_hc.training.logging_utils import (
        log_steady_field_images,
        save_steady_field_images,
    )

    h, w = ctx.meta["shapelist"]
    out = {}
    if ctx.out_dir is None:
        log_steady_field_images(
            ctx.fx, ctx.pred, ctx.target, h, w,
            prefix=ctx.prefix, epoch=ctx.epoch, step=ctx.step,
        )
    else:
        out.update(save_steady_field_images(
            ctx.fx, ctx.pred, ctx.target, h, w,
            out_dir=ctx.out_dir, prefix=ctx.prefix,
        ))
    out.update(_log_boundary_profiles(ctx))
    return out


def train(cfg: dict, *, device, log_fn: Callable | None = None):
    return train_steady_task(
        cfg,
        device=device,
        build_train_val_loaders=build_train_val_loaders,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        prepare_batch=_prepare_batch,
        log_fn=log_fn,
    )


def test(
    cfg: dict,
    *,
    device,
    checkpoint_path: str | Path | None = None,
    log_fn: Callable | None = None,
):
    payload = test_steady_task(
        cfg,
        device=device,
        checkpoint_path=checkpoint_path,
        build_test_loader=build_test_loader,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        prepare_batch=_prepare_batch,
        log_fn=log_fn,
    )
    output_dir = Path(cfg["paths"]["output_dir"])
    with open(output_dir / "test_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return payload


def tune(cfg: dict, *, device):
    return run_optuna_search(cfg, device=device, train_fn=train)
