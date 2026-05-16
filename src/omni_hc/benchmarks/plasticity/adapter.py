from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from omni_hc.training.search import run_optuna_search
from omni_hc.training.tasks.dynamic_conditional import (
    test_dynamic_conditional_task,
    train_dynamic_conditional_task,
)

from .data import build_test_loader, build_train_val_loaders


def _get_meta(loader):
    return loader.plasticity_meta


def _runtime_overrides(meta: dict):
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


def _prepare_batch(batch, *, device):
    return (
        batch["coords"].to(device),
        batch["time"].to(device),
        batch["x"].to(device),
        batch["y"].to(device),
    )


def log_plasticity(ctx) -> dict[str, str]:
    from omni_hc.training.logging_utils import (
        log_plasticity_mesh_consistency_media,
        save_plasticity_mesh_consistency_media,
    )

    h, w = ctx.meta["shapelist"]
    t_out = int(ctx.meta["t_out"])
    out_dim = int(ctx.meta["out_dim"])
    fps = int((ctx.cfg.get("wandb_logging") or {}).get("plasticity_video_fps", 4))

    if ctx.out_dir is None:
        log_plasticity_mesh_consistency_media(
            ctx.pred, ctx.target, h, w,
            t_out=t_out, out_dim=out_dim,
            prefix=ctx.prefix, epoch=ctx.epoch,
            aux_tensors=ctx.aux_tensors, step=ctx.step, fps=fps,
        )
        return {}
    return save_plasticity_mesh_consistency_media(
        ctx.pred, ctx.target, h, w,
        t_out=t_out, out_dim=out_dim,
        out_dir=ctx.out_dir, prefix=ctx.prefix,
        aux_tensors=ctx.aux_tensors, fps=fps,
    )


def train(cfg: dict, *, device, log_fn: Callable | None = None):
    return train_dynamic_conditional_task(
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
    payload = test_dynamic_conditional_task(
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
