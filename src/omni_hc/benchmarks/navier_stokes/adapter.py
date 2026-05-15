from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from omni_hc.training.search import run_optuna_search
from omni_hc.training.tasks.autoregressive import (
    test_autoregressive_task,
    train_autoregressive_task,
)

from .data import build_test_loader, build_train_val_loaders, make_grid


def _get_meta(loader):
    return loader.ns_meta


def _runtime_overrides(meta: dict):
    return {
        "shapelist": tuple(meta["shapelist"]),
        "task": "dynamic_autoregressive",
        "T_in": int(meta["t_in"]),
        "T_out": int(meta["t_out"]),
        "out_dim": int(meta["out_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "loader": "ns",
        "geotype": "structured_2D",
        "space_dim": 2,
    }


def _init_task_state(meta: dict, *, sample_dtype, device):
    h, w = tuple(meta["shapelist"])
    return {"grid_flat": make_grid(h, w, device=device, dtype=sample_dtype)}


def _prepare_batch(batch, *, device, task_state):
    fx = batch["x"].to(device)
    target = batch["y"].to(device)
    coords = task_state["grid_flat"].unsqueeze(0).repeat(int(fx.shape[0]), 1, 1)
    return coords, fx, target


def log_ns(ctx) -> dict[str, str]:
    from omni_hc.training.logging_utils import (
        log_prediction_images,
        log_rollout_gifs,
        save_prediction_images,
        save_rollout_gifs,
    )

    h, w = ctx.meta["shapelist"]
    out_dim = int(ctx.meta["out_dim"])
    fps = int((ctx.cfg.get("wandb_logging") or {}).get("rollout_gif_fps", 4))

    if ctx.out_dir is None:
        log_prediction_images(
            ctx.pred[..., :out_dim],
            ctx.target[..., :out_dim],
            ctx.fx[..., -out_dim:],
            h, w,
            prefix=ctx.prefix, epoch=ctx.epoch, step=ctx.step,
        )
        log_rollout_gifs(
            ctx.pred, ctx.target, h, w,
            out_dim=out_dim, prefix=ctx.prefix,
            epoch=ctx.epoch, step=ctx.step, fps=fps,
        )
        return {}
    paths = {}
    paths.update(
        save_prediction_images(
            ctx.pred[..., :out_dim],
            ctx.target[..., :out_dim],
            ctx.fx[..., -out_dim:],
            h, w,
            out_dir=ctx.out_dir, prefix=ctx.prefix,
        )
    )
    paths.update(
        save_rollout_gifs(
            ctx.pred, ctx.target, h, w,
            out_dim=out_dim, out_dir=ctx.out_dir, prefix=ctx.prefix, fps=fps,
        )
    )
    return paths


def train(cfg: dict, *, device, log_fn: Callable | None = None):
    return train_autoregressive_task(
        cfg,
        device=device,
        build_train_val_loaders=build_train_val_loaders,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        init_task_state=_init_task_state,
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
    payload = test_autoregressive_task(
        cfg,
        device=device,
        checkpoint_path=checkpoint_path,
        build_test_loader=build_test_loader,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        init_task_state=_init_task_state,
        prepare_batch=_prepare_batch,
        log_fn=log_fn,
    )
    output_dir = Path(cfg["paths"]["output_dir"])
    with open(output_dir / "test_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return payload


def tune(cfg: dict, *, device):
    return run_optuna_search(cfg, device=device, train_fn=train)
