from __future__ import annotations

from pathlib import Path
from typing import Callable

import yaml

from omni_hc.training.search import run_optuna_search
from omni_hc.training.tasks.steady import test_steady_task, train_steady_task

from .data import build_test_loader, build_train_val_loaders


def _get_meta(loader):
    return loader.pipe_meta


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


def log_pipe(ctx) -> dict[str, str]:
    from omni_hc.training.logging_utils import (
        log_pipe_flow_images,
        save_pipe_flow_images,
    )

    h, w = ctx.meta["shapelist"]
    if ctx.out_dir is None:
        log_pipe_flow_images(
            ctx.coords, ctx.pred, ctx.target, h, w,
            prefix=ctx.prefix, epoch=ctx.epoch, step=ctx.step,
        )
        return {}
    return save_pipe_flow_images(
        ctx.coords, ctx.pred, ctx.target, h, w,
        out_dir=ctx.out_dir, prefix=ctx.prefix,
    )


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
