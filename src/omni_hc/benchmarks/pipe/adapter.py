from __future__ import annotations

from pathlib import Path

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


def train(cfg: dict, *, nsl_root: str | Path | None, device):
    return train_steady_task(
        cfg,
        nsl_root=nsl_root,
        device=device,
        build_train_val_loaders=build_train_val_loaders,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        prepare_batch=_prepare_batch,
    )


def test(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device,
    checkpoint_path: str | Path | None = None,
):
    payload = test_steady_task(
        cfg,
        nsl_root=nsl_root,
        device=device,
        checkpoint_path=checkpoint_path,
        build_test_loader=build_test_loader,
        get_meta=_get_meta,
        runtime_overrides=_runtime_overrides,
        prepare_batch=_prepare_batch,
    )
    output_dir = Path(cfg["paths"]["output_dir"])
    with open(output_dir / "test_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return payload


def tune(cfg: dict, *, nsl_root: str | Path | None, device):
    return run_optuna_search(
        cfg,
        nsl_root=nsl_root,
        device=device,
        train_fn=train,
    )
