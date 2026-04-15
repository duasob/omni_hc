from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import yaml

from omni_hc.benchmarks.navier_stokes_data import (
    build_test_loader,
    build_train_val_loaders,
    make_grid,
)
from omni_hc.integrations.nsl import create_model, resolve_nsl_root
from omni_hc.training.logging_utils import (
    finish_wandb_if_active,
    init_wandb_if_enabled,
    log_metrics,
    log_prediction_images,
)
from omni_hc.training.optuna_utils import apply_optuna_search_space


def build_optimizer(model, cfg: dict):
    name = str(cfg.get("optimizer", "adamw")).lower()
    lr = float(cfg.get("learning_rate", 1e-3))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict):
    name = str(cfg.get("scheduler", "none")).lower()
    if name in {"", "none", "null"}:
        return None, False
    if name == "onecyclelr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(cfg["learning_rate"]),
            epochs=int(cfg["num_epochs"]),
            steps_per_epoch=max(int(cfg["steps_per_epoch"]), 1),
            pct_start=float(cfg.get("pct_start", 0.3)),
        )
        return scheduler, True
    if name == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", 100)),
            gamma=float(cfg.get("gamma", 0.5)),
        )
        return scheduler, False
    raise ValueError(f"Unknown scheduler: {name}")


def forward_with_optional_aux(model, coords, fx):
    if bool(getattr(model, "supports_aux", False)):
        out = model(coords, fx, return_aux=True)
        if isinstance(out, tuple) and len(out) == 3:
            pred, pred_base, corr = out
            return {
                "pred": pred,
                "pred_mean": float(pred.mean().item()),
                "pred_base_mean": float(pred_base.mean().item()),
                "corr_mean": float(corr.mean().item()),
            }
    pred = model(coords, fx)
    return {
        "pred": pred,
        "pred_mean": float(pred.mean().item()),
        "pred_base_mean": None,
        "corr_mean": None,
    }


def _relative_l2_per_sample(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
):
    diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
    diff_norm = torch.norm(diff, p=2, dim=1)
    tgt_norm = torch.norm(target.reshape(target.shape[0], -1), p=2, dim=1).clamp_min(
        eps
    )
    return diff_norm / tgt_norm


def _prepare_batch(batch, grid_flat, *, device):
    fx = batch["x"].to(device)
    target = batch["y"].to(device)
    coords = grid_flat.unsqueeze(0).repeat(int(fx.shape[0]), 1, 1)
    return coords, fx, target


def _rollout_autoregressive(
    model, coords, fx, target, *, t_out: int, out_dim: int, teacher_forcing: bool
):
    preds = []
    current_fx = fx
    step_rel_l2 = torch.zeros((), device=coords.device)
    pred_mean_sum = 0.0
    pred_base_sum = 0.0
    corr_sum = 0.0
    pred_base_count = 0
    corr_count = 0
    for step in range(int(t_out)):
        out = forward_with_optional_aux(model, coords, current_fx)
        pred_t = out["pred"]
        preds.append(pred_t)
        pred_mean_sum += float(out["pred_mean"])
        if out["pred_base_mean"] is not None:
            pred_base_sum += float(out["pred_base_mean"])
            pred_base_count += 1
        if out["corr_mean"] is not None:
            corr_sum += float(out["corr_mean"])
            corr_count += 1
        y_t = target[..., out_dim * step : out_dim * (step + 1)]
        step_rel_l2 = step_rel_l2 + _relative_l2_per_sample(pred_t, y_t).sum()
        next_tail = y_t if teacher_forcing else pred_t
        current_fx = torch.cat((current_fx[..., out_dim:], next_tail), dim=-1)
    pred = torch.cat(preds, dim=-1)
    aux = {
        "pred_mean": pred_mean_sum / max(int(t_out), 1),
        "pred_base_mean": None
        if pred_base_count == 0
        else pred_base_sum / pred_base_count,
        "corr_mean": None if corr_count == 0 else corr_sum / corr_count,
    }
    return pred, step_rel_l2, aux


def _evaluate(model, loader, *, device, grid_flat, t_out: int, out_dim: int):
    model.eval()
    mse_sum = 0.0
    rel_l2_sum = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            coords, fx, target = _prepare_batch(batch, grid_flat, device=device)
            pred, _, _ = _rollout_autoregressive(
                model,
                coords,
                fx,
                target,
                t_out=t_out,
                out_dim=out_dim,
                teacher_forcing=False,
            )
            batch_size = int(target.shape[0])
            mse_sum += float(
                F.mse_loss(pred, target, reduction="none")
                .reshape(batch_size, -1)
                .mean(dim=1)
                .sum()
                .item()
            )
            rel_l2_sum += float(_relative_l2_per_sample(pred, target).sum().item())
            samples += batch_size
    denom = max(samples, 1)
    return {"mse": mse_sum / denom, "rel_l2": rel_l2_sum / denom}


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


def _grid_from_meta(meta: dict, *, device, dtype):
    h, w = tuple(meta["shapelist"])
    return make_grid(h, w, device=device, dtype=dtype)


def _resolve_output_dir(cfg: dict) -> Path:
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _normalize_interval(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null"}:
            return None
        return int(text)
    return int(value)


def _write_resolved_config(cfg: dict, *, output_dir: Path, resolved_nsl_root: Path):
    payload = deepcopy(cfg)
    payload.setdefault("backend", {})
    payload["backend"]["resolved_nsl_root"] = str(resolved_nsl_root)
    with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _load_checkpoint_state(checkpoint_path: str | Path, *, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f"Expected dict checkpoint at {checkpoint_path}, received {type(checkpoint).__name__}"
        )
    return checkpoint


def _load_model_for_eval(cfg: dict, *, nsl_root: str | Path | None, device, checkpoint_path: str | Path):
    test_loader = build_test_loader(cfg)
    meta = test_loader.ns_meta
    model, model_args, resolved_nsl_root = create_model(
        cfg,
        nsl_root=nsl_root,
        device=device,
        runtime_overrides=_runtime_overrides(meta),
    )
    checkpoint = _load_checkpoint_state(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    sample_dtype = next(iter(test_loader))["x"].dtype
    grid_flat = _grid_from_meta(meta, device=device, dtype=sample_dtype)
    return {
        "model": model,
        "model_args": model_args,
        "resolved_nsl_root": resolved_nsl_root,
        "loader": test_loader,
        "meta": meta,
        "grid_flat": grid_flat,
    }


def train_ns_demo(cfg: dict, *, nsl_root: str | Path | None, device: torch.device):
    train_loader, val_loader = build_train_val_loaders(cfg)
    meta = train_loader.ns_meta
    model, model_args, resolved_nsl_root = create_model(
        cfg,
        nsl_root=nsl_root,
        device=device,
        runtime_overrides=_runtime_overrides(meta),
    )

    output_dir = _resolve_output_dir(cfg)
    _write_resolved_config(cfg, output_dir=output_dir, resolved_nsl_root=resolved_nsl_root)

    sample_dtype = next(iter(train_loader))["x"].dtype
    grid_flat = _grid_from_meta(meta, device=device, dtype=sample_dtype)

    training_cfg = deepcopy(cfg.get("training", {}))
    training_cfg["steps_per_epoch"] = len(train_loader)
    optimizer = build_optimizer(model, training_cfg)
    scheduler, scheduler_step_per_batch = build_scheduler(optimizer, training_cfg)
    teacher_forcing = bool(int(training_cfg.get("teacher_forcing", 1)))
    t_out = int(meta["t_out"])
    out_dim = int(meta["out_dim"])
    h, w = tuple(meta["shapelist"])
    best_val = float("inf")
    best_metrics = None
    wandb_cfg = cfg.get("wandb_logging", {}) or {}
    log_every = _normalize_interval(wandb_cfg.get("log_every", 100))
    image_log_every = _normalize_interval(wandb_cfg.get("image_log_every"))

    init_wandb_if_enabled(cfg)
    try:
        for epoch in range(int(training_cfg.get("num_epochs", 1))):
            model.train()
            train_loss_sum = 0.0
            train_rel_l2_sum = 0.0
            train_pred_mean_sum = 0.0
            train_pred_base_mean_sum = 0.0
            train_corr_mean_sum = 0.0
            pred_base_count = 0
            corr_count = 0
            samples = 0

            for step, batch in enumerate(train_loader):
                coords, fx, target = _prepare_batch(batch, grid_flat, device=device)
                pred, _, aux = _rollout_autoregressive(
                    model,
                    coords,
                    fx,
                    target,
                    t_out=t_out,
                    out_dim=out_dim,
                    teacher_forcing=teacher_forcing,
                )
                loss = F.mse_loss(pred, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                max_grad_norm = training_cfg.get("max_grad_norm")
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(max_grad_norm)
                    )
                optimizer.step()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

                batch_size = int(target.shape[0])
                train_loss_sum += float(loss.item()) * batch_size
                train_rel_l2_sum += float(
                    _relative_l2_per_sample(pred, target).sum().item()
                )
                train_pred_mean_sum += float(aux["pred_mean"]) * batch_size
                if aux["pred_base_mean"] is not None:
                    train_pred_base_mean_sum += float(aux["pred_base_mean"]) * batch_size
                    pred_base_count += batch_size
                if aux["corr_mean"] is not None:
                    train_corr_mean_sum += float(aux["corr_mean"]) * batch_size
                    corr_count += batch_size
                samples += batch_size

                if log_every is not None and log_every > 0:
                    global_step = epoch * len(train_loader) + step + 1
                    if (step + 1) % log_every == 0:
                        payload = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "train/step_mse": float(loss.item()),
                            "train/step_rel_l2": float(
                                _relative_l2_per_sample(pred, target).mean().item()
                            ),
                            "train/pred_mean": float(aux["pred_mean"]),
                        }
                        if aux["pred_base_mean"] is not None:
                            payload["train/pred_base_mean"] = float(
                                aux["pred_base_mean"]
                            )
                        if aux["corr_mean"] is not None:
                            payload["train/corr_mean"] = float(aux["corr_mean"])
                        log_metrics(payload, step=global_step)

            if scheduler is not None and not scheduler_step_per_batch:
                scheduler.step()

            train_metrics = {
                "mse": train_loss_sum / max(samples, 1),
                "rel_l2": train_rel_l2_sum / max(samples, 1),
                "pred_mean": train_pred_mean_sum / max(samples, 1),
                "pred_base_mean": None
                if pred_base_count == 0
                else train_pred_base_mean_sum / pred_base_count,
                "corr_mean": None
                if corr_count == 0
                else train_corr_mean_sum / corr_count,
            }

            if (
                (image_log_every is not None and image_log_every > 0)
                or (log_every is not None and log_every > 0)
            ):
                model.eval()
                with torch.no_grad():
                    max_val_idx = max(len(val_loader) - 1, 0)
                    sampled_idx = random.randint(0, max_val_idx)
                    for val_step, batch in enumerate(val_loader):
                        coords, fx, target = _prepare_batch(
                            batch, grid_flat, device=device
                        )
                        pred, _, aux = _rollout_autoregressive(
                            model,
                            coords,
                            fx,
                            target,
                            t_out=t_out,
                            out_dim=out_dim,
                            teacher_forcing=False,
                        )
                        if (
                            image_log_every is not None
                            and image_log_every > 0
                            and epoch % image_log_every == 0
                            and val_step == sampled_idx
                        ):
                            log_prediction_images(
                                pred[..., :out_dim],
                                target[..., :out_dim],
                                fx[..., -out_dim:],
                                h,
                                w,
                                prefix="validation",
                                epoch=epoch,
                            )
                        if (
                            log_every is not None
                            and log_every > 0
                            and (val_step + 1) % log_every == 0
                        ):
                            payload = {
                                "epoch": epoch + 1,
                                "val/pred_mean": float(aux["pred_mean"]),
                            }
                            if aux["pred_base_mean"] is not None:
                                payload["val/pred_base_mean"] = float(
                                    aux["pred_base_mean"]
                                )
                            if aux["corr_mean"] is not None:
                                payload["val/corr_mean"] = float(aux["corr_mean"])
                            log_metrics(payload)
                model.train()

            val_metrics = _evaluate(
                model,
                val_loader,
                device=device,
                grid_flat=grid_flat,
                t_out=t_out,
                out_dim=out_dim,
            )
            print(
                f"epoch {epoch + 1}/{int(training_cfg.get('num_epochs', 1))} "
                f"train_mse={train_metrics['mse']:.6f} "
                f"train_rel_l2={train_metrics['rel_l2']:.6f} "
                f"val_mse={val_metrics['mse']:.6f} "
                f"val_rel_l2={val_metrics['rel_l2']:.6f}"
            )
            log_metrics(
                {
                    "epoch": epoch + 1,
                    "train/mse": train_metrics["mse"],
                    "train/rel_l2": train_metrics["rel_l2"],
                    "train/pred_mean": train_metrics["pred_mean"],
                    "val/mse": val_metrics["mse"],
                    "val/rel_l2": val_metrics["rel_l2"],
                },
                step=epoch + 1,
            )

            latest_path = output_dir / "latest.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": None
                    if scheduler is None
                    else scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "model_args": vars(model_args),
                    "resolved_nsl_root": str(resolved_nsl_root),
                },
                latest_path,
            )
            if val_metrics["rel_l2"] < best_val:
                best_val = float(val_metrics["rel_l2"])
                best_metrics = deepcopy(val_metrics)
                torch.save(
                    torch.load(latest_path, map_location="cpu"), output_dir / "best.pt"
                )
        return {
            "best_val_rel_l2": best_val,
            "best_val_metrics": best_metrics,
            "output_dir": str(output_dir),
            "resolved_nsl_root": str(resolved_nsl_root),
        }
    finally:
        finish_wandb_if_active()


def test_ns_demo(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
):
    output_dir = _resolve_output_dir(cfg)
    if checkpoint_path is None:
        checkpoint_path = output_dir / "best.pt"
    runtime = _load_model_for_eval(
        cfg,
        nsl_root=nsl_root,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    metrics = _evaluate(
        runtime["model"],
        runtime["loader"],
        device=device,
        grid_flat=runtime["grid_flat"],
        t_out=int(runtime["meta"]["t_out"]),
        out_dim=int(runtime["meta"]["out_dim"]),
    )
    payload = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "metrics": metrics,
        "resolved_nsl_root": str(runtime["resolved_nsl_root"]),
    }
    with open(output_dir / "test_metrics.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    return payload


def run_optuna_ns_demo(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
):
    try:
        import optuna
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ModuleNotFoundError(
            "Optuna is required for search. Install it with `pip install optuna` "
            "or `pip install -e '.[experiments]'`."
        ) from exc

    optuna_cfg = cfg.get("optuna", {}) or {}
    search_space = optuna_cfg.get("search_space", {})
    if not search_space:
        raise ValueError("optuna.search_space is required")
    num_trials = int(optuna_cfg.get("num_trials", 10))
    direction = str(optuna_cfg.get("direction", "minimize"))
    base_output_dir = Path(optuna_cfg.get("save_dir", cfg["paths"]["output_dir"]))

    def objective(trial):
        trial_cfg = apply_optuna_search_space(cfg, trial, search_space)
        trial_cfg.setdefault("paths", {})
        trial_cfg["paths"]["output_dir"] = str(base_output_dir / f"trial_{trial.number:03d}")
        if "wandb_logging" in trial_cfg:
            trial_cfg["wandb_logging"] = {
                **trial_cfg["wandb_logging"],
                "run_name": f"{optuna_cfg.get('run_name', 'optuna')}_trial_{trial.number:03d}",
            }
        result = train_ns_demo(trial_cfg, nsl_root=nsl_root, device=device)
        return float(result["best_val_rel_l2"])

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=num_trials)
    return study
