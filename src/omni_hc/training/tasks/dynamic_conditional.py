from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import (
    MetricAccumulator,
    build_optimizer,
    build_scheduler,
    diagnostic_values,
    forward_with_optional_aux,
    load_checkpoint_state,
    load_model_state_dict,
    normalize_interval,
    prefix_metric_names,
    relative_l2_per_sample,
    restore_training_checkpoint,
    resolve_output_dir,
    save_checkpoint_bundle,
    write_resolved_config,
)
from omni_hc.benchmarks.base import MediaLogContext
from omni_hc.training.logging_utils import (
    finish_wandb_if_active,
    init_wandb_if_enabled,
    log_metrics,
)


def _decode_if_needed(normalizer, tensor: torch.Tensor) -> torch.Tensor:
    return normalizer.decode(tensor) if normalizer is not None else tensor


def _build_nsl_l2_loss():
    from utils.loss import L2Loss

    return L2Loss(size_average=False)


def rollout_dynamic_conditional(
    model,
    coords,
    fx,
    time,
    target,
    *,
    t_out: int,
    out_dim: int,
    y_normalizer=None,
    train_step_per_time: bool = False,
    optimizer=None,
    max_grad_norm=None,
    step_loss_fn=None,
):
    preds = []
    step_rel_l2_sum = torch.zeros((), device=coords.device)
    pred_mean_sum = 0.0
    diag_metrics = MetricAccumulator()
    total_loss = torch.zeros((), device=coords.device)
    metric_loss_sum = 0.0

    for step in range(int(t_out)):
        input_t = time[:, step : step + 1].reshape(coords.shape[0], 1)
        y_t = target[..., out_dim * step : out_dim * (step + 1)]
        out = forward_with_optional_aux(model, coords, fx, T=input_t)
        pred_t = out["pred"]
        preds.append(pred_t)

        pred_t_decoded = _decode_if_needed(y_normalizer, pred_t)
        y_t_decoded = _decode_if_needed(y_normalizer, y_t)
        rel_l2 = relative_l2_per_sample(pred_t_decoded, y_t_decoded)
        if step_loss_fn is None:
            loss = rel_l2.sum()
        else:
            loss = step_loss_fn(
                pred_t_decoded.reshape(pred_t_decoded.shape[0], -1),
                y_t_decoded.reshape(y_t_decoded.shape[0], -1),
            )
        metric_loss = loss / int(coords.shape[0])
        metric_loss_sum += float(metric_loss.item())

        if train_step_per_time:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(max_grad_norm)
                )
            optimizer.step()
            preds[-1] = preds[-1].detach()
        else:
            total_loss = total_loss + loss

        step_rel_l2_sum = step_rel_l2_sum + rel_l2.sum()
        pred_mean_sum += float(pred_t_decoded.mean().item())
        diag_metrics.update(out["diagnostics"], weight=int(coords.shape[0]))

    pred = torch.cat(preds, dim=-1)
    mean_loss = total_loss / max(int(t_out), 1)
    mean_metric_loss = metric_loss_sum / max(int(t_out), 1)
    if train_step_per_time:
        mean_loss = torch.tensor(
            mean_metric_loss,
            device=coords.device,
            dtype=coords.dtype,
        )
    return pred, mean_loss, mean_metric_loss, step_rel_l2_sum, {
        "pred_mean": pred_mean_sum / max(int(t_out), 1),
        "diagnostics": diag_metrics.as_diagnostics(),
    }


def evaluate_dynamic_conditional(
    model,
    loader,
    *,
    device,
    y_normalizer,
    prepare_batch,
    t_out: int,
    out_dim: int,
):
    model.eval()
    nsl_l2_loss = _build_nsl_l2_loss()
    mse_sum = 0.0
    rel_l2_sum = 0.0
    step_rel_l2_sum = 0.0
    diag_metrics = MetricAccumulator()
    samples = 0
    with torch.no_grad():
        for batch in loader:
            coords, time, fx, target = prepare_batch(batch, device=device)
            pred, _, _, step_rel_l2, summary = rollout_dynamic_conditional(
                model,
                coords,
                fx,
                time,
                target,
                t_out=t_out,
                out_dim=out_dim,
                y_normalizer=y_normalizer,
                step_loss_fn=nsl_l2_loss,
            )
            pred_decoded = _decode_if_needed(y_normalizer, pred)
            target_decoded = _decode_if_needed(y_normalizer, target)
            batch_size = int(target.shape[0])
            mse_sum += float(
                F.mse_loss(pred_decoded, target_decoded, reduction="none")
                .reshape(batch_size, -1)
                .mean(dim=1)
                .sum()
                .item()
            )
            rel_l2_sum += float(
                relative_l2_per_sample(pred_decoded, target_decoded).sum().item()
            )
            step_rel_l2_sum += float(step_rel_l2.item())
            diag_metrics.update(summary["diagnostics"], weight=batch_size)
            samples += batch_size

    denom = max(samples, 1)
    metrics = {
        "mse": mse_sum / denom,
        "rel_l2": rel_l2_sum / denom,
        "step_rel_l2": step_rel_l2_sum / (denom * max(int(t_out), 1)),
    }
    metrics.update(diag_metrics.compute())
    return metrics


def train_dynamic_conditional_task(
    cfg: dict,
    *,
    device: torch.device,
    build_train_val_loaders,
    get_meta,
    runtime_overrides,
    prepare_batch,
    log_fn=None,
):
    print("building train/validation loaders", flush=True)
    train_loader, val_loader = build_train_val_loaders(cfg)
    meta = get_meta(train_loader)
    y_normalizer = getattr(train_loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    print(
        f"creating model {cfg.get('model', {}).get('backbone', 'FNO')}",
        flush=True,
    )
    model, model_args, resolved_nsl_root = create_model(
        cfg,
        device=device,
        runtime_overrides=runtime_overrides(meta),
    )
    nsl_l2_loss = _build_nsl_l2_loss()
    output_dir = resolve_output_dir(cfg)
    write_resolved_config(
        cfg, output_dir=output_dir, resolved_nsl_root=resolved_nsl_root
    )

    training_cfg = deepcopy(cfg.get("training", {}))
    seed = int(training_cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    training_cfg["steps_per_epoch"] = len(train_loader)
    optimizer = build_optimizer(model, training_cfg)
    scheduler, scheduler_step_per_batch = build_scheduler(optimizer, training_cfg)
    start_epoch = 0
    resume_checkpoint = training_cfg.get("resume_checkpoint")
    if resume_checkpoint is not None:
        start_epoch = restore_training_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        print(f"resuming training from {resume_checkpoint} at epoch {start_epoch}")
    train_step_per_time = bool(training_cfg.get("step_per_time", True))
    max_grad_norm = training_cfg.get("max_grad_norm")
    t_out = int(meta["t_out"])
    out_dim = int(meta["out_dim"])
    has_validation = val_loader is not None
    best_score = float("inf")
    best_metrics = None
    best_selection_metric = "val/rel_l2" if has_validation else "train/loss"
    wandb_cfg = cfg.get("wandb_logging", {}) or {}
    log_every = normalize_interval(wandb_cfg.get("log_every", 100))
    image_log_every = normalize_interval(wandb_cfg.get("image_log_every"))
    plasticity_video_fps = int(wandb_cfg.get("plasticity_video_fps", 4))

    init_wandb_if_enabled(cfg)
    try:
        num_epochs = int(training_cfg.get("num_epochs", 1))
        if start_epoch >= num_epochs:
            raise ValueError(
                f"Checkpoint epoch {start_epoch} is already at or beyond "
                f"configured num_epochs={num_epochs}."
            )
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss_sum = 0.0
            train_rel_l2_sum = 0.0
            train_step_rel_l2_sum = 0.0
            train_pred_mean_sum = 0.0
            train_diag_metrics = MetricAccumulator()
            samples = 0

            for step, batch in enumerate(train_loader):
                coords, time, fx, target = prepare_batch(batch, device=device)
                pred, loss, metric_loss, step_rel_l2, summary = (
                    rollout_dynamic_conditional(
                        model,
                        coords,
                        fx,
                        time,
                        target,
                        t_out=t_out,
                        out_dim=out_dim,
                        y_normalizer=y_normalizer,
                        train_step_per_time=train_step_per_time,
                        optimizer=optimizer,
                        max_grad_norm=max_grad_norm,
                        step_loss_fn=nsl_l2_loss,
                    )
                )
                if not train_step_per_time:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(max_grad_norm)
                        )
                    optimizer.step()
                if scheduler is not None and scheduler_step_per_batch:
                    scheduler.step()

                pred_decoded = _decode_if_needed(y_normalizer, pred)
                target_decoded = _decode_if_needed(y_normalizer, target)
                batch_size = int(target.shape[0])
                train_loss_sum += float(metric_loss) * batch_size
                train_rel_l2_sum += float(
                    relative_l2_per_sample(pred_decoded, target_decoded).sum().item()
                )
                train_step_rel_l2_sum += float(step_rel_l2.item())
                train_pred_mean_sum += float(summary["pred_mean"]) * batch_size
                train_diag_metrics.update(summary["diagnostics"], weight=batch_size)
                samples += batch_size

                if log_every is not None and log_every > 0:
                    global_step = epoch * len(train_loader) + step + 1
                    if (step + 1) % log_every == 0:
                        payload = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "train/step_loss": float(metric_loss),
                            "train/step_rel_l2": float(
                                step_rel_l2.item()
                                / (batch_size * max(int(t_out), 1))
                            ),
                            "train/pred_mean": float(summary["pred_mean"]),
                        }
                        payload.update(
                            prefix_metric_names(
                                diagnostic_values(summary["diagnostics"]),
                                "train",
                            )
                        )
                        log_metrics(payload, step=global_step)

            if scheduler is not None and not scheduler_step_per_batch:
                scheduler.step()

            train_metrics = {
                "loss": train_loss_sum / max(samples, 1),
                "rel_l2": train_rel_l2_sum / max(samples, 1),
                "step_rel_l2": train_step_rel_l2_sum
                / (max(samples, 1) * max(int(t_out), 1)),
                "pred_mean": train_pred_mean_sum / max(samples, 1),
            }
            train_metrics.update(train_diag_metrics.compute())
            epoch_step = (epoch + 1) * len(train_loader)
            val_metrics = None
            if has_validation:
                if (
                    image_log_every is not None
                    and image_log_every > 0
                    and epoch % image_log_every == 0
                    and str(meta.get("loader", "")) == "plas"
                ):
                    model.eval()
                    with torch.no_grad():
                        sampled_idx = random.randint(0, max(len(val_loader) - 1, 0))
                        for val_step, batch in enumerate(val_loader):
                            if val_step != sampled_idx:
                                continue
                            coords, time, fx, target = prepare_batch(
                                batch,
                                device=device,
                            )
                            pred, *_ = rollout_dynamic_conditional(
                                model,
                                coords,
                                fx,
                                time,
                                target,
                                t_out=t_out,
                                out_dim=out_dim,
                                y_normalizer=y_normalizer,
                                step_loss_fn=nsl_l2_loss,
                            )
                            final_time = time[:, -1:].reshape(coords.shape[0], 1)
                            final_out = forward_with_optional_aux(
                                model,
                                coords,
                                fx,
                                T=final_time,
                            )
                            pred_decoded = _decode_if_needed(y_normalizer, pred)
                            target_decoded = _decode_if_needed(y_normalizer, target)
                            if log_fn is not None:
                                ctx = MediaLogContext(
                                    pred=pred_decoded,
                                    target=target_decoded,
                                    coords=coords,
                                    fx=fx,
                                    aux_tensors=final_out["aux_tensors"],
                                    meta=meta,
                                    cfg=cfg,
                                    prefix="validation",
                                    epoch=epoch,
                                    step=epoch_step,
                                    out_dir=None,
                                )
                                log_fn(ctx)
                                constraint = getattr(model, "constraint", None)
                                if constraint is not None:
                                    type(constraint).log_media(ctx)
                            break
                    model.train()

                val_metrics = evaluate_dynamic_conditional(
                    model,
                    val_loader,
                    device=device,
                    y_normalizer=y_normalizer,
                    prepare_batch=prepare_batch,
                    t_out=t_out,
                    out_dim=out_dim,
                )
                print(
                    f"epoch {epoch + 1}/{int(training_cfg.get('num_epochs', 1))} "
                    f"train_rel_l2={train_metrics['rel_l2']:.6f} "
                    f"val_rel_l2={val_metrics['rel_l2']:.6f} "
                    f"val_mse={val_metrics['mse']:.6f}"
                )
                log_metrics(
                    (
                        {"epoch": epoch + 1}
                        | prefix_metric_names(train_metrics, "train")
                        | prefix_metric_names(val_metrics, "val")
                    ),
                    step=epoch_step,
                )
            else:
                print(
                    f"epoch {epoch + 1}/{int(training_cfg.get('num_epochs', 1))} "
                    f"train_loss={train_metrics['loss']:.6f} "
                    f"train_rel_l2={train_metrics['rel_l2']:.6f}"
                )
                log_metrics(
                    {"epoch": epoch + 1}
                    | prefix_metric_names(train_metrics, "train"),
                    step=epoch_step,
                )

            checkpoint_payload = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": None
                if scheduler is None
                else scheduler.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "selection_metric": best_selection_metric,
                "model_args": vars(model_args),
                "resolved_nsl_root": str(resolved_nsl_root),
            }
            current_score = (
                float(val_metrics["rel_l2"])
                if has_validation and val_metrics is not None
                else float(train_metrics["loss"])
            )
            is_best = current_score < best_score
            if is_best:
                best_score = current_score
                best_metrics = deepcopy(
                    val_metrics if has_validation else train_metrics
                )
            save_checkpoint_bundle(output_dir, checkpoint_payload, is_best=is_best)
        result = {
            "selection_metric": best_selection_metric,
            "best_score": best_score,
            "output_dir": str(output_dir),
            "resolved_nsl_root": str(resolved_nsl_root),
        }
        if has_validation:
            result.update(
                {
                    "best_val_rel_l2": best_score,
                    "best_val_metrics": best_metrics,
                }
            )
        else:
            result.update(
                {
                    "best_train_loss": best_score,
                    "best_train_metrics": best_metrics,
                    "best_val_rel_l2": None,
                    "best_val_metrics": None,
                }
            )
        return result
    finally:
        finish_wandb_if_active()


def test_dynamic_conditional_task(
    cfg: dict,
    *,
    device: torch.device,
    checkpoint_path: str | Path | None,
    build_test_loader,
    get_meta,
    runtime_overrides,
    prepare_batch,
    log_fn=None,
):
    output_dir = resolve_output_dir(cfg)
    if checkpoint_path is None:
        checkpoint_path = output_dir / "best.pt"
    test_loader = build_test_loader(cfg)
    meta = get_meta(test_loader)
    y_normalizer = getattr(test_loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
    model, model_args, resolved_nsl_root = create_model(
        cfg,
        device=device,
        runtime_overrides=runtime_overrides(meta),
    )
    checkpoint = load_checkpoint_state(checkpoint_path, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    metrics = evaluate_dynamic_conditional(
        model,
        test_loader,
        device=device,
        y_normalizer=y_normalizer,
        prepare_batch=prepare_batch,
        t_out=int(meta["t_out"]),
        out_dim=int(meta["out_dim"]),
    )
    media_paths = {}
    if str(meta.get("loader", "")) == "plas":
        media_dir = output_dir / "test_media"
        t_out = int(meta["t_out"])
        out_dim = int(meta["out_dim"])
        plasticity_video_fps = int(
            (cfg.get("wandb_logging", {}) or {}).get("plasticity_video_fps", 4)
        )
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                coords, time, fx, target = prepare_batch(batch, device=device)
                pred, *_ = rollout_dynamic_conditional(
                    model,
                    coords,
                    fx,
                    time,
                    target,
                    t_out=t_out,
                    out_dim=out_dim,
                    y_normalizer=y_normalizer,
                    step_loss_fn=_build_nsl_l2_loss(),
                )
                final_time = time[:, -1:].reshape(coords.shape[0], 1)
                final_out = forward_with_optional_aux(
                    model,
                    coords,
                    fx,
                    T=final_time,
                )
                pred_decoded = _decode_if_needed(y_normalizer, pred)
                target_decoded = _decode_if_needed(y_normalizer, target)
                if log_fn is not None:
                    ctx = MediaLogContext(
                        pred=pred_decoded,
                        target=target_decoded,
                        coords=coords,
                        fx=fx,
                        aux_tensors=final_out["aux_tensors"],
                        meta=meta,
                        cfg=cfg,
                        prefix="test",
                        epoch=0,
                        step=None,
                        out_dir=media_dir,
                    )
                    media_paths.update(log_fn(ctx))
                    constraint = getattr(model, "constraint", None)
                    if constraint is not None:
                        media_paths.update(type(constraint).log_media(ctx))
                break
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "metrics": metrics,
        "media": media_paths,
        "resolved_nsl_root": str(resolved_nsl_root),
        "model_args": vars(model_args),
    }
