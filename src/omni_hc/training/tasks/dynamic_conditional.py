from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

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
from omni_hc.training.reproducibility import seed_everything, training_seed


def _decode_if_needed(normalizer, tensor: torch.Tensor) -> torch.Tensor:
    return normalizer.decode(tensor) if normalizer is not None else tensor


def _build_nsl_l2_loss():
    from utils.loss import L2Loss

    return L2Loss(size_average=False)


def _dynamic_component_rel_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    t_out: int,
    out_dim: int,
    channel_slice: slice,
) -> torch.Tensor:
    """Mean per-step relative L2 for a channel group in a dynamic target."""
    pred_view = pred.reshape(pred.shape[0], pred.shape[1], int(t_out), int(out_dim))
    target_view = target.reshape(
        target.shape[0],
        target.shape[1],
        int(t_out),
        int(out_dim),
    )
    total = torch.zeros((), device=pred.device, dtype=pred.dtype)
    for step in range(int(t_out)):
        total = total + relative_l2_per_sample(
            pred_view[:, :, step, channel_slice],
            target_view[:, :, step, channel_slice],
        ).mean()
    return total / max(int(t_out), 1)


def _dynamic_component_error_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    t_out: int,
    out_dim: int,
    channel_slice: slice,
) -> dict[str, float]:
    pred_view = pred.reshape(pred.shape[0], pred.shape[1], int(t_out), int(out_dim))
    target_view = target.reshape(
        target.shape[0],
        target.shape[1],
        int(t_out),
        int(out_dim),
    )
    residual = pred_view[..., channel_slice] - target_view[..., channel_slice]
    target_values = target_view[..., channel_slice]
    return {
        "mse": float(residual.square().mean().item()),
        "mae": float(residual.abs().mean().item()),
        "target_rms": float(target_values.square().mean().sqrt().item()),
    }


def _plasticity_material_grid(
    shapelist,
    *,
    device,
    dtype,
    x_left: float = 0.35,
    x_right: float = -49.65,
    y_top: float = 14.9,
    y_bottom: float = -0.1,
) -> torch.Tensor | None:
    if shapelist is None:
        return None
    i_count, j_count = (int(v) for v in tuple(shapelist))
    x = torch.linspace(
        float(x_left),
        float(x_right),
        i_count,
        device=device,
        dtype=dtype,
    )
    y = torch.linspace(
        float(y_top),
        float(y_bottom),
        j_count,
        device=device,
        dtype=dtype,
    )
    xx = x[:, None].expand(i_count, j_count)
    yy = y[None, :].expand(i_count, j_count)
    return torch.stack((xx, yy), dim=-1).reshape(1, i_count * j_count, 1, 2)


def _plasticity_material_grid_from_config(
    cfg: dict,
    meta: dict,
    *,
    device,
    dtype,
) -> torch.Tensor | None:
    if str(meta.get("loader", "")) != "plas":
        return None
    constraint_cfg = cfg.get("constraint", {}) or {}
    return _plasticity_material_grid(
        meta.get("shapelist"),
        device=device,
        dtype=dtype,
        x_left=float(constraint_cfg.get("x_left", 0.35)),
        x_right=float(constraint_cfg.get("x_right", -49.65)),
        y_top=float(constraint_cfg.get("y_top", 14.9)),
        y_bottom=float(constraint_cfg.get("y_bottom", -0.1)),
    )


def _plasticity_deviation_consistency_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    t_out: int,
    out_dim: int,
    material_grid: torch.Tensor | None,
) -> dict[str, float]:
    if int(out_dim) < 4 or material_grid is None:
        return {}
    pred_view = pred.reshape(pred.shape[0], pred.shape[1], int(t_out), int(out_dim))
    target_view = target.reshape(
        target.shape[0],
        target.shape[1],
        int(t_out),
        int(out_dim),
    )
    material = material_grid.to(device=pred.device, dtype=pred.dtype)
    pred_residual = pred_view[..., 2:4] - (pred_view[..., 0:2] - material)
    target_residual = target_view[..., 2:4] - (target_view[..., 0:2] - material)
    return {
        "deviation_consistency_pred_mse": float(pred_residual.square().mean().item()),
        "deviation_consistency_pred_abs_max": float(pred_residual.abs().max().item()),
        "deviation_consistency_target_mse": float(
            target_residual.square().mean().item()
        ),
        "deviation_consistency_target_abs_max": float(
            target_residual.abs().max().item()
        ),
    }


def plasticity_component_loss_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    t_out: int,
    out_dim: int,
    material_grid: torch.Tensor | None = None,
) -> dict[str, float]:
    if int(out_dim) < 4:
        return {}
    position_errors = _dynamic_component_error_metrics(
        pred,
        target,
        t_out=t_out,
        out_dim=out_dim,
        channel_slice=slice(0, 2),
    )
    deviation_errors = _dynamic_component_error_metrics(
        pred,
        target,
        t_out=t_out,
        out_dim=out_dim,
        channel_slice=slice(2, 4),
    )
    metrics = {
        "loss_position": float(
            _dynamic_component_rel_l2(
                pred,
                target,
                t_out=t_out,
                out_dim=out_dim,
                channel_slice=slice(0, 2),
            ).item()
        ),
        "loss_deviation": float(
            _dynamic_component_rel_l2(
                pred,
                target,
                t_out=t_out,
                out_dim=out_dim,
                channel_slice=slice(2, 4),
            ).item()
        ),
        "mse_position": position_errors["mse"],
        "mse_deviation": deviation_errors["mse"],
        "mae_position": position_errors["mae"],
        "mae_deviation": deviation_errors["mae"],
        "target_rms_position": position_errors["target_rms"],
        "target_rms_deviation": deviation_errors["target_rms"],
    }
    metrics.update(
        _plasticity_deviation_consistency_metrics(
            pred,
            target,
            t_out=t_out,
            out_dim=out_dim,
            material_grid=material_grid,
        )
    )
    return metrics


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
    collect_aux_keys: tuple[str, ...] | None = None,
):
    preds = []
    aux_history = {key: [] for key in (collect_aux_keys or ())}
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
        for key in aux_history:
            value = out["aux_tensors"].get(key)
            if isinstance(value, torch.Tensor):
                aux_history[key].append(value.detach())

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
    summary = {
        "pred_mean": pred_mean_sum / max(int(t_out), 1),
        "diagnostics": diag_metrics.as_diagnostics(),
    }
    if aux_history:
        summary["aux_tensors"] = {
            key: torch.stack(values, dim=1)
            for key, values in aux_history.items()
            if len(values) == int(t_out)
        }
    return pred, mean_loss, mean_metric_loss, step_rel_l2_sum, summary


def evaluate_dynamic_conditional(
    model,
    loader,
    *,
    device,
    x_normalizer=None,
    y_normalizer,
    prepare_batch,
    t_out: int,
    out_dim: int,
    plasticity_material_grid: torch.Tensor | None = None,
    compute_extra_diagnostics=None,
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
            component_metrics = plasticity_component_loss_metrics(
                pred_decoded,
                target_decoded,
                t_out=t_out,
                out_dim=out_dim,
                material_grid=plasticity_material_grid,
            )
            for name, value in component_metrics.items():
                diag_metrics.update({name: value}, weight=batch_size)
            diag_metrics.update(summary["diagnostics"], weight=batch_size)
            if compute_extra_diagnostics is not None:
                fx_for_diagnostics = _decode_if_needed(x_normalizer, fx)
                extra = compute_extra_diagnostics(
                    pred=pred_decoded,
                    coords=coords,
                    fx=fx_for_diagnostics,
                    target=target_decoded,
                    time=time,
                )
                if extra:
                    diag_metrics.update(extra, weight=batch_size)
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
    seed = training_seed(cfg)
    seed_everything(seed)

    print("building train/validation loaders", flush=True)
    train_loader, val_loader = build_train_val_loaders(cfg)
    meta = get_meta(train_loader)
    x_normalizer = getattr(train_loader, "x_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
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
    if (
        x_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_input_normalizer")
    ):
        model.constraint.set_input_normalizer(x_normalizer)
    nsl_l2_loss = _build_nsl_l2_loss()
    output_dir = resolve_output_dir(cfg)
    write_resolved_config(
        cfg, output_dir=output_dir, resolved_nsl_root=resolved_nsl_root
    )

    training_cfg = deepcopy(cfg.get("training", {}))
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
    plasticity_material_grid = _plasticity_material_grid_from_config(
        cfg,
        meta,
        device=device,
        dtype=torch.float32,
    )
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
            train_component_metrics = MetricAccumulator()
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
                component_metrics = plasticity_component_loss_metrics(
                    pred_decoded,
                    target_decoded,
                    t_out=t_out,
                    out_dim=out_dim,
                    material_grid=plasticity_material_grid,
                )
                train_component_metrics.update(component_metrics, weight=batch_size)
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
                            {
                                f"train/step_{name}": value
                                for name, value in component_metrics.items()
                            }
                        )
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
            train_metrics.update(train_component_metrics.compute())
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
                            pred, _, _, _, rollout_summary = rollout_dynamic_conditional(
                                model,
                                coords,
                                fx,
                                time,
                                target,
                                t_out=t_out,
                                out_dim=out_dim,
                                y_normalizer=y_normalizer,
                                step_loss_fn=nsl_l2_loss,
                                collect_aux_keys=("envelope_x", "envelope_y"),
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
                                    aux_tensors={
                                        **final_out["aux_tensors"],
                                        **rollout_summary.get("aux_tensors", {}),
                                    },
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
                    x_normalizer=x_normalizer,
                    y_normalizer=y_normalizer,
                    prepare_batch=prepare_batch,
                    t_out=t_out,
                    out_dim=out_dim,
                    plasticity_material_grid=plasticity_material_grid,
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
    x_normalizer = getattr(test_loader, "x_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
    y_normalizer = getattr(test_loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)
    model, model_args, resolved_nsl_root = create_model(
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
    from omni_hc.training.benchmark_diagnostics import make_benchmark_diagnostic_fn

    metrics = evaluate_dynamic_conditional(
        model,
        test_loader,
        device=device,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        prepare_batch=prepare_batch,
        t_out=int(meta["t_out"]),
        out_dim=int(meta["out_dim"]),
        plasticity_material_grid=_plasticity_material_grid_from_config(
            cfg,
            meta,
            device=device,
            dtype=torch.float32,
        ),
        compute_extra_diagnostics=make_benchmark_diagnostic_fn(cfg, meta),
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
                pred, _, _, _, rollout_summary = rollout_dynamic_conditional(
                    model,
                    coords,
                    fx,
                    time,
                    target,
                    t_out=t_out,
                    out_dim=out_dim,
                    y_normalizer=y_normalizer,
                    step_loss_fn=_build_nsl_l2_loss(),
                    collect_aux_keys=("envelope_x", "envelope_y"),
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
                        aux_tensors={
                            **final_out["aux_tensors"],
                            **rollout_summary.get("aux_tensors", {}),
                        },
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
