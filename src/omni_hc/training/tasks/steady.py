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
    resolve_output_dir,
    restore_training_checkpoint,
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


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _check_finite(name: str, tensor: torch.Tensor | None, *, raise_on_nonfinite: bool):
    if tensor is None:
        return
    mask = ~torch.isfinite(tensor)
    if mask.any():
        bad = tensor[mask]
        summary = f"count={int(bad.numel())}"
        if bad.numel() > 0:
            summary += f" min={float(bad.min().item())} max={float(bad.max().item())}"
        msg = f"Non-finite values detected in {name}: {summary}"
        if raise_on_nonfinite:
            raise RuntimeError(msg)
        print(msg)


def _tensor_stats(name: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{name}: None"
    if tensor.numel() == 0:
        return f"{name}: shape={tuple(tensor.shape)} empty"
    finite = torch.isfinite(tensor)
    if not finite.any():
        return f"{name}: shape={tuple(tensor.shape)} all_nonfinite"
    finite_vals = tensor[finite]
    return (
        f"{name}: shape={tuple(tensor.shape)} "
        f"min={float(finite_vals.min().item())} "
        f"max={float(finite_vals.max().item())} "
        f"mean={float(finite_vals.mean().item())} "
        f"std={float(finite_vals.std(unbiased=False).item())}"
    )


def _all_boundary_idx(H: int, W: int) -> torch.Tensor:
    """Flat indices for the perimeter of an H×W row-major grid, corners deduplicated."""
    bottom = torch.arange(W)
    top    = torch.arange((H - 1) * W, H * W)
    left   = torch.arange(0, H * W, W)
    right  = torch.arange(W - 1, H * W, W)
    return torch.cat([bottom, top, left[1:-1], right[1:-1]])


def evaluate_steady(
    model,
    loader,
    *,
    device,
    y_normalizer,
    prepare_batch,
    shapelist: tuple[int, int] | None = None,
    debug_nan_checks: bool = False,
    raise_on_nonfinite: bool = True,
    compute_extra_diagnostics=None,
):
    model.eval()
    mse_sum = 0.0
    rel_l2_sum = 0.0
    boundary_mse_sum = 0.0
    boundary_rel_l2_sum = 0.0
    diag_metrics = MetricAccumulator()
    samples = 0
    constraint = getattr(model, "constraint", None)
    boundary_idx = getattr(constraint, "idx_all_boundary", None)
    if boundary_idx is None and shapelist is not None and len(shapelist) == 2:
        boundary_idx = _all_boundary_idx(int(shapelist[0]), int(shapelist[1]))
    with torch.no_grad():
        for batch in loader:
            coords, fx, target = prepare_batch(batch, device=device)
            if debug_nan_checks:
                _check_finite(
                    "eval/coords", coords, raise_on_nonfinite=raise_on_nonfinite
                )
                _check_finite(
                    "eval/target", target, raise_on_nonfinite=raise_on_nonfinite
                )
            out = forward_with_optional_aux(model, coords, fx)
            pred = _decode_if_needed(y_normalizer, out["pred"])
            target_decoded = _decode_if_needed(y_normalizer, target)
            if debug_nan_checks:
                _check_finite("eval/pred", pred, raise_on_nonfinite=raise_on_nonfinite)
                _check_finite(
                    "eval/target_decoded",
                    target_decoded,
                    raise_on_nonfinite=raise_on_nonfinite,
                )
            batch_size = int(target.shape[0])
            mse_sum += float(
                F.mse_loss(pred, target_decoded, reduction="none")
                .reshape(batch_size, -1)
                .mean(dim=1)
                .sum()
                .item()
            )
            rel_l2_sum += float(
                relative_l2_per_sample(pred, target_decoded).sum().item()
            )
            if boundary_idx is not None:
                bidx = boundary_idx.to(pred.device)
                boundary_mse_sum += float(
                    F.mse_loss(pred[:, bidx, :], target_decoded[:, bidx, :], reduction="none")
                    .reshape(batch_size, -1)
                    .mean(dim=1)
                    .sum()
                    .item()
                )
                boundary_rel_l2_sum += float(
                    relative_l2_per_sample(
                        pred[:, bidx, :], target_decoded[:, bidx, :]
                    )
                    .sum()
                    .item()
                )
            diag_metrics.update(out["diagnostics"], weight=batch_size)
            if compute_extra_diagnostics is not None:
                extra = compute_extra_diagnostics(
                    pred=pred,
                    coords=coords,
                    fx=fx,
                    target=target_decoded,
                )
                if extra:
                    diag_metrics.update(extra, weight=batch_size)
            samples += batch_size
    denom = max(samples, 1)
    metrics = {"mse": mse_sum / denom, "rel_l2": rel_l2_sum / denom}
    if boundary_idx is not None:
        metrics["boundary_rmse"]    = float((boundary_mse_sum / denom) ** 0.5)
        metrics["boundary_rel_l2"] = boundary_rel_l2_sum / denom
    metrics.update(diag_metrics.compute())
    return metrics


def train_steady_task(
    cfg: dict,
    *,
    device: torch.device,
    build_train_val_loaders,
    get_meta,
    runtime_overrides,
    prepare_batch,
    log_fn=None,
):
    # TODO: remove this printing
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
    nsl_l2_loss = _build_nsl_l2_loss()
    if (
        y_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_target_normalizer")
    ):
        model.constraint.set_target_normalizer(y_normalizer)
    if (
        x_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_input_normalizer")
    ):
        model.constraint.set_input_normalizer(x_normalizer)
    if hasattr(model, "constraint") and hasattr(model.constraint, "set_grid_shape"):
        shapelist = meta.get("shapelist")
        if shapelist is not None:
            model.constraint.set_grid_shape(tuple(shapelist))
    if hasattr(model, "constraint") and hasattr(model.constraint, "set_domain_bounds"):
        domain_bounds = meta.get("domain_bounds")
        if domain_bounds is not None and len(domain_bounds) == 2:
            model.constraint.set_domain_bounds(
                lower=float(domain_bounds[0]),
                upper=float(domain_bounds[1]),
            )
    uy_normalizer = getattr(train_loader, "uy_normalizer", None)
    if uy_normalizer is not None:
        uy_normalizer = uy_normalizer.to(device)
    if (
        uy_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_uy_normalizer")
    ):
        model.constraint.set_uy_normalizer(uy_normalizer)

    output_dir = resolve_output_dir(cfg)
    write_resolved_config(
        cfg, output_dir=output_dir, resolved_nsl_root=resolved_nsl_root
    )

    training_cfg = deepcopy(cfg.get("training", {}))
    debug_nan_checks = bool(training_cfg.get("debug_nan_checks", False))
    raise_on_nonfinite = bool(training_cfg.get("raise_on_nonfinite", True))
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
    derivloss = _as_bool(
        training_cfg.get("derivloss", getattr(model_args, "derivloss", False))
    )
    derivloss_weight = float(training_cfg.get("derivloss_weight", 0.1))
    shapelist = tuple(meta.get("shapelist", ()))
    if derivloss and len(shapelist) != 2:
        raise ValueError(f"derivloss requires a 2D shapelist, got {shapelist!r}")
    nsl_deriv_loss = None
    if derivloss:
        from utils.loss import DerivLoss

        nsl_deriv_loss = DerivLoss(size_average=False, shapelist=shapelist)
    has_validation = val_loader is not None
    best_score = float("inf")
    best_metrics = None
    best_selection_metric = "val/rel_l2" if has_validation else "train/loss"
    wandb_cfg = cfg.get("wandb_logging", {}) or {}
    log_every = normalize_interval(wandb_cfg.get("log_every", 100))
    image_log_every = normalize_interval(wandb_cfg.get("image_log_every"))

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
            train_mse_sum = 0.0
            train_rel_l2_sum = 0.0
            train_deriv_rel_l2_sum = 0.0
            train_pred_mean_sum = 0.0
            train_diag_metrics = MetricAccumulator()
            samples = 0

            for step, batch in enumerate(train_loader):
                coords, fx, target = prepare_batch(batch, device=device)
                uy_target = batch.get("y_uy")
                if uy_target is not None:
                    uy_target = uy_target.to(device)
                if debug_nan_checks:
                    _check_finite(
                        "train/coords", coords, raise_on_nonfinite=raise_on_nonfinite
                    )
                    _check_finite(
                        "train/target", target, raise_on_nonfinite=raise_on_nonfinite
                    )
                out = forward_with_optional_aux(model, coords, fx, uy_target=uy_target)
                pred = out["pred"]
                if debug_nan_checks:
                    _check_finite(
                        "train/pred", pred, raise_on_nonfinite=raise_on_nonfinite
                    )
                pred_decoded = _decode_if_needed(y_normalizer, pred)
                target_decoded = _decode_if_needed(y_normalizer, target)
                if debug_nan_checks:
                    _check_finite(
                        "train/pred_decoded",
                        pred_decoded,
                        raise_on_nonfinite=raise_on_nonfinite,
                    )
                    _check_finite(
                        "train/target_decoded",
                        target_decoded,
                        raise_on_nonfinite=raise_on_nonfinite,
                    )
                if debug_nan_checks and epoch == 0 and step == 0:
                    print(_tensor_stats("train/coords", coords))
                    print(_tensor_stats("train/target", target))
                    print(_tensor_stats("train/pred", pred))
                    print(_tensor_stats("train/pred_decoded", pred_decoded))
                    print(_tensor_stats("train/target_decoded", target_decoded))
                rel_l2 = relative_l2_per_sample(pred_decoded, target_decoded)
                deriv_loss_value = None
                batch_size = int(target.shape[0])
                loss = nsl_l2_loss(pred_decoded, target_decoded)
                if out.get("extra_loss") is not None:
                    loss = loss + out["extra_loss"]
                batch_loss_sum = float(loss.item())
                if derivloss:
                    deriv_loss = nsl_deriv_loss(pred_decoded, target_decoded)
                    deriv_loss_value = float(deriv_loss.item())
                    loss = loss + derivloss_weight * deriv_loss
                    batch_loss_sum = float(loss.item())
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

                train_loss_sum += batch_loss_sum
                train_mse_sum += float(
                    F.mse_loss(pred_decoded, target_decoded, reduction="none")
                    .reshape(batch_size, -1)
                    .mean(dim=1)
                    .sum()
                    .item()
                )
                train_rel_l2_sum += float(rel_l2.sum().item())
                if deriv_loss_value is not None:
                    train_deriv_rel_l2_sum += deriv_loss_value
                train_pred_mean_sum += float(pred_decoded.mean().item()) * batch_size
                train_diag_metrics.update(out["diagnostics"], weight=batch_size)
                samples += batch_size

                if log_every is not None and log_every > 0:
                    global_step = epoch * len(train_loader) + step + 1
                    if (step + 1) % log_every == 0:
                        payload = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "train/step_loss": batch_loss_sum / max(batch_size, 1),
                            "train/step_rel_l2": float(rel_l2.mean().item()),
                            "train/pred_mean": float(pred_decoded.mean().item()),
                        }
                        if deriv_loss_value is not None:
                            payload["train/step_deriv_rel_l2"] = deriv_loss_value / max(
                                batch_size, 1
                            )
                        payload.update(
                            prefix_metric_names(
                                diagnostic_values(out["diagnostics"]),
                                "train",
                            )
                        )
                        log_metrics(payload, step=global_step)

            if scheduler is not None and not scheduler_step_per_batch:
                scheduler.step()

            train_metrics = {
                "loss": train_loss_sum / max(samples, 1),
                "mse": train_mse_sum / max(samples, 1),
                "rel_l2": train_rel_l2_sum / max(samples, 1),
                "pred_mean": train_pred_mean_sum / max(samples, 1),
            }
            if derivloss:
                train_metrics["deriv_rel_l2"] = train_deriv_rel_l2_sum / max(
                    samples,
                    1,
                )
            train_metrics.update(train_diag_metrics.compute())
            epoch_step = (epoch + 1) * len(train_loader)

            if has_validation and (
                (image_log_every is not None and image_log_every > 0)
                or (log_every is not None and log_every > 0)
            ):
                model.eval()
                with torch.no_grad():
                    max_val_idx = max(len(val_loader) - 1, 0)
                    sampled_idx = random.randint(0, max_val_idx)
                    can_log_image = (
                        log_fn is not None
                        and image_log_every is not None
                        and image_log_every > 0
                    )
                    for val_step, batch in enumerate(val_loader):
                        coords, fx, target = prepare_batch(batch, device=device)
                        out = forward_with_optional_aux(model, coords, fx)
                        pred_decoded = _decode_if_needed(y_normalizer, out["pred"])
                        target_decoded = _decode_if_needed(y_normalizer, target)
                        if (
                            can_log_image
                            and epoch % image_log_every == 0
                            and val_step == sampled_idx
                        ):
                            ctx = MediaLogContext(
                                pred=pred_decoded,
                                target=target_decoded,
                                coords=_decode_if_needed(x_normalizer, coords),
                                fx=_decode_if_needed(x_normalizer, fx) if fx is not None else None,
                                aux_tensors=out["aux_tensors"],
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
                        if (
                            log_every is not None
                            and log_every > 0
                            and (val_step + 1) % log_every == 0
                        ):
                            payload = {
                                "epoch": epoch + 1,
                                "val/pred_mean": float(pred_decoded.mean().item()),
                            }
                            payload.update(
                                prefix_metric_names(
                                    diagnostic_values(out["diagnostics"]),
                                    "val",
                                )
                            )
                            log_metrics(payload, step=epoch_step)
                model.train()

            val_metrics = None
            if has_validation:
                val_metrics = evaluate_steady(
                    model,
                    val_loader,
                    device=device,
                    y_normalizer=y_normalizer,
                    prepare_batch=prepare_batch,
                    shapelist=tuple(meta.get("shapelist", ())) or None,
                    debug_nan_checks=debug_nan_checks,
                    raise_on_nonfinite=raise_on_nonfinite,
                )
                boundary_str = (
                    f" val_boundary_rmse={val_metrics['boundary_rmse']:.6f}"
                    f" val_boundary_rel_l2={val_metrics['boundary_rel_l2']:.6f}"
                    if "boundary_rel_l2" in val_metrics
                    else ""
                )
                print(
                    f"epoch {epoch + 1}/{int(training_cfg.get('num_epochs', 1))} "
                    f"train_rel_l2={train_metrics['rel_l2']:.6f} "
                    f"val_rel_l2={val_metrics['rel_l2']:.6f} "
                    f"val_mse={val_metrics['mse']:.6f}"
                    f"{boundary_str}"
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
                    {"epoch": epoch + 1} | prefix_metric_names(train_metrics, "train"),
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


def test_steady_task(
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
    evaluation_cfg = cfg.get("evaluation", {}) or {}
    debug_nan_checks = bool(evaluation_cfg.get("debug_nan_checks", False))
    raise_on_nonfinite = bool(evaluation_cfg.get("raise_on_nonfinite", True))
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
        y_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_target_normalizer")
    ):
        model.constraint.set_target_normalizer(y_normalizer)
    if (
        x_normalizer is not None
        and hasattr(model, "constraint")
        and hasattr(model.constraint, "set_input_normalizer")
    ):
        model.constraint.set_input_normalizer(x_normalizer)
    if hasattr(model, "constraint") and hasattr(model.constraint, "set_grid_shape"):
        shapelist = meta.get("shapelist")
        if shapelist is not None:
            model.constraint.set_grid_shape(tuple(shapelist))
    if hasattr(model, "constraint") and hasattr(model.constraint, "set_domain_bounds"):
        domain_bounds = meta.get("domain_bounds")
        if domain_bounds is not None and len(domain_bounds) == 2:
            model.constraint.set_domain_bounds(
                lower=float(domain_bounds[0]),
                upper=float(domain_bounds[1]),
            )
    checkpoint = load_checkpoint_state(checkpoint_path, device=device)
    load_model_state_dict(model, checkpoint["model_state_dict"])
    from omni_hc.training.benchmark_diagnostics import make_benchmark_diagnostic_fn

    metrics = evaluate_steady(
        model,
        test_loader,
        device=device,
        y_normalizer=y_normalizer,
        prepare_batch=prepare_batch,
        shapelist=tuple(meta.get("shapelist", ())) or None,
        debug_nan_checks=debug_nan_checks,
        compute_extra_diagnostics=make_benchmark_diagnostic_fn(cfg, meta),
        raise_on_nonfinite=raise_on_nonfinite,
    )
    media_paths = {}
    media_dir = output_dir / "test_media"
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            coords, fx, target = prepare_batch(batch, device=device)
            out = forward_with_optional_aux(model, coords, fx)
            pred_decoded = _decode_if_needed(y_normalizer, out["pred"])
            target_decoded = _decode_if_needed(y_normalizer, target)
            if log_fn is not None:
                ctx = MediaLogContext(
                    pred=pred_decoded,
                    target=target_decoded,
                    coords=_decode_if_needed(x_normalizer, coords),
                    fx=_decode_if_needed(x_normalizer, fx) if fx is not None else None,
                    aux_tensors=out["aux_tensors"],
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
