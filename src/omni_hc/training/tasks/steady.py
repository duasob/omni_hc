from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from omni_hc.constraints import boundary_stats
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import (
    build_optimizer,
    build_scheduler,
    forward_with_optional_aux,
    load_checkpoint_state,
    normalize_interval,
    relative_l2_per_sample,
    resolve_output_dir,
    save_checkpoint_bundle,
    write_resolved_config,
)
from omni_hc.training.logging_utils import (
    finish_wandb_if_active,
    init_wandb_if_enabled,
    log_metrics,
    log_steady_field_images,
)


def _decode_if_needed(normalizer, tensor: torch.Tensor) -> torch.Tensor:
    return normalizer.decode(tensor) if normalizer is not None else tensor


def _resolve_boundary_check(cfg: dict, meta: dict) -> dict[str, float] | None:
    constraint_cfg = cfg.get("constraint", {}) or {}
    constraint_name = str(constraint_cfg.get("name", "")).strip().lower()
    if constraint_name not in {
        "dirichlet_ansatz",
        "dirichlet_boundary_ansatz",
        "darcy_flux_projection",
        "darcy_flux_fft_pad",
    }:
        return None
    if constraint_name in {"darcy_flux_projection", "darcy_flux_fft_pad"} and not bool(
        constraint_cfg.get("enforce_boundary", True)
    ):
        return None

    domain_bounds = meta.get("domain_bounds", (0.0, 1.0))
    if len(domain_bounds) != 2:
        raise ValueError(
            f"Expected meta['domain_bounds'] to contain (lower, upper), got {domain_bounds!r}"
        )
    lower_default, upper_default = domain_bounds
    return {
        "target_value": float(constraint_cfg.get("boundary_value", 0.0)),
        "lower": float(constraint_cfg.get("lower", lower_default)),
        "upper": float(constraint_cfg.get("upper", upper_default)),
        "atol": float(constraint_cfg.get("boundary_atol", 1e-6)),
    }


def evaluate_steady(
    model,
    loader,
    *,
    device,
    y_normalizer,
    prepare_batch,
    boundary_check=None,
):
    model.eval()
    mse_sum = 0.0
    rel_l2_sum = 0.0
    boundary_mean_sum = 0.0
    boundary_max = 0.0
    samples = 0
    with torch.no_grad():
        for batch in loader:
            coords, fx, target = prepare_batch(batch, device=device)
            out = forward_with_optional_aux(model, coords, fx)
            pred = _decode_if_needed(y_normalizer, out["pred"])
            target_decoded = _decode_if_needed(y_normalizer, target)
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
            if boundary_check is not None:
                stats = boundary_stats(
                    pred,
                    coords,
                    target_value=float(boundary_check["target_value"]),
                    lower=float(boundary_check["lower"]),
                    upper=float(boundary_check["upper"]),
                    atol=float(boundary_check["atol"]),
                )
                boundary_mean_sum += stats["boundary_abs_mean"] * batch_size
                boundary_max = max(boundary_max, stats["boundary_abs_max"])
            samples += batch_size
    denom = max(samples, 1)
    metrics = {"mse": mse_sum / denom, "rel_l2": rel_l2_sum / denom}
    if boundary_check is not None:
        metrics["boundary_abs_mean"] = boundary_mean_sum / denom
        metrics["boundary_abs_max"] = boundary_max
    return metrics


def train_steady_task(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
    build_train_val_loaders,
    get_meta,
    runtime_overrides,
    prepare_batch,
):
    train_loader, val_loader = build_train_val_loaders(cfg)
    meta = get_meta(train_loader)
    x_normalizer = getattr(train_loader, "x_normalizer", None)
    if x_normalizer is not None:
        x_normalizer = x_normalizer.to(device)
    y_normalizer = getattr(train_loader, "y_normalizer", None)
    if y_normalizer is not None:
        y_normalizer = y_normalizer.to(device)

    model, model_args, resolved_nsl_root = create_model(
        cfg,
        nsl_root=nsl_root,
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

    output_dir = resolve_output_dir(cfg)
    write_resolved_config(cfg, output_dir=output_dir, resolved_nsl_root=resolved_nsl_root)

    training_cfg = deepcopy(cfg.get("training", {}))
    training_cfg["steps_per_epoch"] = len(train_loader)
    optimizer = build_optimizer(model, training_cfg)
    scheduler, scheduler_step_per_batch = build_scheduler(optimizer, training_cfg)
    best_val = float("inf")
    best_metrics = None
    wandb_cfg = cfg.get("wandb_logging", {}) or {}
    log_every = normalize_interval(wandb_cfg.get("log_every", 100))
    image_log_every = normalize_interval(wandb_cfg.get("image_log_every"))
    boundary_check = _resolve_boundary_check(cfg, meta)

    init_wandb_if_enabled(cfg)
    try:
        for epoch in range(int(training_cfg.get("num_epochs", 1))):
            model.train()
            train_loss_sum = 0.0
            train_mse_sum = 0.0
            train_rel_l2_sum = 0.0
            train_pred_mean_sum = 0.0
            train_pred_base_mean_sum = 0.0
            train_corr_mean_sum = 0.0
            pred_base_count = 0
            corr_count = 0
            train_boundary_mean_sum = 0.0
            train_boundary_max = 0.0
            samples = 0

            for step, batch in enumerate(train_loader):
                coords, fx, target = prepare_batch(batch, device=device)
                out = forward_with_optional_aux(model, coords, fx)
                pred = out["pred"]
                pred_decoded = _decode_if_needed(y_normalizer, pred)
                target_decoded = _decode_if_needed(y_normalizer, target)
                rel_l2 = relative_l2_per_sample(pred_decoded, target_decoded)
                loss = rel_l2.mean()
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
                train_mse_sum += float(
                    F.mse_loss(pred_decoded, target_decoded, reduction="none")
                    .reshape(batch_size, -1)
                    .mean(dim=1)
                    .sum()
                    .item()
                )
                train_rel_l2_sum += float(rel_l2.sum().item())
                train_pred_mean_sum += float(pred_decoded.mean().item()) * batch_size
                if out["pred_base_mean"] is not None:
                    train_pred_base_mean_sum += float(out["pred_base_mean"]) * batch_size
                    pred_base_count += batch_size
                if out["corr_mean"] is not None:
                    train_corr_mean_sum += float(out["corr_mean"]) * batch_size
                    corr_count += batch_size
                if boundary_check is not None:
                    stats = boundary_stats(
                        pred_decoded,
                        coords,
                        target_value=float(boundary_check["target_value"]),
                        lower=float(boundary_check["lower"]),
                        upper=float(boundary_check["upper"]),
                        atol=float(boundary_check["atol"]),
                    )
                    train_boundary_mean_sum += stats["boundary_abs_mean"] * batch_size
                    train_boundary_max = max(
                        train_boundary_max, stats["boundary_abs_max"]
                    )
                samples += batch_size

                if log_every is not None and log_every > 0:
                    global_step = epoch * len(train_loader) + step + 1
                    if (step + 1) % log_every == 0:
                        payload = {
                            "epoch": epoch + 1,
                            "global_step": global_step,
                            "train/step_loss": float(loss.item()),
                            "train/step_rel_l2": float(rel_l2.mean().item()),
                            "train/pred_mean": float(pred_decoded.mean().item()),
                        }
                        if out["pred_base_mean"] is not None:
                            payload["train/pred_base_mean"] = float(
                                out["pred_base_mean"]
                            )
                        if out["corr_mean"] is not None:
                            payload["train/corr_mean"] = float(out["corr_mean"])
                        log_metrics(payload, step=global_step)

            if scheduler is not None and not scheduler_step_per_batch:
                scheduler.step()

            train_metrics = {
                "loss": train_loss_sum / max(samples, 1),
                "mse": train_mse_sum / max(samples, 1),
                "rel_l2": train_rel_l2_sum / max(samples, 1),
                "pred_mean": train_pred_mean_sum / max(samples, 1),
                "pred_base_mean": None
                if pred_base_count == 0
                else train_pred_base_mean_sum / pred_base_count,
                "corr_mean": None
                if corr_count == 0
                else train_corr_mean_sum / corr_count,
            }
            if boundary_check is not None:
                train_metrics["boundary_abs_mean"] = (
                    train_boundary_mean_sum / max(samples, 1)
                )
                train_metrics["boundary_abs_max"] = train_boundary_max

            if (
                (image_log_every is not None and image_log_every > 0)
                or (log_every is not None and log_every > 0)
            ):
                model.eval()
                with torch.no_grad():
                    max_val_idx = max(len(val_loader) - 1, 0)
                    sampled_idx = random.randint(0, max_val_idx)
                    h, w = tuple(meta["shapelist"])
                    for val_step, batch in enumerate(val_loader):
                        coords, fx, target = prepare_batch(batch, device=device)
                        out = forward_with_optional_aux(model, coords, fx)
                        pred_decoded = _decode_if_needed(y_normalizer, out["pred"])
                        target_decoded = _decode_if_needed(y_normalizer, target)
                        if (
                            image_log_every is not None
                            and image_log_every > 0
                            and epoch % image_log_every == 0
                            and val_step == sampled_idx
                        ):
                            log_steady_field_images(
                                fx,
                                pred_decoded,
                                target_decoded,
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
                                "val/pred_mean": float(pred_decoded.mean().item()),
                            }
                            if out["pred_base_mean"] is not None:
                                payload["val/pred_base_mean"] = float(
                                    out["pred_base_mean"]
                                )
                            if out["corr_mean"] is not None:
                                payload["val/corr_mean"] = float(out["corr_mean"])
                            log_metrics(payload)
                model.train()

            val_metrics = evaluate_steady(
                model,
                val_loader,
                device=device,
                y_normalizer=y_normalizer,
                prepare_batch=prepare_batch,
                boundary_check=boundary_check,
            )
            print(
                f"epoch {epoch + 1}/{int(training_cfg.get('num_epochs', 1))} "
                f"train_rel_l2={train_metrics['rel_l2']:.6f} "
                f"val_rel_l2={val_metrics['rel_l2']:.6f} "
                f"val_mse={val_metrics['mse']:.6f}"
            )
            log_metrics(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/mse": train_metrics["mse"],
                    "train/rel_l2": train_metrics["rel_l2"],
                    "val/mse": val_metrics["mse"],
                    "val/rel_l2": val_metrics["rel_l2"],
                },
                step=epoch + 1,
            )
            if boundary_check is not None:
                log_metrics(
                    {
                        "epoch": epoch + 1,
                        "train/boundary_abs_mean": train_metrics["boundary_abs_mean"],
                        "train/boundary_abs_max": train_metrics["boundary_abs_max"],
                        "val/boundary_abs_mean": val_metrics["boundary_abs_mean"],
                        "val/boundary_abs_max": val_metrics["boundary_abs_max"],
                    },
                    step=epoch + 1,
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
                "model_args": vars(model_args),
                "resolved_nsl_root": str(resolved_nsl_root),
            }
            is_best = val_metrics["rel_l2"] < best_val
            if is_best:
                best_val = float(val_metrics["rel_l2"])
                best_metrics = deepcopy(val_metrics)
            save_checkpoint_bundle(output_dir, checkpoint_payload, is_best=is_best)
        return {
            "best_val_rel_l2": best_val,
            "best_val_metrics": best_metrics,
            "output_dir": str(output_dir),
            "resolved_nsl_root": str(resolved_nsl_root),
        }
    finally:
        finish_wandb_if_active()


def test_steady_task(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
    checkpoint_path: str | Path | None,
    build_test_loader,
    get_meta,
    runtime_overrides,
    prepare_batch,
    boundary_check=None,
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
        nsl_root=nsl_root,
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
    model.load_state_dict(checkpoint["model_state_dict"])
    if boundary_check is None:
        boundary_check = _resolve_boundary_check(cfg, meta)
    metrics = evaluate_steady(
        model,
        test_loader,
        device=device,
        y_normalizer=y_normalizer,
        prepare_batch=prepare_batch,
        boundary_check=boundary_check,
    )
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "metrics": metrics,
        "resolved_nsl_root": str(resolved_nsl_root),
        "model_args": vars(model_args),
    }
