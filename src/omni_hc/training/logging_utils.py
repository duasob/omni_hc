from __future__ import annotations

import numpy as np
import torch

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None


def init_wandb_if_enabled(cfg: dict):
    wandb_cfg = cfg.get("wandb_logging", {}) or {}
    if not wandb_cfg.get("wandb", False):
        return False
    if wandb is None:
        print("W&B disabled: wandb package is not available.")
        wandb_cfg["wandb"] = False
        return False
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
        wandb.init(
            project=wandb_cfg["project"],
            name=wandb_cfg["run_name"],
            config=cfg,
            reinit=True,
        )
        return True
    except Exception as exc:
        print(f"W&B init failed ({type(exc).__name__}): {exc}")
        print("Continuing with wandb logging disabled.")
        wandb_cfg["wandb"] = False
        return False


def log_metrics(payload: dict, *, step: int | None = None):
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    wandb.log(payload, step=step)


def _normalize_image(image):
    img = image.detach().cpu().numpy()
    vmin = float(img.min())
    vmax = float(img.max())
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return (img - vmin) / (vmax - vmin)


def _normalize_image_with_range(image, *, vmin: float, vmax: float):
    img = image.detach().cpu().numpy()
    if vmax <= vmin:
        vmax = vmin + 1e-12
    return np.clip((img - vmin) / (vmax - vmin), 0.0, 1.0)


def _to_grayscale_rgb(image, *, vmin: float | None = None, vmax: float | None = None):
    if vmin is None or vmax is None:
        img = _normalize_image(image)
    else:
        img = _normalize_image_with_range(image, vmin=vmin, vmax=vmax)
    rgb = np.stack([img, img, img], axis=-1)
    return (rgb * 255).astype("uint8")


def _to_red_blue_rgb(image):
    img = image.detach().cpu().numpy()
    scale = float(np.abs(img).max())
    if scale <= 0:
        scale = 1e-12
    norm = np.clip(img / scale, -1.0, 1.0)
    red = np.clip(norm, 0.0, 1.0)
    blue = np.clip(-norm, 0.0, 1.0)
    rgb = np.stack([red, np.zeros_like(red), blue], axis=-1)
    return (rgb * 255).astype("uint8")


def log_prediction_images(pred, target, fx, h, w, *, prefix, epoch):
    if wandb is None or getattr(wandb, "run", None) is None:
        return

    input_img = fx[0, :, 0].view(h, w)
    pred_img = pred[0, :, 0].view(h, w)
    target_img = target[0, :, 0].view(h, w)
    diff_img = pred_img - target_img

    pred_merge = torch.cat([input_img, pred_img], dim=1)
    target_merge = torch.cat([input_img, target_img], dim=1)

    wandb.log(
        {
            f"{prefix}/pred": wandb.Image(_to_grayscale_rgb(pred_merge)),
            f"{prefix}/target": wandb.Image(_to_grayscale_rgb(target_merge)),
            f"{prefix}/error": wandb.Image(_to_red_blue_rgb(diff_img)),
            "epoch": epoch + 1,
        }
    )


def log_steady_field_images(coeff, pred, target, h, w, *, prefix, epoch):
    if wandb is None or getattr(wandb, "run", None) is None:
        return

    coeff_img = coeff[0, :, 0].view(h, w)
    pred_img = pred[0, :, 0].view(h, w)
    target_img = target[0, :, 0].view(h, w)
    error_signed = pred_img - target_img
    error_abs = error_signed.abs()

    pred_vmin = float(min(pred_img.min().item(), target_img.min().item()))
    pred_vmax = float(max(pred_img.max().item(), target_img.max().item()))
    abs_vmax = float(error_abs.max().item())
    if abs_vmax <= 0.0:
        abs_vmax = 1e-12

    wandb.log(
        {
            f"{prefix}/coeff": wandb.Image(_to_grayscale_rgb(coeff_img)),
            f"{prefix}/pred": wandb.Image(
                _to_grayscale_rgb(pred_img, vmin=pred_vmin, vmax=pred_vmax)
            ),
            f"{prefix}/target": wandb.Image(
                _to_grayscale_rgb(target_img, vmin=pred_vmin, vmax=pred_vmax)
            ),
            f"{prefix}/error_signed": wandb.Image(_to_red_blue_rgb(error_signed)),
            f"{prefix}/error_abs": wandb.Image(
                _to_grayscale_rgb(error_abs, vmin=0.0, vmax=abs_vmax)
            ),
            "epoch": epoch + 1,
        }
    )


def finish_wandb_if_active():
    if wandb is None:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
