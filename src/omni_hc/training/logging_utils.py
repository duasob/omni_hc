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


def _to_grayscale_rgb(image):
    img = _normalize_image(image)
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


def finish_wandb_if_active():
    if wandb is None:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
