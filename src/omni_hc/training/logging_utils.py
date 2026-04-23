from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

import numpy as np
import torch

try:
    import wandb
except Exception:  # pragma: no cover - optional dependency
    wandb = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None


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
        project = wandb_cfg.get("project", "omni_hc")
        run_name = wandb_cfg.get("run_name")
        if run_name is None:
            output_dir = cfg.get("paths", {}).get("output_dir", "")
            run_name = str(output_dir).strip("/").replace("/", "_") or None
        wandb.init(
            project=project,
            name=run_name,
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

    if coeff.shape[-1] >= 2 and plt is not None:
        log_pipe_flow_images(coeff, pred, target, h, w, prefix=prefix, epoch=epoch)
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


def _plot_pipe_field(ax, x, y, field, *, title, vmin=None, vmax=None, cmap="viridis"):
    mesh = ax.pcolormesh(x, y, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(x[:, 0], y[:, 0], color="white", linewidth=1.2)
    ax.plot(x[:, -1], y[:, -1], color="white", linewidth=1.2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def log_pipe_flow_images(coords, pred, target, h, w, *, prefix, epoch):
    if wandb is None or getattr(wandb, "run", None) is None or plt is None:
        return

    x = coords[0, :, 0].detach().cpu().reshape(h, w).numpy()
    y = coords[0, :, 1].detach().cpu().reshape(h, w).numpy()
    pred_img = pred[0, :, 0].detach().cpu().reshape(h, w).numpy()
    target_img = target[0, :, 0].detach().cpu().reshape(h, w).numpy()
    error_img = pred_img - target_img

    pred_vmin = float(min(pred_img.min(), target_img.min()))
    pred_vmax = float(max(pred_img.max(), target_img.max()))
    err_abs = float(np.abs(error_img).max())
    if err_abs <= 0.0:
        err_abs = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=140)
    im0 = _plot_pipe_field(
        axes[0],
        x,
        y,
        pred_img,
        title="pred ux",
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    im1 = _plot_pipe_field(
        axes[1],
        x,
        y,
        target_img,
        title="target ux",
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    im2 = _plot_pipe_field(
        axes[2],
        x,
        y,
        error_img,
        title="pred - target",
        vmin=-err_abs,
        vmax=err_abs,
        cmap="coolwarm",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()

    wandb.log(
        {
            f"{prefix}/pipe_ux": wandb.Image(fig),
            "epoch": epoch + 1,
        }
    )
    plt.close(fig)


def finish_wandb_if_active():
    if wandb is None:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
