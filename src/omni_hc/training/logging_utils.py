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


def log_prediction_images(pred, target, fx, h, w, *, prefix, epoch, step=None):
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
        },
        step=step,
    )


def log_steady_field_images(
    coeff,
    pred,
    target,
    h,
    w,
    *,
    prefix,
    epoch,
    aux_tensors=None,
    step=None,
):
    if wandb is None or getattr(wandb, "run", None) is None:
        return

    if coeff.shape[-1] >= 2 and plt is not None:
        log_pipe_flow_images(
            coeff,
            pred,
            target,
            h,
            w,
            prefix=prefix,
            epoch=epoch,
            aux_tensors=aux_tensors,
            step=step,
        )
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
        },
        step=step,
    )


def _plot_point_cloud_field(
    ax,
    coords,
    field,
    *,
    title,
    point_size,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=field,
        s=point_size,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        alpha=0.98,
    )
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.4, alpha=0.18)
    return scatter


def log_unstructured_point_cloud_images(
    coords,
    pred,
    target,
    *,
    prefix,
    epoch,
    aux_tensors=None,
    point_size=24.0,
    step=None,
):
    if wandb is None or getattr(wandb, "run", None) is None or plt is None:
        return

    coords_np = coords[0].detach().cpu().numpy()
    pred_np = pred[0, :, 0].detach().cpu().numpy()
    target_np = target[0, :, 0].detach().cpu().numpy()
    error_np = pred_np - target_np

    pred_vmin = float(min(pred_np.min(), target_np.min()))
    pred_vmax = float(max(pred_np.max(), target_np.max()))
    err_abs = float(np.abs(error_np).max())
    if err_abs <= 0.0:
        err_abs = 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=150)
    im0 = _plot_point_cloud_field(
        axes[0],
        coords_np,
        target_np,
        title="target sigma",
        point_size=point_size,
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    im1 = _plot_point_cloud_field(
        axes[1],
        coords_np,
        pred_np,
        title="predicted sigma",
        point_size=point_size,
        vmin=pred_vmin,
        vmax=pred_vmax,
    )
    im2 = _plot_point_cloud_field(
        axes[2],
        coords_np,
        error_np,
        title="pred - target",
        point_size=point_size,
        vmin=-err_abs,
        vmax=err_abs,
        cmap="coolwarm",
    )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()

    payload = {
        f"{prefix}/elasticity_sigma": wandb.Image(fig),
        "epoch": epoch + 1,
    }
    plt.close(fig)

    if aux_tensors is not None:
        latent_keys = [
            ("theta", "theta", "twilight"),
            ("theta_raw", "theta raw", "coolwarm"),
            ("log_lambda", "log lambda", "coolwarm"),
            ("log_lambda_raw", "log lambda raw", "coolwarm"),
            ("lambda", "lambda", "magma"),
            ("det_c", "det C", "coolwarm"),
            ("right_cauchy_green_c11", "C 11", "viridis"),
            ("right_cauchy_green_c12", "C 12", "coolwarm"),
            ("right_cauchy_green_c22", "C 22", "viridis"),
            ("det_fhat", "det F hat", "coolwarm"),
            ("i1", "I1", "plasma"),
            ("i2", "I2", "plasma"),
            ("stress_11", "stress 11", "viridis"),
            ("stress_22", "stress 22", "viridis"),
            ("stress_12", "stress 12", "coolwarm"),
            ("stress_trace", "stress trace", "viridis"),
            ("stress_dev_11", "dev stress 11", "coolwarm"),
            ("stress_dev_22", "dev stress 22", "coolwarm"),
            ("stress_dev_12", "dev stress 12", "coolwarm"),
            ("stress_dev_inner", "dev stress inner", "plasma"),
            ("fhat_11", "F hat 11", "coolwarm"),
            ("fhat_12", "F hat 12", "coolwarm"),
            ("fhat_21", "F hat 21", "coolwarm"),
            ("fhat_22", "F hat 22", "coolwarm"),
            ("deformation_f11", "F 11", "coolwarm"),
            ("deformation_f12", "F 12", "coolwarm"),
            ("deformation_f21", "F 21", "coolwarm"),
            ("deformation_f22", "F 22", "coolwarm"),
            ("stretch_raw", "stretch raw", "coolwarm"),
            ("phi", "phi", "twilight"),
            ("phi_raw", "phi raw", "coolwarm"),
            ("amplitude_raw", "amplitude raw", "coolwarm"),
            ("amplitude", "amplitude", "viridis"),
            ("directional_stretch", "directional stretch", "plasma"),
            ("det_f", "det F", "coolwarm"),
        ]
        available = [
            (key, title, cmap)
            for key, title, cmap in latent_keys
            if key in aux_tensors
        ]
        if available:
            ncols = min(3, len(available))
            nrows = int(np.ceil(len(available) / ncols))
            fig_aux, axes_aux = plt.subplots(
                nrows,
                ncols,
                figsize=(4.5 * ncols, 4.0 * nrows),
                dpi=150,
                squeeze=False,
            )
            for ax, (key, title, cmap) in zip(axes_aux.ravel(), available):
                values = aux_tensors[key][0, :, 0].detach().cpu().numpy()
                vmin = vmax = None
                if key in {"det_f", "det_c"}:
                    spread = max(float(np.abs(values - 1.0).max()), 1e-12)
                    vmin = 1.0 - spread
                    vmax = 1.0 + spread
                im = _plot_point_cloud_field(
                    ax,
                    coords_np,
                    values,
                    title=title,
                    point_size=point_size,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                )
                fig_aux.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            for ax in axes_aux.ravel()[len(available) :]:
                ax.axis("off")
            fig_aux.tight_layout()
            payload[f"{prefix}/elasticity_latents"] = wandb.Image(fig_aux)
            plt.close(fig_aux)

    wandb.log(payload, step=step)


def _plot_pipe_field(ax, x, y, field, *, title, vmin=None, vmax=None, cmap="viridis"):
    mesh = ax.pcolormesh(x, y, field, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.plot(x[:, 0], y[:, 0], color="white", linewidth=1.2)
    ax.plot(x[:, -1], y[:, -1], color="white", linewidth=1.2)
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return mesh


def log_pipe_flow_images(
    coords,
    pred,
    target,
    h,
    w,
    *,
    prefix,
    epoch,
    aux_tensors=None,
    step=None,
):
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
        },
        step=step,
    )
    plt.close(fig)

    if aux_tensors is not None and all(
        key in aux_tensors for key in ("stream_psi", "stream_uy", "stream_div")
    ):
        log_pipe_stream_images(
            coords,
            h,
            w,
            prefix=prefix,
            epoch=epoch,
            psi=aux_tensors["stream_psi"],
            uy=aux_tensors["stream_uy"],
            divergence=aux_tensors["stream_div"],
            psi_bc=aux_tensors.get("stream_psi_bc"),
            mask=aux_tensors.get("stream_mask"),
            step=step,
        )


def log_pipe_stream_images(
    coords,
    h,
    w,
    *,
    prefix,
    epoch,
    step=None,
    psi,
    uy,
    divergence,
    psi_bc=None,
    mask=None,
):
    if wandb is None or getattr(wandb, "run", None) is None or plt is None:
        return

    x = coords[0, :, 0].detach().cpu().reshape(h, w).numpy()
    y = coords[0, :, 1].detach().cpu().reshape(h, w).numpy()
    psi_img = psi[0, :, 0].detach().cpu().reshape(h, w).numpy()
    uy_img = uy[0, :, 0].detach().cpu().reshape(h, w).numpy()
    div_img = divergence[0, :, 0].detach().cpu().reshape(h - 1, w - 1).numpy()

    ncols = 5 if psi_bc is not None and mask is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 3.5), dpi=140)
    if ncols == 3:
        axes_main = axes
    else:
        axes_main = axes[:3]
    im0 = _plot_pipe_field(axes[0], x, y, psi_img, title="stream psi")
    uy_scale = max(float(np.abs(uy_img).max()), 1e-12)
    im1 = _plot_pipe_field(
        axes_main[1],
        x,
        y,
        uy_img,
        title="recovered uy",
        vmin=-uy_scale,
        vmax=uy_scale,
        cmap="coolwarm",
    )
    div_scale = max(float(np.abs(div_img).max()), 1e-12)
    im2 = axes_main[2].pcolormesh(
        x[:-1, :-1],
        y[:-1, :-1],
        div_img,
        shading="nearest",
        cmap="coolwarm",
        vmin=-div_scale,
        vmax=div_scale,
    )
    axes_main[2].set_title("stream div")
    axes_main[2].set_aspect("equal", adjustable="box")
    axes_main[2].set_xlabel("x")
    axes_main[2].set_ylabel("y")
    fig.colorbar(im0, ax=axes_main[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes_main[1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes_main[2], fraction=0.046, pad=0.04)
    if psi_bc is not None and mask is not None:
        psi_bc_img = psi_bc[0, :, 0].detach().cpu().reshape(h, w).numpy()
        mask_img = mask[0, :, 0].detach().cpu().reshape(h, w).numpy()
        im3 = _plot_pipe_field(axes[3], x, y, psi_bc_img, title="psi bc")
        im4 = _plot_pipe_field(axes[4], x, y, mask_img, title="stream mask")
        fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        fig.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    fig.tight_layout()

    wandb.log(
        {
            f"{prefix}/pipe_stream": wandb.Image(fig),
            "epoch": epoch + 1,
        },
        step=step,
    )
    plt.close(fig)


def finish_wandb_if_active():
    if wandb is None:
        return
    if getattr(wandb, "run", None) is not None:
        wandb.finish()
