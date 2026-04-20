from __future__ import annotations

# %%
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - local debug convenience
    plt = None

from omni_hc.benchmarks.darcy.data import build_test_loader
from omni_hc.constraints.spectral import (
    crop_spatial_2d,
    finite_difference_curl_2d,
    finite_difference_divergence_2d,
    finite_difference_gradient_2d,
    normalize_padding_2d,
    pad_spatial_2d,
    sine_poisson_solve_dirichlet_2d,
    spectral_divergence_2d,
    spectral_gradient_2d,
)
from omni_hc.core import load_composed_config
from omni_hc.integrations.nsl.modeling import create_model

torch.set_printoptions(precision=4, sci_mode=False)


@dataclass
class DebugConfig:
    config_path: str = "configs/experiments/darcy/fno_small.yaml"
    nsl_root: str | None = None
    device: str = "cpu"
    batch_index: int = 0
    sample_index: int = 0
    padding: int = 8
    padding_mode: str = "reflect"
    padding_modes: tuple[str, ...] = ("reflect", "replicate", "zeros")
    interior_margin: int = 8
    particular_field: str = "y_only"
    dtype: torch.dtype = torch.float64


CFG = DebugConfig()
DEVICE = torch.device(CFG.device)

print(CFG)
if plt is None:
    print("matplotlib is not available; plots are disabled.")


# %%


def darcy_runtime_overrides(meta: dict) -> dict[str, object]:
    return {
        "shapelist": tuple(meta["shapelist"]),
        "task": str(meta["task"]),
        "loader": str(meta["loader"]),
        "geotype": str(meta["geotype"]),
        "space_dim": int(meta["space_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "out_dim": int(meta["out_dim"]),
    }


def print_scalar_stats(
    name: str, tensor: torch.Tensor, *, target: float | None = None
) -> None:
    value = tensor.detach()
    print(f"[{name}] shape={tuple(value.shape)}")
    print(
        f"  mean={value.mean().item():.6f} min={value.min().item():.6f} max={value.max().item():.6f}"
    )
    print(
        f"  abs_mean={value.abs().mean().item():.6f} abs_max={value.abs().max().item():.6f}"
    )
    if target is not None:
        residual = value - target
        print(
            f"  residual(target={target:.3f}) abs_mean={residual.abs().mean().item():.6f} "
            f"abs_max={residual.abs().max().item():.6f}"
        )


def print_divergence_report(
    name: str,
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
    target: float | None = None,
) -> None:
    print(f"\n=== {name} ===")
    if field.shape[-2] > 2 and field.shape[-1] > 2:
        fd_div = finite_difference_divergence_2d(field, dy=dy, dx=dx)
        print_scalar_stats("finite_difference_div", fd_div, target=target)
    spectral_div = spectral_divergence_2d(field, dy=dy, dx=dx)
    print_scalar_stats("spectral_div", spectral_div, target=target)


def interior_core_2d(tensor: torch.Tensor, *, margin: int) -> torch.Tensor:
    if margin <= 0:
        return tensor
    if tensor.shape[-2] <= 2 * margin or tensor.shape[-1] <= 2 * margin:
        raise ValueError(
            f"Interior margin {margin} is too large for tensor shape {tuple(tensor.shape)!r}"
        )
    return tensor[..., margin:-margin, margin:-margin]


def _plot_ready(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu()
    if value.ndim == 4:
        return value[0, 0]
    if value.ndim == 3:
        return value[0]
    if value.ndim == 2:
        return value
    raise ValueError(f"Unsupported tensor shape for plotting: {tuple(value.shape)!r}")


def plot_scalar_panels(
    title: str,
    panels: list[tuple[str, torch.Tensor]],
    *,
    center_zero: bool = False,
    cmap: str = "viridis",
) -> None:
    if plt is None:
        return
    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    processed = [_plot_ready(tensor) for _, tensor in panels]
    vmin = vmax = None
    if center_zero:
        scale = max(float(value.abs().max().item()) for value in processed)
        if scale <= 0.0:
            scale = 1e-12
        vmin, vmax = -scale, scale

    for ax, (name, _), value in zip(axes, panels, processed):
        image = ax.imshow(
            value.numpy(),
            cmap=cmap,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.8)

    fig.suptitle(title)
    plt.show()


def plot_vector_panels(title: str, field: torch.Tensor) -> None:
    vx = field[:, 0:1]
    vy = field[:, 1:2]
    norm = field.square().sum(dim=1, keepdim=True).sqrt()
    plot_scalar_panels(
        title,
        [
            ("vx", vx),
            ("vy", vy),
            ("norm", norm),
        ],
    )


def reshape_channels_last_to_grid(
    tensor: torch.Tensor,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(
            f"Expected (batch, n_points, channels), got {tuple(tensor.shape)!r}"
        )
    return tensor.transpose(1, 2).reshape(
        tensor.shape[0], tensor.shape[2], height, width
    )


def make_padded_mesh(
    *,
    height: int,
    width: int,
    dy: float,
    dx: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(height, device=DEVICE, dtype=dtype) * dy
    x = torch.arange(width, device=DEVICE, dtype=dtype) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


def particular_y_only(
    *,
    batch_size: int,
    height: int,
    width: int,
    dy: float,
    dx: float,
    force_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy, _ = make_padded_mesh(height=height, width=width, dy=dy, dx=dx, dtype=dtype)
    yy = yy - yy.mean()
    vy = force_value * yy.view(1, 1, height, width).expand(batch_size, 1, height, width)
    vx = torch.zeros_like(vy)
    return torch.cat([vx, vy], dim=1)


def particular_xy_affine(
    *,
    batch_size: int,
    height: int,
    width: int,
    dy: float,
    dx: float,
    force_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    yy, xx = make_padded_mesh(height=height, width=width, dy=dy, dx=dx, dtype=dtype)
    vx = (
        0.5
        * force_value
        * xx.view(1, 1, height, width).expand(batch_size, 1, height, width)
    )
    vy = (
        0.5
        * force_value
        * yy.view(1, 1, height, width).expand(batch_size, 1, height, width)
    )
    return torch.cat([vx, vy], dim=1)


def build_particular_flux(
    *,
    name: str,
    batch_size: int,
    height: int,
    width: int,
    dy: float,
    dx: float,
    force_value: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = str(name).lower()
    if key == "y_only":
        return particular_y_only(
            batch_size=batch_size,
            height=height,
            width=width,
            dy=dy,
            dx=dx,
            force_value=force_value,
            dtype=dtype,
        )
    if key in {"xy_affine", "x_y_affine"}:
        return particular_xy_affine(
            batch_size=batch_size,
            height=height,
            width=width,
            dy=dy,
            dx=dx,
            force_value=force_value,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported particular field '{name}'")


def stream_velocity_from_psi(
    psi: torch.Tensor, *, dy: float, dx: float
) -> torch.Tensor:
    gradient = spectral_gradient_2d(psi, dy=dy, dx=dx)
    dpsi_dx = gradient[:, 0:1]
    dpsi_dy = gradient[:, 1:2]
    return torch.cat([dpsi_dy, -dpsi_dx], dim=1)


def recover_pressure_from_gradient_sine(
    gradient: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rhs = finite_difference_divergence_2d(gradient, dy=dy, dx=dx)
    pressure = sine_poisson_solve_dirichlet_2d(rhs, dy=dy, dx=dx)
    return pressure, rhs


def recover_pressure_from_flux_sine(
    flux: torch.Tensor,
    permeability: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gradient = -flux / permeability
    pressure, rhs = recover_pressure_from_gradient_sine(gradient, dy=dy, dx=dx)
    return pressure, gradient, rhs


def vector_error_norm(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (pred - target).square().sum(dim=1, keepdim=True).sqrt()


def boundary_abs(field: torch.Tensor) -> torch.Tensor:
    top = field[..., 0, :].abs()
    bottom = field[..., -1, :].abs()
    left = field[..., :, 0].abs()
    right = field[..., :, -1].abs()
    return torch.cat(
        [
            top.reshape(field.shape[0], -1),
            bottom.reshape(field.shape[0], -1),
            left.reshape(field.shape[0], -1),
            right.reshape(field.shape[0], -1),
        ],
        dim=1,
    )


def collect_divergence_metrics(
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
    target: float,
) -> dict[str, float]:
    fd_residual = finite_difference_divergence_2d(field, dy=dy, dx=dx) - target
    spectral_residual = spectral_divergence_2d(field, dy=dy, dx=dx) - target
    return {
        "fd_abs_mean": float(fd_residual.abs().mean().item()),
        "fd_abs_max": float(fd_residual.abs().max().item()),
        "spectral_abs_mean": float(spectral_residual.abs().mean().item()),
        "spectral_abs_max": float(spectral_residual.abs().max().item()),
    }


def print_metric_summary(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")


# %%
cfg = load_composed_config(CFG.config_path)
cfg.setdefault("wandb_logging", {})
cfg["wandb_logging"]["wandb"] = False

test_loader = build_test_loader(cfg)
meta = test_loader.darcy_meta
height, width = meta["shapelist"]
lower, upper = meta["domain_bounds"]
dy = (upper - lower) / max(height - 1, 1)
dx = (upper - lower) / max(width - 1, 1)

model, args, resolved_nsl_root = create_model(
    cfg,
    nsl_root=CFG.nsl_root,
    device=DEVICE,
    runtime_overrides=darcy_runtime_overrides(meta),
)
model.eval()

print(f"Loaded config: {CFG.config_path}")
print(f"Resolved NSL root: {resolved_nsl_root}")
print(f"Grid shape: {(height, width)}")
print(f"Model backbone: {cfg.get('model', {}).get('backbone')}")
# %%
batch = None
for batch_idx, candidate in enumerate(test_loader):
    if batch_idx == CFG.batch_index:
        batch = candidate
        break
if batch is None:
    raise IndexError(f"Could not find batch_index={CFG.batch_index}")

coords = batch["coords"].to(device=DEVICE, dtype=CFG.dtype)
coeff_encoded = batch["x"].to(device=DEVICE, dtype=CFG.dtype)
target_encoded = batch["y"].to(device=DEVICE, dtype=CFG.dtype)

coords = coords[CFG.sample_index : CFG.sample_index + 1]
coeff_encoded = coeff_encoded[CFG.sample_index : CFG.sample_index + 1]
target_encoded = target_encoded[CFG.sample_index : CFG.sample_index + 1]
coeff_physical = test_loader.x_normalizer.decode(coeff_encoded.float()).to(
    device=DEVICE,
    dtype=CFG.dtype,
)
target_physical = test_loader.y_normalizer.decode(target_encoded.float()).to(
    device=DEVICE,
    dtype=CFG.dtype,
)
coeff_grid = reshape_channels_last_to_grid(coeff_physical, height=height, width=width)
target_grid = reshape_channels_last_to_grid(target_physical, height=height, width=width)

print(f"Using batch {CFG.batch_index}, sample {CFG.sample_index}")
print_scalar_stats("coeff_physical", coeff_physical)
print_scalar_stats("target_physical", target_physical)
plot_scalar_panels(
    "Darcy sample from real dataset",
    [
        (
            "permeability a(x)",
            reshape_channels_last_to_grid(coeff_physical, height=height, width=width),
        ),
        (
            "pressure u(x)",
            reshape_channels_last_to_grid(target_physical, height=height, width=width),
        ),
    ],
)


# %%
with torch.no_grad():
    psi_flat = model(coords.float().to(DEVICE), coeff_encoded.float().to(DEVICE)).to(
        dtype=CFG.dtype
    )

psi = reshape_channels_last_to_grid(psi_flat, height=height, width=width)
print_scalar_stats("psi", psi)
plot_scalar_panels(
    "Model-predicted stream function psi",
    [
        ("psi", psi),
    ],
    center_zero=True,
)

# %%
padding = normalize_padding_2d(CFG.padding)
pad_left, pad_right, pad_top, pad_bottom = padding
padded_height = height + pad_top + pad_bottom
padded_width = width + pad_left + pad_right
interior_margin = int(CFG.interior_margin)

branches: dict[str, dict[str, torch.Tensor]] = {}
for mode in CFG.padding_modes:
    psi_padded_mode = pad_spatial_2d(psi, padding, mode=mode)
    v_corr_padded_mode = stream_velocity_from_psi(psi_padded_mode, dy=dy, dx=dx)
    v_corr_physical_mode = crop_spatial_2d(v_corr_padded_mode, padding)
    v_corr_interior_mode = interior_core_2d(v_corr_physical_mode, margin=interior_margin)

    v_part_padded_mode = build_particular_flux(
        name=CFG.particular_field,
        batch_size=1,
        height=padded_height,
        width=padded_width,
        dy=dy,
        dx=dx,
        force_value=1.0,
        dtype=CFG.dtype,
    )
    v_part_physical_mode = crop_spatial_2d(v_part_padded_mode, padding)
    v_part_interior_mode = interior_core_2d(v_part_physical_mode, margin=interior_margin)
    v_valid_padded_mode = v_part_padded_mode + v_corr_padded_mode
    v_valid_physical_mode = crop_spatial_2d(v_valid_padded_mode, padding)
    v_valid_interior_mode = interior_core_2d(
        v_valid_physical_mode, margin=interior_margin
    )
    u_recovered_mode, w_input_mode, rhs_mode = recover_pressure_from_flux_sine(
        v_valid_physical_mode,
        coeff_grid,
        dy=dy,
        dx=dx,
    )
    grad_u_mode = finite_difference_gradient_2d(u_recovered_mode, dy=dy, dx=dx)
    w_error_mode = vector_error_norm(grad_u_mode, w_input_mode)
    darcy_flux_mode = -coeff_grid * grad_u_mode
    darcy_residual_mode = finite_difference_divergence_2d(
        darcy_flux_mode,
        dy=dy,
        dx=dx,
    ) - 1.0
    pressure_error_mode = (u_recovered_mode - target_grid).abs()
    boundary_error_mode = boundary_abs(u_recovered_mode)

    branches[mode] = {
        "psi_padded": psi_padded_mode,
        "v_corr_padded": v_corr_padded_mode,
        "v_corr_physical": v_corr_physical_mode,
        "v_corr_interior": v_corr_interior_mode,
        "v_part_padded": v_part_padded_mode,
        "v_part_physical": v_part_physical_mode,
        "v_part_interior": v_part_interior_mode,
        "v_valid_padded": v_valid_padded_mode,
        "v_valid_physical": v_valid_physical_mode,
        "v_valid_interior": v_valid_interior_mode,
        "u_recovered": u_recovered_mode,
        "w_input": w_input_mode,
        "rhs": rhs_mode,
        "grad_u": grad_u_mode,
        "w_error": w_error_mode,
        "darcy_residual": darcy_residual_mode,
        "pressure_error": pressure_error_mode,
        "boundary_error": boundary_error_mode,
    }

print(
    f"Comparing padding modes on the same psi. Domains: padded={(padded_height, padded_width)}, "
    f"physical={(height, width)}, interior_margin={interior_margin}, "
    f"interior={(height - 2 * interior_margin, width - 2 * interior_margin)}"
)
for mode in CFG.padding_modes:
    tensors = branches[mode]
    print(f"\n### Padding mode: {mode}")
    print_metric_summary(
        "v_corr divergence residuals",
        {
            "padded/spectral_abs_mean": collect_divergence_metrics(
                tensors["v_corr_padded"], dy=dy, dx=dx, target=0.0
            )["spectral_abs_mean"],
            "padded/spectral_abs_max": collect_divergence_metrics(
                tensors["v_corr_padded"], dy=dy, dx=dx, target=0.0
            )["spectral_abs_max"],
            "physical/fd_abs_mean": collect_divergence_metrics(
                tensors["v_corr_physical"], dy=dy, dx=dx, target=0.0
            )["fd_abs_mean"],
            "physical/fd_abs_max": collect_divergence_metrics(
                tensors["v_corr_physical"], dy=dy, dx=dx, target=0.0
            )["fd_abs_max"],
            "interior/fd_abs_mean": collect_divergence_metrics(
                tensors["v_corr_interior"], dy=dy, dx=dx, target=0.0
            )["fd_abs_mean"],
            "interior/fd_abs_max": collect_divergence_metrics(
                tensors["v_corr_interior"], dy=dy, dx=dx, target=0.0
            )["fd_abs_max"],
        },
    )
    print_metric_summary(
        "v_valid continuity residuals",
        {
            "physical/fd_abs_mean": collect_divergence_metrics(
                tensors["v_valid_physical"], dy=dy, dx=dx, target=1.0
            )["fd_abs_mean"],
            "physical/fd_abs_max": collect_divergence_metrics(
                tensors["v_valid_physical"], dy=dy, dx=dx, target=1.0
            )["fd_abs_max"],
            "interior/fd_abs_mean": collect_divergence_metrics(
                tensors["v_valid_interior"], dy=dy, dx=dx, target=1.0
            )["fd_abs_mean"],
            "interior/fd_abs_max": collect_divergence_metrics(
                tensors["v_valid_interior"], dy=dy, dx=dx, target=1.0
            )["fd_abs_max"],
        },
    )
    print_metric_summary(
        "sine recovery metrics",
        {
            "pressure_abs_mean": float(tensors["pressure_error"].mean().item()),
            "pressure_abs_max": float(tensors["pressure_error"].max().item()),
            "w_error_abs_mean": float(tensors["w_error"].mean().item()),
            "w_error_abs_max": float(tensors["w_error"].max().item()),
            "darcy_res_abs_mean": float(tensors["darcy_residual"].abs().mean().item()),
            "darcy_res_abs_max": float(tensors["darcy_residual"].abs().max().item()),
            "boundary_abs_mean": float(tensors["boundary_error"].mean().item()),
            "boundary_abs_max": float(tensors["boundary_error"].max().item()),
        },
    )


# %%
selected = branches[CFG.padding_mode]
psi_padded = selected["psi_padded"]
v_corr_padded = selected["v_corr_padded"]
v_corr_physical = selected["v_corr_physical"]
v_corr_interior = selected["v_corr_interior"]
v_part_physical = selected["v_part_physical"]
v_part_interior = selected["v_part_interior"]
v_valid_physical = selected["v_valid_physical"]
v_valid_interior = selected["v_valid_interior"]
u_recovered = selected["u_recovered"]
w_input = selected["w_input"]
rhs_recovered = selected["rhs"]
grad_u = selected["grad_u"]
w_error = selected["w_error"]
darcy_residual = selected["darcy_residual"]
pressure_error = selected["pressure_error"]
boundary_error = selected["boundary_error"]

print(f"Detailed diagnostics for selected padding mode '{CFG.padding_mode}'")
print_scalar_stats("psi_padded", psi_padded)
print_divergence_report(
    "v_corr on padded domain",
    v_corr_padded,
    dy=dy,
    dx=dx,
    target=0.0,
)
print_divergence_report(
    "v_corr on physical domain",
    v_corr_physical,
    dy=dy,
    dx=dx,
    target=0.0,
)
print_divergence_report(
    "v_corr on interior core",
    v_corr_interior,
    dy=dy,
    dx=dx,
    target=0.0,
)
plot_scalar_panels(
    "Padded psi and v_corr divergence",
    [
        ("psi_padded", psi_padded),
        ("spectral div(v_corr_padded)", spectral_divergence_2d(v_corr_padded, dy=dy, dx=dx)),
        ("fd div(v_corr_physical)", finite_difference_divergence_2d(v_corr_physical, dy=dy, dx=dx)),
        ("fd div(v_corr_interior)", finite_difference_divergence_2d(v_corr_interior, dy=dy, dx=dx)),
    ],
    center_zero=True,
)
plot_vector_panels("v_corr on physical domain", v_corr_physical)


# %%
print(f"Build v_valid with particular field '{CFG.particular_field}'")
print_divergence_report(
    "v_part on physical domain",
    v_part_physical,
    dy=dy,
    dx=dx,
    target=1.0,
)
print_divergence_report(
    "v_part on interior core",
    v_part_interior,
    dy=dy,
    dx=dx,
    target=1.0,
)
print_divergence_report(
    "v_valid on physical domain",
    v_valid_physical,
    dy=dy,
    dx=dx,
    target=1.0,
)
print_divergence_report(
    "v_valid on interior core",
    v_valid_interior,
    dy=dy,
    dx=dx,
    target=1.0,
)
plot_scalar_panels(
    "Helmholtz valid flux diagnostics",
    [
        ("fd div(v_part physical) - 1", finite_difference_divergence_2d(v_part_physical, dy=dy, dx=dx) - 1.0),
        ("fd div(v_valid physical) - 1", finite_difference_divergence_2d(v_valid_physical, dy=dy, dx=dx) - 1.0),
        ("fd div(v_valid interior) - 1", finite_difference_divergence_2d(v_valid_interior, dy=dy, dx=dx) - 1.0),
        ("spectral div(v_corr_padded)", spectral_divergence_2d(v_corr_padded, dy=dy, dx=dx)),
    ],
    center_zero=True,
)
plot_vector_panels("v_valid on physical domain", v_valid_physical)


# %%
print("Sine recovery from v_valid")
print_scalar_stats("u_recovered", u_recovered)
print_scalar_stats("pressure_error", pressure_error, target=0.0)
print_scalar_stats("w_error", w_error, target=0.0)
print_scalar_stats("darcy_residual", darcy_residual, target=0.0)
print_scalar_stats("boundary_error", boundary_error, target=0.0)
print_scalar_stats(
    "curl(w_input)",
    finite_difference_curl_2d(w_input, dy=dy, dx=dx),
    target=0.0,
)
print_scalar_stats(
    "curl(grad_u)",
    finite_difference_curl_2d(grad_u, dy=dy, dx=dx),
    target=0.0,
)
plot_vector_panels("Input w = -v_valid / a", w_input)
plot_vector_panels("Recovered grad(u)", grad_u)
plot_scalar_panels(
    "Sine pressure recovery diagnostics",
    [
        ("u_recovered", u_recovered),
        ("u_target", target_grid),
        ("pressure_error", pressure_error),
        ("darcy_residual", darcy_residual),
    ],
    center_zero=True,
)
plot_scalar_panels(
    "Gradient recovery diagnostics",
    [
        ("rhs = div(w_input)", rhs_recovered),
        ("|grad(u) - w_input|", w_error),
        ("curl(w_input)", finite_difference_curl_2d(w_input, dy=dy, dx=dx)),
        ("curl(grad_u)", finite_difference_curl_2d(grad_u, dy=dy, dx=dx)),
    ],
    center_zero=True,
)


# %%

print(
    "\nInterpretation guide:\n"
    "1. psi is the scalar latent output from the real backbone on a real Darcy sample.\n"
    "2. v_corr = grad_perp(psi) is built on the padded domain, so spectral_div(v_corr_padded)\n"
    "   is the construction check and should be near zero.\n"
    "3. The physical-domain check is finite_difference_div(v_valid_physical) - 1.\n"
    "4. The interior-core check is finite_difference_div(v_valid_interior) - 1; if this is much\n"
    "   better than the full physical-domain metric, the remaining issue is mostly edge handling.\n"
    "5. w = -v_valid / a is the gradient candidate used by the sine recovery stage.\n"
    "6. Good recovery signals are pressure_error, |grad(u) - w|, darcy_residual, and boundary_error.\n"
    "7. The padding-mode sweep compares reflect / replicate / zeros on the same psi field.\n"
)
