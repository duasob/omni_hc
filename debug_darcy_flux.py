from __future__ import annotations

# %%
from dataclasses import dataclass

import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - local debug convenience
    plt = None

from omni_hc.constraints.spectral import (
    crop_spatial_2d,
    fft_leray_project_2d,
    finite_difference_curl_2d,
    finite_difference_divergence_2d,
    finite_difference_gradient_2d,
    finite_difference_laplacian_2d,
    normalize_padding_2d,
    pad_spatial_2d,
    spectral_divergence_2d,
    spectral_poisson_solve_2d,
    sine_poisson_solve_dirichlet_2d,
)

torch.set_printoptions(precision=4, sci_mode=False)


@dataclass
class DebugConfig:
    height: int = 32
    width: int = 32
    padding: int = 8
    lower: float = 0.0
    upper: float = 1.0
    force_value: float = 1.0
    permeability_value: float = 4.0
    padding_mode: str = "reflect"
    seed: int = 7
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def dy(self) -> float:
        return (self.upper - self.lower) / max(self.height - 1, 1)

    @property
    def dx(self) -> float:
        return (self.upper - self.lower) / max(self.width - 1, 1)


CFG = DebugConfig()
torch.manual_seed(CFG.seed)
DEVICE = torch.device(CFG.device)

print(CFG)
if plt is None:
    print("matplotlib is not available; plots are disabled.")


# %%


def make_physical_mesh(cfg: DebugConfig) -> tuple[torch.Tensor, torch.Tensor]:
    y = torch.linspace(
        cfg.lower,
        cfg.upper,
        cfg.height,
        device=DEVICE,
        dtype=cfg.dtype,
    )
    x = torch.linspace(
        cfg.lower,
        cfg.upper,
        cfg.width,
        device=DEVICE,
        dtype=cfg.dtype,
    )
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return yy, xx


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


def print_vector_stats(name: str, field: torch.Tensor) -> None:
    norm = field.square().sum(dim=1, keepdim=True).sqrt()
    print_scalar_stats(f"{name}/norm", norm)


def print_divergence_report(
    name: str,
    field: torch.Tensor,
    *,
    dy: float,
    dx: float,
    target: float | None = None,
) -> None:
    print(f"\n=== {name} ===")
    spectral_div = spectral_divergence_2d(field, dy=dy, dx=dx)
    fd_div = finite_difference_divergence_2d(field, dy=dy, dx=dx)
    print_scalar_stats("spectral_div", spectral_div, target=target)
    print_scalar_stats("finite_difference_div", fd_div, target=target)


def print_curl_report(name: str, field: torch.Tensor, *, dy: float, dx: float) -> None:
    print(f"\n=== {name} ===")
    curl = finite_difference_curl_2d(field, dy=dy, dx=dx)
    print_scalar_stats("finite_difference_curl", curl, target=0.0)


def make_constant_permeability(cfg: DebugConfig) -> torch.Tensor:
    return torch.full(
        (1, 1, cfg.height, cfg.width),
        cfg.permeability_value,
        device=DEVICE,
        dtype=cfg.dtype,
    )


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


def make_periodic_pressure(cfg: DebugConfig) -> torch.Tensor:
    yy, xx = make_physical_mesh(cfg)
    u = torch.sin(2.0 * torch.pi * xx) * torch.sin(2.0 * torch.pi * yy)
    return u.view(1, 1, cfg.height, cfg.width)


def make_dirichlet_pressure(cfg: DebugConfig) -> torch.Tensor:
    yy, xx = make_physical_mesh(cfg)
    u = torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
    return u.view(1, 1, cfg.height, cfg.width)


def recover_pressure_from_flux(
    flux: torch.Tensor,
    permeability: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    gradient = -flux / permeability
    rhs = spectral_divergence_2d(gradient, dy=dy, dx=dx)
    pressure = spectral_poisson_solve_2d(rhs, dy=dy, dx=dx)
    return pressure, gradient


def recover_pressure_from_gradient_sine(
    gradient: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rhs = finite_difference_divergence_2d(gradient, dy=dy, dx=dx)
    pressure = sine_poisson_solve_dirichlet_2d(rhs, dy=dy, dx=dx)
    return pressure, rhs


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
    cmap: str = "viridis",
    center_zero: bool = False,
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
            value.numpy(), cmap=cmap, vmin=vmin, vmax=vmax, origin="lower"
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
        center_zero=False,
    )


PADDING = normalize_padding_2d(CFG.padding)
print("Normalized padding:", PADDING)


# %%
# Step 0: establish the baseline for a periodic-compatible problem.
u_periodic = make_periodic_pressure(CFG)
w_periodic = finite_difference_gradient_2d(u_periodic, dy=CFG.dy, dx=CFG.dx)
a_periodic = make_constant_permeability(CFG)
v_periodic = -a_periodic * w_periodic

print("Periodic baseline: u -> grad(u) -> flux")
print_vector_stats("v_periodic", v_periodic)
print_divergence_report(
    "Periodic baseline flux divergence",
    v_periodic,
    dy=CFG.dy,
    dx=CFG.dx,
    target=None,
)
u_recovered_periodic, w_recovered_periodic = recover_pressure_from_flux(
    v_periodic,
    a_periodic,
    dy=CFG.dy,
    dx=CFG.dx,
)
print_scalar_stats(
    "periodic pressure reconstruction error",
    u_recovered_periodic - u_periodic,
    target=0.0,
)
print_curl_report(
    "periodic recovered gradient curl", w_recovered_periodic, dy=CFG.dy, dx=CFG.dx
)
plot_scalar_panels(
    "Periodic baseline pressure recovery",
    [
        ("u_periodic", u_periodic),
        ("u_recovered", u_recovered_periodic),
        ("reconstruction_error", u_recovered_periodic - u_periodic),
    ],
    center_zero=True,
)
plot_scalar_panels(
    "Periodic baseline divergence and curl",
    [
        ("div(v_periodic)", spectral_divergence_2d(v_periodic, dy=CFG.dy, dx=CFG.dx)),
        (
            "curl(w_recovered)",
            finite_difference_curl_2d(w_recovered_periodic, dy=CFG.dy, dx=CFG.dx),
        ),
    ],
    center_zero=True,
)


# %%

# Step 0b: establish the baseline for a zero-Dirichlet pressure using the sine solve.
u_dirichlet = make_dirichlet_pressure(CFG)
w_dirichlet = finite_difference_gradient_2d(u_dirichlet, dy=CFG.dy, dx=CFG.dx)
a_dirichlet = make_constant_permeability(CFG)
v_dirichlet = -a_dirichlet * w_dirichlet
u_recovered_sine, rhs_dirichlet = recover_pressure_from_gradient_sine(
    w_dirichlet,
    dy=CFG.dy,
    dx=CFG.dx,
)
laplace_dirichlet = finite_difference_laplacian_2d(
    u_dirichlet,
    dy=CFG.dy,
    dx=CFG.dx,
)

print("Dirichlet baseline: u -> grad(u) -> sine Poisson solve")
print_vector_stats("v_dirichlet", v_dirichlet)
print_scalar_stats(
    "dirichlet pressure reconstruction error",
    u_recovered_sine - u_dirichlet,
    target=0.0,
)
print_scalar_stats(
    "dirichlet rhs consistency",
    rhs_dirichlet - laplace_dirichlet,
    target=0.0,
)
print_curl_report(
    "dirichlet recovered gradient curl",
    finite_difference_gradient_2d(u_recovered_sine, dy=CFG.dy, dx=CFG.dx),
    dy=CFG.dy,
    dx=CFG.dx,
)
plot_scalar_panels(
    "Dirichlet baseline pressure recovery",
    [
        ("u_dirichlet", u_dirichlet),
        ("u_recovered_sine", u_recovered_sine),
        ("reconstruction_error", u_recovered_sine - u_dirichlet),
    ],
    center_zero=True,
)
plot_scalar_panels(
    "Dirichlet baseline rhs",
    [
        ("div(grad(u))", rhs_dirichlet),
        ("laplacian(u)", laplace_dirichlet),
        ("rhs mismatch", rhs_dirichlet - laplace_dirichlet),
    ],
    center_zero=True,
)


# %%

# Step 1: inspect candidate particular fields directly.
physical_height = CFG.height
physical_width = CFG.width
pad_left, pad_right, pad_top, pad_bottom = PADDING
padded_height = physical_height + pad_top + pad_bottom
padded_width = physical_width + pad_left + pad_right

v_part_y = particular_y_only(
    batch_size=1,
    height=padded_height,
    width=padded_width,
    dy=CFG.dy,
    dx=CFG.dx,
    force_value=CFG.force_value,
    dtype=CFG.dtype,
)
v_part_xy = particular_xy_affine(
    batch_size=1,
    height=padded_height,
    width=padded_width,
    dy=CFG.dy,
    dx=CFG.dx,
    force_value=CFG.force_value,
    dtype=CFG.dtype,
)

print("Particular field sanity checks on the padded domain")
print_divergence_report(
    "Current particular field: v=(0, y)",
    v_part_y,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
print_divergence_report(
    "Alternative affine field: v=(x/2, y/2)",
    v_part_xy,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
plot_vector_panels("Current particular field v=(0, y)", v_part_y)
plot_scalar_panels(
    "Current particular field divergence",
    [
        (
            "spectral div - 1",
            spectral_divergence_2d(v_part_y, dy=CFG.dy, dx=CFG.dx) - CFG.force_value,
        ),
        (
            "fd div - 1",
            finite_difference_divergence_2d(v_part_y, dy=CFG.dy, dx=CFG.dx)
            - CFG.force_value,
        ),
    ],
    center_zero=True,
)
plot_vector_panels("Alternative affine field v=(x/2, y/2)", v_part_xy)
plot_scalar_panels(
    "Alternative affine field divergence",
    [
        (
            "spectral div - 1",
            spectral_divergence_2d(v_part_xy, dy=CFG.dy, dx=CFG.dx) - CFG.force_value,
        ),
        (
            "fd div - 1",
            finite_difference_divergence_2d(v_part_xy, dy=CFG.dy, dx=CFG.dx)
            - CFG.force_value,
        ),
    ],
    center_zero=True,
)


# %%
# Step 2: verify the Leray projection itself on a random correction.
random_flux_physical = torch.randn(
    1,
    2,
    CFG.height,
    CFG.width,
    device=DEVICE,
    dtype=CFG.dtype,
)
random_flux_padded = pad_spatial_2d(
    random_flux_physical,
    PADDING,
    mode=CFG.padding_mode,
)
projected_correction = fft_leray_project_2d(
    random_flux_padded - v_part_y,
    dy=CFG.dy,
    dx=CFG.dx,
)
constrained_flux_padded = v_part_y + projected_correction
constrained_flux_physical = crop_spatial_2d(constrained_flux_padded, PADDING)

print("Random prediction + projection")
print_divergence_report(
    "Projected correction on padded domain",
    projected_correction,
    dy=CFG.dy,
    dx=CFG.dx,
    target=0.0,
)
print_divergence_report(
    "Constrained flux on padded domain",
    constrained_flux_padded,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
print_divergence_report(
    "Constrained flux on cropped physical domain",
    constrained_flux_physical,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
plot_scalar_panels(
    "Projection diagnostics",
    [
        (
            "proj corr div (padded)",
            spectral_divergence_2d(projected_correction, dy=CFG.dy, dx=CFG.dx),
        ),
        (
            "constrained div - 1 (padded)",
            spectral_divergence_2d(constrained_flux_padded, dy=CFG.dy, dx=CFG.dx)
            - CFG.force_value,
        ),
        (
            "constrained div - 1 (physical, fd)",
            finite_difference_divergence_2d(
                constrained_flux_physical, dy=CFG.dy, dx=CFG.dx
            )
            - CFG.force_value,
        ),
    ],
    center_zero=True,
)
plot_vector_panels("Constrained flux on physical domain", constrained_flux_physical)


# %%

# Step 3: recover w = grad(u) from the constrained flux and inspect curl.
permeability_physical = make_constant_permeability(CFG)
permeability_padded = pad_spatial_2d(
    permeability_physical,
    PADDING,
    mode=CFG.padding_mode,
)
w_padded = -constrained_flux_padded / permeability_padded
w_physical = crop_spatial_2d(w_padded, PADDING)

print("Flux -> gradient recovery")
print_vector_stats("w_physical", w_physical)
print_curl_report(
    "Recovered w on the physical domain", w_physical, dy=CFG.dy, dx=CFG.dx
)
plot_vector_panels("Recovered gradient field w on physical domain", w_physical)
plot_scalar_panels(
    "Recovered gradient curl",
    [
        (
            "curl(w_physical)",
            finite_difference_curl_2d(w_physical, dy=CFG.dy, dx=CFG.dx),
        ),
    ],
    center_zero=True,
)


# %%

# Step 4: solve Poisson for u and check whether the reconstructed pressure is self-consistent.
laplace_u_padded = spectral_divergence_2d(w_padded, dy=CFG.dy, dx=CFG.dx)
u_padded = spectral_poisson_solve_2d(laplace_u_padded, dy=CFG.dy, dx=CFG.dx)
u_physical = crop_spatial_2d(u_padded, PADDING)
grad_u_physical = finite_difference_gradient_2d(u_physical, dy=CFG.dy, dx=CFG.dx)
darcy_flux_physical = -permeability_physical * grad_u_physical
darcy_residual = (
    finite_difference_divergence_2d(
        darcy_flux_physical,
        dy=CFG.dy,
        dx=CFG.dx,
    )
    - CFG.force_value
)
u_physical_sine, rhs_physical_sine = recover_pressure_from_gradient_sine(
    w_physical,
    dy=CFG.dy,
    dx=CFG.dx,
)
grad_u_physical_sine = finite_difference_gradient_2d(
    u_physical_sine,
    dy=CFG.dy,
    dx=CFG.dx,
)
darcy_flux_physical_sine = -permeability_physical * grad_u_physical_sine
darcy_residual_sine = (
    finite_difference_divergence_2d(
        darcy_flux_physical_sine,
        dy=CFG.dy,
        dx=CFG.dx,
    )
    - CFG.force_value
)

print("Poisson recovery")
print_scalar_stats("u_physical", u_physical)
print_scalar_stats("Darcy residual from recovered pressure", darcy_residual, target=0.0)
print_curl_report("curl(grad(u_recovered))", grad_u_physical, dy=CFG.dy, dx=CFG.dx)
print_scalar_stats("u_physical_sine", u_physical_sine)
print_scalar_stats(
    "Darcy residual from sine recovered pressure",
    darcy_residual_sine,
    target=0.0,
)
print_curl_report(
    "curl(grad(u_recovered_sine))",
    grad_u_physical_sine,
    dy=CFG.dy,
    dx=CFG.dx,
)
plot_scalar_panels(
    "FFT Poisson recovery outputs",
    [
        ("u_physical", u_physical),
        ("Darcy residual", darcy_residual),
        (
            "curl(grad(u))",
            finite_difference_curl_2d(grad_u_physical, dy=CFG.dy, dx=CFG.dx),
        ),
    ],
    center_zero=True,
)
plot_scalar_panels(
    "Sine Poisson recovery outputs",
    [
        ("u_physical_sine", u_physical_sine),
        ("Darcy residual (sine)", darcy_residual_sine),
        (
            "curl(grad(u_sine))",
            finite_difference_curl_2d(grad_u_physical_sine, dy=CFG.dy, dx=CFG.dx),
        ),
    ],
    center_zero=True,
)
plot_scalar_panels(
    "FFT vs sine pressure comparison",
    [
        ("u_fft - u_sine", u_physical - u_physical_sine),
        ("rhs from w_physical", rhs_physical_sine),
        ("darcy residual diff", darcy_residual - darcy_residual_sine),
    ],
    center_zero=True,
)


# %%

# Step 5: run the same pipeline with the affine v=(x/2, y/2) field so the comparison is explicit.
projected_correction_xy = fft_leray_project_2d(
    random_flux_padded - v_part_xy,
    dy=CFG.dy,
    dx=CFG.dx,
)
constrained_flux_padded_xy = v_part_xy + projected_correction_xy
constrained_flux_physical_xy = crop_spatial_2d(constrained_flux_padded_xy, PADDING)
w_padded_xy = -constrained_flux_padded_xy / permeability_padded
w_physical_xy = crop_spatial_2d(w_padded_xy, PADDING)

print("Alternative affine field comparison")
print_divergence_report(
    "Constrained flux on padded domain with v=(x/2, y/2)",
    constrained_flux_padded_xy,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
print_divergence_report(
    "Constrained flux on physical domain with v=(x/2, y/2)",
    constrained_flux_physical_xy,
    dy=CFG.dy,
    dx=CFG.dx,
    target=CFG.force_value,
)
print_curl_report(
    "Recovered w on physical domain with v=(x/2, y/2)",
    w_physical_xy,
    dy=CFG.dy,
    dx=CFG.dx,
)
plot_scalar_panels(
    "Alternative affine field comparison",
    [
        (
            "div - 1 (padded)",
            spectral_divergence_2d(constrained_flux_padded_xy, dy=CFG.dy, dx=CFG.dx)
            - CFG.force_value,
        ),
        (
            "div - 1 (physical, fd)",
            finite_difference_divergence_2d(
                constrained_flux_physical_xy, dy=CFG.dy, dx=CFG.dx
            )
            - CFG.force_value,
        ),
        (
            "curl(w_physical)",
            finite_difference_curl_2d(w_physical_xy, dy=CFG.dy, dx=CFG.dx),
        ),
    ],
    center_zero=True,
)


# %%

print(
    "\nInterpretation guide:\n"
    "1. If the padded-domain spectral divergence of the particular field is already bad,\n"
    "   the affine construction is incompatible with the FFT backend.\n"
    "2. If the projected correction has near-zero spectral divergence but the final field does not,\n"
    "   the particular field is the source of the failure.\n"
    "3. If divergence looks acceptable but curl(w) grows, the flux-to-gradient step is not\n"
    "   producing an integrable gradient field.\n"
    "4. If the sine Dirichlet baseline reconstructs well, the scalar sine backend is numerically sound.\n"
    "5. If FFT and sine pressure recovery disagree strongly on the same w field, the pressure solve\n"
    "   boundary handling is still a major source of error.\n"
)
