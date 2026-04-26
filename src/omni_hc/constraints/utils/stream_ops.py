from __future__ import annotations

import torch

from .spectral import finite_difference_derivative_2d, spectral_gradient_2d


def stream_velocity_from_psi_cartesian_spectral(
    psi: torch.Tensor,
    *,
    dy: float,
    dx: float,
) -> torch.Tensor:
    if psi.ndim != 4 or psi.shape[1] != 1:
        raise ValueError(
            "Expected a scalar field with shape (batch, 1, height, width), "
            f"got {tuple(psi.shape)!r}"
        )
    gradient = spectral_gradient_2d(psi, dy=dy, dx=dx)
    dpsi_dx = gradient[:, 0:1]
    dpsi_dy = gradient[:, 1:2]
    return torch.cat([dpsi_dy, -dpsi_dx], dim=1)


def stream_velocity_from_psi_curvilinear(
    psi: torch.Tensor,
    coords_grid: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    if psi.ndim != 4 or psi.shape[1] != 1:
        raise ValueError(
            "Expected a scalar field with shape (batch, 1, height, width), "
            f"got {tuple(psi.shape)!r}"
        )
    if coords_grid.ndim != 4 or coords_grid.shape[1] != 2:
        raise ValueError(
            "Expected coordinates with shape (batch, 2, height, width), "
            f"got {tuple(coords_grid.shape)!r}"
        )
    if psi.shape[0] != coords_grid.shape[0] or psi.shape[-2:] != coords_grid.shape[-2:]:
        raise ValueError(
            f"psi shape {tuple(psi.shape)!r} and coords shape {tuple(coords_grid.shape)!r} "
            "must share batch/height/width"
        )

    x = coords_grid[:, 0:1]
    y = coords_grid[:, 1:2]
    x_s = finite_difference_derivative_2d(x, spacing=1.0, axis=-2)
    x_t = finite_difference_derivative_2d(x, spacing=1.0, axis=-1)
    y_s = finite_difference_derivative_2d(y, spacing=1.0, axis=-2)
    y_t = finite_difference_derivative_2d(y, spacing=1.0, axis=-1)
    psi_s = finite_difference_derivative_2d(psi, spacing=1.0, axis=-2)
    psi_t = finite_difference_derivative_2d(psi, spacing=1.0, axis=-1)

    jac = x_s * y_t - x_t * y_s
    safe_jac = torch.where(
        jac.abs() < eps,
        torch.full_like(jac, eps) * torch.where(jac >= 0, 1.0, -1.0),
        jac,
    )
    ux = (-x_t * psi_s + x_s * psi_t) / safe_jac
    uy = (y_s * psi_t - y_t * psi_s) / safe_jac
    return torch.cat([ux, uy], dim=1), jac


def finite_volume_divergence_curvilinear(
    field: torch.Tensor,
    coords_grid: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    if field.ndim != 4 or field.shape[1] != 2:
        raise ValueError(
            "Expected a vector field with shape (batch, 2, height, width), "
            f"got {tuple(field.shape)!r}"
        )
    if coords_grid.ndim != 4 or coords_grid.shape[1] != 2:
        raise ValueError(
            "Expected coordinates with shape (batch, 2, height, width), "
            f"got {tuple(coords_grid.shape)!r}"
        )
    if field.shape[0] != coords_grid.shape[0] or field.shape[-2:] != coords_grid.shape[-2:]:
        raise ValueError(
            f"field shape {tuple(field.shape)!r} and coords shape {tuple(coords_grid.shape)!r} "
            "must share batch/height/width"
        )

    x = coords_grid[:, 0]
    y = coords_grid[:, 1]
    ux = field[:, 0]
    uy = field[:, 1]

    x00 = x[:, :-1, :-1]
    x10 = x[:, 1:, :-1]
    x11 = x[:, 1:, 1:]
    x01 = x[:, :-1, 1:]
    y00 = y[:, :-1, :-1]
    y10 = y[:, 1:, :-1]
    y11 = y[:, 1:, 1:]
    y01 = y[:, :-1, 1:]

    ux00 = ux[:, :-1, :-1]
    ux10 = ux[:, 1:, :-1]
    ux11 = ux[:, 1:, 1:]
    ux01 = ux[:, :-1, 1:]
    uy00 = uy[:, :-1, :-1]
    uy10 = uy[:, 1:, :-1]
    uy11 = uy[:, 1:, 1:]
    uy01 = uy[:, :-1, 1:]

    dx1 = x10 - x00
    dy1 = y10 - y00
    dx2 = x11 - x10
    dy2 = y11 - y10
    dx3 = x01 - x11
    dy3 = y01 - y11
    dx4 = x00 - x01
    dy4 = y00 - y01

    ux1 = 0.5 * (ux00 + ux10)
    ux2 = 0.5 * (ux10 + ux11)
    ux3 = 0.5 * (ux11 + ux01)
    ux4 = 0.5 * (ux01 + ux00)
    uy1 = 0.5 * (uy00 + uy10)
    uy2 = 0.5 * (uy10 + uy11)
    uy3 = 0.5 * (uy11 + uy01)
    uy4 = 0.5 * (uy01 + uy00)

    flux = (
        ux1 * dy1
        - uy1 * dx1
        + ux2 * dy2
        - uy2 * dx2
        + ux3 * dy3
        - uy3 * dx3
        + ux4 * dy4
        - uy4 * dx4
    )

    area = 0.5 * (
        x00 * y10
        + x10 * y11
        + x11 * y01
        + x01 * y00
        - y00 * x10
        - y10 * x11
        - y11 * x01
        - y01 * x00
    ).abs()
    return flux / area.clamp_min(eps)
