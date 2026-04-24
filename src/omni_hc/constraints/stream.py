from __future__ import annotations

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .spectral import reshape_channels_last_to_grid, reshape_grid_to_channels_last
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


class PipeStreamFunctionUxConstraint(ConstraintModule):
    """
    Interprets the scalar backbone output as a stream function psi on the
    curvilinear pipe mesh and returns ux = dpsi/dy in physical coordinates.
    """

    name = "pipe_stream_function_ux"

    def __init__(
        self,
        *,
        shapelist=None,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.shapelist = None if shapelist is None else tuple(int(v) for v in shapelist)
        self.eps = float(eps)
        self.input_normalizer = None
        self.target_normalizer = None

    def set_grid_shape(self, shapelist) -> None:
        self.shapelist = tuple(int(v) for v in shapelist)

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _grid_shape(self) -> tuple[int, int]:
        if self.shapelist is None:
            raise ValueError("PipeStreamFunctionUxConstraint requires a 2D grid shape")
        if len(self.shapelist) != 2:
            raise ValueError(
                f"PipeStreamFunctionUxConstraint expects a 2D grid shape, got {self.shapelist!r}"
            )
        return int(self.shapelist[0]), int(self.shapelist[1])

    def _decode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return coords
        return self.input_normalizer.decode(coords)

    def _encode_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_normalizer is None:
            return target
        return self.target_normalizer.encode(target)

    def forward(self, *, pred, coords=None, return_aux=False, **_unused):
        if coords is None:
            raise ValueError("coords are required for PipeStreamFunctionUxConstraint")
        if pred.ndim != 3 or pred.shape[-1] != 1:
            raise ValueError(
                "PipeStreamFunctionUxConstraint expects a scalar backbone output "
                f"with shape (batch, n_points, 1), got {tuple(pred.shape)!r}"
            )
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(
                "PipeStreamFunctionUxConstraint expects coords with shape "
                f"(batch, n_points, 2), got {tuple(coords.shape)!r}"
            )

        height, width = self._grid_shape()
        psi = reshape_channels_last_to_grid(pred, shapelist=(height, width))
        coords_grid = reshape_channels_last_to_grid(
            self._decode_coords(coords),
            shapelist=(height, width),
        )
        velocity, jac = stream_velocity_from_psi_curvilinear(
            psi,
            coords_grid,
            eps=self.eps,
        )
        ux = velocity[:, 0:1]
        uy = velocity[:, 1:2]
        ux_flat = reshape_grid_to_channels_last(ux)
        ux_encoded = self._encode_target(ux_flat)

        if return_aux:
            div = finite_volume_divergence_curvilinear(
                velocity,
                coords_grid,
                eps=self.eps,
            )
            diagnostics = {
                "constraint/stream_div_abs_mean": ConstraintDiagnostic(
                    value=div.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_div_abs_max": ConstraintDiagnostic(
                    value=div.abs().max(),
                    reduce="max",
                ),
                "constraint/stream_uy_abs_mean": ConstraintDiagnostic(
                    value=uy.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_uy_abs_max": ConstraintDiagnostic(
                    value=uy.abs().max(),
                    reduce="max",
                ),
                "constraint/stream_jac_min": ConstraintDiagnostic(
                    value=jac.min(),
                    reduce="min",
                ),
                "constraint/stream_jac_max": ConstraintDiagnostic(
                    value=jac.max(),
                    reduce="max",
                ),
                "constraint/stream_psi_mean": ConstraintDiagnostic(
                    value=psi.mean(),
                    reduce="mean",
                ),
                "constraint/stream_psi_std": ConstraintDiagnostic(
                    value=psi.std(unbiased=False),
                    reduce="mean",
                ),
            }
            return self.as_output(
                ux_encoded,
                aux={
                    "pred_base": pred,
                    "stream_psi": reshape_grid_to_channels_last(psi),
                    "stream_uy": reshape_grid_to_channels_last(uy),
                    "stream_div": div.reshape(div.shape[0], 1, -1).transpose(1, 2),
                },
                diagnostics=diagnostics,
            )
        return ux_encoded
