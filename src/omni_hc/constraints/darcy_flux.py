from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .spectral import (
    crop_spatial_2d,
    finite_difference_curl_2d,
    finite_difference_divergence_2d,
    finite_difference_gradient_2d,
    normalize_padding_2d,
    pad_spatial_2d,
    reshape_channels_last_to_grid,
    reshape_grid_to_channels_last,
    sine_poisson_solve_dirichlet_2d,
    spectral_divergence_2d,
    spectral_gradient_2d,
)


class DarcyFluxConstraint(ConstraintModule):
    """
    Darcy-specific pressure recovery from a scalar stream function.

    The backbone predicts a scalar stream function psi on the physical Darcy
    grid. The constraint then:
    1. pads psi on a larger computational box,
    2. builds a divergence-free correction v_corr = grad_perp(psi) spectrally,
    3. adds a fixed particular field v_part with div(v_part) = 1,
    4. computes w = -v_valid / a on the physical domain,
    5. recovers pressure with a Dirichlet-aware sine Poisson solve.

    This gives a hard continuity construction for the padded stream correction,
    while keeping the final output as a scalar pressure field.
    """

    name = "darcy_flux_projection"

    _SUPPORTED_BACKENDS = {"helmholtz", "helmholtz_sine", "sine"}

    def __init__(
        self,
        *,
        spectral_backend: str = "helmholtz_sine",
        force_value: float = 1.0,
        permeability_eps: float = 1e-6,
        padding: int | Sequence[int] = 8,
        padding_mode: str = "reflect",
        pressure_out_dim: int = 1,
        enforce_boundary: bool = True,
        boundary_value: float = 0.0,
        particular_field: str = "y_only",
        shapelist: Sequence[int] | None = None,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        super().__init__()
        backend = str(spectral_backend).lower()
        if backend not in self._SUPPORTED_BACKENDS:
            raise NotImplementedError(
                "DarcyFluxConstraint only supports the Helmholtz+sine path right now. "
                f"Received spectral_backend={spectral_backend!r}."
            )
        self.spectral_backend = backend
        self.force_value = float(force_value)
        self.permeability_eps = float(permeability_eps)
        self.padding = normalize_padding_2d(padding)
        self.padding_mode = str(padding_mode)
        self.pressure_out_dim = int(pressure_out_dim)
        if self.pressure_out_dim != 1:
            raise ValueError(
                f"DarcyFluxConstraint currently recovers a scalar pressure field, got {pressure_out_dim}"
            )
        self.enforce_boundary = bool(enforce_boundary)
        self.boundary_value = float(boundary_value)
        self.particular_field = str(particular_field).lower()
        self.shapelist = None if shapelist is None else tuple(int(v) for v in shapelist)
        self.lower = float(lower)
        self.upper = float(upper)
        self.input_normalizer = None
        self.target_normalizer = None

    def set_grid_shape(self, shapelist: Sequence[int]) -> None:
        self.shapelist = tuple(int(v) for v in shapelist)

    def set_domain_bounds(self, *, lower: float, upper: float) -> None:
        self.lower = float(lower)
        self.upper = float(upper)

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def _grid_shape(self) -> tuple[int, int]:
        if self.shapelist is None:
            raise ValueError("DarcyFluxConstraint requires a 2D grid shape")
        if len(self.shapelist) != 2:
            raise ValueError(
                f"DarcyFluxConstraint expects a 2D grid shape, got {self.shapelist!r}"
            )
        return int(self.shapelist[0]), int(self.shapelist[1])

    def _decode_permeability(self, fx: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return fx
        return self.input_normalizer.decode(fx)

    def _encode_pressure(self, pressure: torch.Tensor) -> torch.Tensor:
        if self.target_normalizer is None:
            return pressure
        return self.target_normalizer.encode(pressure)

    def _spacing(self, height: int, width: int) -> tuple[float, float]:
        extent = self.upper - self.lower
        dy = extent / max(height - 1, 1)
        dx = extent / max(width - 1, 1)
        return float(dy), float(dx)

    def _particular_flux(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        dy: float,
        dx: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.arange(height, device=device, dtype=dtype) * dy
        x = torch.arange(width, device=device, dtype=dtype) * dx
        yy, xx = torch.meshgrid(y, x, indexing="ij")

        if self.particular_field == "y_only":
            vy = self.force_value * (yy - yy.mean())
            vx = torch.zeros_like(vy)
        elif self.particular_field in {"xy_affine", "x_y_affine"}:
            vx = 0.5 * self.force_value * xx
            vy = 0.5 * self.force_value * yy
        else:
            raise ValueError(
                "Unsupported Darcy particular field "
                f"{self.particular_field!r}. Expected 'y_only' or 'xy_affine'."
            )

        vx = vx.view(1, 1, height, width).expand(batch_size, 1, height, width)
        vy = vy.view(1, 1, height, width).expand(batch_size, 1, height, width)
        return torch.cat([vx, vy], dim=1)

    def _stream_velocity_from_psi(
        self,
        psi: torch.Tensor,
        *,
        dy: float,
        dx: float,
    ) -> torch.Tensor:
        gradient = spectral_gradient_2d(psi, dy=dy, dx=dx)
        dpsi_dx = gradient[:, 0:1]
        dpsi_dy = gradient[:, 1:2]
        return torch.cat([dpsi_dy, -dpsi_dx], dim=1)

    def _recover_pressure_from_gradient_sine(
        self,
        gradient: torch.Tensor,
        *,
        dy: float,
        dx: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rhs = finite_difference_divergence_2d(gradient, dy=dy, dx=dx)
        pressure = sine_poisson_solve_dirichlet_2d(rhs, dy=dy, dx=dx)
        if self.enforce_boundary and self.boundary_value != 0.0:
            pressure = pressure + self.boundary_value
        return pressure, rhs

    def _recover_pressure_from_flux_sine(
        self,
        flux: torch.Tensor,
        permeability: torch.Tensor,
        *,
        dy: float,
        dx: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gradient = -flux / permeability.clamp_min(self.permeability_eps)
        pressure, rhs = self._recover_pressure_from_gradient_sine(
            gradient,
            dy=dy,
            dx=dx,
        )
        return pressure, gradient, rhs

    def _boundary_residual(self, pressure: torch.Tensor) -> torch.Tensor:
        residual = pressure - self.boundary_value
        top = residual[..., 0, :].abs()
        bottom = residual[..., -1, :].abs()
        left = residual[..., :, 0].abs()
        right = residual[..., :, -1].abs()
        return torch.cat(
            [
                top.reshape(pressure.shape[0], -1),
                bottom.reshape(pressure.shape[0], -1),
                left.reshape(pressure.shape[0], -1),
                right.reshape(pressure.shape[0], -1),
            ],
            dim=1,
        )

    def _vector_error_norm(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return (pred - target).square().sum(dim=1, keepdim=True).sqrt()

    def forward(self, *, pred, fx=None, return_aux=False, **_unused):
        if fx is None:
            raise ValueError("fx is required for DarcyFluxConstraint")
        if pred.ndim != 3:
            raise ValueError(
                "Expected a flattened stream prediction with shape (batch, n_points, 1), "
                f"got {tuple(pred.shape)!r}"
            )
        if pred.shape[-1] != 1:
            raise ValueError(
                "DarcyFluxConstraint expects a scalar backbone output "
                f"(predicted stream function), got last dimension {pred.shape[-1]}"
            )
        if fx.ndim != 3 or fx.shape[-1] != 1:
            raise ValueError(
                "DarcyFluxConstraint expects a single-channel permeability input, "
                f"got {tuple(fx.shape)!r}"
            )

        height, width = self._grid_shape()
        dy, dx = self._spacing(height, width)

        psi = reshape_channels_last_to_grid(pred, shapelist=(height, width))
        permeability = reshape_channels_last_to_grid(
            self._decode_permeability(fx),
            shapelist=(height, width),
        )

        psi_padded = pad_spatial_2d(psi, self.padding, mode=self.padding_mode)
        padded_height, padded_width = psi_padded.shape[-2], psi_padded.shape[-1]

        stream_correction_padded = self._stream_velocity_from_psi(
            psi_padded,
            dy=dy,
            dx=dx,
        )
        particular_flux_padded = self._particular_flux(
            batch_size=psi_padded.shape[0],
            height=padded_height,
            width=padded_width,
            dy=dy,
            dx=dx,
            device=psi_padded.device,
            dtype=psi_padded.dtype,
        )
        constrained_flux_padded = particular_flux_padded + stream_correction_padded
        constrained_flux = crop_spatial_2d(constrained_flux_padded, self.padding)

        pressure, gradient_input, _rhs = self._recover_pressure_from_flux_sine(
            constrained_flux,
            permeability,
            dy=dy,
            dx=dx,
        )
        pressure_flat = reshape_grid_to_channels_last(pressure)
        pressure_encoded = self._encode_pressure(pressure_flat)

        if return_aux:
            stream_correction = crop_spatial_2d(stream_correction_padded, self.padding)
            flux_divergence = finite_difference_divergence_2d(
                constrained_flux,
                dy=dy,
                dx=dx,
            )
            flux_divergence_residual = flux_divergence - self.force_value
            stream_divergence_padded = spectral_divergence_2d(
                stream_correction_padded,
                dy=dy,
                dx=dx,
            )
            pressure_gradient = finite_difference_gradient_2d(
                pressure,
                dy=dy,
                dx=dx,
            )
            w_error = self._vector_error_norm(pressure_gradient, gradient_input)
            w_curl = finite_difference_curl_2d(
                gradient_input,
                dy=dy,
                dx=dx,
            )
            darcy_flux_from_pressure = -permeability * pressure_gradient
            darcy_residual = finite_difference_divergence_2d(
                darcy_flux_from_pressure,
                dy=dy,
                dx=dx,
            ) - self.force_value
            boundary_residual = self._boundary_residual(pressure)

            diagnostics = {
                "constraint/stream_div_abs_mean": ConstraintDiagnostic(
                    value=stream_divergence_padded.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_div_abs_max": ConstraintDiagnostic(
                    value=stream_divergence_padded.abs().max(),
                    reduce="max",
                ),
                "constraint/flux_div_abs_mean": ConstraintDiagnostic(
                    value=flux_divergence_residual.abs().mean(),
                    reduce="mean",
                ),
                "constraint/flux_div_abs_max": ConstraintDiagnostic(
                    value=flux_divergence_residual.abs().max(),
                    reduce="max",
                ),
                "constraint/w_error_abs_mean": ConstraintDiagnostic(
                    value=w_error.mean(),
                    reduce="mean",
                ),
                "constraint/w_error_abs_max": ConstraintDiagnostic(
                    value=w_error.max(),
                    reduce="max",
                ),
                "constraint/w_curl_abs_mean": ConstraintDiagnostic(
                    value=w_curl.abs().mean(),
                    reduce="mean",
                ),
                "constraint/w_curl_abs_max": ConstraintDiagnostic(
                    value=w_curl.abs().max(),
                    reduce="max",
                ),
                "constraint/darcy_res_abs_mean": ConstraintDiagnostic(
                    value=darcy_residual.abs().mean(),
                    reduce="mean",
                ),
                "constraint/darcy_res_abs_max": ConstraintDiagnostic(
                    value=darcy_residual.abs().max(),
                    reduce="max",
                ),
                "constraint/boundary_abs_mean": ConstraintDiagnostic(
                    value=boundary_residual.mean(),
                    reduce="mean",
                ),
                "constraint/boundary_abs_max": ConstraintDiagnostic(
                    value=boundary_residual.max(),
                    reduce="max",
                ),
            }
            return self.as_output(
                pressure_encoded,
                aux={
                    "pred_base": pred,
                    "stream_correction": reshape_grid_to_channels_last(stream_correction),
                    "constrained_flux": reshape_grid_to_channels_last(constrained_flux),
                },
                diagnostics=diagnostics,
            )
        return pressure_encoded
