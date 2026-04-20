from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .boundary import boundary_stats
from .spectral import (
    crop_spatial_2d,
    finite_difference_curl_2d,
    finite_difference_divergence_2d,
    finite_difference_gradient_2d,
    fft_leray_project_2d,
    normalize_padding_2d,
    pad_spatial_2d,
    reshape_channels_last_to_grid,
    reshape_grid_to_channels_last,
    spectral_divergence_2d,
    spectral_poisson_solve_2d,
)


class DarcyFluxConstraint(ConstraintModule):
    """
    Darcy-specific pressure recovery from a latent flux field using an fft+padding
    backend.

    The backbone predicts a flattened 2-channel flux field. The constraint:
    1. decodes the physical permeability a(x),
    2. pads the flux/permeability grids,
    3. projects the flux correction to a divergence-free field with a Fourier
       Leray projection,
    4. converts flux to a pressure gradient estimate via w = -v / a,
    5. recovers pressure from a padded Poisson solve, then crops back.

    For the current MVP, only the `fft_pad` backend is implemented. A future
    Dirichlet-aware sine backend can reuse the same wrapper contract.
    """

    name = "darcy_flux_projection"

    def __init__(
        self,
        *,
        spectral_backend: str = "fft_pad",
        force_value: float = 1.0,
        permeability_eps: float = 1e-6,
        padding: int | Sequence[int] = 8,
        padding_mode: str = "reflect",
        pressure_out_dim: int = 1,
        enforce_boundary: bool = True,
        boundary_value: float = 0.0,
        shapelist: Sequence[int] | None = None,
        lower: float = 0.0,
        upper: float = 1.0,
    ) -> None:
        super().__init__()
        self.spectral_backend = str(spectral_backend).lower()
        if self.spectral_backend != "fft_pad":
            raise NotImplementedError(
                "Only spectral_backend='fft_pad' is implemented right now. "
                "A future sine backend can be added behind the same API."
            )
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
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        y = torch.arange(height, device=device, dtype=dtype) * dy
        y = y - y.mean()
        v_y = self.force_value * y.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        v_x = torch.zeros_like(v_y)
        return torch.cat([v_x, v_y], dim=1)

    def _enforce_boundary_values(self, pressure: torch.Tensor) -> torch.Tensor:
        if not self.enforce_boundary:
            return pressure
        out = pressure.clone()
        out[..., 0, :] = self.boundary_value
        out[..., -1, :] = self.boundary_value
        out[..., :, 0] = self.boundary_value
        out[..., :, -1] = self.boundary_value
        return out

    def forward(self, *, pred, fx=None, coords=None, return_aux=False, **_unused):
        if fx is None:
            raise ValueError("fx is required for DarcyFluxConstraint")
        if pred.ndim != 3:
            raise ValueError(
                "Expected a flattened flux prediction with shape (batch, n_points, 2), "
                f"got {tuple(pred.shape)!r}"
            )
        if pred.shape[-1] != 2:
            raise ValueError(
                "DarcyFluxConstraint expects a 2-channel backbone output "
                f"(predicted flux), got last dimension {pred.shape[-1]}"
            )
        if fx.ndim != 3 or fx.shape[-1] != 1:
            raise ValueError(
                "DarcyFluxConstraint expects a single-channel permeability input, "
                f"got {tuple(fx.shape)!r}"
            )

        height, width = self._grid_shape()
        dy, dx = self._spacing(height, width)

        flux = reshape_channels_last_to_grid(pred, shapelist=(height, width))
        permeability = reshape_channels_last_to_grid(
            self._decode_permeability(fx),
            shapelist=(height, width),
        )

        flux_padded = pad_spatial_2d(flux, self.padding, mode=self.padding_mode)
        permeability_padded = pad_spatial_2d(
            permeability,
            self.padding,
            mode=self.padding_mode,
        )

        padded_height, padded_width = flux_padded.shape[-2], flux_padded.shape[-1]
        particular_flux = self._particular_flux(
            batch_size=flux_padded.shape[0],
            height=padded_height,
            width=padded_width,
            dy=dy,
            device=flux_padded.device,
            dtype=flux_padded.dtype,
        )
        projected_correction = fft_leray_project_2d(
            flux_padded - particular_flux,
            dy=dy,
            dx=dx,
        )
        constrained_flux = particular_flux + projected_correction

        pressure_gradient = -constrained_flux / permeability_padded.clamp_min(
            self.permeability_eps
        )
        laplace_pressure = spectral_divergence_2d(
            pressure_gradient,
            dy=dy,
            dx=dx,
        )
        pressure_padded = spectral_poisson_solve_2d(
            laplace_pressure,
            dy=dy,
            dx=dx,
        )
        pressure = crop_spatial_2d(pressure_padded, self.padding)
        pressure = self._enforce_boundary_values(pressure)

        pressure_flat = reshape_grid_to_channels_last(pressure)
        pressure_encoded = self._encode_pressure(pressure_flat)

        if return_aux:
            constrained_flux_physical = crop_spatial_2d(constrained_flux, self.padding)
            constrained_flux_flat = reshape_grid_to_channels_last(constrained_flux_physical)
            flux_correction_flat = constrained_flux_flat - pred
            physical_gradient = crop_spatial_2d(pressure_gradient, self.padding)
            flux_divergence = finite_difference_divergence_2d(
                constrained_flux_physical,
                dy=dy,
                dx=dx,
            )
            flux_divergence_residual = flux_divergence - self.force_value
            gradient_curl = finite_difference_curl_2d(
                physical_gradient,
                dy=dy,
                dx=dx,
            )
            pressure_gradient_from_u = finite_difference_gradient_2d(
                pressure,
                dy=dy,
                dx=dx,
            )
            darcy_flux_from_pressure = -permeability * pressure_gradient_from_u
            darcy_residual = finite_difference_divergence_2d(
                darcy_flux_from_pressure,
                dy=dy,
                dx=dx,
            ) - self.force_value
            particular_divergence_padded = spectral_divergence_2d(
                particular_flux,
                dy=dy,
                dx=dx,
            ) - self.force_value
            correction_divergence_padded = spectral_divergence_2d(
                projected_correction,
                dy=dy,
                dx=dx,
            )
            constrained_divergence_padded = spectral_divergence_2d(
                constrained_flux,
                dy=dy,
                dx=dx,
            ) - self.force_value

            diagnostics = {
                "constraint/flux_div_abs_mean": ConstraintDiagnostic(
                    value=flux_divergence_residual.abs().mean(),
                    reduce="mean",
                ),
                "constraint/flux_div_abs_max": ConstraintDiagnostic(
                    value=flux_divergence_residual.abs().max(),
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
                "constraint/grad_curl_abs_mean": ConstraintDiagnostic(
                    value=gradient_curl.abs().mean(),
                    reduce="mean",
                ),
                "constraint/grad_curl_abs_max": ConstraintDiagnostic(
                    value=gradient_curl.abs().max(),
                    reduce="max",
                ),
                "constraint/flux_correction_rel_l2": ConstraintDiagnostic(
                    value=flux_correction_flat.reshape(flux_correction_flat.shape[0], -1)
                    .norm(dim=1)
                    .div(
                        pred.reshape(pred.shape[0], -1).norm(dim=1).clamp_min(1e-12)
                    )
                    .mean(),
                    reduce="mean",
                ),
                "constraint/debug/fft_particular_div_padded_abs_mean": ConstraintDiagnostic(
                    value=particular_divergence_padded.abs().mean(),
                    reduce="mean",
                ),
                "constraint/debug/fft_particular_div_padded_abs_max": ConstraintDiagnostic(
                    value=particular_divergence_padded.abs().max(),
                    reduce="max",
                ),
                "constraint/debug/fft_correction_div_padded_abs_mean": ConstraintDiagnostic(
                    value=correction_divergence_padded.abs().mean(),
                    reduce="mean",
                ),
                "constraint/debug/fft_correction_div_padded_abs_max": ConstraintDiagnostic(
                    value=correction_divergence_padded.abs().max(),
                    reduce="max",
                ),
                "constraint/debug/fft_constrained_div_padded_abs_mean": ConstraintDiagnostic(
                    value=constrained_divergence_padded.abs().mean(),
                    reduce="mean",
                ),
                "constraint/debug/fft_constrained_div_padded_abs_max": ConstraintDiagnostic(
                    value=constrained_divergence_padded.abs().max(),
                    reduce="max",
                ),
            }
            if coords is not None:
                boundary = boundary_stats(
                    pressure_flat,
                    coords,
                    target_value=self.boundary_value,
                    lower=self.lower,
                    upper=self.upper,
                )
                diagnostics["constraint/boundary_abs_mean"] = ConstraintDiagnostic(
                    value=boundary["boundary_abs_mean"],
                    reduce="mean",
                )
                diagnostics["constraint/boundary_abs_max"] = ConstraintDiagnostic(
                    value=boundary["boundary_abs_max"],
                    reduce="max",
                )
            return self.as_output(
                pressure_encoded,
                aux={
                    "pred_base": pred,
                    "flux_correction": flux_correction_flat,
                    "constrained_flux": constrained_flux_flat,
                },
                diagnostics=diagnostics,
            )
        return pressure_encoded
