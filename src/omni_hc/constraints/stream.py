from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .utils.spectral import (
    reshape_channels_last_to_grid,
    reshape_grid_to_channels_last,
)
from .utils.stream_ops import (
    finite_volume_divergence_curvilinear,
    stream_velocity_from_psi_curvilinear,
)


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


class PipeStreamFunctionBoundaryAnsatz(ConstraintModule):
    """
    Builds a hard-constrained stream function on the curvilinear pipe mesh:

        psi = psi_bc(eta) + xi^p * eta^2 * (1 - eta)^2 * N

    with

        psi_bc(eta) = C + Umax * H * (2 eta^2 - 4/3 eta^3)

    and returns ux recovered from the stream function in physical
    coordinates. This preserves the inlet parabolic profile at xi=0 and
    keeps the correction from changing wall values.
    """

    name = "pipe_stream_function_boundary_ansatz"

    def __init__(
        self,
        *,
        shapelist: Sequence[int] | None = None,
        amplitude: float = 0.25,
        inlet_axis: int = 0,
        transverse_axis: int = 1,
        coordinate_channel: int = 1,
        boundary_constant: float = 0.0,
        decay_power: float = 4.0,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.shapelist = None if shapelist is None else tuple(int(v) for v in shapelist)
        self.amplitude = float(amplitude)
        self.inlet_axis = int(inlet_axis)
        self.transverse_axis = int(transverse_axis)
        self.coordinate_channel = int(coordinate_channel)
        self.boundary_constant = float(boundary_constant)
        self.decay_power = float(decay_power)
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
            raise ValueError(
                "PipeStreamFunctionBoundaryAnsatz requires a 2D grid shape"
            )
        if len(self.shapelist) != 2:
            raise ValueError(
                "PipeStreamFunctionBoundaryAnsatz expects a 2D grid shape, "
                f"got {self.shapelist!r}"
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

    def _transverse_coordinate(self, coords_grid: torch.Tensor) -> torch.Tensor:
        if self.coordinate_channel not in {0, 1}:
            raise ValueError(
                f"coordinate_channel must be 0 or 1, got {self.coordinate_channel}"
            )
        return coords_grid[:, self.coordinate_channel : self.coordinate_channel + 1]

    def _normalized_transverse_coordinate(
        self, transverse_coord: torch.Tensor
    ) -> torch.Tensor:
        if self.transverse_axis not in {0, 1}:
            raise ValueError(
                f"transverse_axis must be 0 or 1, got {self.transverse_axis}"
            )
        reduce_dim = -1 if self.transverse_axis == 1 else -2
        coord_min = transverse_coord.amin(dim=reduce_dim, keepdim=True)
        coord_max = transverse_coord.amax(dim=reduce_dim, keepdim=True)
        return (transverse_coord - coord_min) / (coord_max - coord_min).clamp_min(
            self.eps
        )

    def _inlet_extent(self, transverse_coord: torch.Tensor) -> torch.Tensor:
        if self.inlet_axis == 0:
            inlet_coord = transverse_coord[:, :, 0, :]
        elif self.inlet_axis == 1:
            inlet_coord = transverse_coord[:, :, :, 0]
        else:
            raise ValueError(f"inlet_axis must be 0 or 1, got {self.inlet_axis}")
        inlet_min = inlet_coord.amin(dim=-1, keepdim=True)
        inlet_max = inlet_coord.amax(dim=-1, keepdim=True)
        return (inlet_max - inlet_min).view(transverse_coord.shape[0], 1, 1, 1)

    def _normalized_streamwise_coordinate(
        self,
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.inlet_axis not in {0, 1}:
            raise ValueError(f"inlet_axis must be 0 or 1, got {self.inlet_axis}")
        if self.inlet_axis == 0:
            xi_line = torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype)
            return xi_line.view(1, 1, height, 1).expand(batch_size, 1, height, width)
        xi_line = torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype)
        return xi_line.view(1, 1, 1, width).expand(batch_size, 1, height, width)

    def _stream_function_bc(
        self,
        *,
        eta: torch.Tensor,
        inlet_extent: torch.Tensor,
    ) -> torch.Tensor:
        primitive = 2.0 * eta.square() - (4.0 / 3.0) * eta.pow(3)
        return self.boundary_constant + self.amplitude * inlet_extent * primitive

    def _correction_mask(self, *, xi: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
        return xi.pow(self.decay_power) * eta.square() * (1.0 - eta).square()

    def forward(self, *, pred, coords=None, return_aux=False, **_unused):
        if coords is None:
            raise ValueError("coords are required for PipeStreamFunctionBoundaryAnsatz")
        if pred.ndim != 3 or pred.shape[-1] != 1:
            raise ValueError(
                "PipeStreamFunctionBoundaryAnsatz expects a scalar backbone output "
                f"with shape (batch, n_points, 1), got {tuple(pred.shape)!r}"
            )
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError(
                "PipeStreamFunctionBoundaryAnsatz expects coords with shape "
                f"(batch, n_points, 2), got {tuple(coords.shape)!r}"
            )

        height, width = self._grid_shape()
        pred_base = reshape_channels_last_to_grid(pred, shapelist=(height, width))
        coords_grid = reshape_channels_last_to_grid(
            self._decode_coords(coords),
            shapelist=(height, width),
        )
        transverse_coord = self._transverse_coordinate(coords_grid)
        eta = self._normalized_transverse_coordinate(transverse_coord)
        inlet_extent = self._inlet_extent(transverse_coord)
        xi = self._normalized_streamwise_coordinate(
            batch_size=pred.shape[0],
            height=height,
            width=width,
            device=pred.device,
            dtype=pred.dtype,
        )
        psi_bc = self._stream_function_bc(eta=eta, inlet_extent=inlet_extent)
        correction_mask = self._correction_mask(xi=xi, eta=eta)
        psi = psi_bc + correction_mask * pred_base

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
            if self.inlet_axis == 0:
                inlet_eta = eta[:, :, 0, :]
                inlet_ux = ux[:, :, 0, :].reshape(ux.shape[0], -1)
                wall_ux = torch.cat(
                    [
                        ux[:, :, :, 0].reshape(ux.shape[0], -1),
                        ux[:, :, :, -1].reshape(ux.shape[0], -1),
                    ],
                    dim=1,
                )
            else:
                inlet_eta = eta[:, :, :, 0]
                inlet_ux = ux[:, :, :, 0].reshape(ux.shape[0], -1)
                wall_ux = torch.cat(
                    [
                        ux[:, :, 0, :].reshape(ux.shape[0], -1),
                        ux[:, :, -1, :].reshape(ux.shape[0], -1),
                    ],
                    dim=1,
                )
            inlet_profile = (
                self.amplitude * 4.0 * inlet_eta * (1.0 - inlet_eta)
            ).reshape(inlet_ux.shape[0], -1)
            inlet_residual = (inlet_ux - inlet_profile).abs()
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
                "constraint/stream_inlet_abs_mean": ConstraintDiagnostic(
                    value=inlet_residual.mean(),
                    reduce="mean",
                ),
                "constraint/stream_inlet_abs_max": ConstraintDiagnostic(
                    value=inlet_residual.max(),
                    reduce="max",
                ),
                "constraint/stream_wall_ux_abs_mean": ConstraintDiagnostic(
                    value=wall_ux.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_wall_ux_abs_max": ConstraintDiagnostic(
                    value=wall_ux.abs().max(),
                    reduce="max",
                ),
                "constraint/stream_mask_mean": ConstraintDiagnostic(
                    value=correction_mask.mean(),
                    reduce="mean",
                ),
                "constraint/stream_mask_max": ConstraintDiagnostic(
                    value=correction_mask.max(),
                    reduce="max",
                ),
            }
            return self.as_output(
                ux_encoded,
                aux={
                    "pred_base": pred,
                    "stream_psi": reshape_grid_to_channels_last(psi),
                    "stream_uy": reshape_grid_to_channels_last(uy),
                    "stream_div": div.reshape(div.shape[0], 1, -1).transpose(1, 2),
                    "stream_psi_bc": reshape_grid_to_channels_last(psi_bc),
                    "stream_mask": reshape_grid_to_channels_last(correction_mask),
                },
                diagnostics=diagnostics,
            )
        return ux_encoded
