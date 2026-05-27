from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from .base import ConstraintDiagnostic, ConstraintModule
from .utils.boundary_ops import encode_target
from .utils.spectral import (
    reshape_channels_last_to_grid,
    reshape_grid_to_channels_last,
)
from .utils.stream_ops import (
    finite_volume_divergence_curvilinear,
    stream_velocity_from_psi_curvilinear,
)


def _uy_supervision_loss(
    uy_pred_flat: torch.Tensor,
    uy_target: torch.Tensor,
    uy_normalizer,
    weight: float,
) -> torch.Tensor:
    """Relative L2 loss between predicted and target uy, both encoded with uy_normalizer."""
    uy_pred_enc = encode_target(uy_pred_flat, uy_normalizer)
    uy_tgt_enc = encode_target(uy_target, uy_normalizer)
    n = uy_pred_enc.shape[0]
    diff = (uy_pred_enc - uy_tgt_enc).reshape(n, -1)
    tgt_norm = uy_tgt_enc.reshape(n, -1).norm(p=2, dim=1).clamp_min(1e-12)
    return weight * (diff.norm(p=2, dim=1) / tgt_norm).sum()


class PipeStreamFunctionUxConstraint(ConstraintModule):
    """
    Interprets the scalar backbone output as a stream function psi on the
    curvilinear pipe mesh and returns ux = dpsi/dy in physical coordinates.

    When uy_loss_weight > 0, an additional relative-L2 loss on uy is computed
    against the uy_target tensor passed through the batch (key "y_uy") and added
    to the returned extra_loss. The benchmark target and out_dim stay at 1 (ux only).
    Set data.load_uy: true in the config to supply uy_target.
    """

    name = "pipe_stream_function_ux"

    def __init__(
        self,
        *,
        shapelist=None,
        eps: float = 1e-12,
        uy_loss_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.shapelist = None if shapelist is None else tuple(int(v) for v in shapelist)
        self.eps = float(eps)
        self.uy_loss_weight = float(uy_loss_weight)
        self.input_normalizer = None
        self.target_normalizer = None
        self.uy_normalizer = None

    def set_grid_shape(self, shapelist) -> None:
        self.shapelist = tuple(int(v) for v in shapelist)

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_uy_normalizer(self, normalizer) -> None:
        self.uy_normalizer = normalizer

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

    def forward(self, *, pred, coords=None, return_aux=False, uy_target=None, **_unused):
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

        uy_flat = reshape_grid_to_channels_last(uy)

        extra_loss = None
        if self.uy_loss_weight > 0.0 and uy_target is not None:
            extra_loss = _uy_supervision_loss(
                uy_flat, uy_target, self.uy_normalizer, self.uy_loss_weight
            )

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
            aux_out = {
                "pred_base": pred,
                "stream_psi": reshape_grid_to_channels_last(psi),
                "stream_uy": uy_flat,
                "stream_div": div.reshape(div.shape[0], 1, -1).transpose(1, 2),
            }
            if uy_target is not None:
                aux_out["stream_uy_target"] = uy_target
                uy_pred_enc = encode_target(uy_flat, self.uy_normalizer)
                uy_tgt_enc = encode_target(uy_target, self.uy_normalizer)
                n = uy_pred_enc.shape[0]
                diff = (uy_pred_enc - uy_tgt_enc).reshape(n, -1)
                tgt_norm = uy_tgt_enc.reshape(n, -1).norm(p=2, dim=1).clamp_min(1e-12)
                diagnostics["constraint/stream_uy_rel_l2"] = ConstraintDiagnostic(
                    value=(diff.norm(p=2, dim=1) / tgt_norm).mean(),
                    reduce="mean",
                )
            return self.as_output(
                ux_encoded,
                aux=aux_out,
                diagnostics=diagnostics,
                extra_loss=extra_loss,
            )
        return ux_encoded

    @classmethod
    def log_media(cls, ctx) -> dict[str, str]:
        aux = ctx.aux_tensors
        if not all(k in aux for k in ("stream_psi", "stream_uy", "stream_div")):
            return {}
        h, w = ctx.meta["shapelist"]
        uy_target = aux.get("stream_uy_target")
        if ctx.out_dir is None:
            from omni_hc.training.logging_utils import log_pipe_stream_images
            log_pipe_stream_images(
                ctx.coords, h, w,
                prefix=ctx.prefix, epoch=ctx.epoch, step=ctx.step,
                psi=aux["stream_psi"], uy=aux["stream_uy"], divergence=aux["stream_div"],
                uy_target=uy_target,
            )
            return {}
        from omni_hc.training.logging_utils import save_pipe_stream_images
        return save_pipe_stream_images(
            ctx.coords, h, w,
            out_dir=ctx.out_dir, prefix=ctx.prefix,
            psi=aux["stream_psi"], uy=aux["stream_uy"], divergence=aux["stream_div"],
            uy_target=uy_target,
        )


class PipeStreamFunctionBoundaryAnsatz(ConstraintModule):
    """
    Builds a hard-constrained stream function on the curvilinear pipe mesh:

        psi = psi_bc(eta) + xi^p * eta^2 * (1 - eta)^2 * N

    with

        psi_bc(eta) = C + Umax * H * (2 eta^2 - 4/3 eta^3)

    and returns ux recovered from the stream function in physical
    coordinates. This preserves the inlet parabolic profile at xi=0 and
    keeps the correction from changing wall values.

    When uy_loss_weight > 0, an additional relative-L2 loss on uy is computed
    against the uy_target tensor passed through the batch (key "y_uy") and added
    to the returned extra_loss. The benchmark target and out_dim stay at 1 (ux only).
    Set data.load_uy: true in the config to supply uy_target.
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
        uy_loss_weight: float = 0.0,
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
        self.uy_loss_weight = float(uy_loss_weight)
        self.input_normalizer = None
        self.target_normalizer = None
        self.uy_normalizer = None

    def set_grid_shape(self, shapelist) -> None:
        self.shapelist = tuple(int(v) for v in shapelist)

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_uy_normalizer(self, normalizer) -> None:
        self.uy_normalizer = normalizer

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

    @property
    def idx_all_boundary(self):
        """Flat node indices of the CONSTRAINED boundary (inlet + walls).

        The steady eval uses this to compute boundary_rel_l2. The outlet face is
        excluded on purpose: the ansatz does not constrain it (the correction
        window l = xi^p * eta^2 * (1 - eta)^2 is largest there), so scoring it
        would report fit error on a free edge as if it were a constraint miss.
        """
        if self.shapelist is None or len(self.shapelist) != 2:
            return None
        h, w = int(self.shapelist[0]), int(self.shapelist[1])
        grid = torch.arange(h * w).reshape(h, w)
        if self.inlet_axis == 0:
            inlet = grid[0, :]       # inlet face (streamwise start)
            wall_a = grid[1:, 0]     # walls run along the streamwise axis
            wall_b = grid[1:, -1]
        else:
            inlet = grid[:, 0]
            wall_a = grid[0, 1:]
            wall_b = grid[-1, 1:]
        return torch.cat([inlet.reshape(-1), wall_a.reshape(-1), wall_b.reshape(-1)])

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

    def forward(self, *, pred, coords=None, return_aux=False, uy_target=None, **_unused):
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
        uy_flat = reshape_grid_to_channels_last(uy)

        extra_loss = None
        if self.uy_loss_weight > 0.0 and uy_target is not None:
            extra_loss = _uy_supervision_loss(
                uy_flat, uy_target, self.uy_normalizer, self.uy_loss_weight
            )

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
            inlet_rmse = (inlet_ux - inlet_profile).square().mean(dim=-1).sqrt()
            div_rmse = div.square().reshape(div.shape[0], -1).mean(dim=-1).sqrt()
            diagnostics = {
                "constraint/stream_div_abs_mean": ConstraintDiagnostic(
                    value=div.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_div_abs_max": ConstraintDiagnostic(
                    value=div.abs().max(),
                    reduce="max",
                ),
                "constraint/div_rmse": ConstraintDiagnostic(
                    value=div_rmse.mean(),
                    reduce="mean",
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
                "constraint/inlet_rmse": ConstraintDiagnostic(
                    value=inlet_rmse.mean(),
                    reduce="mean",
                ),
                "constraint/stream_wall_ux_abs_mean": ConstraintDiagnostic(
                    value=wall_ux.abs().mean(),
                    reduce="mean",
                ),
                "constraint/stream_wall_ux_abs_max": ConstraintDiagnostic(
                    value=wall_ux.abs().max(),
                    reduce="max",
                ),
                "constraint/wall_abs_max": ConstraintDiagnostic(
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
            aux_out = {
                "pred_base": pred,
                "stream_psi": reshape_grid_to_channels_last(psi),
                "stream_uy": uy_flat,
                "stream_div": div.reshape(div.shape[0], 1, -1).transpose(1, 2),
                "stream_psi_bc": reshape_grid_to_channels_last(psi_bc),
                "stream_mask": reshape_grid_to_channels_last(correction_mask),
            }
            if uy_target is not None:
                aux_out["stream_uy_target"] = uy_target
                uy_pred_enc = encode_target(uy_flat, self.uy_normalizer)
                uy_tgt_enc = encode_target(uy_target, self.uy_normalizer)
                n = uy_pred_enc.shape[0]
                diff = (uy_pred_enc - uy_tgt_enc).reshape(n, -1)
                tgt_norm = uy_tgt_enc.reshape(n, -1).norm(p=2, dim=1).clamp_min(1e-12)
                diagnostics["constraint/stream_uy_rel_l2"] = ConstraintDiagnostic(
                    value=(diff.norm(p=2, dim=1) / tgt_norm).mean(),
                    reduce="mean",
                )
            return self.as_output(
                ux_encoded,
                aux=aux_out,
                diagnostics=diagnostics,
                extra_loss=extra_loss,
            )
        return ux_encoded

    @classmethod
    def log_media(cls, ctx) -> dict[str, str]:
        aux = ctx.aux_tensors
        if not all(k in aux for k in ("stream_psi", "stream_uy", "stream_div")):
            return {}
        h, w = ctx.meta["shapelist"]
        uy_target = aux.get("stream_uy_target")
        if ctx.out_dir is None:
            from omni_hc.training.logging_utils import log_pipe_stream_images
            log_pipe_stream_images(
                ctx.coords, h, w,
                prefix=ctx.prefix, epoch=ctx.epoch, step=ctx.step,
                psi=aux["stream_psi"], uy=aux["stream_uy"], divergence=aux["stream_div"],
                psi_bc=aux.get("stream_psi_bc"), mask=aux.get("stream_mask"),
                uy_target=uy_target,
            )
            return {}
        from omni_hc.training.logging_utils import save_pipe_stream_images
        return save_pipe_stream_images(
            ctx.coords, h, w,
            out_dir=ctx.out_dir, prefix=ctx.prefix,
            psi=aux["stream_psi"], uy=aux["stream_uy"], divergence=aux["stream_div"],
            psi_bc=aux.get("stream_psi_bc"), mask=aux.get("stream_mask"),
            uy_target=uy_target,
        )
