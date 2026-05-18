from __future__ import annotations

import math
from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .utils.boundary_ops import (
    apply_boundary_ansatz,
    decode_target,
    encode_target,
    validate_channels_last_prediction,
)
from .utils.structured_grid import axis_coordinate, edge_values, resolve_grid_shape

# Reference freestream conditions (ρ∞=1, p∞=1, M∞=0.8, AoA=0, γ=1.4)
GAMMA: float = 1.4
RHO_INF: float = 1.0
P_INF: float = 1.0
M_INF: float = 0.8
A_INF: float = math.sqrt(GAMMA)   # ≈ 1.1832
U_INF: float = M_INF * A_INF      # ≈ 0.9466
V_INF: float = 0.0

# Freestream state vectors for common prediction targets
# [ρ, u, v, p] — primitive variables excluding derived Mach
FREESTREAM_PRIMITIVE: tuple[float, ...] = (RHO_INF, U_INF, V_INF, P_INF)
# Mach only — matches the NSL airfoil loader (out_dim=1)
FREESTREAM_MACH: tuple[float, ...] = (M_INF,)
# Full 5-channel [ρ, u, v, p, M]
FREESTREAM_FULL: tuple[float, ...] = (RHO_INF, U_INF, V_INF, P_INF, M_INF)


def derived_mach(
    rho: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    *,
    gamma: float = GAMMA,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute Mach from primitives: M = |q| / sqrt(γ p / ρ).

    This is exact in the NACA dataset (verified zero residual). Use it as the
    Mach output head instead of predicting Mach as an independent channel.
    """
    a2 = gamma * p / rho.clamp_min(eps)
    a = a2.clamp_min(eps).sqrt()
    return (u**2 + v**2).sqrt() / a


class AirfoilFarFieldAnsatz(ConstraintModule):
    """Enforces far-field Dirichlet BCs at the outer C-grid boundary (j = J-1).

    The C-grid layout:
      - axis 0 (i): circumferential, 221 points wrapping around the airfoil
      - axis 1 (j): radial, 51 points from wall (j=0) to far-field (j=50, r≈40)

    Ansatz: Q = Q_inf + d(j) * N
    where d(j) = (1 - j/(J-1))^power — 1 at wall, 0 at far-field.

    At j=J-1: Q = Q_inf exactly (far-field enforced as hard constraint).
    At j=0:   Q = Q_inf + N    (backbone has full freedom at the wall).

    The backbone learns the perturbation field N, which is zero at the far-field
    by construction — the geometry is encoded through the (x, y) input coordinates.
    """

    name = "airfoil_far_field_ansatz"

    def __init__(
        self,
        *,
        out_dim: int,
        freestream_values: Sequence[float],
        grid_shape: Sequence[int] | None = None,
        radial_axis: int = 1,
        distance_power: float = 1.0,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.freestream_values = tuple(float(v) for v in freestream_values)
        if len(self.freestream_values) != self.out_dim:
            raise ValueError(
                f"freestream_values length {len(self.freestream_values)} "
                f"must match out_dim={self.out_dim}"
            )
        self.grid_shape = (
            None if grid_shape is None else tuple(int(v) for v in grid_shape)
        )
        self.radial_axis = int(radial_axis)
        self.distance_power = float(distance_power)
        self.target_normalizer = None

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_grid_shape(self, grid_shape: Sequence[int]) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)

    def _resolve_grid_shape(self, pred: torch.Tensor) -> tuple[int, int]:
        return resolve_grid_shape(
            self.grid_shape, pred, name="airfoil far-field ansatz"
        )

    def _freestream_tensor(self, pred: torch.Tensor) -> torch.Tensor:
        """Q_inf tensor in prediction (possibly normalised) space — shape (1, N, C)."""
        g = pred.new_tensor(self.freestream_values)          # (out_dim,)
        g = g.expand(pred.shape[0], pred.shape[1], self.out_dim).contiguous()
        return encode_target(g, self.target_normalizer)

    def _distance(
        self, grid_shape: Sequence[int], pred: torch.Tensor
    ) -> torch.Tensor:
        # eta ∈ [0, 1]: 0 at j=0 (wall), 1 at j=J-1 (far-field)
        eta = axis_coordinate(
            grid_shape,
            axis=self.radial_axis,
            dtype=pred.dtype,
            device=pred.device,
        )
        d = (1.0 - eta).clamp_min(0.0)   # 1 at wall, 0 at far-field
        if self.distance_power != 1.0:
            d = d.pow(self.distance_power)
        return d  # (1, N, 1)

    def _far_field_residual(
        self, field: torch.Tensor, grid_shape: tuple[int, int]
    ) -> torch.Tensor:
        far_vals = edge_values(
            field, grid_shape, axis=self.radial_axis, edge="upper"
        )  # (B, I, C)
        q_inf = field.new_tensor(self.freestream_values)
        return (far_vals - q_inf).abs()

    def forward(self, *, pred, return_aux=False, **_unused):
        validate_channels_last_prediction(
            pred, out_dim=self.out_dim, name=self.__class__.__name__
        )
        grid_shape = self._resolve_grid_shape(pred)
        g = self._freestream_tensor(pred)
        d = self._distance(grid_shape, pred)
        out = apply_boundary_ansatz(pred=pred, particular=g, distance=d)

        if return_aux:
            field = decode_target(out, self.target_normalizer)
            residual = self._far_field_residual(field, grid_shape)
            pred_field = decode_target(pred, self.target_normalizer)
            base_residual = self._far_field_residual(pred_field, grid_shape)
            diagnostics = {
                "constraint/far_field_abs_mean": ConstraintDiagnostic(
                    value=residual.mean(), reduce="mean"
                ),
                "constraint/far_field_abs_max": ConstraintDiagnostic(
                    value=residual.max(), reduce="max"
                ),
                "constraint/far_field_base_abs_mean": ConstraintDiagnostic(
                    value=base_residual.mean(), reduce="mean"
                ),
                "constraint/far_field_base_abs_max": ConstraintDiagnostic(
                    value=base_residual.max(), reduce="max"
                ),
                "constraint/distance_mean": ConstraintDiagnostic(
                    value=d.mean(), reduce="mean"
                ),
                "constraint/distance_min": ConstraintDiagnostic(
                    value=d.min(), reduce="min"
                ),
            }
            return self.as_output(
                out,
                aux={"pred_base": pred, "distance": d},
                diagnostics=diagnostics,
            )
        return out
