from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule
from .utils.spectral import (
    reshape_channels_last_to_grid,
    reshape_grid_to_channels_last,
    sine_poisson_solve_dirichlet_2d,
)


class DarcyDefectCorrectionConstraint(ConstraintModule):
    """
    Hard Darcy interior constraint via defect correction (no boundary enforcement).

    Steps:
    1. Decode backbone output to physical pressure space.
    2. Compute the discrete Darcy residual r = -div_h(a_hm ∇_h u) - f using
       harmonic-mean face permeabilities for the variable-coefficient operator.
    3. Correct: delta_u solves Δ(delta_u) = r/a with Dirichlet BCs; u += delta_u.
       Repeat n_correction_steps times.
    4. Re-encode and return.

    Boundary values are left as predicted by the backbone. The correction delta_u
    uses zero Dirichlet BCs, so it does not alter whatever the backbone placed on
    the boundary.

    The correction is exact inside uniform-permeability regions. For binary {3, 12}
    permeability, the interface convergence factor is |1 - a_hm/a_local| ≈ 0.6 per step.
    """

    name = "darcy_defect_correction"

    def __init__(
        self,
        *,
        force_value: float = 1.0,
        n_correction_steps: int = 1,
        lower: float = 0.0,
        upper: float = 1.0,
        shapelist: Sequence[int] | None = None,
        permeability_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.force_value = float(force_value)
        self.n_correction_steps = int(n_correction_steps)
        self.lower = float(lower)
        self.upper = float(upper)
        self.shapelist = None if shapelist is None else tuple(int(v) for v in shapelist)
        self.permeability_eps = float(permeability_eps)
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
            raise ValueError(
                "DarcyDefectCorrectionConstraint requires a 2D grid shape; "
                "call set_grid_shape() before forward."
            )
        if len(self.shapelist) != 2:
            raise ValueError(f"Expected a 2D shapelist, got {self.shapelist!r}")
        return int(self.shapelist[0]), int(self.shapelist[1])

    def _spacing(self, height: int, width: int) -> tuple[float, float]:
        extent = self.upper - self.lower
        dy = extent / max(height - 1, 1)
        dx = extent / max(width - 1, 1)
        return float(dy), float(dx)

    def _decode_permeability(self, fx: torch.Tensor) -> torch.Tensor:
        return fx if self.input_normalizer is None else self.input_normalizer.decode(fx)

    def _decode_pressure(self, u: torch.Tensor) -> torch.Tensor:
        return u if self.target_normalizer is None else self.target_normalizer.decode(u)

    def _encode_pressure(self, u: torch.Tensor) -> torch.Tensor:
        return u if self.target_normalizer is None else self.target_normalizer.encode(u)

    def _darcy_residual(
        self,
        u: torch.Tensor,  # (B, 1, H, W) physical
        a: torch.Tensor,  # (B, 1, H, W) physical
        *,
        dy: float,
        dx: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Discrete Darcy residual r = -div_h(a_hm ∇_h u) - f at interior grid points.

        Uses harmonic-mean face permeabilities:
            a_face = 2 * a_L * a_R / (a_L + a_R)

        Returns:
            r_interior  : (B, 1, H-2, W-2)  residual at interior points
            is_interface: (B, 1, H-2, W-2)  True where permeability changes across a face
            a_interior  : (B, 1, H-2, W-2)  permeability at interior points
        """
        # Harmonic-mean face permeabilities
        a_face_x = (2.0 * a[:, :, :, :-1] * a[:, :, :, 1:]) / (
            a[:, :, :, :-1] + a[:, :, :, 1:] + self.permeability_eps
        )  # (B, 1, H, W-1)
        a_face_y = (2.0 * a[:, :, :-1, :] * a[:, :, 1:, :]) / (
            a[:, :, :-1, :] + a[:, :, 1:, :] + self.permeability_eps
        )  # (B, 1, H-1, W)

        # Face fluxes: a_hm * ∇_h u
        flux_x = a_face_x * (u[:, :, :, 1:] - u[:, :, :, :-1]) / dx   # (B, 1, H, W-1)
        flux_y = a_face_y * (u[:, :, 1:, :] - u[:, :, :-1, :]) / dy   # (B, 1, H-1, W)

        # Divergence at interior points i∈[1,H-2], j∈[1,W-2]
        div_x = (flux_x[:, :, 1:-1, 1:] - flux_x[:, :, 1:-1, :-1]) / dx   # (B,1,H-2,W-2)
        div_y = (flux_y[:, :, 1:, 1:-1] - flux_y[:, :, :-1, 1:-1]) / dy   # (B,1,H-2,W-2)

        r_interior = -(div_x + div_y) - self.force_value   # (B,1,H-2,W-2)

        # Interface detection: interior point adjacent to a cell with different permeability
        a_int = a[:, :, 1:-1, 1:-1]
        is_interface = (
            (a_int != a[:, :, 1:-1, 2:])    # right neighbour
            | (a_int != a[:, :, 1:-1, :-2]) # left neighbour
            | (a_int != a[:, :, 2:, 1:-1])  # bottom neighbour
            | (a_int != a[:, :, :-2, 1:-1]) # top neighbour
        )  # (B, 1, H-2, W-2) bool

        return r_interior, is_interface, a_int

    def forward(self, *, pred, coords=None, fx=None, return_aux=False, **_unused):
        if fx is None:
            raise ValueError("fx (permeability) is required for DarcyDefectCorrectionConstraint")
        if pred.ndim != 3 or pred.shape[-1] != 1:
            raise ValueError(
                f"Expected backbone output (B, N, 1), got {tuple(pred.shape)!r}"
            )
        if fx.ndim != 3 or fx.shape[-1] != 1:
            raise ValueError(
                f"Expected permeability input (B, N, 1), got {tuple(fx.shape)!r}"
            )

        height, width = self._grid_shape()
        dy, dx = self._spacing(height, width)

        # Decode permeability to physical values and reshape to grid
        a = reshape_channels_last_to_grid(
            self._decode_permeability(fx), shapelist=(height, width)
        )  # (B, 1, H, W)

        # Decode backbone prediction and reshape to grid (no boundary ansatz)
        u_phys = self._decode_pressure(pred)                                          # (B, N, 1)
        u = reshape_channels_last_to_grid(u_phys, shapelist=(height, width))          # (B,1,H,W)

        # Iterative defect correction
        delta_u_last = None
        for _ in range(self.n_correction_steps):
            r_int, _is_intf, a_int = self._darcy_residual(u, a, dy=dy, dx=dx)
            rhs = torch.zeros_like(u)
            rhs[:, :, 1:-1, 1:-1] = r_int / a_int.clamp_min(self.permeability_eps)
            delta_u_last = sine_poisson_solve_dirichlet_2d(rhs, dy=dy, dx=dx)
            u = u + delta_u_last

        u_encoded = self._encode_pressure(reshape_grid_to_channels_last(u))

        if not return_aux:
            return u_encoded

        # Diagnostics on the final corrected output
        r_int, is_intf, _ = self._darcy_residual(u, a, dy=dy, dx=dx)
        r_abs = r_int.abs()

        bulk_mask = ~is_intf
        n_bulk = bulk_mask.float().sum().clamp_min(1.0)
        n_intf = is_intf.float().sum().clamp_min(1.0)
        r_bulk_mean = (r_abs * bulk_mask.float()).sum() / n_bulk
        r_intf_mean = (r_abs * is_intf.float()).sum() / n_intf

        diagnostics: dict[str, ConstraintDiagnostic] = {
            "constraint/darcy_res_abs_mean": ConstraintDiagnostic(
                value=r_abs.mean(), reduce="mean"
            ),
            "constraint/darcy_res_abs_max": ConstraintDiagnostic(
                value=r_abs.max(), reduce="max"
            ),
            "constraint/darcy_res_bulk_abs_mean": ConstraintDiagnostic(
                value=r_bulk_mean, reduce="mean"
            ),
            "constraint/darcy_res_intf_abs_mean": ConstraintDiagnostic(
                value=r_intf_mean, reduce="mean"
            ),
        }
        if delta_u_last is not None:
            diagnostics["constraint/correction_norm_mean"] = ConstraintDiagnostic(
                value=delta_u_last.abs().mean(), reduce="mean"
            )

        return self.as_output(
            u_encoded,
            aux={"pred_base": u_phys},
            diagnostics=diagnostics,
        )
