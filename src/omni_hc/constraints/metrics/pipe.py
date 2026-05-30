"""Constraint-error metrics for the Pipe 2D benchmark.

Methodology table rows (primary, mean-based metrics):
    Pipe / Zero wall velocity  -> mean |û_x| at wall nodes
    Pipe / Parabolic inlet     -> mean |û_x|inlet - u_x_par|
    Pipe / Divergence-free     -> mean |∇·û|

(max / rmse / relative variants are also emitted as secondary diagnostics.)

Channel convention: pred channel 0 is u_x (streamwise velocity). For
out_dim == 1 (stream-function ansatz output) and for the unconstrained
baseline (out_dim == 3: u_x, u_y, p) this is consistent.

Wall indices: pipe meshes are curvilinear (H, W) grids with j=0 and
j=W-1 as the lower/upper walls; i=0 is the inlet, i=H-1 the outlet.
"""

from __future__ import annotations

from typing import Any

import torch

from ..base import ConstraintDiagnostic


def _reshape_to_grid(pred: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    h, w = grid_shape
    if pred.dim() == 2:
        pred = pred.unsqueeze(-1)
    b = pred.shape[0]
    c = pred.shape[-1] if pred.dim() == 3 else 1
    return pred.reshape(b, h, w, c)


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    """
    Args:
        pred: (B, H*W, C) — at minimum the streamwise velocity in channel 0.
        batch: provides ``coords`` (B, H*W, 2) — physical (x, y) coordinates,
            needed for curvilinear divergence and the inlet parabolic target.
        meta: provides ``shapelist`` = (H, W).
    """
    h, w = tuple(meta["shapelist"])
    grid = _reshape_to_grid(pred, (h, w))  # (B, H, W, C)
    ux = grid[..., 0]  # (B, H, W) — streamwise velocity

    wall_vals = torch.cat([ux[:, :, 0].abs(), ux[:, :, -1].abs()], dim=-1)  # (B, 2W)

    out: dict[str, ConstraintDiagnostic] = {
        "constraint/wall_abs_mean": ConstraintDiagnostic(
            value=wall_vals.mean(), reduce="mean"
        ),
        "constraint/wall_abs_max": ConstraintDiagnostic(
            value=wall_vals.max(), reduce="max"
        ),
    }

    # Inlet residual: u_x at i=0 vs parabolic target A * 4 * t * (1 - t).
    coords = batch.get("coords")
    if coords is not None:
        coords_grid = coords.reshape(coords.shape[0], h, w, 2)
        inlet_y = coords_grid[:, 0, :, 1]  # (B, W) transverse coord at inlet
        y_min = inlet_y.min(dim=-1, keepdim=True).values
        y_max = inlet_y.max(dim=-1, keepdim=True).values
        t = (inlet_y - y_min) / (y_max - y_min).clamp_min(1e-12)
        amplitude = float(meta.get("inlet_amplitude", 0.25))
        target = amplitude * 4.0 * t * (1.0 - t)  # (B, W)
        inlet_ux = ux[:, 0, :]  # (B, W)
        residual = (inlet_ux - target).abs()
        out["constraint/inlet_abs_mean"] = ConstraintDiagnostic(
            value=residual.mean(), reduce="mean"
        )
        out["constraint/inlet_abs_max"] = ConstraintDiagnostic(
            value=residual.max(), reduce="max"
        )
        residual2 = (inlet_ux - target).square()
        rmse_per_sample = residual2.mean(dim=-1).sqrt()
        out["constraint/inlet_rmse"] = ConstraintDiagnostic(
            value=rmse_per_sample.mean(), reduce="mean"
        )

    # Divergence ∇·u on the curvilinear (i, j) mesh via chain rule:
    #   ∂f/∂x = ( y_η f_ξ - y_ξ f_η) / det(J)
    #   ∂f/∂y = (-x_η f_ξ + x_ξ f_η) / det(J)
    # Needs u_y (channel 1) and physical coords from batch.
    coords = batch.get("coords")
    if coords is not None and grid.shape[-1] >= 2:
        coords_grid = coords.reshape(coords.shape[0], h, w, 2)
        x_phys = coords_grid[..., 0]  # (B, H, W)
        y_phys = coords_grid[..., 1]
        uy = grid[..., 1]  # (B, H, W)

        def _grad(t, dim):
            return torch.gradient(t, dim=dim)[0]

        x_xi = _grad(x_phys, 1)
        x_eta = _grad(x_phys, 2)
        y_xi = _grad(y_phys, 1)
        y_eta = _grad(y_phys, 2)
        det_J = x_xi * y_eta - x_eta * y_xi

        u_xi = _grad(ux, 1)
        u_eta = _grad(ux, 2)
        v_xi = _grad(uy, 1)
        v_eta = _grad(uy, 2)

        du_dx = (y_eta * u_xi - y_xi * u_eta) / det_J.clamp_min(1e-12)
        dv_dy = (-x_eta * v_xi + x_xi * v_eta) / det_J.clamp_min(1e-12)
        div = du_dx + dv_dy  # (B, H, W)
        div_abs = div.abs()
        out["constraint/div_abs_mean"] = ConstraintDiagnostic(
            value=div_abs.mean(), reduce="mean"
        )
        out["constraint/div_abs_max"] = ConstraintDiagnostic(
            value=div_abs.max(), reduce="max"
        )
        div2 = div.square().reshape(div.shape[0], -1)
        rmse_per_sample = div2.mean(dim=-1).sqrt()
        out["constraint/div_rmse"] = ConstraintDiagnostic(
            value=rmse_per_sample.mean(), reduce="mean"
        )

        # Relative divergence: fraction of the local gradient budget that fails
        # to cancel. Dimensionless, robust to mesh stretching and dataset scale.
        # Per sample: mean_grid(|div|) / mean_grid(|du/dx| + |dv/dy|), then mean
        # over samples.
        grad_budget = (du_dx.abs() + dv_dy.abs())
        num = div_abs.reshape(div_abs.shape[0], -1).mean(dim=-1)
        den = grad_budget.reshape(grad_budget.shape[0], -1).mean(dim=-1)
        rel_per_sample = num / den.clamp_min(1e-12)
        out["constraint/div_rel_mean"] = ConstraintDiagnostic(
            value=rel_per_sample.mean(), reduce="mean"
        )

    return out
