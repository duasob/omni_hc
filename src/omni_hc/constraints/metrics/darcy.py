"""Constraint-error metrics for the Darcy 2D benchmark.

Methodology table rows:
    Darcy / Dirichlet strict       -> ‖û|∂Ω‖∞
    Darcy / Dirichlet corner-only  -> ‖û|corners‖∞
    Darcy / Darcy residual         -> ‖∇·(-a∇û) - 1‖₂
"""

from __future__ import annotations

from typing import Any

import torch

from ..base import ConstraintDiagnostic
from ..utils.spectral import (
    finite_difference_divergence_2d,
    finite_difference_gradient_2d,
)


def _reshape_to_grid(pred: torch.Tensor, grid_shape: tuple[int, int]) -> torch.Tensor:
    """(B, H*W, C) -> (B, H, W, C). Tolerates missing channel dim."""
    h, w = grid_shape
    if pred.dim() == 2:
        pred = pred.unsqueeze(-1)
    b = pred.shape[0]
    c = pred.shape[-1] if pred.dim() == 3 else 1
    return pred.reshape(b, h, w, c)


def _boundary_mask(h: int, w: int, device, dtype) -> torch.Tensor:
    mask = torch.zeros((h, w), device=device, dtype=dtype)
    mask[0, :] = 1
    mask[-1, :] = 1
    mask[:, 0] = 1
    mask[:, -1] = 1
    return mask  # (H, W)


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    """
    Args:
        pred: (B, H*W, C=1) pressure field (or scalar field — the target ``y``).
        batch: dict with at least ``"x"`` (input, including the permeability field
            ``k`` as one of its channels) — only used by the flux metric.
        meta: provides ``shapelist`` = (H, W).

    Returns:
        ``constraint/dirichlet_strict_{mean,max}``,
        ``constraint/dirichlet_corner_{mean,max}``.
        ``constraint/darcy_res_{abs_mean,abs_max,rmse}``.
    """
    h, w = tuple(meta["shapelist"])
    grid = _reshape_to_grid(pred, (h, w))  # (B, H, W, C)
    # Take channel 0: Darcy pred is scalar pressure.
    field = grid[..., 0].abs()  # (B, H, W)

    boundary_mask = _boundary_mask(h, w, device=field.device, dtype=torch.bool)
    boundary_vals = field[:, boundary_mask]  # (B, n_boundary)

    corner_vals = torch.stack(
        [field[:, 0, 0], field[:, 0, -1], field[:, -1, 0], field[:, -1, -1]],
        dim=-1,
    )  # (B, 4)

    out: dict[str, ConstraintDiagnostic] = {
        "constraint/dirichlet_strict_mean": ConstraintDiagnostic(
            value=boundary_vals.mean(), reduce="mean"
        ),
        "constraint/dirichlet_strict_max": ConstraintDiagnostic(
            value=boundary_vals.max(), reduce="max"
        ),
        "constraint/dirichlet_corner_mean": ConstraintDiagnostic(
            value=corner_vals.mean(), reduce="mean"
        ),
        "constraint/dirichlet_corner_max": ConstraintDiagnostic(
            value=corner_vals.max(), reduce="max"
        ),
    }

    # Pressure-induced Darcy residual: ∇·(-k ∇û) - 1.
    fx = batch.get("x")
    if fx is not None:
        k = _reshape_to_grid(fx, (h, w))[..., 0]  # (B, H, W) permeability (channel 0)
        pressure = grid[..., 0]  # (B, H, W) unsigned pred
        # Convert to NCHW for the spectral helpers.
        pressure_nchw = pressure.unsqueeze(1)  # (B, 1, H, W)
        k_nchw = k.unsqueeze(1)  # (B, 1, H, W)
        dy = 1.0 / max(h - 1, 1)
        dx_ = 1.0 / max(w - 1, 1)
        grad = finite_difference_gradient_2d(pressure_nchw, dy=dy, dx=dx_)
        flux = -k_nchw * grad  # (B, 2, H, W) -- [v_x, v_y]
        div = finite_difference_divergence_2d(flux, dy=dy, dx=dx_)  # (B, 1, H, W)
        residual = (div - 1.0).abs()  # source term = 1
        out["constraint/darcy_res_abs_mean"] = ConstraintDiagnostic(
            value=residual.mean(), reduce="mean"
        )
        out["constraint/darcy_res_abs_max"] = ConstraintDiagnostic(
            value=residual.max(), reduce="max"
        )
        residual2 = (div - 1.0).square().reshape(div.shape[0], -1)
        rmse_per_sample = residual2.mean(dim=-1).sqrt()
        out["constraint/darcy_res_rmse"] = ConstraintDiagnostic(
            value=rmse_per_sample.mean(), reduce="mean"
        )

        # Backwards-compatible aliases for existing report artifacts. New report
        # rows should prefer the darcy_res_* keys to avoid confusing this
        # pressure-induced residual with the flux constraint's constructed flux.
        out["constraint/flux_abs_mean"] = ConstraintDiagnostic(
            value=residual.mean(), reduce="mean"
        )
        out["constraint/flux_abs_max"] = ConstraintDiagnostic(
            value=residual.max(), reduce="max"
        )
        out["constraint/flux_rmse"] = ConstraintDiagnostic(
            value=rmse_per_sample.mean(), reduce="mean"
        )

    return out
