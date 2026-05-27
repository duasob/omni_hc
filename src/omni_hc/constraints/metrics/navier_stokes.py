"""Constraint-error metrics for the Navier-Stokes benchmark.

Methodology table row:
    NS / Global vorticity mean -> |ω̄_pred|

The dataset target ``y`` is the vorticity field, so a model trained to
predict ``y`` is predicting vorticity directly. The constraint is that
the *spatial mean* of the predicted vorticity is identically zero. The
error is the absolute spatial mean.
"""

from __future__ import annotations

from typing import Any

import torch

from ..base import ConstraintDiagnostic


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    """
    Args:
        pred: (B, N, C) or (B, T, N, C) — vorticity field flattened over space.
            For autoregressive rollouts, the leading T axis is the rollout
            horizon; we measure the mean per (sample, timestep, channel) and
            reduce across all of them.
        batch, meta: unused (kept for signature parity).

    Returns:
        ``constraint/vorticity_abs_mean`` and ``constraint/vorticity_abs_max``
        — both measure |ω̄|; the "_max" variant takes the worst sample's
        mean rather than max-abs of values, since the constraint is on the
        spatial mean, not on a residual field.
    """
    del batch, meta

    # per-sample spatial mean.
    spatial_mean = pred.mean(dim=-2)  # (B, ...) or (B, T, C)
    abs_mean = spatial_mean.abs()

    return {
        "constraint/vorticity_abs_mean": ConstraintDiagnostic(
            value=abs_mean.mean(),  # mean over all samples and timesteps.
            reduce="mean",
        ),
        "constraint/vorticity_abs_max": ConstraintDiagnostic(
            value=abs_mean.max(),  # max over all samples and timesteps.
            reduce="max",
        ),
    }
