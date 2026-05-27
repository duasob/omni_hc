"""Constraint-error metrics for the Plasticity 2D benchmark.

Methodology table row:
    Plasticity / Mesh consistency -> ‖Δŝ < 0‖₀  (count of negative spacings)

The constrained model emits ``constraint/min_dx``, ``constraint/min_dy``,
``constraint/axis_order_margin_min``, etc. (see
``src/omni_hc/constraints/plasticity.py``). The methodology metric is the
fraction (or count) of *negative* spacings — distinct from a minimum,
since one violation among many would dominate the methodology metric but
not the minimum.

For an unconstrained baseline that predicts node coordinates (B, T, N, 2)
or similar, the residual can be computed by taking finite differences of
the predicted coords along the spatial axes and counting negatives. We
hold off implementing this until we confirm the pred-tensor layout
(dynamic-conditional task), so that the implementation is correct for
both rollout horizons and the baseline pred shape.
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
    del pred, batch, meta
    # TODO: once we confirm the pred shape for dynamic_conditional plasticity
    # (the (B, T, H, W, 2)-ish layout of predicted coords), implement
    # neg_spacing_count and neg_spacing_fraction. Reference math:
    # src/omni_hc/constraints/plasticity.py around the MeshConsistency module.
    return {}
