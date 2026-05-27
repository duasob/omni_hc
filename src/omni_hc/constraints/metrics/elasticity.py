"""Constraint-error metrics for the Elasticity 2D benchmark.

Methodology table row:
    Elasticity / Incompressibility -> |det Ĉ - 1|_max

Important asymmetry: the *constrained* model emits the right Cauchy-Green
tensor C as part of its forward pass (DeviatoricStressConstraint), and
|det C - 1| is directly measurable. The *unconstrained* baseline outputs
a raw stress tensor (no C), so the incompressibility residual cannot be
computed from pred alone. The methodology table marks this row "/" for
the baseline; we propagate that by emitting no diagnostics when C is
absent from the batch/aux.

For now this returns an empty dict — the constrained-model diagnostics
already live in ``src/omni_hc/constraints/elasticity.py`` under the keys
``constraint/det_c_abs_error_{mean,max}``. We can re-export them here once
the test pipeline integration is in place, so the registry doesn't need
to special-case elasticity.
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
    # TODO: once test-pipeline integration lands, surface
    # constraint/det_c_abs_error_{mean,max} from the constraint module's
    # aux tensors here so the registry sees a single canonical key.
    return {}
