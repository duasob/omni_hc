"""Constraint-error metrics for the Elasticity 2D benchmark.

The plane-stress constraint reports 3D incompressibility and out-of-plane
stress residuals directly from its latent stretch field. An unconstrained
scalar-stress baseline has no deformation tensor, so neither residual can be
reconstructed from its prediction alone.

For now this returns an empty dict. The constrained-model diagnostics live in
``src/omni_hc/constraints/elasticity.py`` under
``constraint/full_det_f_abs_error_*`` and
``constraint/plane_stress_abs_error_*``.
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
    # TODO: surface the constraint module's residuals here once the test
    # pipeline passes auxiliary tensors to metric functions.
    return {}
