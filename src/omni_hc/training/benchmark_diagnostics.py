"""Helper that wires :mod:`omni_hc.constraints.metrics` into the test/eval path.

The metric functions are pure ``(pred, batch, meta) -> dict[str, ConstraintDiagnostic]``;
the eval loops don't carry the full ``batch`` dict (they receive
``(coords, fx, target)`` from ``prepare_batch``), so this helper packages
the available tensors back into the shape the metric functions expect.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from omni_hc.constraints.base import ConstraintDiagnostic
from omni_hc.constraints.metrics import BENCHMARK_METRICS


def make_benchmark_diagnostic_fn(
    cfg: dict, meta: dict
) -> Callable[..., dict[str, ConstraintDiagnostic]] | None:
    """Return a per-batch diagnostic callable, or ``None`` if no metric is
    registered for this benchmark.

    The returned callable takes ``(pred, coords, fx, **extra)`` and dispatches
    to the appropriate ``BENCHMARK_METRICS`` function.
    """
    benchmark_name = str((cfg.get("benchmark") or {}).get("name") or "")
    fn = BENCHMARK_METRICS.get(benchmark_name)
    if fn is None:
        return None

    diagnostic_meta = dict(meta)
    diagnostics_cfg = cfg.get("diagnostics") or {}
    constraint_cfg = cfg.get("constraint") or {}
    for key in (
        "x_left",
        "x_right",
        "y_top",
        "y_bottom",
        "y_bottom_min",
        "y_bottom_max",
        "top_height",
        "die_speed",
        "time_duration",
        "top_envelope_tolerance",
        "bottom_boundary_tolerance",
        "below_y_bottom_tolerance",
        "collapse_spacing_threshold",
        "top_collapse_rows",
    ):
        if key in diagnostics_cfg:
            diagnostic_meta[key] = diagnostics_cfg[key]
        if key in constraint_cfg:
            diagnostic_meta[key] = constraint_cfg[key]

    def call(
        pred: torch.Tensor,
        coords: torch.Tensor | None = None,
        fx: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        **extra: Any,
    ) -> dict[str, ConstraintDiagnostic]:
        batch = {"coords": coords, "x": fx, "target": target, **extra}
        return fn(pred, batch, diagnostic_meta)

    return call
