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

    def call(
        pred: torch.Tensor,
        coords: torch.Tensor | None = None,
        fx: torch.Tensor | None = None,
        **extra: Any,
    ) -> dict[str, ConstraintDiagnostic]:
        batch = {"coords": coords, "x": fx, **extra}
        return fn(pred, batch, meta)

    return call
