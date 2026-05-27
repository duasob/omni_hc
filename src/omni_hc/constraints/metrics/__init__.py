"""Benchmark-level constraint-error metrics.

Pure functions that compute the constraint-error metrics defined in the
report's methodology table (chapter 5) from a prediction tensor and the
batch it came from. Decoupled from the constraint enforcement mechanism:
they fire identically for hard-constrained models (where values should be
machine epsilon) and for unconstrained baselines (where values measure
the violation).

Each benchmark module exports a single ``compute`` function::

    compute(pred, batch, meta) -> dict[str, ConstraintDiagnostic]

The dispatch table ``BENCHMARK_METRICS`` maps ``cfg["benchmark"]["name"]``
to the appropriate compute function.
"""

from __future__ import annotations

from typing import Callable

from . import darcy, elasticity, navier_stokes, pipe, plasticity


BENCHMARK_METRICS: dict[str, Callable] = {
    "navier_stokes_2d": navier_stokes.compute,
    "darcy_2d": darcy.compute,
    "pipe_2d": pipe.compute,
    "elasticity_2d": elasticity.compute,
    "plasticity_2d": plasticity.compute,
}


__all__ = ["BENCHMARK_METRICS"]
