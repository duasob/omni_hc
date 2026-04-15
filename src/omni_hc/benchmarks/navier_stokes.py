from __future__ import annotations

from . import BENCHMARKS
from .base import BenchmarkSpec

NAVIER_STOKES_2D = BenchmarkSpec(
    name="navier_stokes_2d",
    domain="incompressible viscous flow on a periodic 2D grid",
    dataset_family="fno",
    primary_invariant="global vorticity preservation",
    example_config="configs/benchmarks/navier_stokes/demo.yaml",
    notes=(
        "First migration target from hc_fluid. The initial hard constraint is a "
        "mean-preserving correction wrapper attached to the backbone with a forward hook."
    ),
)

BENCHMARKS.register(NAVIER_STOKES_2D.name, NAVIER_STOKES_2D)

