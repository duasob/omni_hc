from __future__ import annotations

from omni_hc.benchmarks import BENCHMARKS
from omni_hc.benchmarks.base import BenchmarkAdapter, BenchmarkSpec

from .adapter import test, train, tune

NAVIER_STOKES_2D_SPEC = BenchmarkSpec(
    name="navier_stokes_2d",
    domain="incompressible viscous flow on a periodic 2D grid",
    dataset_family="fno",
    primary_invariant="global vorticity preservation",
    example_config="configs/benchmarks/navier_stokes/base.yaml",
    notes=(
        "First migration target from hc_fluid. The initial hard constraint is a "
        "mean-preserving correction wrapper attached to the backbone with a forward hook."
    ),
)

NAVIER_STOKES_2D = BenchmarkAdapter(
    spec=NAVIER_STOKES_2D_SPEC,
    train_fn=train,
    test_fn=test,
    tune_fn=tune,
)

BENCHMARKS.register(NAVIER_STOKES_2D.spec.name, NAVIER_STOKES_2D)
