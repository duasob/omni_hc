from __future__ import annotations

from omni_hc.benchmarks import BENCHMARKS
from omni_hc.benchmarks.base import BenchmarkAdapter, BenchmarkSpec

from .adapter import test, train, tune

DARCY_2D_SPEC = BenchmarkSpec(
    name="darcy_2d",
    domain="steady-state porous media flow on a structured 2D grid",
    dataset_family="fno",
    primary_invariant="boundary-aware pressure field",
    example_config="configs/benchmarks/darcy/base.yaml",
    notes=(
        "Initial Darcy integration uses the standard FNO benchmark pair with a "
        "generic steady-state training loop. Hard constraints can be layered on top later."
    ),
)

DARCY_2D = BenchmarkAdapter(
    spec=DARCY_2D_SPEC,
    train_fn=train,
    test_fn=test,
    tune_fn=tune,
)

BENCHMARKS.register(DARCY_2D.spec.name, DARCY_2D)
