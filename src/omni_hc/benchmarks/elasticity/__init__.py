from __future__ import annotations

from omni_hc.benchmarks import BENCHMARKS
from omni_hc.benchmarks.base import BenchmarkAdapter, BenchmarkSpec

from .adapter import test, train, tune

ELASTICITY_2D_SPEC = BenchmarkSpec(
    name="elasticity_2d",
    domain="scalar stress prediction on an unstructured 2D elasticity point cloud",
    dataset_family="geo-fno",
    primary_invariant="positive scalar stress from an area-preserving stretch tensor",
    example_config="configs/benchmarks/elasticity/base.yaml",
    notes=(
        "Uses the upstream Neural-Solver-Library elasticity point-cloud setup: "
        "2D coordinates as input and one scalar sigma target per material point."
    ),
)

ELASTICITY_2D = BenchmarkAdapter(
    spec=ELASTICITY_2D_SPEC,
    train_fn=train,
    test_fn=test,
    tune_fn=tune,
)

BENCHMARKS.register(ELASTICITY_2D.spec.name, ELASTICITY_2D)
