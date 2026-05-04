from __future__ import annotations

from omni_hc.benchmarks import BENCHMARKS
from omni_hc.benchmarks.base import BenchmarkAdapter, BenchmarkSpec

from .adapter import test, train, tune

PLASTICITY_2D_SPEC = BenchmarkSpec(
    name="plasticity_2d",
    domain="dynamic plastic forging response on a structured 2D grid",
    dataset_family="geo-fno",
    primary_invariant="unknown four-channel deformation/state response",
    example_config="configs/benchmarks/plasticity/base.yaml",
    notes=(
        "Initial plasticity integration mirrors NSL's dynamic_conditional setup: "
        "a 1D die profile is broadcast across a 101x31 grid and queried with a "
        "scalar time input for each of 20 output steps."
    ),
)

PLASTICITY_2D = BenchmarkAdapter(
    spec=PLASTICITY_2D_SPEC,
    train_fn=train,
    test_fn=test,
    tune_fn=tune,
)

BENCHMARKS.register(PLASTICITY_2D.spec.name, PLASTICITY_2D)
