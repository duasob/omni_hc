from __future__ import annotations

from omni_hc.benchmarks import BENCHMARKS
from omni_hc.benchmarks.base import BenchmarkAdapter, BenchmarkSpec

from .adapter import test, train, tune

PIPE_2D_SPEC = BenchmarkSpec(
    name="pipe_2d",
    domain="steady pipe flow on a structured 2D grid",
    dataset_family="fno",
    primary_invariant="unconstrained field response baseline",
    example_config="configs/benchmarks/pipe/base.yaml",
    notes=(
        "Initial pipe integration is a plain FNO baseline with no hard constraints. "
        "It mirrors the upstream NSL/FNO pipe setup closely enough to exercise the "
        "shared OmniHC steady-task runtime."
    ),
)

PIPE_2D = BenchmarkAdapter(
    spec=PIPE_2D_SPEC,
    train_fn=train,
    test_fn=test,
    tune_fn=tune,
)

BENCHMARKS.register(PIPE_2D.spec.name, PIPE_2D)
