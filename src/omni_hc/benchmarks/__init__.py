from omni_hc.benchmarks.base import BenchmarkAdapter

# Defined before any adapter imports so runner.py can import this name without
# hitting a circular-import error mid-initialisation. Populated below.
BENCHMARKS: dict[str, BenchmarkAdapter] = {}

from .darcy.adapter import test as _darcy_test, train as _darcy_train, tune as _darcy_tune
from .elasticity.adapter import test as _elast_test, train as _elast_train, tune as _elast_tune
from .navier_stokes.adapter import test as _ns_test, train as _ns_train, tune as _ns_tune
from .pipe.adapter import test as _pipe_test, train as _pipe_train, tune as _pipe_tune
from .plasticity.adapter import test as _plas_test, train as _plas_train, tune as _plas_tune

for _adapter in [
    BenchmarkAdapter(
        name="darcy_2d",
        domain="steady-state porous media flow on a structured 2D grid",
        train_fn=_darcy_train,
        test_fn=_darcy_test,
        tune_fn=_darcy_tune,
    ),
    BenchmarkAdapter(
        name="elasticity_2d",
        domain="scalar stress prediction on an unstructured 2D elasticity point cloud",
        train_fn=_elast_train,
        test_fn=_elast_test,
        tune_fn=_elast_tune,
    ),
    BenchmarkAdapter(
        name="navier_stokes_2d",
        domain="incompressible viscous flow on a periodic 2D grid",
        train_fn=_ns_train,
        test_fn=_ns_test,
        tune_fn=_ns_tune,
    ),
    BenchmarkAdapter(
        name="pipe_2d",
        domain="steady pipe flow on a structured 2D grid",
        train_fn=_pipe_train,
        test_fn=_pipe_test,
        tune_fn=_pipe_tune,
    ),
    BenchmarkAdapter(
        name="plasticity_2d",
        domain="dynamic plastic forging response on a structured 2D grid",
        train_fn=_plas_train,
        test_fn=_plas_test,
        tune_fn=_plas_tune,
    ),
]:
    BENCHMARKS[_adapter.name] = _adapter

__all__ = ["BENCHMARKS", "BenchmarkAdapter"]
