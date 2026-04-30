from omni_hc.core import Registry

from .base import BenchmarkAdapter, BenchmarkSpec

BENCHMARKS = Registry[BenchmarkAdapter]("benchmarks")

from . import darcy, elasticity, navier_stokes, pipe  # noqa: F401

__all__ = ["BENCHMARKS", "BenchmarkAdapter", "BenchmarkSpec"]
