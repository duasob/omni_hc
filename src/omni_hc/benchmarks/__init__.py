from omni_hc.core import Registry

from .base import BenchmarkSpec

BENCHMARKS = Registry[BenchmarkSpec]("benchmarks")

from . import navier_stokes  # noqa: F401

__all__ = ["BENCHMARKS", "BenchmarkSpec"]

