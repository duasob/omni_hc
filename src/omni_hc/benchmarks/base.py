from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    domain: str
    dataset_family: str
    primary_invariant: str
    example_config: str
    notes: str = ""

