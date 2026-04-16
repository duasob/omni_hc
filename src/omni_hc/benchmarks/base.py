from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    domain: str
    dataset_family: str
    primary_invariant: str
    example_config: str
    notes: str = ""


BenchmarkHandler = Callable[..., Any]


@dataclass(frozen=True)
class BenchmarkAdapter:
    spec: BenchmarkSpec
    train_fn: BenchmarkHandler
    test_fn: BenchmarkHandler
    tune_fn: BenchmarkHandler | None = None

    def train(
        self, cfg: dict, *, nsl_root: str | Path | None, device: torch.device
    ) -> Any:
        return self.train_fn(cfg, nsl_root=nsl_root, device=device)

    def test(
        self,
        cfg: dict,
        *,
        nsl_root: str | Path | None,
        device: torch.device,
        checkpoint_path: str | Path | None = None,
    ) -> Any:
        return self.test_fn(
            cfg,
            nsl_root=nsl_root,
            device=device,
            checkpoint_path=checkpoint_path,
        )

    def tune(
        self, cfg: dict, *, nsl_root: str | Path | None, device: torch.device
    ) -> Any:
        if self.tune_fn is None:
            raise NotImplementedError(
                f"Benchmark '{self.spec.name}' does not define a tuning entrypoint."
            )
        return self.tune_fn(cfg, nsl_root=nsl_root, device=device)


BenchmarkRuntime = BenchmarkAdapter
