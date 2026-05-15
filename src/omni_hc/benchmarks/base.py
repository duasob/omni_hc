from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    domain: str
    train_fn: Callable[..., Any]
    test_fn: Callable[..., Any]
    tune_fn: Callable[..., Any] | None = None

    def train(self, cfg: dict, *, device: torch.device) -> Any:
        return self.train_fn(cfg, device=device)

    def test(
        self,
        cfg: dict,
        *,
        device: torch.device,
        checkpoint_path: str | Path | None = None,
    ) -> Any:
        return self.test_fn(cfg, device=device, checkpoint_path=checkpoint_path)

    def tune(self, cfg: dict, *, device: torch.device) -> Any:
        if self.tune_fn is None:
            raise NotImplementedError(
                f"Benchmark '{self.name}' does not define a tuning entrypoint."
            )
        return self.tune_fn(cfg, device=device)
