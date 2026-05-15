from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


@dataclass
class MediaLogContext:
    """Everything a benchmark or constraint log function might need."""

    pred: torch.Tensor
    target: torch.Tensor
    coords: torch.Tensor        # spatial coords, decoded if x_normalizer was used
    fx: torch.Tensor | None     # input field/coefficient, decoded; None for point-cloud benchmarks
    aux_tensors: dict[str, Any]
    meta: dict[str, Any]        # shapelist, out_dim, t_out, geotype, …
    cfg: dict[str, Any]         # full run config (wandb_logging.point_size, fps, …)
    prefix: str                 # "validation" or "test"
    epoch: int
    step: int | None
    out_dir: Path | None        # None → W&B only; Path → save files there


@dataclass(frozen=True)
class BenchmarkAdapter:
    name: str
    domain: str
    train_fn: Callable[..., Any]
    test_fn: Callable[..., Any]
    tune_fn: Callable[..., Any] | None = None
    log_fn: Callable[[MediaLogContext], dict[str, str]] | None = None

    def train(self, cfg: dict, *, device: torch.device) -> Any:
        return self.train_fn(cfg, device=device, log_fn=self.log_fn)

    def test(
        self,
        cfg: dict,
        *,
        device: torch.device,
        checkpoint_path: str | Path | None = None,
    ) -> Any:
        return self.test_fn(
            cfg, device=device, checkpoint_path=checkpoint_path, log_fn=self.log_fn
        )

    def tune(self, cfg: dict, *, device: torch.device) -> Any:
        if self.tune_fn is None:
            raise NotImplementedError(
                f"Benchmark '{self.name}' does not define a tuning entrypoint."
            )
        return self.tune_fn(cfg, device=device)
