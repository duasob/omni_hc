from __future__ import annotations

from pathlib import Path

import torch

from omni_hc.benchmarks import BENCHMARKS, BenchmarkAdapter


def get_benchmark_adapter(cfg: dict) -> BenchmarkAdapter:
    benchmark_cfg = cfg.get("benchmark", {}) or {}
    benchmark_name = benchmark_cfg.get("name")
    if not benchmark_name:
        raise KeyError("Config must define benchmark.name for runtime dispatch.")
    return BENCHMARKS.get(str(benchmark_name))


def train_benchmark(cfg: dict, *, device: torch.device):
    adapter = get_benchmark_adapter(cfg)
    return adapter.train(cfg, device=device)


def test_benchmark(
    cfg: dict,
    *,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
):
    adapter = get_benchmark_adapter(cfg)
    return adapter.test(cfg, device=device, checkpoint_path=checkpoint_path)


def tune_benchmark(cfg: dict, *, device: torch.device):
    adapter = get_benchmark_adapter(cfg)
    return adapter.tune(cfg, device=device)
