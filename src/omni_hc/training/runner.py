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


def train_benchmark(
    cfg: dict, *, nsl_root: str | Path | None, device: torch.device
):
    adapter = get_benchmark_adapter(cfg)
    return adapter.train(cfg, nsl_root=nsl_root, device=device)


def test_benchmark(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
    checkpoint_path: str | Path | None = None,
):
    adapter = get_benchmark_adapter(cfg)
    return adapter.test(
        cfg,
        nsl_root=nsl_root,
        device=device,
        checkpoint_path=checkpoint_path,
    )


def tune_benchmark(
    cfg: dict, *, nsl_root: str | Path | None, device: torch.device
):
    adapter = get_benchmark_adapter(cfg)
    return adapter.tune(cfg, nsl_root=nsl_root, device=device)


get_benchmark_runtime = get_benchmark_adapter
