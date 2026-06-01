"""Compute Chapter 5 ground-truth data diagnostics.

Most model-test metrics are read directly from ``outputs/**/test_metrics.yaml``.
This command fills the report-only source that requires evaluating constraint
diagnostics on the underlying test-set targets:

    artifacts/report/metrics/ch5_gt_metrics.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omni_hc.benchmarks.darcy import adapter as darcy_adapter
from omni_hc.benchmarks.darcy.data import build_test_loader as build_darcy_test_loader
from omni_hc.benchmarks.navier_stokes import adapter as ns_adapter
from omni_hc.benchmarks.navier_stokes.data import build_test_loader as build_ns_test_loader
from omni_hc.benchmarks.pipe import adapter as pipe_adapter
from omni_hc.benchmarks.pipe.data import build_test_loader as build_pipe_test_loader
from omni_hc.benchmarks.plasticity import adapter as plasticity_adapter
from omni_hc.benchmarks.plasticity.data import build_test_loader as build_plasticity_test_loader
from omni_hc.constraints.metrics import BENCHMARK_METRICS
from omni_hc.core import load_yaml_file
from omni_hc.training.common import MetricAccumulator


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "ns_constrained": "outputs/navier_stokes/mean_constraint/transolver/final/seed_42",
    "darcy_baseline": "outputs/darcy/none/transolver/final/seed_42",
    "pipe_baseline": "outputs/pipe/none/transolver/final/seed_42",
    "plasticity_constrained": "outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/final/seed_42",
}


def _decode_if_needed(loader, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    normalizer = getattr(loader, "y_normalizer", None)
    if normalizer is None:
        return tensor
    return normalizer.to(device).decode(tensor)


def _decode_uy_if_needed(loader, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    normalizer = getattr(loader, "uy_normalizer", None)
    if normalizer is None:
        return tensor
    return normalizer.to(device).decode(tensor)


def _decode_x_if_needed(
    loader,
    tensor: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    normalizer = getattr(loader, "x_normalizer", None)
    if normalizer is None:
        return tensor
    return normalizer.to(device).decode(tensor)


def _target_for_metric(benchmark: str, target: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    if benchmark == "navier_stokes_2d":
        b, n, c = target.shape
        t_out = int(meta["t_out"])
        out_dim = int(meta["out_dim"])
        return target.reshape(b, n, t_out, out_dim).permute(0, 2, 1, 3)
    return target


def _accumulate_gt(
    *,
    benchmark: str,
    cfg_path: str,
    build_loader: Callable[[dict], Any],
    get_meta: Callable[[Any], dict],
    batch_tensors: Callable[[dict[str, torch.Tensor], torch.device], tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    cfg = load_yaml_file(REPO_ROOT / cfg_path)
    loader = build_loader(cfg)
    meta = get_meta(loader)
    metric_fn = BENCHMARK_METRICS[benchmark]
    acc = MetricAccumulator()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        coords, fx, target = batch_tensors(batch, device)
        fx = _decode_x_if_needed(loader, fx, device)
        target = _decode_if_needed(loader, target.to(device), device)
        pred = _target_for_metric(benchmark, target, meta)
        diagnostics = metric_fn(pred, {"coords": coords, "x": fx}, meta)
        acc.update(diagnostics, weight=int(target.shape[0]))
    return acc.compute()


def _accumulate_pipe_gt(device: torch.device, max_batches: int | None) -> dict[str, float]:
    cfg = load_yaml_file(REPO_ROOT / RUNS["pipe_baseline"] / "resolved_config.yaml")
    cfg.setdefault("data", {})["load_uy"] = True
    loader = build_pipe_test_loader(cfg)
    meta = pipe_adapter._get_meta(loader)
    metric_fn = BENCHMARK_METRICS["pipe_2d"]
    acc = MetricAccumulator()
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        coords = batch["coords"].to(device)
        fx = batch["x"].to(device)
        ux = _decode_if_needed(loader, batch["y"].to(device), device)
        uy = _decode_uy_if_needed(loader, batch["y_uy"].to(device), device)
        pred = torch.cat([ux, uy], dim=-1)
        diagnostics = metric_fn(pred, {"coords": coords, "x": fx}, meta)
        acc.update(diagnostics, weight=int(ux.shape[0]))
    return acc.compute()


def compute_gt_metrics(device: torch.device, *, max_batches: int | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    print("computing GT metrics: navier_stokes")
    metrics.update(
        _accumulate_gt(
            benchmark="navier_stokes_2d",
            cfg_path=f"{RUNS['ns_constrained']}/resolved_config.yaml",
            build_loader=build_ns_test_loader,
            get_meta=ns_adapter._get_meta,
            batch_tensors=lambda batch, dev: (None, batch["x"].to(dev), batch["y"].to(dev)),
            device=device,
            max_batches=max_batches,
        )
    )
    print("computing GT metrics: darcy")
    metrics.update(
        _accumulate_gt(
            benchmark="darcy_2d",
            cfg_path=f"{RUNS['darcy_baseline']}/resolved_config.yaml",
            build_loader=build_darcy_test_loader,
            get_meta=darcy_adapter._get_meta,
            batch_tensors=lambda batch, dev: (
                batch["coords"].to(dev),
                batch["x"].to(dev),
                batch["y"].to(dev),
            ),
            device=device,
            max_batches=max_batches,
        )
    )
    print("computing GT metrics: pipe")
    metrics.update(_accumulate_pipe_gt(device, max_batches))
    print("computing GT metrics: plasticity")
    metrics.update(
        _accumulate_gt(
            benchmark="plasticity_2d",
            cfg_path=f"{RUNS['plasticity_constrained']}/resolved_config.yaml",
            build_loader=build_plasticity_test_loader,
            get_meta=plasticity_adapter._get_meta,
            batch_tensors=lambda batch, dev: (
                batch["coords"].to(dev),
                batch["x"].to(dev),
                batch["y"].to(dev),
            ),
            device=device,
            max_batches=max_batches,
        )
    )
    return metrics


def write_metrics(path: Path, metrics: dict[str, float], *, warnings: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": metrics}
    if warnings:
        payload["warnings"] = warnings
    path.write_text(yaml.safe_dump(payload, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="reporting.compute_ch5")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report"))
    parser.add_argument("--device", default="cpu", help="cpu | cuda")
    parser.add_argument(
        "--max-gt-batches",
        type=int,
        default=None,
        help="Limit GT metric computation for quick smoke checks. Omit for the full test split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    gt_path = args.output_dir / "metrics" / "ch5_gt_metrics.yaml"
    write_metrics(
        gt_path,
        compute_gt_metrics(device, max_batches=args.max_gt_batches),
    )
    print(f"wrote {gt_path}")


if __name__ == "__main__":
    main()
