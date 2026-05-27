"""Compute auxiliary generated metrics used by the Chapter 5 report macros.

Most model-test metrics are read directly from ``outputs/**/test_metrics.yaml``.
This command fills report-only sources that require extra computation:

    artifacts/report/metrics/ch5_gt_metrics.yaml
    artifacts/report/metrics/ch5_cost_metrics.yaml
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from omni_hc.benchmarks.darcy import adapter as darcy_adapter
from omni_hc.benchmarks.darcy.data import build_test_loader as build_darcy_test_loader
from omni_hc.benchmarks.elasticity import adapter as elasticity_adapter
from omni_hc.benchmarks.elasticity.data import build_test_loader as build_elasticity_test_loader
from omni_hc.benchmarks.navier_stokes import adapter as ns_adapter
from omni_hc.benchmarks.navier_stokes.data import build_test_loader as build_ns_test_loader
from omni_hc.benchmarks.pipe import adapter as pipe_adapter
from omni_hc.benchmarks.pipe.data import build_test_loader as build_pipe_test_loader
from omni_hc.benchmarks.plasticity import adapter as plasticity_adapter
from omni_hc.benchmarks.plasticity.data import build_test_loader as build_plasticity_test_loader
from omni_hc.constraints.metrics import BENCHMARK_METRICS
from omni_hc.core import load_yaml_file
from omni_hc.integrations.nsl.modeling import create_model
from omni_hc.training.common import MetricAccumulator, forward_with_optional_aux


REPO_ROOT = Path(__file__).resolve().parents[2]

RUNS = {
    "ns_baseline": "outputs/navier_stokes/none/galerkin_transformer/final/seed_42",
    "ns_constrained": "outputs/navier_stokes/mean_constraint/transolver/final/seed_42",
    "darcy_baseline": "outputs/darcy/none/transolver/final/seed_42",
    "darcy_dirichlet": "outputs/darcy/dirichlet_ansatz_zero/transolver/final/seed_42",
    "darcy_flux": "outputs/darcy/darcy_flux_constraint/transolver/final/seed_42",
    "pipe_baseline": "outputs/pipe/none/transolver/final/seed_42",
    "pipe_stream": "outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/final/seed_42",
    "elasticity_constrained": "outputs/elasticity/elasticity_deviatoric_stress_constraint/transolver/final/seed_42",
    "plasticity_constrained": "outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/final/seed_42",
    "plasticity_baseline": "outputs/plasticity/none/transolver/final/seed_42",
    "plasticity_baseline_smoke": "outputs/plasticity/none/transolver/smoke/seed_42",
}


def _decode_if_needed(loader, tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    normalizer = getattr(loader, "y_normalizer", None)
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
        target = _decode_if_needed(loader, target.to(device), device)
        pred = _target_for_metric(benchmark, target, meta)
        diagnostics = metric_fn(pred, {"coords": coords, "x": fx}, meta)
        acc.update(diagnostics, weight=int(target.shape[0]))
    return acc.compute()


def compute_gt_metrics(device: torch.device, *, max_batches: int | None = None) -> dict[str, float]:
    metrics: dict[str, float] = {}
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
    metrics.update(
        _accumulate_gt(
            benchmark="pipe_2d",
            cfg_path=f"{RUNS['pipe_baseline']}/resolved_config.yaml",
            build_loader=build_pipe_test_loader,
            get_meta=pipe_adapter._get_meta,
            batch_tensors=lambda batch, dev: (
                batch["coords"].to(dev),
                batch["x"].to(dev),
                batch["y"].to(dev),
            ),
            device=device,
            max_batches=max_batches,
        )
    )
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


def _local_cfg(run_key: str) -> dict:
    cfg = load_yaml_file(REPO_ROOT / RUNS[run_key] / "resolved_config.yaml")
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("backend", {})["nsl_root"] = str(REPO_ROOT / "external" / "Neural-Solver-Library")
    return cfg


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _flops_for_forward(fn: Callable[[], Any], device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            fn()
    return int(sum(getattr(evt, "flops", 0) or 0 for evt in prof.key_averages()))


def _strip_constraint(cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["constraint"] = {}
    return out


def _first_batch(loader, device: torch.device):
    return next(iter(loader))


def _profile_pair(
    *,
    run_key: str,
    build_loader: Callable[[dict], Any],
    get_meta: Callable[[Any], dict],
    runtime_overrides: Callable[[dict], dict],
    make_forward: Callable[[torch.nn.Module, dict[str, torch.Tensor], torch.device, dict], Callable[[], Any]],
    device: torch.device,
) -> tuple[int, int]:
    cfg_hc = _local_cfg(run_key)
    cfg_bl = _strip_constraint(cfg_hc)
    loader = build_loader(cfg_hc)
    meta = get_meta(loader)
    batch = _first_batch(loader, device)
    hc, *_ = create_model(cfg_hc, device=device, runtime_overrides=runtime_overrides(meta))
    bl, *_ = create_model(cfg_bl, device=device, runtime_overrides=runtime_overrides(meta))
    hc.eval()
    bl.eval()
    param_delta = _count_trainable_params(hc) - _count_trainable_params(bl)
    hc_flops = _flops_for_forward(make_forward(hc, batch, device, meta), device)
    bl_flops = _flops_for_forward(make_forward(bl, batch, device, meta), device)
    return int(param_delta), int(hc_flops - bl_flops)


def compute_cost_metrics(device: torch.device) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    def steady_forward(model, batch, dev, meta):
        del meta
        coords = batch["coords"].to(dev)
        fx = batch.get("x")
        fx = None if fx is None else fx.to(dev)
        return lambda: forward_with_optional_aux(model, coords, fx)

    def ns_forward(model, batch, dev, meta):
        fx = batch["x"].to(dev)
        h, w = tuple(meta["shapelist"])
        from omni_hc.benchmarks.navier_stokes.data import make_grid

        coords = make_grid(h, w, device=dev, dtype=fx.dtype).unsqueeze(0).repeat(fx.shape[0], 1, 1)
        return lambda: forward_with_optional_aux(model, coords, fx)

    def plasticity_forward(model, batch, dev, meta):
        del meta
        coords = batch["coords"].to(dev)
        fx = batch["x"].to(dev)
        time = batch["time"].to(dev)
        return lambda: forward_with_optional_aux(model, coords, fx, T=time[:, :1])

    specs = [
        ("ns", "ns_constrained", build_ns_test_loader, ns_adapter._get_meta, ns_adapter._runtime_overrides, ns_forward),
        ("darcy_dirichlet", "darcy_dirichlet", build_darcy_test_loader, darcy_adapter._get_meta, darcy_adapter._runtime_overrides, steady_forward),
        ("darcy_flux", "darcy_flux", build_darcy_test_loader, darcy_adapter._get_meta, darcy_adapter._runtime_overrides, steady_forward),
        ("pipe_stream", "pipe_stream", build_pipe_test_loader, pipe_adapter._get_meta, pipe_adapter._runtime_overrides, steady_forward),
        ("elasticity", "elasticity_constrained", build_elasticity_test_loader, elasticity_adapter._get_meta, elasticity_adapter._runtime_overrides, steady_forward),
        ("plasticity", "plasticity_constrained", build_plasticity_test_loader, plasticity_adapter._get_meta, plasticity_adapter._runtime_overrides, plasticity_forward),
    ]
    for prefix, run_key, build_loader, get_meta, runtime_overrides, make_forward in specs:
        try:
            params, flops = _profile_pair(
                run_key=run_key,
                build_loader=build_loader,
                get_meta=get_meta,
                runtime_overrides=runtime_overrides,
                make_forward=make_forward,
                device=device,
            )
        except Exception as exc:
            warnings.append(f"{prefix}: profiler failed ({exc}); emitted 0 cost placeholders")
            params, flops = 0, 0
        metrics[f"{prefix}/params_overhead"] = float(params)
        metrics[f"{prefix}/flops_overhead"] = float(flops)
    metrics["pipe_wall/params_overhead"] = 0.0
    metrics["pipe_wall/flops_overhead"] = 0.0
    return metrics, warnings


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
        "--skip-cost-profile",
        action="store_true",
        help="Emit zero cost placeholders instead of constructing/profiling models.",
    )
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
    metrics_dir = args.output_dir / "metrics"
    write_metrics(
        metrics_dir / "ch5_gt_metrics.yaml",
        compute_gt_metrics(device, max_batches=args.max_gt_batches),
    )
    if args.skip_cost_profile:
        cost = {
            key: 0.0
            for key in [
                "ns/params_overhead",
                "ns/flops_overhead",
                "darcy_dirichlet/flops_overhead",
                "darcy_flux/flops_overhead",
                "pipe_wall/flops_overhead",
                "pipe_stream/flops_overhead",
                "elasticity/params_overhead",
                "elasticity/flops_overhead",
                "plasticity/flops_overhead",
            ]
        }
        write_metrics(
            metrics_dir / "ch5_cost_metrics.yaml",
            cost,
            warnings=["cost profiling skipped by --skip-cost-profile"],
        )
    else:
        cost, warnings = compute_cost_metrics(device)
        write_metrics(metrics_dir / "ch5_cost_metrics.yaml", cost, warnings=warnings)
    print(f"wrote Chapter 5 metrics to {metrics_dir}")


if __name__ == "__main__":
    main()
