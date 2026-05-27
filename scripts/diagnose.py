#!/usr/bin/env python
"""
Model diagnostics: parameter count and training FLOPs per epoch.

Usage:
  # Component flags (requires data on disk):
  python scripts/diagnose.py --benchmark darcy_2d --backbone FNO --budget final

  # Pre-resolved or experiment config:
  python scripts/diagnose.py --config path/to/resolved_config.yaml --budget final

  # Checkpoint (loads resolved_config.yaml from the same directory):
  python scripts/diagnose.py --checkpoint outputs/.../seed_42/best.pt [--budget final]
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omni_hc.core import (
    compose_run_config,
    deep_merge,
    load_yaml_file,
    parse_dotted_overrides,
)
from omni_hc.integrations.nsl import create_model
from omni_hc.training.common import forward_with_optional_aux


# ── benchmark dispatch ────────────────────────────────────────────────────────

def _steady_overrides(meta: dict) -> dict:
    return {
        "shapelist": tuple(meta["shapelist"]),
        "task": str(meta["task"]),
        "loader": str(meta["loader"]),
        "geotype": str(meta["geotype"]),
        "space_dim": int(meta["space_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "out_dim": int(meta["out_dim"]),
    }


_BENCHMARK_INFO: dict[str, dict[str, Any]] = {
    "darcy_2d": dict(
        loader_module="omni_hc.benchmarks.darcy.data",
        meta_attr="darcy_meta",
        task_type="steady",
        runtime_overrides_fn=_steady_overrides,
    ),
    "elasticity_2d": dict(
        loader_module="omni_hc.benchmarks.elasticity.data",
        meta_attr="elasticity_meta",
        task_type="steady",
        runtime_overrides_fn=_steady_overrides,
    ),
    "pipe_2d": dict(
        loader_module="omni_hc.benchmarks.pipe.data",
        meta_attr="pipe_meta",
        task_type="steady",
        runtime_overrides_fn=_steady_overrides,
    ),
    "plasticity_2d": dict(
        loader_module="omni_hc.benchmarks.plasticity.data",
        meta_attr="plasticity_meta",
        task_type="dynamic_conditional",
        runtime_overrides_fn=lambda meta: {
            "shapelist": tuple(meta["shapelist"]),
            "task": "dynamic_conditional",
            "loader": "plas",
            "geotype": "structured_2D",
            "space_dim": int(meta["space_dim"]),
            "fun_dim": int(meta["fun_dim"]),
            "out_dim": int(meta["out_dim"]),
            "T_out": int(meta["t_out"]),
            "time_input": True,
        },
    ),
    "navier_stokes_2d": dict(
        loader_module="omni_hc.benchmarks.navier_stokes.data",
        meta_attr="ns_meta",
        task_type="autoregressive",
        runtime_overrides_fn=lambda meta: {
            "shapelist": tuple(meta["shapelist"]),
            "task": "dynamic_autoregressive",
            "T_in": int(meta["t_in"]),
            "T_out": int(meta["t_out"]),
            "out_dim": int(meta["out_dim"]),
            "fun_dim": int(meta["fun_dim"]),
            "loader": "ns",
            "geotype": "structured_2D",
            "space_dim": 2,
        },
    ),
}


def _build_loaders(benchmark_name: str, cfg: dict):
    mod = importlib.import_module(_BENCHMARK_INFO[benchmark_name]["loader_module"])
    return mod.build_train_val_loaders(cfg)


def _get_meta(benchmark_name: str, loader) -> dict:
    return getattr(loader, _BENCHMARK_INFO[benchmark_name]["meta_attr"])


def _first_batch(loader) -> dict:
    for batch in loader:
        return batch
    raise RuntimeError("Dataloader returned no batches")


def _prepare_batch(
    benchmark_name: str, batch: dict, meta: dict, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    """Return (coords, fx, target, time). fx and time are None when not applicable."""
    if benchmark_name == "navier_stokes_2d":
        from omni_hc.benchmarks.navier_stokes.data import make_grid

        fx = batch["x"].to(device)
        target = batch["y"].to(device)
        h, w = tuple(meta["shapelist"])
        grid = make_grid(h, w, device=device, dtype=fx.dtype)
        coords = grid.unsqueeze(0).expand(int(fx.shape[0]), -1, -1)
        return coords, fx, target, None

    if benchmark_name == "plasticity_2d":
        return (
            batch["coords"].to(device),
            batch["x"].to(device),
            batch["y"].to(device),
            batch["time"].to(device),
        )

    # Steady tasks (darcy, elasticity, pipe)
    coords = batch["coords"].to(device)
    fx_raw = batch.get("x")
    fx = fx_raw.to(device) if fx_raw is not None else None
    target = batch["y"].to(device)
    return coords, fx, target, None


def _build_step_fn(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    coords: torch.Tensor,
    fx: torch.Tensor | None,
    target: torch.Tensor,
    *,
    task_type: str,
    meta: dict,
    device: torch.device,
    time: torch.Tensor | None = None,
):
    t_out = int(meta.get("t_out", 1))
    out_dim = int(meta.get("out_dim", 1))

    if task_type == "steady":
        def step():
            out = forward_with_optional_aux(model, coords, fx)
            pred = out["pred"]
            loss = F.mse_loss(
                pred.reshape(pred.shape[0], -1),
                target.reshape(target.shape[0], -1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    elif task_type == "dynamic_conditional":
        def step():
            total = torch.zeros((), device=device)
            for t in range(t_out):
                input_t = time[:, t : t + 1].reshape(coords.shape[0], 1)
                out = forward_with_optional_aux(model, coords, fx, T=input_t)
                pred_t = out["pred"]
                y_t = target[..., out_dim * t : out_dim * (t + 1)]
                total = total + F.mse_loss(
                    pred_t.reshape(pred_t.shape[0], -1),
                    y_t.reshape(y_t.shape[0], -1),
                )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

    elif task_type == "autoregressive":
        def step():
            current_fx = fx.clone()
            total = torch.zeros((), device=device)
            for t in range(t_out):
                out = forward_with_optional_aux(model, coords, current_fx)
                pred_t = out["pred"]
                y_t = target[..., out_dim * t : out_dim * (t + 1)]
                total = total + F.mse_loss(
                    pred_t.reshape(pred_t.shape[0], -1),
                    y_t.reshape(y_t.shape[0], -1),
                )
                current_fx = torch.cat((current_fx[..., out_dim:], pred_t), dim=-1)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    return step


def _profile_step(step_fn, *, device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return int(sum(e.flops for e in prof.key_averages() if e.flops))


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _effective_train_samples(cfg: dict) -> int:
    ntrain = int(cfg.get("data", {}).get("ntrain", 1000))
    val_size = int(cfg.get("training", {}).get("val_size", 0))
    if val_size <= 0:
        return ntrain
    return max(ntrain - min(max(val_size, 1), ntrain - 1), 1)


def _format_flops(value: float) -> str:
    for scale, suffix in [
        (1e18, "EFLOPs"), (1e15, "PFLOPs"), (1e12, "TFLOPs"),
        (1e9, "GFLOPs"), (1e6, "MFLOPs"), (1e3, "KFLOPs"),
    ]:
        if abs(value) >= scale:
            return f"{value / scale:.4g} {suffix}"
    return f"{value:.4g} FLOPs"


def _run_dir_from_args(args: argparse.Namespace, cfg: dict) -> Path | None:
    if args.checkpoint is not None:
        return Path(args.checkpoint).resolve().parent
    output_dir = (cfg.get("paths") or {}).get("output_dir")
    if output_dir:
        return Path(str(output_dir)).resolve()
    return None


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


# ── config resolution ─────────────────────────────────────────────────────────

def _resolve_cfg(args: argparse.Namespace) -> dict:
    if args.checkpoint is not None:
        checkpoint_dir = Path(args.checkpoint).parent
        resolved_path = checkpoint_dir / "resolved_config.yaml"
        if not resolved_path.exists():
            raise FileNotFoundError(
                f"No resolved_config.yaml found at {resolved_path}. "
                "Pass --config or component flags instead."
            )
        cfg = load_yaml_file(resolved_path)
        cfg.setdefault("experiment", {})["mode"] = "train"
        if args.budget is not None:
            budget_path = ROOT / "configs" / "budgets" / f"{args.budget}.yaml"
            if not budget_path.exists():
                raise FileNotFoundError(f"Budget config not found: {budget_path}")
            cfg = deep_merge(cfg, load_yaml_file(budget_path))
        return cfg

    return compose_run_config(
        benchmark=args.benchmark,
        backbone=args.backbone,
        constraint=args.constraint,
        budget=args.budget,
        experiment=args.config,
        mode="train",
        seed=args.seed,
        extra_overrides=parse_dotted_overrides(args.override),
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Report parameter count and training FLOPs per epoch."
    )
    p.add_argument("--benchmark", type=str, default=None)
    p.add_argument("--backbone", type=str, default=None)
    p.add_argument("--constraint", type=str, default=None)
    p.add_argument("--budget", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--config", type=str, default=None,
        help="Pre-resolved or experiment YAML (alternative to component flags).",
    )
    p.add_argument(
        "--checkpoint", type=str, default=None,
        help="Checkpoint .pt file; resolved_config.yaml is read from the same directory.",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help="Dotted config override. Repeatable.",
    )
    p.add_argument(
        "--write-yaml",
        type=str,
        nargs="?",
        const="diagnostics.yaml",
        default=None,
        help=(
            "Write machine-readable diagnostics. With no value, writes "
            "diagnostics.yaml next to --checkpoint or paths.output_dir."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)

    print("Resolving config...", flush=True)
    cfg = _resolve_cfg(args)

    benchmark_name = str((cfg.get("benchmark") or {}).get("name") or "")
    if benchmark_name not in _BENCHMARK_INFO:
        raise ValueError(
            f"Unsupported benchmark '{benchmark_name}'. "
            f"Supported: {sorted(_BENCHMARK_INFO)}"
        )

    info = _BENCHMARK_INFO[benchmark_name]
    task_type: str = info["task_type"]

    print(f"Building train loader for {benchmark_name}...", flush=True)
    train_loader, _ = _build_loaders(benchmark_name, cfg)
    meta = _get_meta(benchmark_name, train_loader)

    print(f"Creating model...", flush=True)
    runtime_overrides = info["runtime_overrides_fn"](meta)
    model, _model_args, _ = create_model(cfg, device=device, runtime_overrides=runtime_overrides)
    model.train()

    batch = _first_batch(train_loader)
    coords, fx, target, time = _prepare_batch(benchmark_name, batch, meta, device=device)

    total_params, trainable_params = _count_params(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    step_fn = _build_step_fn(
        model, optimizer, coords, fx, target,
        task_type=task_type, meta=meta, device=device, time=time,
    )

    print("Profiling training step...", flush=True)
    step_flops = _profile_step(step_fn, device=device)

    batch_size = int(cfg.get("training", {}).get("batch_size", 4))
    samples_per_epoch = _effective_train_samples(cfg)
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    epoch_flops = step_flops * steps_per_epoch
    num_epochs = int(cfg.get("training", {}).get("num_epochs", 0))

    backbone_name = str((cfg.get("model") or {}).get("backbone") or "")
    constraint_name = str((cfg.get("experiment") or {}).get("constraint") or "unconstrained")
    budget_label = str((cfg.get("experiment") or {}).get("budget") or args.budget or "")

    print()
    print("=" * 60)
    print(f"  benchmark  : {benchmark_name}")
    print(f"  backbone   : {backbone_name}")
    print(f"  constraint : {constraint_name}")
    print(f"  budget     : {budget_label}")
    print(f"  device     : {device}")
    print(f"  task_type  : {task_type}")
    print("=" * 60)
    print()
    print("Parameters")
    print(f"  total      : {total_params:,}")
    print(f"  trainable  : {trainable_params:,}")
    print()
    print("Training FLOPs")
    print(f"  batch_size        : {batch_size}")
    print(f"  samples_per_epoch : {samples_per_epoch}")
    print(f"  steps_per_epoch   : {steps_per_epoch}")
    print(f"  step_flops        : {step_flops:,}  ({_format_flops(step_flops)})")
    print(f"  epoch_flops       : {epoch_flops:,}  ({_format_flops(epoch_flops)})")
    if num_epochs > 0:
        total_flops = epoch_flops * num_epochs
        print(f"  num_epochs        : {num_epochs}")
        print(f"  total_flops       : {total_flops:,}  ({_format_flops(total_flops)})")
    else:
        total_flops = 0
    if task_type != "steady":
        t_out = int(meta.get("t_out", 1))
        label = "rollout steps" if task_type == "autoregressive" else "time steps"
        print(f"\n  Note: each step covers the full {t_out}-step {label}.")
    print(
        "\n  Note: torch.profiler with_flops=True counts matmul/conv/linear ops.\n"
        "  Attention kernels, normalisation layers, and element-wise ops may be\n"
        "  under-reported."
    )

    if args.write_yaml is not None:
        yaml_path = Path(args.write_yaml)
        if not yaml_path.is_absolute():
            run_dir = _run_dir_from_args(args, cfg)
            if run_dir is None:
                raise ValueError(
                    "--write-yaml needs an absolute path when no checkpoint or "
                    "paths.output_dir is available."
                )
            yaml_path = run_dir / yaml_path

        payload = {
            "provenance": {
                "benchmark": benchmark_name,
                "backbone": backbone_name,
                "constraint": constraint_name,
                "budget": budget_label,
                "device": str(device),
                "task_type": task_type,
                "num_epochs": num_epochs,
                "training_samples": samples_per_epoch,
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "seed": int((cfg.get("training") or {}).get("seed", args.seed)),
                "checkpoint": str(Path(args.checkpoint).resolve())
                if args.checkpoint is not None
                else None,
            },
            "cost": {
                "total_params": total_params,
                "trainable_params": trainable_params,
                "step_flops": step_flops,
                "epoch_flops": epoch_flops,
                "total_flops": total_flops,
            },
            "notes": [
                "torch.profiler with_flops=True counts matmul/conv/linear ops; some kernels may be under-reported.",
            ],
        }
        _write_yaml(yaml_path, payload)
        print(f"\nwrote diagnostics YAML: {yaml_path}")


if __name__ == "__main__":
    main()
