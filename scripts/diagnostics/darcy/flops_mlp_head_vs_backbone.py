#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from omni_hc.constraints.sine_boundary import SineBoundaryConstraint
from omni_hc.core import compose_run_config, deep_merge, parse_dotted_overrides
from omni_hc.integrations.nsl import create_model
from omni_hc.integrations.nsl.modeling import ensure_nsl_path


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be HxW, e.g. 85x85")
    return int(parts[0]), int(parts[1])


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _resolve_grid_shape(cfg: dict[str, Any], explicit: tuple[int, int] | None) -> tuple[int, int]:
    if explicit is not None:
        return explicit

    model_shape = cfg.get("model", {}).get("args", {}).get("shapelist")
    if model_shape is not None:
        return int(model_shape[0]), int(model_shape[1])

    dataset_name = str(cfg.get("benchmark", {}).get("dataset", ""))
    full_resolution = 421 if "421" in dataset_name else int(
        cfg.get("diagnostics", {}).get("full_resolution", 421)
    )
    data_cfg = cfg.get("data", {})
    r1 = int(data_cfg.get("downsamplex", 1))
    r2 = int(data_cfg.get("downsampley", 1))
    return int(((full_resolution - 1) / r1) + 1), int(((full_resolution - 1) / r2) + 1)


def _train_samples(cfg: dict[str, Any]) -> int:
    ntrain = int(cfg.get("data", {}).get("ntrain", 1000))
    val_size = int(cfg.get("training", {}).get("val_size", 0))
    if val_size <= 0:
        return ntrain
    return max(ntrain - min(max(val_size, 1), ntrain - 1), 1)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _profile_training_step(step_fn, *, device: torch.device) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, with_flops=True) as prof:
        step_fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    return int(sum(event.flops for event in prof.key_averages() if event.flops))


def _check_backbone_dependencies(cfg: dict[str, Any]) -> None:
    model_cfg = cfg.get("model", {}) or {}
    backbone = str(model_cfg.get("backbone", ""))
    args = model_cfg.get("args", {}) or {}
    if backbone == "ONO" and str(args.get("attn_type", "nystrom")) == "nystrom":
        if importlib.util.find_spec("nystrom_attention") is None:
            raise ModuleNotFoundError(
                "Darcy ONO is configured with attn_type='nystrom', which requires "
                "the optional package 'nystrom_attention'. Install it in this Python "
                "environment, or profile ONO with a built-in attention path, e.g. "
                "`--backbone-attn-type selfAttention` or "
                "`--backbone-override model.args.attn_type=linear`."
            )


def _format_count(value: float) -> str:
    units = [
        (1e18, "EFLOPs"),
        (1e15, "PFLOPs"),
        (1e12, "TFLOPs"),
        (1e9, "GFLOPs"),
        (1e6, "MFLOPs"),
        (1e3, "KFLOPs"),
    ]
    for scale, suffix in units:
        if abs(value) >= scale:
            return f"{value / scale:.4g} {suffix}"
    return f"{value:.4g} FLOPs"


def _count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(param.numel() for param in module.parameters())
    trainable = sum(param.numel() for param in module.parameters() if param.requires_grad)
    return total, trainable


def _mlp_head_step_flops(
    *,
    cfg: dict[str, Any],
    grid_shape: tuple[int, int],
    batch_size: int,
    latent_dim: int,
    device: torch.device,
) -> tuple[int, torch.nn.Module]:
    constraint_cfg = cfg.get("constraint", {}) or {}
    head = SineBoundaryConstraint(
        n_modes=int(constraint_cfg.get("n_modes", 21)),
        grid_shape=grid_shape,
        hidden_dim=int(constraint_cfg.get("hidden_dim", 256)),
        n_layers=int(constraint_cfg.get("n_layers", 3)),
        act=str(constraint_cfg.get("act", "gelu")),
        latent_dim=latent_dim,
    ).to(device)
    head.train()
    opt = torch.optim.Adam(head.coeff_head.parameters(), lr=1e-3)

    h, w = grid_shape
    n_points = h * w
    boundary_dim = 2 * w + 2 * (h - 2)
    fx = torch.randn(batch_size, n_points, 1, device=device)
    target = torch.randn(batch_size, boundary_dim, device=device)

    def step() -> None:
        feats = head._boundary_feats(fx)
        if latent_dim:
            feats = torch.cat([feats, torch.zeros(batch_size, latent_dim, device=device)], dim=-1)
        coeffs = head.coeff_head(feats).view(batch_size, 4, head.n_modes)
        bottom = coeffs[:, 0] @ head.basis_h.T
        top = coeffs[:, 1] @ head.basis_h.T
        left = coeffs[:, 2] @ head.basis_v.T
        right = coeffs[:, 3] @ head.basis_v.T
        pred = torch.cat([bottom, top, left[:, 1:-1], right[:, 1:-1]], dim=-1)
        loss = torch.nn.functional.mse_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    flops = _profile_training_step(step, device=device)
    return flops, head.coeff_head


def _backbone_step_flops(
    *,
    cfg: dict[str, Any],
    grid_shape: tuple[int, int],
    batch_size: int,
    device: torch.device,
) -> tuple[int, torch.nn.Module]:
    h, w = grid_shape
    meta = {
        "shapelist": grid_shape,
        "task": "steady",
        "loader": "darcy",
        "geotype": "structured_2D",
        "space_dim": 2,
        "fun_dim": 1,
        "out_dim": 1,
    }
    runtime_overrides = {
        "shapelist": tuple(meta["shapelist"]),
        "task": str(meta["task"]),
        "loader": str(meta["loader"]),
        "geotype": str(meta["geotype"]),
        "space_dim": int(meta["space_dim"]),
        "fun_dim": int(meta["fun_dim"]),
        "out_dim": int(meta["out_dim"]),
    }
    model, model_args, _ = create_model(
        cfg,
        device=device,
        runtime_overrides=runtime_overrides,
    )
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ensure_nsl_path(cfg)
    from utils.loss import DerivLoss, L2Loss

    n_points = h * w
    coords = torch.rand(batch_size, n_points, 2, device=device)
    fx = torch.randn(batch_size, n_points, 1, device=device)
    target = torch.randn(batch_size, n_points, 1, device=device)
    l2_loss = L2Loss(size_average=False)
    derivloss = _as_bool(
        cfg.get("training", {}).get("derivloss", getattr(model_args, "derivloss", False))
    )
    deriv_weight = float(cfg.get("training", {}).get("derivloss_weight", 0.1))
    deriv_loss = DerivLoss(size_average=False, shapelist=grid_shape) if derivloss else None

    def step() -> None:
        pred = model(coords, fx)
        loss = l2_loss(pred, target)
        if deriv_loss is not None:
            loss = loss + deriv_weight * deriv_loss(pred, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    flops = _profile_training_step(step, device=device)
    return flops, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate counted training FLOPs for the Darcy sine-boundary MLP head "
            "and for one epoch of a backbone model."
        )
    )
    parser.add_argument("--benchmark", default="darcy")
    parser.add_argument(
        "--backbone",
        default="ONO",
        help="Backbone to compare against. Defaults to ONO.",
    )
    parser.add_argument("--backbone-budget", default="final")
    parser.add_argument(
        "--backbone-attn-type",
        choices=["nystrom", "linear", "selfAttention"],
        default=None,
        help=(
            "Shortcut for ONO model.args.attn_type. Defaults to the backbone config "
            "(Darcy ONO currently resolves to nystrom)."
        ),
    )
    parser.add_argument("--constraint", default="darcy_sine_boundary_constraint")
    parser.add_argument("--constraint-budget", default="final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grid-shape", type=_parse_shape, default=None)
    parser.add_argument("--mlp-batch-size", type=int, default=64)
    parser.add_argument("--mlp-epochs", type=int, default=100)
    parser.add_argument("--mlp-max-samples", type=int, default=None)
    parser.add_argument("--latent-dim", type=int, default=0)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override applied to both composed configs.",
    )
    parser.add_argument(
        "--backbone-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override applied only to the backbone config.",
    )
    parser.add_argument(
        "--ono-backbone",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ono-budget",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--ono-override",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--constraint-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted override applied only to the MLP-head constraint config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _device(args.device)
    torch.manual_seed(args.seed)

    backbone = args.ono_backbone or args.backbone
    backbone_budget = args.ono_budget or args.backbone_budget
    backbone_only_overrides = parse_dotted_overrides(args.backbone_override)
    if args.backbone_attn_type is not None:
        backbone_only_overrides = deep_merge(
            backbone_only_overrides,
            {"model": {"args": {"attn_type": args.backbone_attn_type}}},
        )
    if args.ono_override:
        backbone_only_overrides = deep_merge(
            backbone_only_overrides,
            parse_dotted_overrides(args.ono_override),
        )

    shared_overrides = parse_dotted_overrides(args.override)
    backbone_cfg = compose_run_config(
        benchmark=args.benchmark,
        backbone=backbone,
        constraint=None,
        budget=backbone_budget,
        seed=args.seed,
        extra_overrides=deep_merge(shared_overrides, backbone_only_overrides),
    )
    _check_backbone_dependencies(backbone_cfg)
    mlp_cfg = compose_run_config(
        benchmark=args.benchmark,
        backbone="FNO",
        constraint=args.constraint,
        budget=args.constraint_budget,
        seed=args.seed,
        extra_overrides=deep_merge(shared_overrides, parse_dotted_overrides(args.constraint_override)),
    )
    grid_shape = _resolve_grid_shape(backbone_cfg, args.grid_shape)

    mlp_ntrain = int(mlp_cfg.get("data", {}).get("ntrain", 1000))
    if args.mlp_max_samples is not None:
        mlp_ntrain = min(mlp_ntrain, int(args.mlp_max_samples))
    mlp_steps = math.ceil(mlp_ntrain / args.mlp_batch_size)
    backbone_batch_size = int(backbone_cfg.get("training", {}).get("batch_size", 4))
    backbone_samples = _train_samples(backbone_cfg)
    backbone_steps = math.ceil(backbone_samples / backbone_batch_size)

    mlp_step_flops, mlp_head = _mlp_head_step_flops(
        cfg=mlp_cfg,
        grid_shape=grid_shape,
        batch_size=args.mlp_batch_size,
        latent_dim=args.latent_dim,
        device=device,
    )
    backbone_step_flops, backbone_model = _backbone_step_flops(
        cfg=backbone_cfg,
        grid_shape=grid_shape,
        batch_size=backbone_batch_size,
        device=device,
    )

    mlp_epoch_flops = mlp_step_flops * mlp_steps
    mlp_total_flops = mlp_epoch_flops * int(args.mlp_epochs)
    backbone_epoch_flops = backbone_step_flops * backbone_steps
    ratio = mlp_total_flops / backbone_epoch_flops if backbone_epoch_flops else float("nan")
    mlp_total_params, mlp_trainable_params = _count_parameters(mlp_head)
    backbone_total_params, backbone_trainable_params = _count_parameters(backbone_model)

    print("FLOP diagnostic")
    print(f"device: {device}")
    print(f"grid_shape: {grid_shape[0]}x{grid_shape[1]}")
    print()
    print("MLP head pretrain")
    print(f"batch_size: {args.mlp_batch_size}")
    print(f"samples_per_epoch: {mlp_ntrain}")
    print(f"steps_per_epoch: {mlp_steps}")
    print(f"epochs: {args.mlp_epochs}")
    print(f"parameters: {mlp_total_params:,} total / {mlp_trainable_params:,} trainable")
    print(f"counted_step_flops: {mlp_step_flops:,} ({_format_count(mlp_step_flops)})")
    print(f"counted_epoch_flops: {mlp_epoch_flops:,} ({_format_count(mlp_epoch_flops)})")
    print(f"counted_total_flops: {mlp_total_flops:,} ({_format_count(mlp_total_flops)})")
    print()
    print("Backbone training")
    print(f"backbone: {backbone}")
    print(f"budget: {backbone_budget}")
    print(f"batch_size: {backbone_batch_size}")
    print(f"samples_per_epoch: {backbone_samples}")
    print(f"steps_per_epoch: {backbone_steps}")
    print(f"parameters: {backbone_total_params:,} total / {backbone_trainable_params:,} trainable")
    print(f"counted_step_flops: {backbone_step_flops:,} ({_format_count(backbone_step_flops)})")
    print(f"counted_epoch_flops: {backbone_epoch_flops:,} ({_format_count(backbone_epoch_flops)})")
    print()
    print(f"MLP total / {backbone} one epoch: {ratio:.6g}x")
    print(
        "Note: torch.profiler(with_flops=True) reports counted FLOPs for supported "
        "operators, mainly matmul/linear/conv; some elementwise, normalization, "
        "optimizer, and attention-kernel work may be absent."
    )


if __name__ == "__main__":
    main()
