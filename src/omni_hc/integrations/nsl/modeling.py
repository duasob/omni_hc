import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from omni_hc.constraints import (
    ConstrainedModel,
    DarcyFluxConstraint,
    DirichletBoundaryAnsatz,
    ForwardHookLatentExtractor,
    MeanConstraint,
)
from omni_hc.core import load_yaml_file
from omni_hc.integrations.nsl.defaults import get_nsl_default_args
from omni_hc.integrations.nsl.paths import resolve_nsl_root

MODEL_REQUIRED_ARGS = {
    "Galerkin_Transformer": [
        "n_hidden",
        "n_heads",
        "dropout",
        "mlp_ratio",
        "n_layers",
        "out_dim",
    ],
    "FNO": [
        "n_hidden",
        "modes",
        "out_dim",
    ],
}


def ensure_nsl_path(nsl_root: str | Path | None, cfg: dict | None = None) -> Path:
    path = resolve_nsl_root(nsl_root, cfg=cfg)
    if not path.exists():
        raise FileNotFoundError(
            f"Neural-Solver-Library root does not exist: {path}"
        )
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def _resolve_backbone(cfg: dict) -> str:
    backbone = cfg.get("model", {}).get("backbone", "FNO")
    return str(backbone)


def _validate_required_args(backbone: str, args_dict: dict[str, Any]) -> None:
    required = MODEL_REQUIRED_ARGS.get(backbone, [])
    missing = [name for name in required if args_dict.get(name) is None]
    if missing:
        raise ValueError(
            f"Missing required args for {backbone}: {missing}"
        )


def build_model_args(cfg: dict, runtime_overrides: dict[str, Any] | None = None):
    backbone = _resolve_backbone(cfg)
    args_dict = get_nsl_default_args()
    args_dict["model"] = backbone

    model_cfg_path = cfg.get("model", {}).get("config")
    if model_cfg_path:
        loaded = load_yaml_file(model_cfg_path)
        if "model" in loaded:
            loaded_model = loaded.get("model", {})
            if not isinstance(loaded_model, dict):
                raise ValueError(
                    f"Expected 'model' mapping in backbone config: {model_cfg_path}"
                )
            args_dict.update(loaded_model.get("args", {}))
        else:
            args_dict.update(loaded)

    args_dict.update(cfg.get("model", {}).get("args", {}))
    if runtime_overrides:
        args_dict.update(runtime_overrides)

    shapelist = args_dict.get("shapelist")
    if isinstance(shapelist, list):
        args_dict["shapelist"] = tuple(shapelist)

    _validate_required_args(backbone, args_dict)
    return SimpleNamespace(**args_dict)


def _build_constraint(backbone: torch.nn.Module, args, cfg: dict):
    constraint_cfg = cfg.get("constraint", {})
    if not constraint_cfg:
        return backbone

    name = str(constraint_cfg.get("name", "")).strip().lower()
    if name in {"dirichlet_ansatz", "dirichlet_boundary_ansatz"}:
        constraint = DirichletBoundaryAnsatz(
            out_dim=int(args.out_dim),
            boundary_value=float(constraint_cfg.get("boundary_value", 0.0)),
            lower=float(constraint_cfg.get("lower", 0.0)),
            upper=float(constraint_cfg.get("upper", 1.0)),
            distance_power=float(constraint_cfg.get("distance_power", 1.0)),
            distance_reduce=str(constraint_cfg.get("distance_reduce", "product")),
        )
        wrapped = ConstrainedModel(backbone=backbone, constraint=constraint)
        if bool(constraint_cfg.get("freeze_base", False)):
            for param in wrapped.backbone.parameters():
                param.requires_grad = False
        return wrapped

    if name in {
        "darcy_flux_projection",
        "darcy_flux_fft_pad",
        "darcy_helmholtz",
        "darcy_streamfunction",
    }:
        constraint = DarcyFluxConstraint(
            spectral_backend=str(
                constraint_cfg.get("spectral_backend", "helmholtz_sine")
            ),
            force_value=float(constraint_cfg.get("force_value", 1.0)),
            permeability_eps=float(constraint_cfg.get("permeability_eps", 1e-6)),
            padding=constraint_cfg.get("padding", 8),
            padding_mode=str(constraint_cfg.get("padding_mode", "reflect")),
            particular_field=str(constraint_cfg.get("particular_field", "y_only")),
            pressure_out_dim=int(
                constraint_cfg.get(
                    "pressure_out_dim",
                    1,
                )
            ),
            enforce_boundary=bool(constraint_cfg.get("enforce_boundary", True)),
            boundary_value=float(constraint_cfg.get("boundary_value", 0.0)),
            shapelist=getattr(args, "shapelist", None),
            lower=float(constraint_cfg.get("lower", 0.0)),
            upper=float(constraint_cfg.get("upper", 1.0)),
        )
        wrapped = ConstrainedModel(backbone=backbone, constraint=constraint)
        if bool(constraint_cfg.get("freeze_base", False)):
            for param in wrapped.backbone.parameters():
                param.requires_grad = False
        return wrapped

    if name not in {"mean_correction", "mean_constraint"}:
        raise ValueError(
            "Unsupported constraint "
            f"'{name}'. Currently supported: mean_correction, dirichlet_ansatz, darcy_flux_projection"
        )

    mode = str(constraint_cfg.get("mode", "post_output")).lower()
    latent_extractor = None
    latent_dim = constraint_cfg.get("latent_dim")
    if mode == "latent_head":
        latent_module = constraint_cfg.get("latent_module")
        if not latent_module:
            raise ValueError("constraint.latent_module is required for latent_head mode")
        latent_extractor = ForwardHookLatentExtractor(backbone, str(latent_module))
        if latent_dim is None:
            latent_dim = getattr(args, "n_hidden", None)

    constraint = MeanConstraint(
        mode=mode,
        out_dim=int(args.out_dim),
        hidden_dim=constraint_cfg.get("correction_hidden"),
        n_layers=int(constraint_cfg.get("correction_layers", 0)),
        act=constraint_cfg.get("correction_act", "gelu"),
        latent_dim=None if latent_dim is None else int(latent_dim),
        channel_dim=int(constraint_cfg.get("channel_dim", -1)),
        reduce_dims=constraint_cfg.get("reduce_dims"),
    )
    wrapped = ConstrainedModel(
        backbone=backbone,
        constraint=constraint,
        latent_extractor=latent_extractor,
    )
    if bool(constraint_cfg.get("freeze_base", False)):
        for param in wrapped.backbone.parameters():
            param.requires_grad = False
    return wrapped


def create_model(
    cfg: dict,
    *,
    nsl_root: str | Path | None,
    device: torch.device,
    runtime_overrides: dict[str, Any] | None = None,
):
    resolved_nsl_root = ensure_nsl_path(nsl_root, cfg=cfg)
    from models.model_factory import get_model

    args = build_model_args(cfg, runtime_overrides=runtime_overrides)
    backbone = get_model(args).to(device)
    model = _build_constraint(backbone, args, cfg).to(device)
    return model, args, resolved_nsl_root
