import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from omni_hc.constraints import (
    DarcyFluxConstraint,
    DirichletBoundaryAnsatz,
    ElasticityPlaneStressVMConstraint,
    MeanConstraint,
    PipeInletParabolicAnsatz,
    PipeStreamFunctionBoundaryAnsatz,
    PipeStreamFunctionUxConstraint,
    PipeUxBoundaryAnsatz,
    PlasticityEnvelopeConstraint,
    PlasticityEnvelopeYFreeXConstraint,
    PlasticityIsotonicRegression,
    PlasticityMeshConsistencyConstraint,
    SineBoundaryConstraint,
    StructuredWallDirichletAnsatz,
)
from omni_hc.core import load_yaml_file
from omni_hc.integrations.nsl.defaults import get_nsl_default_args
from omni_hc.integrations.nsl.paths import resolve_nsl_root

# Args validated eagerly before model construction, per backbone. Backbones
# not listed here are still buildable; they just rely on NSL defaults and
# fail later if a required arg is missing.
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

# Maps constraint name → class. The name matches the constraint YAML filename
# (without .yaml) and the snake_case form of the class name.
# Matching is case-insensitive (normalised in _build_constraint).
_CONSTRAINT_CLASSES: dict[str, type] = {
    "dirichlet_boundary_ansatz": DirichletBoundaryAnsatz,
    "structured_wall_dirichlet_ansatz": StructuredWallDirichletAnsatz,
    "pipe_inlet_parabolic_ansatz": PipeInletParabolicAnsatz,
    "pipe_ux_boundary_ansatz": PipeUxBoundaryAnsatz,
    "pipe_stream_function_ux_constraint": PipeStreamFunctionUxConstraint,
    "pipe_stream_function_boundary_ansatz": PipeStreamFunctionBoundaryAnsatz,
    "darcy_flux_constraint": DarcyFluxConstraint,
    "elasticity_plane_stress_vm_constraint": ElasticityPlaneStressVMConstraint,
    "plasticity_envelope_constraint": PlasticityEnvelopeConstraint,
    "plasticity_envelope_y_free_x": PlasticityEnvelopeYFreeXConstraint,
    "plasticity_isotonic_regression": PlasticityIsotonicRegression,
    "plasticity_mesh_consistency_constraint": PlasticityMeshConsistencyConstraint,
    "mean_constraint": MeanConstraint,
    "sine_boundary_constraint": SineBoundaryConstraint,
}


def ensure_nsl_path(cfg: dict | None = None) -> Path:
    path = resolve_nsl_root(cfg=cfg)
    if not path.exists():
        raise FileNotFoundError(f"Neural-Solver-Library root does not exist: {path}")
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
        raise ValueError(f"Missing required args for {backbone}: {missing}")


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

    constraint_cfg = cfg.get("constraint", {}) or {}
    backbone_out_dim = constraint_cfg.get("backbone_out_dim")
    if backbone_out_dim is not None:
        args_dict["constraint_target_out_dim"] = int(args_dict.get("out_dim", 1))
        args_dict["out_dim"] = int(backbone_out_dim)

    shapelist = args_dict.get("shapelist")
    if isinstance(shapelist, list):
        args_dict["shapelist"] = tuple(shapelist)

    _validate_required_args(backbone, args_dict)
    return SimpleNamespace(**args_dict)


def _model_context(args: SimpleNamespace) -> dict[str, Any]:
    shapelist = getattr(args, "shapelist", None)
    return {
        "out_dim": int(args.out_dim),
        "shapelist": shapelist,
        "grid_shape": shapelist,  # boundary constraints use grid_shape instead of shapelist
        "backbone_out_dim": int(args.out_dim),
        "target_out_dim": int(getattr(args, "constraint_target_out_dim", 1)),
        "n_hidden": getattr(args, "n_hidden", None),
    }


def _build_constraint(backbone: torch.nn.Module, args: SimpleNamespace, cfg: dict):
    constraint_section = cfg.get("constraint", {}) or {}
    if not constraint_section:
        return backbone

    name = str(constraint_section.get("name", "")).strip().lower()
    cls = _CONSTRAINT_CLASSES.get(name)
    if cls is None:
        raise ValueError(
            f"Unsupported constraint '{name}'. "
            f"Supported: {sorted(_CONSTRAINT_CLASSES)}"
        )
    return cls.build(backbone, _model_context(args), cfg)


def create_model(
    cfg: dict,
    *,
    device: torch.device,
    runtime_overrides: dict[str, Any] | None = None,
):
    resolved_nsl_root = ensure_nsl_path(cfg)
    from models.model_factory import get_model

    args = build_model_args(cfg, runtime_overrides=runtime_overrides)
    backbone = get_model(args).to(device)  # from NSL's model factory
    model = _build_constraint(backbone, args, cfg).to(device)
    return model, args, resolved_nsl_root
