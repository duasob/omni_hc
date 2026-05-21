from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import deep_merge, load_composed_config, load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_name(value: str) -> str:
    return value.strip().lower().replace("/", "_").replace(" ", "_").replace("-", "_")


def default_output_root(mode: str) -> Path:
    env_value = os.environ.get("OMNI_HC_OUTPUT_ROOT")
    if env_value:
        return repo_path(env_value)

    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "omni_hc"

    return PROJECT_ROOT / "outputs"


# Search paths for each config component kind.
# Each entry is a list of candidate paths, tried in order.
# Callables receive (value, benchmark) and return a Path.
_COMPONENT_CANDIDATES: dict[str, Any] = {
    "benchmark": lambda v, b: [
        PROJECT_ROOT / "configs/benchmarks" / v / "base.yaml",
    ],
    "backbone": lambda v, b: [
        PROJECT_ROOT / "configs/backbones" / b / f"{v}.yaml",
        PROJECT_ROOT / "configs/backbones" / b / f"{v.lower()}.yaml",
    ],
    "constraint": lambda v, b: [
        PROJECT_ROOT / "configs/constraints" / b / f"{v}.yaml",
        PROJECT_ROOT / "configs/constraints" / f"{v}.yaml",
    ],
    "budget": lambda v, b: [
        PROJECT_ROOT / "configs/budgets" / f"{v}.yaml",
    ],
    "optuna": lambda v, b: [
        PROJECT_ROOT / "configs/optuna" / b / f"{v}.yaml",
        PROJECT_ROOT / "configs/optuna" / f"{v}.yaml",
    ],
}

_REQUIRES_BENCHMARK = {"backbone", "constraint", "optuna"}


def _component_path(
    value: str | Path | None,
    *,
    kind: str,
    benchmark: str | None = None,
    required: bool = True,
) -> Path | None:
    if value is None or str(value).lower() in {"", "none", "null", "unconstrained"}:
        if required:
            raise ValueError(f"{kind} is required")
        return None

    value_str = str(value)

    explicit = repo_path(value_str)
    if explicit.exists():
        return explicit

    if kind in _REQUIRES_BENCHMARK and benchmark is None:
        raise ValueError(f"{kind} resolution requires a benchmark")

    factory = _COMPONENT_CANDIDATES.get(kind)
    if factory is None:
        raise ValueError(f"Unknown config component kind: {kind!r}")

    candidates: list[Path] = factory(value_str, benchmark)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if required:
        searched = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Could not resolve {kind}={value_str!r}. Searched: {searched}"
        )
    return None


def _load_component(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_composed_config(path)


def _load_experiment(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_yaml_file(repo_path(path))


def _experiment_value(
    experiment: dict[str, Any], key: str, cli_value: str | None
) -> str | None:
    if cli_value is not None:
        return cli_value
    value = experiment.get(key)
    return str(value) if value is not None else None


def _has_component_value(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "none", "null", "unconstrained"}


def _is_resolved_run_config(cfg: dict[str, Any]) -> bool:
    benchmark_cfg = cfg.get("benchmark")
    model_cfg = cfg.get("model")
    return (
        isinstance(benchmark_cfg, dict)
        and isinstance(model_cfg, dict)
        and model_cfg.get("backbone") is not None
    )


def _default_optuna_name(
    *, benchmark: str, constraint: str | None, backbone: str
) -> str | None:
    names = []
    if constraint and constraint.lower() not in {"none", "unconstrained"}:
        names.append(constraint)
    names.append(backbone)
    for name in names:
        if (
            _component_path(name, kind="optuna", benchmark=benchmark, required=False)
            is not None
        ):
            return name
    return None


def _run_label(
    *, benchmark: str, backbone: str, constraint: str | None, budget: str, seed: int
) -> str:
    return "_".join(
        safe_name(item)
        for item in [
            benchmark,
            backbone,
            constraint or "unconstrained",
            budget,
            f"seed_{seed}",
        ]
    )


def _load_components(
    benchmark: str,
    backbone: str,
    constraint: str | None,
    budget: str,
    optuna: str | None,
    mode: str,
) -> tuple[dict[str, Any], list[str], str | None]:
    """Load and merge all component configs. Returns (cfg, source_configs, resolved_optuna_name)."""
    cfg: dict[str, Any] = {}
    source_configs: list[str] = []

    paths = [
        _component_path(benchmark, kind="benchmark"),
        _component_path(backbone, kind="backbone", benchmark=benchmark),
        _component_path(
            constraint,
            kind="constraint",
            benchmark=benchmark,
            required=_has_component_value(constraint),
        ),
    ]

    if mode == "tune":
        if optuna is None:
            optuna = _default_optuna_name(
                benchmark=benchmark, constraint=constraint, backbone=backbone
            )
        paths.append(
            _component_path(
                optuna, kind="optuna", benchmark=benchmark, required=optuna is not None
            )
        )

    paths.append(_component_path(budget, kind="budget"))

    for path in paths:
        if path is None:
            continue
        cfg = deep_merge(cfg, _load_component(path))
        source_configs.append(str(path.relative_to(PROJECT_ROOT)))

    return cfg, source_configs, optuna


def _apply_run_metadata(
    cfg: dict[str, Any],
    *,
    benchmark: str,
    backbone: str,
    constraint: str | None,
    budget: str,
    optuna: str | None,
    seed: int,
    mode: str,
    run_name: str,
    output_root: str | Path | None,
    source_configs: list[str],
) -> None:
    root = repo_path(output_root) if output_root else default_output_root(mode)
    cfg.setdefault("paths", {}).setdefault(
        "output_dir",
        str(
            root
            / safe_name(benchmark)
            / safe_name(constraint or "unconstrained")
            / safe_name(backbone)
            / safe_name(budget)
            / f"seed_{seed}"
        ),
    )

    cfg.setdefault("wandb_logging", {})
    cfg["wandb_logging"].setdefault("project", "omni_hc")
    cfg["wandb_logging"].setdefault("run_name", run_name)
    cfg["wandb_logging"].setdefault(
        "tags",
        [
            safe_name(benchmark),
            safe_name(backbone),
            safe_name(constraint or "unconstrained"),
            safe_name(budget),
        ],
    )

    if mode == "tune":
        cfg.setdefault("optuna", {})
        cfg["optuna"].setdefault(
            "save_dir", str(Path(cfg["paths"]["output_dir"]) / "trials")
        )
        cfg["optuna"].setdefault("run_name", run_name)

    ntest = cfg.get("data", {}).get("ntest", 200)
    cfg["experiment"] = {
        "mode": mode,
        "benchmark": benchmark,
        "backbone": backbone,
        "constraint": constraint or "unconstrained",
        "budget": budget,
        "optuna": optuna,
        "seed": seed,
        "source_configs": source_configs,
        "split_policy": (
            "Training and tuning use train/validation data. "
            f"The canonical test split is data.ntest={ntest} and should only "
            "be evaluated with scripts/test.py after final model selection."
        ),
    }


def _check_output_dir_conflict(candidate_dir: Path, resolved_cfg: dict) -> Path:
    """Return candidate_dir if it is safe to reuse, else a timestamped subfolder.

    A mismatch means the seed_N folder was produced by a different experiment
    (e.g. different constraint/val config). We never silently overwrite that run.
    """
    existing = candidate_dir / "resolved_config.yaml"
    if not existing.exists():
        return candidate_dir
    try:
        with open(existing, encoding="utf-8") as fh:
            stored = yaml.safe_load(fh) or {}
    except Exception:
        return candidate_dir

    def _strip(d: dict) -> dict:
        d = deepcopy(d)
        d.pop("backend", None)
        return d

    if _strip(stored) == _strip(resolved_cfg):
        return candidate_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return candidate_dir / timestamp


def compose_run_config(
    *,
    benchmark: str | None = None,
    backbone: str | None = None,
    constraint: str | None = None,
    budget: str | None = None,
    experiment: str | Path | None = None,
    optuna: str | None = None,
    mode: str = "train",
    seed: int | None = None,
    output_root: str | Path | None = None,
    extra_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose a runnable config from named components or a previously-resolved config file.

    Two entry paths:

    A. ``experiment`` points to a fully-resolved run config (has ``benchmark.name`` and
       ``model.backbone`` as dicts) — load it as-is, apply seed/mode/overrides, return.

    B. Named component composition — resolve benchmark, backbone, constraint, budget (and
       optuna for tune mode) to their YAML files, merge them in order, then apply any
       experiment-spec overrides and extra_overrides.  An experiment spec YAML (e.g.
       ``configs/experiments/darcy/fno_small.yaml``) may supply the component names and
       an ``overrides`` block instead of passing them all as CLI args.
    """
    experiment_cfg = _load_experiment(experiment)

    # Path A: already-resolved run config → pass through with overrides/mode applied.
    if experiment is not None and _is_resolved_run_config(experiment_cfg):
        cfg = deepcopy(experiment_cfg)
        if extra_overrides:
            cfg = deep_merge(cfg, extra_overrides)
        if seed is not None:
            cfg.setdefault("training", {})["seed"] = int(seed)
        cfg.setdefault("experiment", {})["mode"] = mode
        return cfg

    # Path B: named component composition.
    benchmark_name = _experiment_value(experiment_cfg, "benchmark", benchmark)
    backbone_name = _experiment_value(experiment_cfg, "backbone", backbone)
    constraint_name = _experiment_value(experiment_cfg, "constraint", constraint)
    budget_name = _experiment_value(experiment_cfg, "budget", budget) or (
        "tune_debug" if mode == "tune" else "debug"
    )
    optuna_name = _experiment_value(experiment_cfg, "optuna", optuna)

    if benchmark_name is None:
        raise ValueError("benchmark is required")
    if backbone_name is None:
        raise ValueError("backbone is required")

    cfg, source_configs, optuna_name = _load_components(
        benchmark_name, backbone_name, constraint_name, budget_name, optuna_name, mode
    )
    cfg = deep_merge(cfg, deepcopy(experiment_cfg.get("overrides", {}) or {}))
    if extra_overrides:
        cfg = deep_merge(cfg, extra_overrides)

    cfg.setdefault("training", {})
    seed_value = int(seed if seed is not None else cfg["training"].get("seed", 42))
    cfg["training"]["seed"] = seed_value

    run_name = str(
        experiment_cfg.get("name")
        or _run_label(
            benchmark=benchmark_name,
            backbone=backbone_name,
            constraint=constraint_name,
            budget=budget_name,
            seed=seed_value,
        )
    )

    _apply_run_metadata(
        cfg,
        benchmark=benchmark_name,
        backbone=backbone_name,
        constraint=constraint_name,
        budget=budget_name,
        optuna=optuna_name,
        seed=seed_value,
        mode=mode,
        run_name=run_name,
        output_root=output_root,
        source_configs=source_configs,
    )

    seed_dir = Path(cfg["paths"]["output_dir"])
    actual_dir = _check_output_dir_conflict(seed_dir, cfg)
    if actual_dir != seed_dir:
        cfg["paths"]["output_dir"] = str(actual_dir)
        if cfg.get("optuna", {}).get("save_dir") == str(seed_dir / "trials"):
            cfg["optuna"]["save_dir"] = str(actual_dir / "trials")

    return cfg
