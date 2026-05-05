from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any

from .config import deep_merge, load_composed_config, load_yaml_file

PROJECT_ROOT = Path(__file__).resolve().parents[3]


BUDGET_CONFIGS = {
    "debug": PROJECT_ROOT / "configs/budgets/debug.yaml",
    "smoke": PROJECT_ROOT / "configs/budgets/smoke.yaml",
    "search": PROJECT_ROOT / "configs/budgets/search.yaml",
    "final": PROJECT_ROOT / "configs/budgets/final.yaml",
    "tune_debug": PROJECT_ROOT / "configs/budgets/tune_debug.yaml",
    "tune_colab": PROJECT_ROOT / "configs/budgets/tune_colab.yaml",
}


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def safe_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def default_output_root(mode: str) -> Path:
    env_value = os.environ.get("OMNI_HC_OUTPUT_ROOT")
    if env_value:
        return repo_path(env_value) / mode

    colab_drive = Path("/content/drive/MyDrive")
    if colab_drive.exists():
        return colab_drive / "omni_hc" / mode

    return PROJECT_ROOT / "outputs" / mode


def _component_path(
    value: str | Path | None,
    *,
    kind: str,
    benchmark: str | None = None,
    required: bool = True,
) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"{kind} is required")
        return None

    value_str = str(value)
    if value_str.lower() in {"", "none", "null", "unconstrained"}:
        if required:
            raise ValueError(f"{kind} is required")
        return None

    candidate = repo_path(value_str)
    if candidate.exists():
        return candidate

    if kind == "benchmark":
        candidates = [PROJECT_ROOT / "configs/benchmarks" / value_str / "base.yaml"]
    elif kind == "backbone":
        if benchmark is None:
            raise ValueError("backbone resolution requires a benchmark")
        candidates = [
            PROJECT_ROOT / "configs/backbones" / benchmark / f"{value_str}.yaml",
            PROJECT_ROOT / "configs/backbones" / benchmark / f"{value_str.lower()}.yaml",
        ]
    elif kind == "constraint":
        if benchmark is None:
            raise ValueError("constraint resolution requires a benchmark")
        candidates = [
            PROJECT_ROOT / "configs/constraints" / benchmark / f"{value_str}.yaml",
            PROJECT_ROOT / "configs/constraints" / f"{value_str}.yaml",
        ]
    elif kind == "budget":
        candidates = [BUDGET_CONFIGS.get(value_str) or PROJECT_ROOT / "configs/budgets" / f"{value_str}.yaml"]
    elif kind == "optuna":
        if benchmark is None:
            raise ValueError("optuna resolution requires a benchmark")
        candidates = [
            PROJECT_ROOT / "configs/optuna" / benchmark / f"{value_str}.yaml",
            PROJECT_ROOT / "configs/optuna" / f"{value_str}.yaml",
        ]
    else:
        raise ValueError(f"Unknown config component kind: {kind}")

    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate

    if required:
        searched = ", ".join(str(item) for item in candidates if item is not None)
        raise FileNotFoundError(f"Could not resolve {kind}={value_str!r}. Searched: {searched}")
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
    experiment: dict[str, Any],
    key: str,
    cli_value: str | None,
) -> str | None:
    if cli_value is not None:
        return cli_value
    value = experiment.get(key)
    if value is None:
        return None
    return str(value)


def _default_optuna_name(
    *,
    benchmark: str,
    constraint: str | None,
    backbone: str,
) -> str | None:
    names = []
    if constraint and constraint.lower() not in {"none", "unconstrained"}:
        names.append(constraint)
    names.append(backbone)

    for name in names:
        path = _component_path(
            name,
            kind="optuna",
            benchmark=benchmark,
            required=False,
        )
        if path is not None:
            return name
    return None


def _run_label(
    *,
    benchmark: str,
    backbone: str,
    constraint: str | None,
    budget: str,
    seed: int,
) -> str:
    constraint_name = constraint or "unconstrained"
    return "_".join(safe_name(item) for item in [benchmark, backbone, constraint_name, budget, f"seed_{seed}"])


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
    """Compose a runnable config from named config components.

    Experiment files are composition specs. They may set benchmark, backbone,
    constraint, budget, optuna, and an overrides mapping.
    """

    experiment_cfg = _load_experiment(experiment)
    if "extends" in experiment_cfg:
        cfg = load_composed_config(repo_path(experiment))
        cfg = deep_merge(cfg, deepcopy(experiment_cfg.get("overrides", {}) or {}))
        if extra_overrides:
            cfg = deep_merge(cfg, extra_overrides)
        return cfg

    benchmark_name = _experiment_value(experiment_cfg, "benchmark", benchmark)
    backbone_name = _experiment_value(experiment_cfg, "backbone", backbone)
    constraint_name = _experiment_value(experiment_cfg, "constraint", constraint)
    budget_name = _experiment_value(experiment_cfg, "budget", budget) or (
        "tune_debug" if mode == "tune" else "debug"
    )
    optuna_name = _experiment_value(experiment_cfg, "optuna", optuna)

    if benchmark_name is None:
        raise ValueError("benchmark is required when no legacy extends config is used")
    if backbone_name is None:
        raise ValueError("backbone is required when no legacy extends config is used")

    cfg: dict[str, Any] = {}
    source_configs: list[str] = []
    component_paths = [
        _component_path(benchmark_name, kind="benchmark"),
        _component_path(backbone_name, kind="backbone", benchmark=benchmark_name),
        _component_path(
            constraint_name,
            kind="constraint",
            benchmark=benchmark_name,
            required=False,
        ),
    ]

    if mode == "tune":
        if optuna_name is None:
            optuna_name = _default_optuna_name(
                benchmark=benchmark_name,
                constraint=constraint_name,
                backbone=backbone_name,
            )
        component_paths.append(
            _component_path(
                optuna_name,
                kind="optuna",
                benchmark=benchmark_name,
                required=optuna_name is not None,
            )
        )

    component_paths.append(_component_path(budget_name, kind="budget"))

    for path in component_paths:
        if path is None:
            continue
        cfg = deep_merge(cfg, _load_component(path))
        source_configs.append(str(path.relative_to(PROJECT_ROOT)))

    cfg = deep_merge(cfg, deepcopy(experiment_cfg.get("overrides", {}) or {}))
    if extra_overrides:
        cfg = deep_merge(cfg, extra_overrides)

    cfg.setdefault("training", {})
    seed_value = int(seed if seed is not None else cfg["training"].get("seed", 42))
    cfg["training"]["seed"] = seed_value

    run_name = str(experiment_cfg.get("name") or _run_label(
        benchmark=benchmark_name,
        backbone=backbone_name,
        constraint=constraint_name,
        budget=budget_name,
        seed=seed_value,
    ))

    root = repo_path(output_root) if output_root else default_output_root(mode)
    cfg.setdefault("paths", {})
    cfg["paths"].setdefault(
        "output_dir",
        str(root / safe_name(benchmark_name) / safe_name(backbone_name) / safe_name(constraint_name or "unconstrained") / safe_name(budget_name) / f"seed_{seed_value}"),
    )

    cfg.setdefault("wandb_logging", {})
    cfg["wandb_logging"].setdefault("project", "omni_hc")
    cfg["wandb_logging"].setdefault("run_name", run_name)
    cfg["wandb_logging"].setdefault(
        "tags",
        [
            safe_name(benchmark_name),
            safe_name(backbone_name),
            safe_name(constraint_name or "unconstrained"),
            safe_name(budget_name),
        ],
    )

    if mode == "tune":
        cfg.setdefault("optuna", {})
        cfg["optuna"].setdefault(
            "save_dir",
            str(Path(cfg["paths"]["output_dir"]) / "trials"),
        )
        cfg["optuna"].setdefault("run_name", run_name)

    ntest = cfg.get("data", {}).get("ntest", 200)
    cfg["experiment"] = {
        "mode": mode,
        "benchmark": benchmark_name,
        "backbone": backbone_name,
        "constraint": constraint_name or "unconstrained",
        "budget": budget_name,
        "optuna": optuna_name,
        "seed": seed_value,
        "source_configs": source_configs,
        "split_policy": (
            "Training and tuning use train/validation data. "
            f"The canonical test split is data.ntest={ntest} and should only "
            "be evaluated with scripts/test.py after final model selection."
        ),
    }
    return cfg

