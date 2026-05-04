from __future__ import annotations

import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omni_hc.core.config import deep_merge, load_composed_config, load_yaml_file


BUDGET_CONFIGS = {
    "debug": PROJECT_ROOT / "configs/budgets/debug.yaml",
    "smoke": PROJECT_ROOT / "configs/budgets/smoke.yaml",
    "search": PROJECT_ROOT / "configs/budgets/search.yaml",
    "final": PROJECT_ROOT / "configs/budgets/final.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resolved configs and launch a batch of training runs."
    )
    parser.add_argument("--sweep", required=True, help="Path to a sweep YAML file.")
    parser.add_argument(
        "--budget",
        default="debug",
        help="Budget name from configs/budgets or a path to a YAML override.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional seed list. Defaults to the budget/config seed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without launching training.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed through to scripts/train.py: auto | cpu | cuda.",
    )
    parser.add_argument(
        "--nsl-root",
        default=None,
        help="Optional Neural-Solver-Library root passed through to scripts/train.py.",
    )
    parser.add_argument(
        "--config-root",
        default="artifacts/generated_configs",
        help="Directory for generated resolved configs.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/batch",
        help="Root directory for batch run outputs.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue launching later runs when one run fails.",
    )
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_budget(path_or_name: str) -> tuple[str, dict[str, Any]]:
    budget_path = BUDGET_CONFIGS.get(path_or_name)
    budget_name = path_or_name
    if budget_path is None:
        budget_path = repo_path(path_or_name)
        budget_name = budget_path.stem
    return budget_name, load_yaml_file(budget_path)


def safe_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace("/", "_")
        .replace(" ", "_")
        .replace("-", "_")
    )


def resolve_run_config(
    *,
    sweep_name: str,
    budget_name: str,
    run: dict[str, Any],
    budget_cfg: dict[str, Any],
    seed: int | None,
    output_root: Path,
) -> dict[str, Any]:
    if "config" not in run:
        raise KeyError(f"Sweep run {run!r} must define a config path.")

    cfg = load_composed_config(repo_path(run["config"]))
    cfg = deep_merge(cfg, budget_cfg)
    cfg = deep_merge(cfg, deepcopy(run.get("overrides", {}) or {}))

    run_name = str(run["name"])
    seed_value = int(seed if seed is not None else cfg.get("training", {}).get("seed", 42))
    cfg.setdefault("training", {})
    cfg["training"]["seed"] = seed_value

    output_dir = output_root / sweep_name / budget_name / run_name / f"seed_{seed_value}"
    cfg.setdefault("paths", {})
    cfg["paths"]["output_dir"] = str(output_dir)

    cfg.setdefault("wandb_logging", {})
    cfg["wandb_logging"]["run_name"] = f"{sweep_name}_{budget_name}_{run_name}_seed_{seed_value}"
    if "tags" in run:
        cfg["wandb_logging"]["tags"] = list(run["tags"])

    cfg["experiment"] = {
        "sweep": sweep_name,
        "budget": budget_name,
        "run": run_name,
        "seed": seed_value,
        "source_config": str(run["config"]),
        "tags": list(run.get("tags", [])),
        "split_policy": (
            "Batch training and tuning use only train/validation data. "
            "The canonical test split is data.ntest=200 and should only be "
            "evaluated with scripts/test.py after final model selection."
        ),
    }
    return cfg


def write_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def command_for_config(config_path: Path, *, device: str, nsl_root: str | None) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train.py"),
        "--config",
        str(config_path),
        "--device",
        device,
    ]
    if nsl_root:
        command.extend(["--nsl-root", nsl_root])
    return command


def main() -> int:
    args = parse_args()
    sweep_path = repo_path(args.sweep)
    sweep = load_yaml_file(sweep_path)
    sweep_name = safe_name(str(sweep.get("name") or sweep_path.stem))
    budget_name, budget_cfg = load_budget(args.budget)
    budget_name = safe_name(budget_name)
    runs = sweep.get("runs", [])
    if not runs:
        raise ValueError(f"Sweep {sweep_path} must define at least one run.")

    config_root = repo_path(args.config_root) / sweep_name / budget_name
    output_root = repo_path(args.output_root)
    seeds = args.seeds if args.seeds is not None else [None]

    commands: list[list[str]] = []
    for run in runs:
        run_name = safe_name(str(run["name"]))
        for seed in seeds:
            cfg = resolve_run_config(
                sweep_name=sweep_name,
                budget_name=budget_name,
                run={**run, "name": run_name},
                budget_cfg=budget_cfg,
                seed=seed,
                output_root=output_root,
            )
            seed_value = int(cfg["training"]["seed"])
            generated_config = config_root / run_name / f"seed_{seed_value}.yaml"
            write_config(cfg, generated_config)
            commands.append(
                command_for_config(
                    generated_config,
                    device=args.device,
                    nsl_root=args.nsl_root,
                )
            )

    print(f"Generated {len(commands)} config(s) under {config_root}")
    for command in commands:
        print(" ".join(command))

    if args.dry_run:
        return 0

    for command in commands:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0 and not args.continue_on_failure:
            return int(completed.returncode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
