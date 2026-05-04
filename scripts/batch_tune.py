from __future__ import annotations

import argparse
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from batch_train import (
    PROJECT_ROOT,
    default_output_root,
    deep_merge,
    load_budget,
    load_run_base_config,
    repo_path,
    safe_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate resolved configs and launch a batch of Optuna tuning runs."
    )
    parser.add_argument("--sweep", required=True, help="Path to a sweep YAML file.")
    parser.add_argument(
        "--budget",
        default="tune_debug",
        help="Tune budget name from configs/budgets or a path to a YAML override.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Optional run names to tune from the sweep.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=None,
        help="Optional run names to skip from the sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write configs and print commands without launching tuning.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device passed through to scripts/tune.py: auto | cpu | cuda.",
    )
    parser.add_argument(
        "--nsl-root",
        default=None,
        help="Optional Neural-Solver-Library root passed through to scripts/tune.py.",
    )
    parser.add_argument(
        "--config-root",
        default="artifacts/generated_configs",
        help="Directory for generated resolved configs.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for durable tuning outputs. Defaults to Drive on Colab.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue launching later tuning runs when one run fails.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging during tuning. Disabled by default for Colab stability.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_config(cfg: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def selected_runs(runs: list[dict[str, Any]], *, only, skip) -> list[dict[str, Any]]:
    only_set = {safe_name(name) for name in only or []}
    skip_set = {safe_name(name) for name in skip or []}
    selected = []
    for run in runs:
        run_name = safe_name(str(run["name"]))
        if only_set and run_name not in only_set:
            continue
        if run_name in skip_set:
            continue
        selected.append({**run, "name": run_name})
    return selected


def resolve_tune_config(
    *,
    sweep_name: str,
    budget_name: str,
    run: dict[str, Any],
    budget_cfg: dict[str, Any],
    output_root: Path,
    enable_wandb: bool,
) -> dict[str, Any]:
    cfg, source_configs = load_run_base_config(run)
    cfg = deep_merge(cfg, budget_cfg)
    cfg = deep_merge(cfg, deepcopy(run.get("overrides", {}) or {}))

    run_name = str(run["name"])
    seed = int(cfg.get("training", {}).get("seed", 42))
    tune_dir = output_root / sweep_name / budget_name / run_name / f"seed_{seed}"
    trial_dir = tune_dir / "trials"

    cfg.setdefault("paths", {})
    cfg["paths"]["output_dir"] = str(tune_dir / "base")

    cfg.setdefault("optuna", {})
    cfg["optuna"]["save_dir"] = str(trial_dir)
    cfg["optuna"]["run_name"] = f"{sweep_name}_{budget_name}_{run_name}_seed_{seed}"

    cfg.setdefault("wandb_logging", {})
    cfg["wandb_logging"]["wandb"] = bool(enable_wandb)
    cfg["wandb_logging"]["run_name"] = f"{sweep_name}_{budget_name}_{run_name}_seed_{seed}"
    if "tags" in run:
        cfg["wandb_logging"]["tags"] = list(run["tags"])

    cfg["experiment"] = {
        "sweep": sweep_name,
        "budget": budget_name,
        "run": run_name,
        "seed": seed,
        "source_configs": source_configs,
        "tags": list(run.get("tags", [])),
        "mode": "tune",
        "split_policy": (
            "Optuna tuning uses validation metrics only. The canonical test split "
            "is data.ntest=200 and should only be evaluated with scripts/test.py "
            "after final model selection."
        ),
    }
    return cfg


def command_for_config(config_path: Path, *, device: str, nsl_root: str | None) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/tune.py"),
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
    sweep = load_yaml(sweep_path)
    sweep_name = safe_name(str(sweep.get("name") or sweep_path.stem))
    budget_name, budget_cfg = load_budget(args.budget)
    budget_name = safe_name(budget_name)
    runs = selected_runs(sweep.get("runs", []), only=args.only, skip=args.skip)
    if not runs:
        raise ValueError("No sweep runs selected.")

    config_root = repo_path(args.config_root) / sweep_name / budget_name
    output_root = repo_path(args.output_root) if args.output_root else default_output_root("tune")

    jobs: list[tuple[str, Path, list[str]]] = []
    for run in runs:
        cfg = resolve_tune_config(
            sweep_name=sweep_name,
            budget_name=budget_name,
            run=run,
            budget_cfg=budget_cfg,
            output_root=output_root,
            enable_wandb=args.wandb,
        )
        generated_config = config_root / str(run["name"]) / "tune.yaml"
        write_config(cfg, generated_config)
        jobs.append(
            (
                str(run["name"]),
                generated_config,
                command_for_config(
                    generated_config,
                    device=args.device,
                    nsl_root=args.nsl_root,
                ),
            )
        )

    print(f"Generated {len(jobs)} tune config(s) under {config_root}", flush=True)
    print(f"Tune outputs will be written under {output_root / sweep_name / budget_name}", flush=True)
    if not args.wandb:
        print("W&B logging is disabled for tuning. Pass --wandb to enable it.", flush=True)
    for _, _, command in jobs:
        print(" ".join(command), flush=True)

    if args.dry_run:
        return 0

    for index, (run_name, generated_config, command) in enumerate(jobs, start=1):
        print(
            f"\n[{index}/{len(jobs)}] Tuning {run_name}: {generated_config}",
            flush=True,
        )
        env = dict(os.environ)
        if not args.wandb:
            env["WANDB_MODE"] = "disabled"
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False, env=env)
        if completed.returncode != 0:
            print(
                f"[{index}/{len(jobs)}] {run_name} failed with exit code {completed.returncode}",
                flush=True,
            )
            if not args.continue_on_failure:
                return int(completed.returncode)
        else:
            print(f"[{index}/{len(jobs)}] Finished tuning {run_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
