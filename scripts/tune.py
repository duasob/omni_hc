import argparse
from pathlib import Path

import torch
import yaml

from omni_hc.core import compose_run_config, parse_dotted_overrides
from omni_hc.training import tune_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment YAML composition spec. Alias for --experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Experiment YAML composition spec.",
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument(
        "--constraint",
        type=str,
        default=None,
        help="Constraint name/path. Use none or unconstrained to skip.",
    )
    parser.add_argument("--budget", type=str, default=None)
    parser.add_argument(
        "--optuna",
        type=str,
        default=None,
        help="Optuna search-space config name/path. Defaults from benchmark+constraint when available.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root for generated output dirs when the config does not set one.",
    )
    parser.add_argument(
        "--nsl-root",
        type=str,
        default=None,
        help="Optional explicit path to Neural-Solver-Library.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config override, e.g. constraint.latent_module=blocks.-1.ln_3. Repeatable.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


if __name__ == "__main__":
    args = parse_args()
    cfg = compose_run_config(
        benchmark=args.benchmark,
        backbone=args.backbone,
        constraint=args.constraint,
        budget=args.budget,
        experiment=args.experiment or args.config,
        optuna=args.optuna,
        mode="tune",
        seed=args.seed,
        output_root=args.output_root,
        extra_overrides=parse_dotted_overrides(args.override),
    )
    study = tune_benchmark(
        cfg,
        nsl_root=None if args.nsl_root is None else Path(args.nsl_root),
        device=resolve_device(args.device),
    )
    print("best_trial_value", study.best_value)
    print("best_trial_params", study.best_trial.params)

    save_dir = cfg.get("optuna", {}).get("save_dir")
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "best_trial_value": float(study.best_value),
            "best_trial_number": int(study.best_trial.number),
            "best_trial_params": dict(study.best_trial.params),
            "num_trials": len(study.trials),
        }
        with open(save_dir / "best_trial.yaml", "w", encoding="utf-8") as handle:
            yaml.safe_dump(summary, handle, sort_keys=False)
