import argparse
from pathlib import Path

import torch
import yaml

from omni_hc.core import load_composed_config
from omni_hc.training import tune_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/navier_stokes/fno_small_mean.yaml",
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
    return parser.parse_args()


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


if __name__ == "__main__":
    args = parse_args()
    cfg = load_composed_config(args.config)
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
