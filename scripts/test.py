import argparse
from pathlib import Path

import torch

from omni_hc.core import compose_run_config, load_yaml_file
from omni_hc.training import test_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Full resolved run YAML config. Alias for --experiment.",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Full resolved run YAML config.",
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root for generated output dirs when the config does not set one.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Defaults to <paths.output_dir>/best.pt",
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


def load_test_config(args):
    config_path = args.experiment or args.config
    if config_path is None:
        return compose_run_config(
            benchmark=args.benchmark,
            backbone=args.backbone,
            constraint=args.constraint,
            budget=args.budget,
            mode="test",
            seed=args.seed,
            output_root=args.output_root,
        )

    cfg = load_yaml_file(config_path)
    if args.seed is not None:
        cfg.setdefault("training", {})
        cfg["training"]["seed"] = int(args.seed)
    if args.output_root is not None:
        cfg.setdefault("paths", {})
        cfg["paths"]["output_dir"] = str(Path(args.output_root))
    elif args.checkpoint is not None:
        cfg.setdefault("paths", {})
        cfg["paths"]["output_dir"] = str(Path(args.checkpoint).expanduser().parent)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["mode"] = "test"
    return cfg


if __name__ == "__main__":
    args = parse_args()
    cfg = load_test_config(args)
    result = test_benchmark(
        cfg,
        nsl_root=None if args.nsl_root is None else Path(args.nsl_root),
        device=resolve_device(args.device),
        checkpoint_path=args.checkpoint,
    )
    metrics = result["metrics"]
    print(
        f"test_mse={metrics['mse']:.6f} "
        f"test_rel_l2={metrics['rel_l2']:.6f}"
    )
