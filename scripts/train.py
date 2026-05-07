import argparse
from pathlib import Path

import torch

from omni_hc.core import compose_run_config, parse_dotted_overrides
from omni_hc.training import train_benchmark


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root for generated output dirs when the config does not set one.",
    )
    parser.add_argument(  # TODO: Remove this argument and assume NSL installed in external/
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
        mode="train",
        seed=args.seed,
        output_root=args.output_root,
        extra_overrides=parse_dotted_overrides(args.override),
    )
    train_benchmark(
        cfg,
        nsl_root=None if args.nsl_root is None else Path(args.nsl_root),
        device=resolve_device(args.device),
    )
