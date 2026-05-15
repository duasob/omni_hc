import argparse

import torch

from omni_hc.core import compose_run_config, parse_dotted_overrides
from omni_hc.training import train_benchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Experiment YAML composition spec.",
    )
    parser.add_argument("--benchmark", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--constraint", type=str, default=None)
    parser.add_argument("--budget", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Root for generated output dirs when the config does not set one.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cpu | cuda",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional full training checkpoint to resume from, typically latest.pt.",
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
        experiment=args.config,
        mode="train",
        seed=args.seed,
        output_root=args.output_root,
        extra_overrides=parse_dotted_overrides(args.override),
    )  # TODO: clean the internals of this function
    if args.checkpoint is not None:
        cfg.setdefault("training", {})["resume_checkpoint"] = args.checkpoint
    train_benchmark(cfg, device=resolve_device(args.device))
