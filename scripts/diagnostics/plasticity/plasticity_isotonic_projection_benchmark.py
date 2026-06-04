from __future__ import annotations

import argparse
import time

import torch

from omni_hc.constraints import (
    PlasticityEnvelopeConstraint,
    PlasticityIsotonicRegression,
    PlasticityMeshConsistencyConstraint,
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_forward(module, *, pred: torch.Tensor, time_input: torch.Tensor, reps: int):
    for _ in range(2):
        module(pred=pred, T=time_input, return_aux=True)
    _sync(pred.device)

    start = time.perf_counter()
    for _ in range(reps):
        module(pred=pred, T=time_input, return_aux=True)
    _sync(pred.device)
    return (time.perf_counter() - start) / max(reps, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark plasticity isotonic projection backends.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--i-count", type=int, default=101)
    parser.add_argument("--j-count", type=int, default=31)
    parser.add_argument("--reps", type=int, default=10)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")

    device = torch.device(args.device)
    shape = (args.i_count, args.j_count)
    n_points = args.i_count * args.j_count
    time_input = torch.zeros(args.batch_size, 1, device=device)

    modules = [
        (
            "isotonic_native",
            PlasticityIsotonicRegression(
                shapelist=shape,
                envelope_source="constant",
                projection_device="native",
            ),
        ),
        (
            "isotonic_cpu",
            PlasticityIsotonicRegression(
                shapelist=shape,
                envelope_source="constant",
                projection_device="cpu",
            ),
        ),
        (
            "isotonic_auto",
            PlasticityIsotonicRegression(
                shapelist=shape,
                envelope_source="constant",
                projection_device="auto",
            ),
        ),
        (
            "mesh_consistency",
            PlasticityMeshConsistencyConstraint(shapelist=shape),
        ),
        (
            "envelope",
            PlasticityEnvelopeConstraint(shapelist=shape, envelope_source="constant"),
        ),
    ]

    print(f"device={device} shape={shape} batch_size={args.batch_size} reps={args.reps}")
    for name, module in modules:
        module = module.to(device)
        pred = torch.randn(
            args.batch_size,
            n_points,
            module.backbone_out_dim,
            device=device,
        )
        seconds = _time_forward(module, pred=pred, time_input=time_input, reps=args.reps)
        print(f"{name}: {seconds:.6f} s/forward")


if __name__ == "__main__":
    main()
