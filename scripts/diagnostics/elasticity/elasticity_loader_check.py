from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import load_elasticity_arrays, select_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check Elasticity dataset files and NSL-style train/test tensor shapes."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/elasticity"),
        help=(
            "Directory containing the elasticity files. Accepts data/elasticity, "
            "data/fno, or a directory containing the .npy files directly."
        ),
    )
    parser.add_argument("--ntrain", type=int, default=1000)
    parser.add_argument("--ntest", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Report UnitTransformer-style sigma statistics from the train split.",
    )
    return parser.parse_args()


def print_stats(name: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    print(
        f"{name}: mean={values.mean(): .6e}, std={values.std(): .6e}, "
        f"min={values.min(): .6e}, max={values.max(): .6e}"
    )


def main() -> None:
    args = parse_args()
    coords, sigma, sigma_path, xy_path = load_elasticity_arrays(args.data_dir)
    train_xy, train_sigma = select_split(
        coords,
        sigma,
        split="train",
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    test_xy, test_sigma = select_split(
        coords,
        sigma,
        split="test",
        ntrain=args.ntrain,
        ntest=args.ntest,
    )

    print("Loaded Elasticity dataset")
    print(f"  sigma_path: {sigma_path}")
    print(f"  xy_path:    {xy_path}")
    print(f"  coords:     {coords.shape}  # NSL order: (samples, points, 2)")
    print(f"  sigma:      {sigma.shape}  # NSL order: (samples, points)")
    print(f"  train_xy:   {train_xy.shape}")
    print(f"  train_s:    {train_sigma.shape}")
    print(f"  test_xy:    {test_xy.shape}")
    print(f"  test_s:     {test_sigma.shape}")
    print(f"  shapelist:  [{train_sigma.shape[1]}]")

    batch = min(int(args.batch_size), int(train_sigma.shape[0]))
    print("NSL TensorDataset batch shapes")
    print(f"  x:          {train_xy[:batch].shape}")
    print(f"  fx:         {train_xy[:batch].shape}")
    print(f"  y:          {train_sigma[:batch].shape}")

    print_stats("sigma all", sigma)
    print_stats("sigma train", train_sigma)
    print_stats("sigma test", test_sigma)

    coord_min = coords.reshape(-1, 2).min(axis=0)
    coord_max = coords.reshape(-1, 2).max(axis=0)
    print(
        "coordinate bounds: "
        f"x=[{coord_min[0]: .6e}, {coord_max[0]: .6e}], "
        f"y=[{coord_min[1]: .6e}, {coord_max[1]: .6e}]"
    )

    if args.normalize:
        mean = train_sigma.mean(axis=(0, 1), keepdims=True)
        std = train_sigma.std(axis=(0, 1), keepdims=True) + 1e-8
        encoded = (train_sigma - mean) / std
        print_stats("sigma train encoded", encoded)


if __name__ == "__main__":
    main()
