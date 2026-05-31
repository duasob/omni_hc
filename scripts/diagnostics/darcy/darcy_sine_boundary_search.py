from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch
import yaml

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from _common import load_darcy_arrays, write_csv

from omni_hc.constraints import SineBoundaryConstraint


def _int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def _str_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Random/grid search for SineBoundaryConstraint coeff_head pretraining. "
            "This trains only the boundary head and ranks trials by validation "
            "boundary rel-L2."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--downsamplex", type=int, default=5)
    p.add_argument("--downsampley", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=1000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--n-modes", type=_int_list, default=_int_list("21"))
    p.add_argument("--hidden-dims", type=_int_list, default=_int_list("256,512"))
    p.add_argument("--n-layers", type=_int_list, default=_int_list("3,5,10"))
    p.add_argument("--lrs", type=_float_list, default=_float_list("3e-4,1e-3,3e-3"))
    p.add_argument(
        "--weight-decays", type=_float_list, default=_float_list("0,1e-5,1e-4")
    )
    p.add_argument("--batch-sizes", type=_int_list, default=_int_list("64,128"))
    p.add_argument(
        "--feature-modes",
        type=_str_list,
        default=_str_list(
            "boundary,boundary_inner,boundary_stats,boundary_inner_stats"
        ),
        help=(
            "Comma-separated feature modes: boundary, boundary_inner, "
            "boundary_stats, boundary_inner_stats, full."
        ),
    )
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument(
        "--max-trials",
        type=int,
        default=32,
        help="Randomly sample this many grid points. Use 0 with --exhaustive for all.",
    )
    p.add_argument("--exhaustive", action="store_true")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/darcy/darcy_sine_boundary_search"),
    )
    p.add_argument(
        "--save-best",
        type=Path,
        default=None,
        help="Optional path for a retrained best coeff_head checkpoint.",
    )
    return p.parse_args()


def _trial_grid(args: argparse.Namespace) -> list[dict[str, object]]:
    keys = (
        "feature_mode",
        "n_modes",
        "hidden_dim",
        "n_layers",
        "lr",
        "weight_decay",
        "batch_size",
    )
    values = (
        args.feature_modes,
        args.n_modes,
        args.hidden_dims,
        args.n_layers,
        args.lrs,
        args.weight_decays,
        args.batch_sizes,
    )
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _select_trials(args: argparse.Namespace, grid: list[dict[str, object]]):
    if args.exhaustive:
        return grid
    if args.max_trials <= 0 or args.max_trials >= len(grid):
        return grid
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(grid), size=args.max_trials, replace=False)
    return [grid[int(i)] for i in chosen]


def _load_arrays(args: argparse.Namespace):
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
        verbose=True,
    )
    n_total = min(args.max_samples, int(sol.shape[0]))
    if n_total <= 1:
        raise ValueError("Need at least two samples for a train/validation search")
    return coeff[:n_total], sol[:n_total], mat_path


def _train_trial(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    H: int,
    W: int,
    device: torch.device,
    args: argparse.Namespace,
    trial: dict[str, object],
    trial_number: int,
    save_path: Path | None = None,
):
    torch.manual_seed(args.seed + trial_number)
    constraint = SineBoundaryConstraint(
        n_modes=int(trial["n_modes"]),
        grid_shape=(H, W),
        hidden_dim=int(trial["hidden_dim"]),
        n_layers=int(trial["n_layers"]),
        feature_mode=str(trial["feature_mode"]),
    ).to(device)
    log = constraint.pretrain_coeff_head(
        coeff,
        sol,
        device=device,
        epochs=args.epochs,
        lr=float(trial["lr"]),
        weight_decay=float(trial["weight_decay"]),
        batch_size=int(trial["batch_size"]),
        val_frac=args.val_frac,
        val_seed=args.val_seed,
        save_path=save_path,
    )
    param_count = sum(p.numel() for p in constraint.coeff_head.parameters())
    score = log["best_val_rel_l2"]
    if score is None:
        score = log["final_val_rel_l2"]
    return {
        "trial": trial_number,
        "param_count": int(param_count),
        "best_val_rel_l2": float(score),
        "best_epoch": log["best_epoch"],
        "final_train_rel_l2": log["final_train_rel_l2"],
        "final_val_rel_l2": log["final_val_rel_l2"],
        **trial,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    coeff, sol, mat_path = _load_arrays(args)
    H, W = int(sol.shape[1]), int(sol.shape[2])

    grid = _trial_grid(args)
    trials = _select_trials(args, grid)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded Darcy: path={mat_path}, grid={H}x{W}, samples={len(sol)}, "
        f"grid_size={len(grid)}, trials={len(trials)}, device={device}"
    )

    rows: list[dict[str, object]] = []
    for i, trial in enumerate(trials):
        print(f"\n[trial {i + 1}/{len(trials)}] {trial}")
        row = _train_trial(
            coeff,
            sol,
            H=H,
            W=W,
            device=device,
            args=args,
            trial=trial,
            trial_number=i,
        )
        rows.append(row)
        print(
            f"[trial {i}] best_val_rel_l2={row['best_val_rel_l2']:.4e} "
            f"params={row['param_count']}"
        )

    rows.sort(key=lambda r: float(r["best_val_rel_l2"]))
    csv_path = args.out_dir / "sine_boundary_search.csv"
    write_csv(csv_path, rows)

    best = rows[0]
    summary = {
        "best_trial": best,
        "num_trials": len(rows),
        "grid_size": len(grid),
        "data": {
            "path": str(mat_path),
            "split": args.split,
            "samples": int(len(sol)),
            "grid_shape": [H, W],
            "downsamplex": args.downsamplex,
            "downsampley": args.downsampley,
        },
        "recommended_constraint": {
            "name": "sine_boundary_constraint",
            "feature_mode": best["feature_mode"],
            "n_modes": int(best["n_modes"]),
            "hidden_dim": int(best["hidden_dim"]),
            "n_layers": int(best["n_layers"]),
            "act": "gelu",
            "coeff_head_pretrain": {
                "epochs": args.epochs,
                "lr": float(best["lr"]),
                "weight_decay": float(best["weight_decay"]),
                "batch_size": int(best["batch_size"]),
                "max_samples": args.max_samples,
                "val_frac": args.val_frac,
                "val_seed": args.val_seed,
            },
        },
    }
    summary_path = args.out_dir / "best_sine_boundary_trial.yaml"
    with open(summary_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(summary, handle, sort_keys=False)

    print(f"\nwrote {csv_path}")
    print(f"wrote {summary_path}")
    print(
        "best: "
        f"feature_mode={best['feature_mode']} n_modes={best['n_modes']} "
        f"hidden_dim={best['hidden_dim']} n_layers={best['n_layers']} "
        f"lr={best['lr']} weight_decay={best['weight_decay']} "
        f"batch_size={best['batch_size']} "
        f"val_rel_l2={best['best_val_rel_l2']:.4e}"
    )

    if args.save_best is not None:
        print(f"\nretraining best trial and saving checkpoint to {args.save_best}")
        _train_trial(
            coeff,
            sol,
            H=H,
            W=W,
            device=device,
            args=args,
            trial=best,
            trial_number=len(rows),
            save_path=args.save_best,
        )


if __name__ == "__main__":
    main()
