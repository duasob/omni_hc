from __future__ import annotations

import argparse
import copy
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
            "TPE/random/grid search for SineBoundaryConstraint coeff_head pretraining. "
            "This trains only the boundary head on Darcy train samples and ranks "
            "trials by boundary rel-L2 on a separate validation split. For the "
            "current Darcy diagnostic workflow, the default validation split is "
            "the 200-sample test split."
        )
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    p.add_argument("--train-split", choices=("train", "test"), default="train")
    p.add_argument("--validation-split", choices=("train", "test"), default="test")
    p.add_argument("--downsamplex", type=int, default=5)
    p.add_argument("--downsampley", type=int, default=5)
    p.add_argument("--train-samples", type=int, default=1000)
    p.add_argument("--validation-samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")

    p.add_argument("--n-modes", type=_int_list, default=_int_list("21,31,41"))
    p.add_argument("--hidden-dims", type=_int_list, default=_int_list("512,768,1024"))
    p.add_argument("--n-layers", type=_int_list, default=_int_list("5,8,12"))
    p.add_argument("--lrs", type=_float_list, default=_float_list("1e-4,3e-4,1e-3"))
    p.add_argument(
        "--weight-decays", type=_float_list, default=_float_list("0,1e-6,1e-5")
    )
    p.add_argument("--batch-sizes", type=_int_list, default=_int_list("64,128"))
    p.add_argument(
        "--feature-modes",
        type=_str_list,
        default=_str_list("boundary"),
        help=(
            "Comma-separated feature modes: boundary, boundary_inner, "
            "boundary_stats, boundary_inner_stats, full. Defaults to boundary "
            "so the first search isolates larger coeff_head capacity."
        ),
    )
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=100,
        help=(
            "Stop a trial after this many epochs without validation improvement. "
            "Use 0 to disable early stopping."
        ),
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation rel-L2 improvement required to reset patience.",
    )
    p.add_argument(
        "--max-trials",
        type=int,
        default=100,
        help=(
            "Number of trials for random/TPE search. For random search, use 0 "
            "with --exhaustive for all grid points."
        ),
    )
    p.add_argument(
        "--search-method",
        choices=("tpe", "random"),
        default="tpe",
        help="Hyperparameter search strategy. TPE uses Optuna's TPESampler.",
    )
    p.add_argument(
        "--tpe-startup-trials",
        type=int,
        default=10,
        help="Number of random startup trials before TPE begins modelling.",
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


def _suggest_tpe_trial(optuna_trial, args: argparse.Namespace) -> dict[str, object]:
    return {
        "feature_mode": optuna_trial.suggest_categorical(
            "feature_mode", args.feature_modes
        ),
        "n_modes": optuna_trial.suggest_categorical("n_modes", args.n_modes),
        "hidden_dim": optuna_trial.suggest_categorical(
            "hidden_dim", args.hidden_dims
        ),
        "n_layers": optuna_trial.suggest_categorical("n_layers", args.n_layers),
        "lr": optuna_trial.suggest_categorical("lr", args.lrs),
        "weight_decay": optuna_trial.suggest_categorical(
            "weight_decay", args.weight_decays
        ),
        "batch_size": optuna_trial.suggest_categorical(
            "batch_size", args.batch_sizes
        ),
    }


def _trial_key(trial: dict[str, object]) -> tuple[object, ...]:
    return (
        trial["feature_mode"],
        int(trial["n_modes"]),
        int(trial["hidden_dim"]),
        int(trial["n_layers"]),
        float(trial["lr"]),
        float(trial["weight_decay"]),
        int(trial["batch_size"]),
    )


def _load_split(
    args: argparse.Namespace,
    *,
    split: str,
    max_samples: int,
):
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
        verbose=True,
    )
    n_total = min(max_samples, int(sol.shape[0]))
    if n_total <= 0:
        raise ValueError(f"Need at least one sample from split={split!r}")
    return coeff[:n_total], sol[:n_total], mat_path


def _boundary_target(
    constraint: SineBoundaryConstraint,
    sol_t: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        [
            sol_t[:, constraint.idx_bottom, 0],
            sol_t[:, constraint.idx_top, 0],
            sol_t[:, constraint.idx_left[1:-1], 0],
            sol_t[:, constraint.idx_right[1:-1], 0],
        ],
        dim=-1,
    )


def _boundary_pred(
    constraint: SineBoundaryConstraint,
    fx: torch.Tensor,
) -> torch.Tensor:
    feats = constraint._boundary_feats(fx)
    coeffs = constraint.coeff_head(feats).view(fx.shape[0], 4, constraint.n_modes)
    u_bottom = coeffs[:, 0] @ constraint.basis_h.T
    u_top = coeffs[:, 1] @ constraint.basis_h.T
    u_left = coeffs[:, 2] @ constraint.basis_v.T
    u_right = coeffs[:, 3] @ constraint.basis_v.T
    return torch.cat(
        [u_bottom, u_top, u_left[:, 1:-1], u_right[:, 1:-1]],
        dim=-1,
    )


def _rel_l2(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    diff = torch.linalg.norm(pred - targ, dim=-1)
    norm = torch.linalg.norm(targ, dim=-1).clamp_min(1e-12)
    return diff / norm


@torch.no_grad()
def _eval_boundary_rel_l2(
    constraint: SineBoundaryConstraint,
    fx: torch.Tensor,
    targ: torch.Tensor,
    *,
    batch_size: int,
) -> float:
    constraint.coeff_head.eval()
    total = 0.0
    n_seen = 0
    for start in range(0, fx.shape[0], batch_size):
        end = start + batch_size
        pred = _boundary_pred(constraint, fx[start:end])
        rel = _rel_l2(pred, targ[start:end])
        total += float(rel.sum().item())
        n_seen += int(rel.numel())
    return total / max(n_seen, 1)


def _bake_target_scale(
    constraint: SineBoundaryConstraint,
    y_scale: float,
) -> None:
    with torch.no_grad():
        last_linear = None
        for module in reversed(list(constraint.coeff_head.modules())):
            if isinstance(module, torch.nn.Linear):
                last_linear = module
                break
        if last_linear is None:
            raise RuntimeError("coeff_head has no Linear layer to absorb y_scale")
        last_linear.weight.mul_(y_scale)
        if last_linear.bias is not None:
            last_linear.bias.mul_(y_scale)


def _train_trial(
    coeff_train: np.ndarray,
    sol_train: np.ndarray,
    coeff_val: np.ndarray,
    sol_val: np.ndarray,
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

    n_train = coeff_train.shape[0]
    n_val = coeff_val.shape[0]
    fx_train = torch.as_tensor(
        coeff_train.reshape(n_train, -1, 1),
        dtype=torch.float32,
        device=device,
    )
    sol_train_t = torch.as_tensor(
        sol_train.reshape(n_train, -1, 1),
        dtype=torch.float32,
        device=device,
    )
    fx_val = torch.as_tensor(
        coeff_val.reshape(n_val, -1, 1),
        dtype=torch.float32,
        device=device,
    )
    sol_val_t = torch.as_tensor(
        sol_val.reshape(n_val, -1, 1),
        dtype=torch.float32,
        device=device,
    )

    with torch.no_grad():
        raw_feats = constraint._raw_permeability_feats(fx_train)
        constraint._feat_mean = raw_feats.mean(0)
        constraint._feat_std = raw_feats.std(0).clamp(min=1e-6)
        y_train_boundary = _boundary_target(constraint, sol_train_t)
        y_scale = float(torch.sqrt((y_train_boundary**2).mean()).item()) + 1e-12

    target_train = _boundary_target(constraint, sol_train_t) / y_scale
    target_val = _boundary_target(constraint, sol_val_t) / y_scale

    opt = torch.optim.Adam(
        constraint.coeff_head.parameters(),
        lr=float(trial["lr"]),
        weight_decay=float(trial["weight_decay"]),
    )
    batch_size = int(trial["batch_size"])
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    final_train_rel_l2 = float("nan")
    final_val_rel_l2 = float("nan")
    stopped_epoch = args.epochs

    print(
        f"[coeff_head search] {args.epochs} epochs  lr={trial['lr']}  "
        f"n_train={n_train}  n_val={n_val}  "
        f"patience={args.early_stopping_patience}",
        flush=True,
    )
    for epoch in range(args.epochs):
        constraint.coeff_head.train()
        perm = torch.randperm(n_train, device=device)
        train_total = 0.0
        train_seen = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            pred = _boundary_pred(constraint, fx_train[idx])
            loss = torch.nn.functional.mse_loss(pred, target_train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                rel = _rel_l2(pred, target_train[idx])
                train_total += float(rel.sum().item())
                train_seen += int(rel.numel())

        final_train_rel_l2 = train_total / max(train_seen, 1)
        final_val_rel_l2 = _eval_boundary_rel_l2(
            constraint,
            fx_val,
            target_val,
            batch_size=batch_size,
        )
        if final_val_rel_l2 < best_val - args.early_stopping_min_delta:
            best_val = final_val_rel_l2
            best_epoch = epoch + 1
            best_state = copy.deepcopy(constraint.coeff_head.state_dict())

        if epoch == 0 or (epoch + 1) % 25 == 0:
            print(
                f"[coeff_head search] epoch {epoch + 1}/{args.epochs}"
                f"  train_rel_l2={final_train_rel_l2:.4e}"
                f"  val_rel_l2={final_val_rel_l2:.4e}"
                f"  best_val_rel_l2={best_val:.4e}@{best_epoch}",
                flush=True,
            )

        no_improve_epochs = epoch + 1 - best_epoch
        if (
            args.early_stopping_patience > 0
            and best_epoch > 0
            and no_improve_epochs >= args.early_stopping_patience
        ):
            stopped_epoch = epoch + 1
            print(
                f"[coeff_head search] early stop at epoch {stopped_epoch}/{args.epochs}"
                f"  best_val_rel_l2={best_val:.4e}@{best_epoch}"
                f"  patience={args.early_stopping_patience}",
                flush=True,
            )
            break

    if best_state is not None:
        constraint.coeff_head.load_state_dict(best_state)

    if save_path is not None:
        _bake_target_scale(constraint, y_scale)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": constraint.coeff_head.state_dict(),
                "feat_mean": constraint._feat_mean.cpu(),
                "feat_std": constraint._feat_std.cpu(),
            },
            str(save_path),
        )
        print(f"[coeff_head search] saved best checkpoint to {save_path}", flush=True)

    param_count = sum(p.numel() for p in constraint.coeff_head.parameters())
    return {
        "trial": trial_number,
        "param_count": int(param_count),
        "best_val_rel_l2": float(best_val),
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "final_train_rel_l2": float(final_train_rel_l2),
        "final_val_rel_l2": float(final_val_rel_l2),
        **trial,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    coeff_train, sol_train, train_path = _load_split(
        args,
        split=args.train_split,
        max_samples=args.train_samples,
    )
    coeff_val, sol_val, validation_path = _load_split(
        args,
        split=args.validation_split,
        max_samples=args.validation_samples,
    )
    H, W = int(sol_train.shape[1]), int(sol_train.shape[2])
    if tuple(sol_val.shape[1:3]) != (H, W):
        raise ValueError(
            f"Train grid {(H, W)} does not match validation grid {sol_val.shape[1:3]}"
        )

    grid = _trial_grid(args)
    use_tpe = args.search_method == "tpe" and not args.exhaustive
    trials = [] if use_tpe else _select_trials(args, grid)
    n_trials = min(args.max_trials, len(grid)) if use_tpe else len(trials)
    if use_tpe and n_trials <= 0:
        raise ValueError("--max-trials must be > 0 when --search-method tpe")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded Darcy: train_path={train_path}, validation_path={validation_path}, "
        f"grid={H}x{W}, train_samples={len(sol_train)}, "
        f"validation_samples={len(sol_val)}, "
        f"grid_size={len(grid)}, trials={n_trials}, search_method={args.search_method}, "
        f"device={device}"
    )

    rows: list[dict[str, object]] = []
    if use_tpe:
        import optuna

        sampler = optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=args.tpe_startup_trials,
        )
        study = optuna.create_study(direction="minimize", sampler=sampler)
        target_trials = n_trials
        seen_scores: dict[tuple[object, ...], float] = {}
        max_suggestions = max(target_trials * 20, target_trials + 100)
        n_suggestions = 0
        while len(rows) < target_trials:
            if n_suggestions >= max_suggestions:
                print(
                    f"[tpe] stopped after {n_suggestions} suggestions because "
                    f"unique configs stalled at {len(rows)}/{target_trials}",
                    flush=True,
                )
                break
            n_suggestions += 1
            optuna_trial = study.ask()
            trial = _suggest_tpe_trial(optuna_trial, args)
            key = _trial_key(trial)
            if key in seen_scores:
                study.tell(optuna_trial, seen_scores[key])
                print(
                    f"[tpe] skipped duplicate suggestion optuna_trial="
                    f"{optuna_trial.number}: {trial}",
                    flush=True,
                )
                continue

            i = len(rows)
            print(f"\n[trial {i + 1}/{target_trials}] {trial}")
            row = _train_trial(
                coeff_train,
                sol_train,
                coeff_val,
                sol_val,
                H=H,
                W=W,
                device=device,
                args=args,
                trial=trial,
                trial_number=i,
            )
            row["optuna_trial"] = int(optuna_trial.number)
            rows.append(row)
            score = float(row["best_val_rel_l2"])
            seen_scores[key] = score
            study.tell(optuna_trial, score)
            print(
                f"[trial {i}] best_val_rel_l2={score:.4e} "
                f"params={row['param_count']}"
            )
    else:
        for i, trial in enumerate(trials):
            print(f"\n[trial {i + 1}/{len(trials)}] {trial}")
            row = _train_trial(
                coeff_train,
                sol_train,
                coeff_val,
                sol_val,
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
        "search_method": args.search_method,
        "tpe_startup_trials": args.tpe_startup_trials if use_tpe else None,
        "data": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "train_split": args.train_split,
            "validation_split": args.validation_split,
            "train_samples": int(len(sol_train)),
            "validation_samples": int(len(sol_val)),
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
                "train_samples": args.train_samples,
                "early_stopping_patience": args.early_stopping_patience,
                "early_stopping_min_delta": args.early_stopping_min_delta,
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
        f"best_epoch={best['best_epoch']} stopped_epoch={best['stopped_epoch']} "
        f"val_rel_l2={best['best_val_rel_l2']:.4e}"
    )

    if args.save_best is not None:
        print(f"\nretraining best trial and saving checkpoint to {args.save_best}")
        _train_trial(
            coeff_train,
            sol_train,
            coeff_val,
            sol_val,
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
