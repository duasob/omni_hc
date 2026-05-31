"""Search SineBoundaryConstraint coeff_head configs under a *no-validation*
protocol: train on the FULL train split (no held-out val) and rank trials by
boundary rel-L2 on a fixed slice of the test split.

This mirrors the way outputs/darcy/coeff_head.pt was produced (train on all
available data, no val holdout). Use darcy_sine_boundary_search.py instead when
you want an honest held-out (val) ranking.
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

from _common import load_darcy_arrays, write_csv
from omni_hc.constraints import SineBoundaryConstraint


def _int_list(v: str) -> list[int]:
    return [int(x) for x in v.split(",") if x.strip()]


def _float_list(v: str) -> list[float]:
    return [float(x) for x in v.split(",") if x.strip()]


def _str_list(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    p.add_argument("--downsamplex", type=int, default=5)
    p.add_argument("--downsampley", type=int, default=5)
    p.add_argument("--train-samples", type=int, default=1024)
    p.add_argument("--test-samples", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--n-modes", type=_int_list, default=_int_list("21"))
    p.add_argument("--hidden-dims", type=_int_list, default=_int_list("256"))
    p.add_argument("--n-layers", type=_int_list, default=_int_list("3"))
    p.add_argument("--lrs", type=_float_list, default=_float_list("3e-4,1e-3,3e-3"))
    p.add_argument("--weight-decays", type=_float_list, default=_float_list("0,1e-5,1e-4"))
    p.add_argument("--batch-sizes", type=_int_list, default=_int_list("64,128"))
    p.add_argument(
        "--feature-modes",
        type=_str_list,
        default=_str_list("boundary,boundary_stats,boundary_inner_stats,full"),
    )
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--max-trials", type=int, default=0, help="0 = exhaustive grid")
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("artifacts/darcy/darcy_sine_boundary_test_search"),
    )
    p.add_argument("--save-best", type=Path, default=None)
    return p.parse_args()


def _grid(args):
    keys = ("feature_mode", "n_modes", "hidden_dim", "n_layers", "lr", "weight_decay", "batch_size")
    vals = (args.feature_modes, args.n_modes, args.hidden_dims, args.n_layers,
            args.lrs, args.weight_decays, args.batch_sizes)
    grid = [dict(zip(keys, c)) for c in itertools.product(*vals)]
    if args.max_trials and args.max_trials < len(grid):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(grid), size=args.max_trials, replace=False)
        return [grid[int(i)] for i in idx], len(grid)
    return grid, len(grid)


def _boundary_target(c: SineBoundaryConstraint, sol_t: torch.Tensor) -> torch.Tensor:
    return torch.cat([
        sol_t[:, c.idx_bottom, 0], sol_t[:, c.idx_top, 0],
        sol_t[:, c.idx_left[1:-1], 0], sol_t[:, c.idx_right[1:-1], 0],
    ], dim=-1)


def _boundary_pred(c: SineBoundaryConstraint, fx: torch.Tensor) -> torch.Tensor:
    feats = c._boundary_feats(fx)
    coeffs = c.coeff_head(feats).view(fx.shape[0], 4, c.n_modes)
    u_b = coeffs[:, 0] @ c.basis_h.T
    u_t = coeffs[:, 1] @ c.basis_h.T
    u_l = coeffs[:, 2] @ c.basis_v.T
    u_r = coeffs[:, 3] @ c.basis_v.T
    return torch.cat([u_b, u_t, u_l[:, 1:-1], u_r[:, 1:-1]], dim=-1)


def _rel_l2(pred, targ):
    num = torch.linalg.norm(pred - targ, dim=-1)
    den = torch.linalg.norm(targ, dim=-1).clamp_min(1e-12)
    return num / den


def train_eval(coeff_tr, sol_tr, coeff_te, sol_te, *, H, W, device, args, trial, tnum, save_path=None):
    torch.manual_seed(args.seed + tnum)
    c = SineBoundaryConstraint(
        n_modes=int(trial["n_modes"]), grid_shape=(H, W),
        hidden_dim=int(trial["hidden_dim"]), n_layers=int(trial["n_layers"]),
        feature_mode=str(trial["feature_mode"]),
    ).to(device)

    n_tr = coeff_tr.shape[0]
    fx_tr = torch.as_tensor(coeff_tr.reshape(n_tr, -1, 1), dtype=torch.float32).to(device)
    sol_tr_t = torch.as_tensor(sol_tr.reshape(n_tr, -1, 1), dtype=torch.float32).to(device)
    n_te = coeff_te.shape[0]
    fx_te = torch.as_tensor(coeff_te.reshape(n_te, -1, 1), dtype=torch.float32).to(device)
    sol_te_t = torch.as_tensor(sol_te.reshape(n_te, -1, 1), dtype=torch.float32).to(device)

    # Feature Z-score stats + target scale from the (full) train split.
    with torch.no_grad():
        raw = c._raw_permeability_feats(fx_tr)
        c._feat_mean = raw.mean(0)
        c._feat_std = raw.std(0).clamp(min=1e-6)
        y_tr_b = _boundary_target(c, sol_tr_t)
        y_scale = float(torch.sqrt((y_tr_b ** 2).mean()).item()) + 1e-12

    targ_tr = _boundary_target(c, sol_tr_t) / y_scale
    targ_te = _boundary_target(c, sol_te_t) / y_scale

    opt = torch.optim.Adam(c.coeff_head.parameters(), lr=float(trial["lr"]),
                           weight_decay=float(trial["weight_decay"]))
    bs = int(trial["batch_size"])

    best_te = float("inf")
    best_epoch = -1
    best_state = None
    final_tr = float("nan")
    for epoch in range(args.epochs):
        c.coeff_head.train()
        perm = torch.randperm(n_tr, device=device)
        tr_rl2 = 0.0
        nb = 0
        for i in range(0, n_tr, bs):
            idx = perm[i:i + bs]
            pred = _boundary_pred(c, fx_tr[idx])
            loss = torch.nn.functional.mse_loss(pred, targ_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                tr_rl2 += float(_rel_l2(pred, targ_tr[idx]).mean().item())
            nb += 1
        final_tr = tr_rl2 / max(nb, 1)

        c.coeff_head.eval()
        with torch.no_grad():
            te_rl2 = float(_rel_l2(_boundary_pred(c, fx_te), targ_te).mean().item())
        if te_rl2 < best_te:
            best_te = te_rl2
            best_epoch = epoch + 1
            if save_path is not None:
                import copy
                best_state = copy.deepcopy(c.coeff_head.state_dict())

    if save_path is not None and best_state is not None:
        c.coeff_head.load_state_dict(best_state)
        with torch.no_grad():
            last = [m for m in c.coeff_head.modules() if isinstance(m, torch.nn.Linear)][-1]
            last.weight.mul_(y_scale)
            if last.bias is not None:
                last.bias.mul_(y_scale)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": c.coeff_head.state_dict(),
                    "feat_mean": c._feat_mean.cpu(), "feat_std": c._feat_std.cpu()},
                   str(save_path))
        print(f"saved best checkpoint to {save_path}", flush=True)

    params = sum(p.numel() for p in c.coeff_head.parameters())
    return {"trial": tnum, "param_count": int(params),
            "best_test_rel_l2": float(best_te), "best_epoch": best_epoch,
            "final_train_rel_l2": float(final_tr), **trial}


def main():
    args = parse_args()
    device = torch.device(args.device)

    coeff_tr, sol_tr, tr_path = load_darcy_arrays(
        args.data_dir, split="train", downsamplex=args.downsamplex,
        downsampley=args.downsampley, verbose=True)
    coeff_te, sol_te, te_path = load_darcy_arrays(
        args.data_dir, split="test", downsamplex=args.downsamplex,
        downsampley=args.downsampley, verbose=True)

    n_tr = min(args.train_samples, int(sol_tr.shape[0]))
    n_te = min(args.test_samples, int(sol_te.shape[0]))
    coeff_tr, sol_tr = coeff_tr[:n_tr], sol_tr[:n_tr]
    coeff_te, sol_te = coeff_te[:n_te], sol_te[:n_te]
    H, W = int(sol_tr.shape[1]), int(sol_tr.shape[2])

    trials, grid_size = _grid(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train={tr_path} ({n_tr})  test={te_path} ({n_te})  grid={H}x{W}  "
          f"grid_size={grid_size}  trials={len(trials)}  device={device}")

    rows = []
    for i, trial in enumerate(trials):
        print(f"\n[trial {i + 1}/{len(trials)}] {trial}", flush=True)
        row = train_eval(coeff_tr, sol_tr, coeff_te, sol_te, H=H, W=W,
                         device=device, args=args, trial=trial, tnum=i)
        rows.append(row)
        print(f"[trial {i}] best_test_rel_l2={row['best_test_rel_l2']:.4e} "
              f"@{row['best_epoch']} train={row['final_train_rel_l2']:.4e} "
              f"params={row['param_count']}", flush=True)

    rows.sort(key=lambda r: r["best_test_rel_l2"])
    csv_path = args.out_dir / "sine_boundary_test_search.csv"
    write_csv(csv_path, rows)
    print(f"\nwrote {csv_path}")
    best = rows[0]
    print(f"BEST: feature_mode={best['feature_mode']} n_modes={best['n_modes']} "
          f"hidden={best['hidden_dim']} n_layers={best['n_layers']} lr={best['lr']} "
          f"wd={best['weight_decay']} bs={best['batch_size']} "
          f"-> test_rel_l2={best['best_test_rel_l2']:.4e} @epoch {best['best_epoch']}")

    if args.save_best is not None:
        print(f"\nretraining best and saving to {args.save_best}")
        train_eval(coeff_tr, sol_tr, coeff_te, sol_te, H=H, W=W, device=device,
                   args=args, trial=best, tnum=len(rows), save_path=args.save_best)


if __name__ == "__main__":
    main()
