from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from _common import load_darcy_arrays, require_matplotlib, write_csv
from darcy_boundary_models import build_dataset, evaluate, train_one


# --------------------------------------------------------------------------- #
# Sweep n_modes for the sine head to find where val rel-L2 plateaus.
#
# Reuses build_dataset / train_one / evaluate from darcy_boundary_models so the
# data pipeline and training loop are identical to the 3-head comparison.
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep --n-modes for one reconstruction head and report the "
        "best validation boundary rel-L2 at each truncation order."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--downsamplex", type=int, default=5)
    p.add_argument("--downsampley", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--mode",
        choices=("sine", "env_cosine"),
        default="sine",
        help="which spectral head to sweep (env_mlp has no mode knob)",
    )
    p.add_argument("--modes-min", type=int, default=1)
    p.add_argument(
        "--modes-max",
        type=int,
        default=0,
        help="0 => use min(H, W) - 2 (max independent DST-I modes)",
    )
    p.add_argument("--modes-step", type=int, default=5)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/darcy/darcy_boundary_n_modes")
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def prepare_split(args, device):
    """Same pipeline as darcy_boundary_models.main(): load, dedup boundary,
    standardize inputs, scalar-normalize targets, fixed train/val split."""
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    n_total = min(args.max_samples, int(sol.shape[0]))
    coeff, sol = coeff[:n_total], sol[:n_total]
    H, W = int(sol.shape[1]), int(sol.shape[2])

    x_np, y_np = build_dataset(coeff, sol)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(round(args.val_frac * n_total)))
    va_idx, tr_idx = perm[:n_val], perm[n_val:]

    mu = x_np[tr_idx].mean(axis=0, keepdims=True)
    sd = x_np[tr_idx].std(axis=0, keepdims=True) + 1e-8
    x_std = (x_np - mu) / sd

    y_scale = float(np.sqrt((y_np[tr_idx] ** 2).mean())) + 1e-12
    y_s = y_np / y_scale

    data = {
        "x_tr": torch.as_tensor(x_std[tr_idx], device=device),
        "y_tr": torch.as_tensor(y_s[tr_idx], device=device),
        "x_va": torch.as_tensor(x_std[va_idx], device=device),
        "y_va": torch.as_tensor(y_s[va_idx], device=device),
    }
    return data, H, W, mat_path, len(tr_idx), len(va_idx)


def plot_sweep(rows, mode, out_dir: Path, show: bool) -> Path:
    require_matplotlib(plt)
    ks = [r["n_modes"] for r in rows]
    rl2 = [r["best_val_rel_l2"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(ks, rl2, marker="o", linewidth=1.5, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("n_modes (truncation order)")
    ax.set_ylabel("best val boundary rel-L2")
    ax.set_title(f"Darcy boundary: rel-L2 vs n_modes ({mode} head)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    out_path = out_dir / f"darcy_boundary_n_modes_{mode}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

    device = torch.device(args.device)
    data, H, W, mat_path, n_tr, n_va = prepare_split(args, device)

    cap = min(H, W) - 2
    modes_max = cap if args.modes_max <= 0 else min(args.modes_max, cap)
    modes = list(range(args.modes_min, modes_max + 1, args.modes_step))
    if modes and modes[-1] != modes_max:
        modes.append(modes_max)

    print(f"Loaded Darcy: path={mat_path}, grid={H}x{W}, train={n_tr}, val={n_va}")
    print(f"sweeping {args.mode} head over n_modes={modes}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for k in modes:
        _model, best_va, _hist = train_one(
            args.mode,
            data["x_tr"],
            data["y_tr"],
            data["x_va"],
            data["y_va"],
            H=H,
            W=W,
            hidden=args.hidden_dim,
            n_layers=args.n_layers,
            n_modes=k,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )
        metrics = evaluate(_model, data["x_va"], data["y_va"])
        rows.append({"n_modes": k, "best_val_rel_l2": best_va, **metrics})
        print(
            f"  n_modes={k:>3}  best_val_rel_l2={best_va:.4e}  "
            f"pred_neg_frac={metrics['pred_neg_frac']:.2e}"
        )

    best = min(rows, key=lambda r: r["best_val_rel_l2"])
    print(
        f"\nbest: n_modes={best['n_modes']} -> "
        f"rel-L2={best['best_val_rel_l2']:.4e}"
    )

    csv_path = args.out_dir / f"darcy_boundary_n_modes_{args.mode}.csv"
    write_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    if args.no_plot:
        return
    if plt is None:
        print("matplotlib not installed; skipping plot.")
        return
    plot_path = plot_sweep(rows, args.mode, args.out_dir, args.show)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
