from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

import sys
# Archived study: relocated from scripts/diagnostics/darcy/; re-expose its _common helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "diagnostics" / "darcy"))

from _common import (
    load_darcy_arrays,
    require_matplotlib,
    write_csv,
)

# --------------------------------------------------------------------------- #
# Boundary extraction
# --------------------------------------------------------------------------- #
#
# Empirical test for the SineBoundaryConstraint design question: how well can a
# small MLP map the boundary of the permeability field a (=coeff) to the
# boundary of the solution u (=sol), under three reconstruction heads:
#
#   1. sine        : u = sum_k c_k sin(k pi j / (L-1))           (sign-free)
#   2. env+cosine  : u = sin(pi j /(L-1)) * softplus(N),         N = cosine series
#   3. env+mlp     : u = sin(pi j /(L-1)) * softplus(N),         N = per-node MLP
#
# Models 2 and 3 are non-negative and corner-zero by construction; model 1 is
# the current constraint (sign-unconstrained, corner-zero only).
#
# Input  feats : a on [bottom(W), top(W), left_inner(H-2), right_inner(H-2)]
# Target       : u on the same unique boundary node set
# Reported     : per-sample rel-L2 of predicted vs true boundary.


def edge_split(field: np.ndarray) -> tuple[np.ndarray, ...]:
    """(..., H, W) -> (bottom[W], top[W], left[H], right[H]) along last axes."""
    bottom = field[..., 0, :]
    top = field[..., -1, :]
    left = field[..., :, 0]
    right = field[..., :, -1]
    return bottom, top, left, right


def unique_boundary(bottom, top, left, right):
    """Concatenate edges with corners deduplicated (matches sine_boundary)."""
    cat = torch.cat if isinstance(bottom, torch.Tensor) else np.concatenate
    return cat([bottom, top, left[..., 1:-1], right[..., 1:-1]], axis=-1)


def build_dataset(coeff: np.ndarray, sol: np.ndarray):
    a_b, a_t, a_l, a_r = edge_split(coeff)
    u_b, u_t, u_l, u_r = edge_split(sol)
    x = unique_boundary(a_b, a_t, a_l, a_r).astype(np.float32)
    y = unique_boundary(u_b, u_t, u_l, u_r).astype(np.float32)
    return x, y


# --------------------------------------------------------------------------- #
# Reconstruction heads
# --------------------------------------------------------------------------- #
def _trunk(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.GELU()]
    for _ in range(max(n_layers, 0)):
        layers += [nn.Linear(hidden, hidden), nn.GELU()]
    return nn.Sequential(*layers)


def _sine_basis(L: int, n_modes: int) -> torch.Tensor:
    j = torch.arange(L, dtype=torch.float32)
    k = torch.arange(1, n_modes + 1, dtype=torch.float32)
    return torch.sin(math.pi * j[:, None] * k[None, :] / (L - 1))


def _cosine_basis(L: int, n_modes: int) -> torch.Tensor:
    """Includes the k=0 (DC) term so the interior shape can be non-zero-mean."""
    j = torch.arange(L, dtype=torch.float32)
    k = torch.arange(0, n_modes, dtype=torch.float32)
    return torch.cos(math.pi * j[:, None] * k[None, :] / (L - 1))


def _envelope(L: int) -> torch.Tensor:
    j = torch.arange(L, dtype=torch.float32)
    # sin(pi) is ~ -1e-7 in float; clamp so the envelope is exactly >= 0.
    return torch.sin(math.pi * j / (L - 1)).clamp_min(0.0)


class BoundaryModel(nn.Module):
    """Shared trunk + one of three edge-reconstruction heads.

    Edge lengths: bottom/top = W, left/right = H. Output is assembled back into
    the deduplicated unique-boundary vector for a like-for-like comparison.
    """

    def __init__(
        self,
        *,
        mode: str,
        in_dim: int,
        H: int,
        W: int,
        hidden: int,
        n_layers: int,
        n_modes: int,
    ):
        super().__init__()
        self.mode = mode
        self.H, self.W = H, W
        self.trunk = _trunk(in_dim, hidden, n_layers)

        if mode == "sine":
            self.head = nn.Linear(hidden, 2 * n_modes + 2 * n_modes)
            self.register_buffer("basis_h", _sine_basis(W, n_modes))
            self.register_buffer("basis_v", _sine_basis(H, n_modes))
            self.n_modes = n_modes
        elif mode == "env_cosine":
            self.head = nn.Linear(hidden, 4 * n_modes)
            self.register_buffer("cos_h", _cosine_basis(W, n_modes))
            self.register_buffer("cos_v", _cosine_basis(H, n_modes))
            self.register_buffer("env_h", _envelope(W))
            self.register_buffer("env_v", _envelope(H))
            self.n_modes = n_modes
        elif mode == "env_mlp":
            self.head = nn.Linear(hidden, 2 * W + 2 * H)
            self.register_buffer("env_h", _envelope(W))
            self.register_buffer("env_v", _envelope(H))
        else:
            raise ValueError(f"unknown mode {mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.trunk(x)
        h = self.head(z)
        H, W, K = self.H, self.W, getattr(self, "n_modes", 0)

        if self.mode == "sine":
            cb, ct = h[:, :K], h[:, K : 2 * K]
            cl, cr = h[:, 2 * K : 3 * K], h[:, 3 * K : 4 * K]
            u_b = cb @ self.basis_h.T
            u_t = ct @ self.basis_h.T
            u_l = cl @ self.basis_v.T
            u_r = cr @ self.basis_v.T
        elif self.mode == "env_cosine":
            cb, ct = h[:, :K], h[:, K : 2 * K]
            cl, cr = h[:, 2 * K : 3 * K], h[:, 3 * K : 4 * K]
            u_b = self.env_h * torch.nn.functional.softplus(cb @ self.cos_h.T)
            u_t = self.env_h * torch.nn.functional.softplus(ct @ self.cos_h.T)
            u_l = self.env_v * torch.nn.functional.softplus(cl @ self.cos_v.T)
            u_r = self.env_v * torch.nn.functional.softplus(cr @ self.cos_v.T)
        else:  # env_mlp
            n_b = h[:, :W]
            n_t = h[:, W : 2 * W]
            n_l = h[:, 2 * W : 2 * W + H]
            n_r = h[:, 2 * W + H : 2 * W + 2 * H]
            u_b = self.env_h * torch.nn.functional.softplus(n_b)
            u_t = self.env_h * torch.nn.functional.softplus(n_t)
            u_l = self.env_v * torch.nn.functional.softplus(n_l)
            u_r = self.env_v * torch.nn.functional.softplus(n_r)

        return unique_boundary(u_b, u_t, u_l, u_r)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #
def rel_l2_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = torch.linalg.norm(pred - target, dim=-1)
    den = torch.linalg.norm(target, dim=-1).clamp_min(1e-12)
    return num / den


def train_one(
    mode: str,
    x_tr,
    y_tr,
    x_va,
    y_va,
    *,
    H,
    W,
    hidden,
    n_layers,
    n_modes,
    epochs,
    lr,
    batch_size,
    weight_decay,
    device,
    seed,
):
    torch.manual_seed(seed)
    model = BoundaryModel(
        mode=mode,
        in_dim=x_tr.shape[1],
        H=H,
        W=W,
        hidden=hidden,
        n_layers=n_layers,
        n_modes=n_modes,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n = x_tr.shape[0]
    bs = min(batch_size, n)
    best_va = math.inf
    best_state = None
    history: list[tuple[int, float, float]] = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            opt.zero_grad(set_to_none=True)
            pred = model(x_tr[idx])
            loss = torch.nn.functional.mse_loss(pred, y_tr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        tr_mse = total / n

        model.eval()
        with torch.no_grad():
            va_rl2 = rel_l2_per_sample(model(x_va), y_va).mean().item()
        history.append((epoch, tr_mse, va_rl2))
        if va_rl2 < best_va:
            best_va = va_rl2
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_va, history


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        pred = model(x)
        rl2 = rel_l2_per_sample(pred, y)
        neg_frac_pred = (pred < 0).float().mean().item()
    return {
        "rel_l2_mean": float(rl2.mean()),
        "rel_l2_p95": float(torch.quantile(rl2, 0.95)),
        "rel_l2_max": float(rl2.max()),
        "pred_neg_frac": neg_frac_pred,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train 3 boundary-prediction heads (sine / env+cosine / "
        "env+mlp) mapping the Darcy permeability boundary to the solution "
        "boundary, and compare rel-L2."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    p.add_argument("--split", choices=("train", "test"), default="train")
    p.add_argument("--downsamplex", type=int, default=5)
    p.add_argument("--downsampley", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=1024)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--out-dir", type=Path, default=Path("artifacts/darcy/darcy_boundary_models")
    )
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def plot_history(histories: dict, out_dir: Path, show: bool) -> Path:
    require_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for mode, hist in histories.items():
        ep = [h[0] for h in hist]
        va = [h[2] for h in hist]
        ax.plot(ep, va, label=mode, linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val boundary rel-L2")
    ax.set_title("Boundary-prediction heads: validation rel-L2")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "darcy_boundary_models_history.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

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

    device = torch.device(args.device)
    # Standardize inputs on the training split (targets left raw; rel-L2 is
    # scale-invariant).
    mu = x_np[tr_idx].mean(axis=0, keepdims=True)
    sd = x_np[tr_idx].std(axis=0, keepdims=True) + 1e-8
    x_std = (x_np - mu) / sd

    # The Darcy boundary solution is O(1e-5); divide by a single scalar so MSE
    # gradients are well-conditioned. rel-L2 is invariant to this global
    # scalar, so the reported metric is unaffected.
    y_scale = float(np.sqrt((y_np[tr_idx] ** 2).mean())) + 1e-12
    y_s = y_np / y_scale

    x_tr = torch.as_tensor(x_std[tr_idx], device=device)
    y_tr = torch.as_tensor(y_s[tr_idx], device=device)
    x_va = torch.as_tensor(x_std[va_idx], device=device)
    y_va = torch.as_tensor(y_s[va_idx], device=device)

    gt_neg_frac = float((y_np < 0).mean())
    print(f"Loaded Darcy: path={mat_path}, sol={sol.shape}, split={args.split}")
    print(
        f"grid HxW={H}x{W}, boundary dim={x_np.shape[1]}, "
        f"train={len(tr_idx)}, val={len(va_idx)}"
    )
    print(
        f"ground-truth boundary negative-node fraction = {gt_neg_frac:.4e}  "
        f"(irreducible rel-L2 floor for non-negative heads)"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)

    modes = ["sine", "env_cosine", "env_mlp"]
    histories: dict[str, list] = {}
    rows: list[dict[str, object]] = []
    for mode in modes:
        model, best_va, hist = train_one(
            mode,
            x_tr,
            y_tr,
            x_va,
            y_va,
            H=H,
            W=W,
            hidden=args.hidden_dim,
            n_layers=args.n_layers,
            n_modes=args.n_modes,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            device=device,
            seed=args.seed,
        )
        histories[mode] = hist
        metrics = evaluate(model, x_va, y_va)
        rows.append(
            {
                "model": mode,
                "best_val_rel_l2": best_va,
                "gt_neg_frac": gt_neg_frac,
                **metrics,
            }
        )
        print(f"\n[{mode}]")
        print(f"  best val rel-L2 = {best_va:.4e}")
        for k, v in metrics.items():
            print(f"  {k:<14} = {v:.4e}")

    csv_path = args.out_dir / "darcy_boundary_models_summary.csv"
    write_csv(csv_path, rows)
    print(f"\nwrote {csv_path}")

    if args.no_plot:
        return
    if plt is None:
        print("matplotlib not installed; skipping plot.")
        return
    plot_path = plot_history(histories, args.out_dir, args.show)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
