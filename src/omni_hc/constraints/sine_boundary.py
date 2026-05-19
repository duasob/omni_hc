from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import ConstrainedModel, ConstraintDiagnostic, ConstraintModule, _BUILD_META_KEYS
from .mean import build_mlp
from .utils.boundary_ops import apply_boundary_ansatz, channel_mask
from .utils.hooks import ForwardHookLatentExtractor


def _sine_basis(n_points: int, n_modes: int) -> torch.Tensor:
    """B[j, k] = sin(k * pi * j / (n_points - 1)).  Zero at j=0 and j=n_points-1."""
    j = torch.arange(n_points, dtype=torch.float32)
    k = torch.arange(1, n_modes + 1, dtype=torch.float32)
    return torch.sin(math.pi * j.unsqueeze(1) * k.unsqueeze(0) / (n_points - 1))


def _boundary_indices_2d(H: int, W: int) -> dict[str, torch.Tensor]:
    """Flat indices for each edge of an H×W row-major grid (y-first ordering)."""
    return {
        "bottom": torch.arange(W),
        "top":    torch.arange((H - 1) * W, H * W),
        "left":   torch.arange(0, H * W, W),
        "right":  torch.arange(W - 1, H * W, W),
    }


class SineBoundaryConstraint(ConstraintModule):
    """
    Predicts per-edge boundary values via a sine basis expansion and hard-enforces
    them on the main model prediction.

    For each edge of length L, the predicted profile is:
        u(x_j) = sum_{k=1}^{n_modes} c_k * sin(k * pi * j / (L - 1))

    Corner-zero is satisfied by construction for all k.

    The MLP that predicts the coefficients takes as input:
        [boundary permeability features] ++ [boundary-pooled transformer latent (optional)]

    Optionally, a latent representation from the backbone can be wired in via
    ForwardHookLatentExtractor (same mechanism as MeanConstraint latent_head).
    The latent is mean-pooled over boundary nodes before concatenation.
    """

    name = "sine_boundary_constraint"

    def __init__(
        self,
        *,
        n_modes: int,
        grid_shape: tuple[int, int],
        hidden_dim: int,
        n_layers: int = 0,
        act: str = "gelu",
        latent_dim: int = 0,
        extractor=None,
    ):
        super().__init__()
        H, W = int(grid_shape[0]), int(grid_shape[1])
        self.H, self.W = H, W
        self.n_modes = n_modes
        self.extractor = extractor

        # Sine basis matrices — (n_points, n_modes)
        self.register_buffer("basis_h", _sine_basis(W, n_modes))
        self.register_buffer("basis_v", _sine_basis(H, n_modes))

        # Flat boundary indices per edge
        idx = _boundary_indices_2d(H, W)
        for edge, tensor in idx.items():
            self.register_buffer(f"idx_{edge}", tensor)

        # Unique boundary nodes for latent pooling (corners deduplicated)
        all_boundary = torch.cat([
            idx["bottom"],
            idx["top"],
            idx["left"][1:-1],
            idx["right"][1:-1],
        ])
        self.register_buffer("idx_all_boundary", all_boundary)

        # Ansatz distance l for f = g + l * N. The indicator (0 on the
        # perimeter, 1 in the interior) reproduces the original hard-overwrite
        # exactly: on the boundary f = g (the sine profile), in the interior
        # f = N (the backbone, untouched). Swapping this for a smooth l would
        # blend the correction inward instead.
        distance = torch.ones(H * W, dtype=torch.float32)
        distance[all_boundary] = 0.0
        self.register_buffer("boundary_distance", distance.view(1, H * W, 1))

        # MLP input: bottom(W) + top(W) + left_inner(H-2) + right_inner(H-2) + latent
        boundary_feat_dim = 2 * W + 2 * (H - 2) + latent_dim
        self.coeff_head = build_mlp(
            boundary_feat_dim, hidden_dim, 4 * n_modes, n_layers=n_layers, act=act
        )

    @classmethod
    def build(
        cls,
        backbone: torch.nn.Module,
        model_context: dict[str, Any],
        constraint_cfg: dict[str, Any],
    ) -> ConstrainedModel:
        _SINE_META = _BUILD_META_KEYS | {"latent_module"}
        params = {k: v for k, v in constraint_cfg.items() if k not in _SINE_META}

        if "grid_shape" not in params:
            shapelist = model_context.get("shapelist")
            if shapelist is None:
                raise ValueError("model_context must contain 'shapelist' for SineBoundaryConstraint")
            params["grid_shape"] = tuple(int(s) for s in shapelist)

        extractor = None
        latent_module = constraint_cfg.get("latent_module")
        if latent_module:
            extractor = ForwardHookLatentExtractor(backbone, latent_module)
            backbone.register_forward_pre_hook(lambda *_: extractor.reset())
            if "latent_dim" not in params:
                n_hidden = model_context.get("n_hidden")
                if n_hidden is not None:
                    params["latent_dim"] = int(n_hidden)

        constraint = cls(**params, extractor=extractor)
        wrapped = ConstrainedModel(backbone=backbone, constraint=constraint)
        if constraint_cfg.get("freeze_base", False):
            for param in wrapped.backbone.parameters():
                param.requires_grad = False
        return wrapped

    @classmethod
    def log_media(cls, ctx) -> dict[str, str]:
        import matplotlib.pyplot as plt

        H, W = ctx.meta["shapelist"]
        pred   = ctx.pred[0, :, 0].cpu().numpy()    # [N]
        target = ctx.target[0, :, 0].cpu().numpy()  # [N]

        pred_2d   = pred.reshape(H, W)
        target_2d = target.reshape(H, W)

        xs = np.linspace(0, 1, W)
        ys = np.linspace(0, 1, H)

        edges = {
            "Bottom ($y=0$)":  (xs, pred_2d[0,  :], target_2d[0,  :]),
            "Top ($y=1$)":     (xs, pred_2d[-1, :], target_2d[-1, :]),
            "Left ($x=0$)":    (ys, pred_2d[:,  0], target_2d[:,  0]),
            "Right ($x=1$)":   (ys, pred_2d[:, -1], target_2d[:, -1]),
        }

        fig, axes = plt.subplots(1, 4, figsize=(14, 3), constrained_layout=True)
        for ax, (label, (pos, pred_edge, gt_edge)) in zip(axes, edges.items()):
            ax.plot(pos, gt_edge,   color="steelblue", linewidth=1.5, label="Ground truth")
            ax.plot(pos, pred_edge, color="darkorange", linewidth=1.5, label="Prediction", linestyle="--")
            ax.axhline(0, color="0.5", linewidth=0.8, linestyle=":")
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("Position")
            ax.set_ylabel("$u$")
        axes[0].legend(fontsize=8, frameon=False)

        if ctx.out_dir is not None:
            out_path = Path(ctx.out_dir) / f"boundary_profiles_epoch{ctx.epoch:04d}.png"
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            return {"constraint/boundary_profiles": str(out_path)}

        try:
            import wandb
            wandb.log(
                {"constraint/boundary_profiles": wandb.Image(fig)},
                step=ctx.step,
            )
        except Exception:
            pass
        plt.close(fig)
        return {}

    def _boundary_feats(self, fx: torch.Tensor) -> torch.Tensor:
        a = fx[:, :, 0]
        return torch.cat([
            a[:, self.idx_bottom],          # [B, W]
            a[:, self.idx_top],             # [B, W]
            a[:, self.idx_left[1:-1]],      # [B, H-2]
            a[:, self.idx_right[1:-1]],     # [B, H-2]
        ], dim=-1)

    def forward(self, *, pred, return_aux=False, **kwargs):
        fx = kwargs.get("fx")
        if fx is None:
            raise ValueError("SineBoundaryConstraint requires fx (input permeability)")

        B = pred.shape[0]
        feats = self._boundary_feats(fx)

        if self.extractor is not None:
            latent = self.extractor.get()
            if latent is not None:
                boundary_latent = latent[:, self.idx_all_boundary, :]  # [B, n_bnd, n_hidden]
                feats = torch.cat([feats, boundary_latent.mean(dim=1)], dim=-1)

        coeffs = self.coeff_head(feats).view(B, 4, self.n_modes)  # [B, 4, n_modes]

        u_bottom = coeffs[:, 0] @ self.basis_h.T   # [B, W]
        u_top    = coeffs[:, 1] @ self.basis_h.T   # [B, W]
        u_left   = coeffs[:, 2] @ self.basis_v.T   # [B, H]
        u_right  = coeffs[:, 3] @ self.basis_v.T   # [B, H]

        # Particular solution g: the sine profile on the perimeter, zero in the
        # interior. With l = boundary_distance (0 on perimeter, 1 interior),
        #   f = g + l * N
        # collapses to g on the boundary and N elsewhere. The write order
        # (bottom, top, left, right) matches the original overwrite; corners
        # are zero in every edge basis so the order is immaterial.
        g_flat = pred.new_zeros(B, self.H * self.W)
        g_flat[:, self.idx_bottom] = u_bottom
        g_flat[:, self.idx_top]    = u_top
        g_flat[:, self.idx_left]   = u_left
        g_flat[:, self.idx_right]  = u_right

        out = apply_boundary_ansatz(
            pred=pred,
            particular=g_flat.unsqueeze(-1),
            distance=self.boundary_distance,
            channel_mask=channel_mask(pred, [0]),
        )

        if return_aux:
            correction = (out[:, self.idx_all_boundary, 0] - pred[:, self.idx_all_boundary, 0]).abs()
            diagnostics = {
                "constraint/boundary_correction_mean_abs": ConstraintDiagnostic(
                    value=correction.mean(), reduce="mean"
                ),
            }
            return self.as_output(
                out,
                aux={"boundary_pred": out[:, self.idx_all_boundary, 0]},
                diagnostics=diagnostics,
            )
        return out
