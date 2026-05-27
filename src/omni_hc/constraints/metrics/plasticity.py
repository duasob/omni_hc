"""Constraint-error metrics for the Plasticity 2D benchmark."""

from __future__ import annotations

from typing import Any

import torch

from ..base import ConstraintDiagnostic


def _reshape_coords(pred: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Return predicted physical coordinates with shape ``(B, T, H, W, 2)``."""
    h, w = tuple(meta["shapelist"])
    t_out = int(meta.get("t_out", meta.get("T_out", 1)))
    out_dim = int(meta.get("out_dim", 4))

    if pred.dim() == 5:
        if pred.shape[-1] < 2:
            raise ValueError(f"Expected at least 2 coordinate channels, got {pred.shape}")
        if pred.shape[1] == t_out and pred.shape[2] == h and pred.shape[3] == w:
            return pred[..., :2]
        if pred.shape[1] == h and pred.shape[2] == w and pred.shape[3] == t_out:
            return pred.permute(0, 3, 1, 2, 4)[..., :2]

    if pred.dim() == 4:
        if pred.shape[-1] == out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0, 3, 1, 2, 4
            )[..., :2]
        if pred.shape[-1] == t_out * out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0, 3, 1, 2, 4
            )[..., :2]

    if pred.dim() == 3:
        if pred.shape[1] == h * w and pred.shape[-1] == t_out * out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0, 3, 1, 2, 4
            )[..., :2]
        if pred.shape[1] == t_out and pred.shape[2] == h * w * out_dim:
            return pred.reshape(pred.shape[0], t_out, h, w, out_dim)[..., :2]

    raise ValueError(
        "Unsupported plasticity prediction shape "
        f"{tuple(pred.shape)} for shapelist={(h, w)}, t_out={t_out}, out_dim={out_dim}"
    )


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    del batch
    coords = _reshape_coords(pred, meta)
    dx = coords[:, :, 1:, :, 0] - coords[:, :, :-1, :, 0]
    dy = coords[:, :, :, 1:, 1] - coords[:, :, :, :-1, 1]
    neg = torch.cat([dx.reshape(-1), dy.reshape(-1)], dim=0) < 0
    count = neg.to(torch.float32).sum()
    fraction = neg.to(torch.float32).mean()
    return {
        "constraint/neg_spacing_count": ConstraintDiagnostic(
            value=count, reduce="sum"
        ),
        "constraint/neg_spacing_fraction": ConstraintDiagnostic(
            value=fraction, reduce="mean"
        ),
    }
