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


def _signed_cell_areas(coords: torch.Tensor) -> torch.Tensor:
    p00 = coords[:, :, :-1, :-1]
    p10 = coords[:, :, 1:, :-1]
    p11 = coords[:, :, 1:, 1:]
    p01 = coords[:, :, :-1, 1:]
    x = torch.stack([p00[..., 0], p10[..., 0], p11[..., 0], p01[..., 0]], dim=-1)
    y = torch.stack([p00[..., 1], p10[..., 1], p11[..., 1], p01[..., 1]], dim=-1)
    return 0.5 * torch.sum(
        x * torch.roll(y, shifts=-1, dims=-1)
        - y * torch.roll(x, shifts=-1, dims=-1),
        dim=-1,
    )


def _reference_orientation(
    coords: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> torch.Tensor:
    target = batch.get("target")
    if isinstance(target, torch.Tensor):
        try:
            ref_coords = _reshape_coords(target, meta).to(
                device=coords.device,
                dtype=coords.dtype,
            )
            ref_area = _signed_cell_areas(ref_coords)
            sign = torch.sign(ref_area.flatten(start_dim=2).median(dim=2).values)
            return torch.where(sign == 0, torch.ones_like(sign), sign)
        except ValueError:
            pass

    area0 = _signed_cell_areas(coords[:, :1])
    sign = torch.sign(area0.flatten(start_dim=2).median(dim=2).values)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return sign.expand(coords.shape[0], coords.shape[1])


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    coords = _reshape_coords(pred, meta)
    batch_size = int(coords.shape[0])
    dx = coords[:, :, :-1, :, 0] - coords[:, :, 1:, :, 0]
    dy = coords[:, :, :, :-1, 1] - coords[:, :, :, 1:, 1]
    neg_per_sample = torch.cat(
        [dx.reshape(batch_size, -1), dy.reshape(batch_size, -1)],
        dim=1,
    ) < 0
    neg = neg_per_sample.reshape(-1)
    count = neg.to(torch.float32).sum()
    global_fraction = neg.to(torch.float32).mean()
    worst_sample_fraction = neg_per_sample.to(torch.float32).mean(dim=1).max()

    signed_area = _signed_cell_areas(coords)
    orientation = _reference_orientation(coords, batch, meta)
    oriented_area = signed_area * orientation[:, :, None, None]
    flipped = oriented_area < 0
    flipped_per_frame = flipped.to(torch.float32).sum(dim=(-1, -2))
    flipped_fraction_per_frame = flipped.to(torch.float32).mean(dim=(-1, -2))
    return {
        "constraint/neg_spacing_count": ConstraintDiagnostic(
            value=count, reduce="sum"
        ),
        "constraint/neg_spacing_global_fraction": ConstraintDiagnostic(
            value=global_fraction, reduce="mean"
        ),
        "constraint/neg_spacing_worst_sample_fraction": ConstraintDiagnostic(
            value=worst_sample_fraction, reduce="max"
        ),
        "constraint/neg_spacing_fraction": ConstraintDiagnostic(
            value=worst_sample_fraction, reduce="max"
        ),
        "constraint/flipped_cell_count_mean": ConstraintDiagnostic(
            value=flipped_per_frame.mean(), reduce="mean"
        ),
        "constraint/flipped_cell_count_worst": ConstraintDiagnostic(
            value=flipped_per_frame.max(), reduce="max"
        ),
        "constraint/flipped_cell_fraction_mean": ConstraintDiagnostic(
            value=flipped_fraction_per_frame.mean(), reduce="mean"
        ),
        "constraint/flipped_cell_fraction_worst": ConstraintDiagnostic(
            value=flipped_fraction_per_frame.max(), reduce="max"
        ),
    }
