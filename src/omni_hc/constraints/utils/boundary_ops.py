from __future__ import annotations

from collections.abc import Sequence

import torch


def encode_target(target: torch.Tensor, normalizer) -> torch.Tensor:
    if normalizer is None:
        return target
    return normalizer.encode(target)


def decode_target(target: torch.Tensor, normalizer) -> torch.Tensor:
    if normalizer is None:
        return target
    return normalizer.decode(target)


def apply_boundary_ansatz(
    *,
    pred: torch.Tensor,
    particular: torch.Tensor,
    distance: torch.Tensor,
    channel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the direct boundary ansatz f = g + l * N."""
    constrained = particular + distance * pred
    if channel_mask is None:
        return constrained
    return torch.where(channel_mask, constrained, pred)


def validate_channels_last_prediction(
    pred: torch.Tensor,
    *,
    out_dim: int,
    name: str,
) -> None:
    if pred.ndim != 3:
        raise ValueError(f"pred must have shape (B, N, C), got {tuple(pred.shape)}")
    if pred.shape[-1] != out_dim:
        raise ValueError(f"Expected pred with out_dim={out_dim}, got {pred.shape[-1]}")


def channel_mask(
    pred: torch.Tensor,
    channel_indices: Sequence[int] | None,
) -> torch.Tensor:
    if channel_indices is None:
        return torch.ones((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)

    mask = torch.zeros((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)
    for channel_idx in channel_indices:
        if channel_idx < 0 or channel_idx >= pred.shape[-1]:
            raise ValueError(
                f"channel index {channel_idx} is out of range for output with "
                f"{pred.shape[-1]} channel(s)"
            )
        mask[..., channel_idx] = True
    return mask


def select_channels(
    field: torch.Tensor,
    channel_indices: Sequence[int] | None,
) -> torch.Tensor:
    if channel_indices is None:
        return field
    return field[..., list(channel_indices)]
