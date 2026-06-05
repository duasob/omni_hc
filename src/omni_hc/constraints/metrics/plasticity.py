"""Constraint-error metrics for the Plasticity 2D benchmark."""

from __future__ import annotations

from typing import Any

import torch

from ..base import ConstraintDiagnostic


def _reshape_field(pred: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Return predicted field with shape ``(B, T, H, W, C)``."""
    h, w = tuple(meta["shapelist"])
    t_out = int(meta.get("t_out", meta.get("T_out", 1)))
    out_dim = int(meta.get("out_dim", 4))

    if pred.dim() == 5:
        if pred.shape[1] == t_out and pred.shape[2] == h and pred.shape[3] == w:
            return pred
        if pred.shape[1] == h and pred.shape[2] == w and pred.shape[3] == t_out:
            return pred.permute(0, 3, 1, 2, 4)

    if pred.dim() == 4:
        if pred.shape[-1] == out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0,
                3,
                1,
                2,
                4,
            )
        if pred.shape[-1] == t_out * out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0,
                3,
                1,
                2,
                4,
            )

    if pred.dim() == 3:
        if pred.shape[1] == h * w and pred.shape[-1] == t_out * out_dim:
            return pred.reshape(pred.shape[0], h, w, t_out, out_dim).permute(
                0,
                3,
                1,
                2,
                4,
            )
        if pred.shape[1] == t_out and pred.shape[2] == h * w * out_dim:
            return pred.reshape(pred.shape[0], t_out, h, w, out_dim)

    raise ValueError(
        "Unsupported plasticity prediction shape "
        f"{tuple(pred.shape)} for shapelist={(h, w)}, t_out={t_out}, out_dim={out_dim}"
    )


def _reshape_coords(pred: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Return predicted physical coordinates with shape ``(B, T, H, W, 2)``."""
    field = _reshape_field(pred, meta)
    if field.shape[-1] < 2:
        raise ValueError(f"Expected at least 2 coordinate channels, got {field.shape}")
    return field[..., :2]


def _count_fraction(
    values: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    mask = values > float(threshold)
    return mask.to(torch.float32).sum(), mask.to(torch.float32).mean()


def _per_sample_fraction(values: torch.Tensor, threshold: float = 0.0) -> torch.Tensor:
    mask = values > float(threshold)
    return mask.reshape(mask.shape[0], -1).to(torch.float32).mean(dim=1)


def _axis_violation_metrics(
    *,
    name: str,
    margin: torch.Tensor,
) -> dict[str, ConstraintDiagnostic]:
    # margin >= 0 is valid. Negative margins mean the axis order is violated.
    violation = (-margin).clamp_min(0.0)
    count, fraction = _count_fraction(violation)
    worst_sample_fraction = _per_sample_fraction(violation).max()
    return {
        f"constraint/{name}_order_violation_count": ConstraintDiagnostic(
            value=count,
            reduce="sum",
        ),
        f"constraint/{name}_order_violation_fraction": ConstraintDiagnostic(
            value=fraction,
            reduce="mean",
        ),
        f"constraint/{name}_order_violation_worst_sample_fraction": ConstraintDiagnostic(
            value=worst_sample_fraction,
            reduce="max",
        ),
        f"constraint/{name}_order_violation_mean": ConstraintDiagnostic(
            value=violation.mean(),
            reduce="mean",
        ),
        f"constraint/{name}_order_violation_max": ConstraintDiagnostic(
            value=violation.max(),
            reduce="max",
        ),
        f"constraint/{name}_order_margin_min": ConstraintDiagnostic(
            value=margin.min(),
            reduce="min",
        ),
    }


def _spacing_compression_metrics(
    *,
    name: str,
    spacing: torch.Tensor,
    threshold: float,
) -> dict[str, ConstraintDiagnostic]:
    collapsed = spacing <= float(threshold)
    collapsed_float = collapsed.to(torch.float32)
    collapsed_per_sample = (
        collapsed.reshape(collapsed.shape[0], -1).to(torch.float32).mean(dim=1)
    )
    return {
        f"constraint/{name}_spacing_min": ConstraintDiagnostic(
            value=spacing.min(),
            reduce="min",
        ),
        f"constraint/{name}_spacing_mean": ConstraintDiagnostic(
            value=spacing.mean(),
            reduce="mean",
        ),
        f"constraint/{name}_spacing_collapse_count": ConstraintDiagnostic(
            value=collapsed_float.sum(),
            reduce="sum",
        ),
        f"constraint/{name}_spacing_collapse_fraction": ConstraintDiagnostic(
            value=collapsed_float.mean(),
            reduce="mean",
        ),
        f"constraint/{name}_spacing_collapse_worst_sample_fraction": ConstraintDiagnostic(
            value=collapsed_per_sample.max(),
            reduce="max",
        ),
    }


def _interp_envelope_y(
    *,
    profile_y: torch.Tensor,
    reference_x: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    if reference_x[0] > reference_x[-1]:
        interp_x = torch.flip(reference_x, dims=(0,))
        interp_y = torch.flip(profile_y, dims=(1,))
    else:
        interp_x = reference_x
        interp_y = profile_y
    x_clamped = x_query.clamp(
        min=float(interp_x[0].item()),
        max=float(interp_x[-1].item()),
    )
    upper = torch.searchsorted(interp_x, x_clamped.contiguous()).clamp(
        min=1,
        max=interp_x.numel() - 1,
    )
    lower = upper - 1
    x_lower = interp_x[lower]
    x_upper = interp_x[upper]
    y_lower = torch.gather(interp_y, dim=1, index=lower)
    y_upper = torch.gather(interp_y, dim=1, index=upper)
    weight = (x_clamped - x_lower) / (x_upper - x_lower).clamp_min(1.0e-12)
    return y_lower + weight * (y_upper - y_lower)


def _envelope_cap_from_input(
    *,
    coords: torch.Tensor,
    field: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> torch.Tensor | None:
    fx = batch.get("x")
    if fx is None:
        return None
    h, w = tuple(meta["shapelist"])
    if fx.dim() != 3 or fx.shape[1] != h * w:
        return None
    target = batch.get("target")
    target_field = _reshape_field(target, meta) if target is not None else None
    if target_field is not None and target_field.shape[-1] >= 4:
        material = target_field[..., :2] - target_field[..., 2:4]
        reference_x = material[:, :, :, 0, 0].mean(dim=(0, 1))
    elif field.shape[-1] >= 4:
        material = field[..., :2] - field[..., 2:4]
        reference_x = material[:, :, :, 0, 0].mean(dim=(0, 1))
    else:
        reference_x = torch.linspace(
            float(meta.get("x_left", 0.35)),
            float(meta.get("x_right", -49.65)),
            h,
            device=coords.device,
            dtype=coords.dtype,
        )
    fx_grid = fx.to(device=coords.device, dtype=coords.dtype).reshape(fx.shape[0], h, w, -1)
    die_profile = torch.flip(fx_grid[:, :, 0, 0], dims=(1,))
    time = batch.get("time")
    if time is None:
        time_values = torch.arange(
            coords.shape[1],
            device=coords.device,
            dtype=coords.dtype,
        ) / max(int(coords.shape[1]), 1)
        time_values = time_values[None, :].expand(coords.shape[0], -1)
    else:
        time_values = time.to(device=coords.device, dtype=coords.dtype)
        if time_values.dim() == 1:
            time_values = time_values[None, :].expand(coords.shape[0], -1)
    die_speed = float(meta.get("die_speed", 6.0))
    time_duration = float(meta.get("time_duration", 1.0))
    top_height = float(meta.get("top_height", 15.1))
    caps = []
    for timestep in range(coords.shape[1]):
        moved_profile = (
            die_profile
            - die_speed * time_duration * time_values[:, timestep : timestep + 1]
        )
        capped_profile = torch.minimum(moved_profile, torch.full_like(moved_profile, top_height))
        caps.append(
            _interp_envelope_y(
                profile_y=capped_profile,
                reference_x=reference_x,
                x_query=coords[:, timestep, :, 0, 0],
            )
        )
    return torch.stack(caps, dim=1)


def _envelope_violation_metrics(
    *,
    coords: torch.Tensor,
    field: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    out: dict[str, ConstraintDiagnostic] = {}
    cap = _envelope_cap_from_input(coords=coords, field=field, batch=batch, meta=meta)
    if cap is not None:
        top_y = coords[:, :, :, 0, 1]
        top_violation = (top_y - cap).clamp_min(0.0)
        count, fraction = _count_fraction(top_violation)
        out.update(
            {
                "constraint/top_envelope_violation_count": ConstraintDiagnostic(
                    value=count,
                    reduce="sum",
                ),
                "constraint/top_envelope_violation_fraction": ConstraintDiagnostic(
                    value=fraction,
                    reduce="mean",
                ),
                "constraint/top_envelope_violation_mean": ConstraintDiagnostic(
                    value=top_violation.mean(),
                    reduce="mean",
                ),
                "constraint/top_envelope_violation_max": ConstraintDiagnostic(
                    value=top_violation.max(),
                    reduce="max",
                ),
                "constraint/top_envelope_clearance_min": ConstraintDiagnostic(
                    value=(cap - top_y).min(),
                    reduce="min",
                ),
            }
        )

    target = batch.get("target")
    target_field = _reshape_field(target, meta) if target is not None else None
    if target_field is not None and target_field.shape[-1] >= 4:
        material = target_field[..., :2] - target_field[..., 2:4]
        bottom_target = material[:, :, :, -1, 1]
    elif field.shape[-1] >= 4:
        material = field[..., :2] - field[..., 2:4]
        bottom_target = material[:, :, :, -1, 1]
    else:
        bottom_target = torch.full_like(
            coords[:, :, :, -1, 1],
            float(meta.get("y_bottom", -0.1)),
        )
    bottom_abs = (coords[:, :, :, -1, 1] - bottom_target).abs()
    out["constraint/bottom_envelope_violation_mean"] = ConstraintDiagnostic(
        value=bottom_abs.mean(),
        reduce="mean",
    )
    out["constraint/bottom_envelope_violation_max"] = ConstraintDiagnostic(
        value=bottom_abs.max(),
        reduce="max",
    )
    y_bottom = float(meta.get("y_bottom", -0.1))
    below_y_bottom = (y_bottom - coords[..., 1]).clamp_min(0.0)
    below_count, below_fraction = _count_fraction(below_y_bottom)
    out["constraint/below_y_bottom_violation_count"] = ConstraintDiagnostic(
        value=below_count,
        reduce="sum",
    )
    out["constraint/below_y_bottom_violation_fraction"] = ConstraintDiagnostic(
        value=below_fraction,
        reduce="mean",
    )
    out["constraint/below_y_bottom_violation_max"] = ConstraintDiagnostic(
        value=below_y_bottom.max(),
        reduce="max",
    )
    return out


def compute(
    pred: torch.Tensor,
    batch: dict[str, Any],
    meta: dict[str, Any],
) -> dict[str, ConstraintDiagnostic]:
    field = _reshape_field(pred, meta)
    coords = field[..., :2]
    dx = coords[:, :, :-1, :, 0] - coords[:, :, 1:, :, 0]
    dy = coords[:, :, :, :-1, 1] - coords[:, :, :, 1:, 1]
    collapse_threshold = float(meta.get("collapse_spacing_threshold", 1.0e-3))
    top_collapse_rows = max(1, int(meta.get("top_collapse_rows", 3)))
    top_dy = dy[..., : min(top_collapse_rows, int(dy.shape[-1]))]
    x_metrics = _axis_violation_metrics(name="x", margin=dx)
    y_metrics = _axis_violation_metrics(name="y", margin=dy)
    axis_violation = torch.cat(
        [
            (-dx).clamp_min(0.0).reshape(coords.shape[0], -1),
            (-dy).clamp_min(0.0).reshape(coords.shape[0], -1),
        ],
        dim=1,
    )
    axis_count, axis_fraction = _count_fraction(axis_violation)
    out = {
        **x_metrics,
        **y_metrics,
        "constraint/axis_order_violation_count": ConstraintDiagnostic(
            value=axis_count,
            reduce="sum",
        ),
        "constraint/axis_order_violation_fraction": ConstraintDiagnostic(
            value=axis_fraction,
            reduce="mean",
        ),
        "constraint/axis_order_violation_worst_sample_fraction": ConstraintDiagnostic(
            value=_per_sample_fraction(axis_violation).max(),
            reduce="max",
        ),
        "constraint/axis_order_violation_mean": ConstraintDiagnostic(
            value=axis_violation.mean(),
            reduce="mean",
        ),
        "constraint/axis_order_violation_max": ConstraintDiagnostic(
            value=axis_violation.max(),
            reduce="max",
        ),
        # Backward-compatible aliases used by existing reports.
        "constraint/neg_spacing_count": ConstraintDiagnostic(value=axis_count, reduce="sum"),
        "constraint/neg_spacing_global_fraction": ConstraintDiagnostic(
            value=axis_fraction,
            reduce="mean",
        ),
        "constraint/neg_spacing_worst_sample_fraction": ConstraintDiagnostic(
            value=_per_sample_fraction(axis_violation).max(),
            reduce="max",
        ),
        "constraint/neg_spacing_fraction": ConstraintDiagnostic(
            value=_per_sample_fraction(axis_violation).max(),
            reduce="max",
        ),
    }
    out.update(
        _spacing_compression_metrics(
            name="y",
            spacing=dy,
            threshold=collapse_threshold,
        )
    )
    out.update(
        _spacing_compression_metrics(
            name="top_y",
            spacing=top_dy,
            threshold=collapse_threshold,
        )
    )
    out.update(_envelope_violation_metrics(coords=coords, field=field, batch=batch, meta=meta))
    return out
