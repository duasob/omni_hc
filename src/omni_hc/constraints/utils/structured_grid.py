from __future__ import annotations

from collections.abc import Sequence

import torch


def normalize_grid_axis(axis: int) -> int:
    resolved = axis if axis >= 0 else 2 + axis
    if resolved not in {0, 1}:
        raise ValueError(f"axis must select one grid axis, got {axis}")
    return resolved


def resolve_grid_shape(
    grid_shape: Sequence[int] | None,
    pred: torch.Tensor,
    *,
    name: str,
) -> tuple[int, int]:
    if grid_shape is not None:
        resolved = tuple(int(v) for v in grid_shape)
    else:
        n_points = int(pred.shape[1])
        side = int(n_points**0.5)
        if side * side != n_points:
            raise ValueError(
                f"grid_shape is required for non-square {name}; got {n_points} points"
            )
        resolved = (side, side)

    if len(resolved) != 2:
        raise ValueError(f"{name} expects a 2D grid shape, got {resolved!r}")
    height, width = int(resolved[0]), int(resolved[1])
    if height <= 1 or width <= 1:
        raise ValueError(f"grid_shape entries must be > 1, got {resolved!r}")
    if height * width != int(pred.shape[1]):
        raise ValueError(
            f"grid_shape={(height, width)} does not match pred point count {pred.shape[1]}"
        )
    return height, width


def edge_values(
    field: torch.Tensor,
    grid_shape: Sequence[int],
    *,
    axis: int,
    edge: str,
) -> torch.Tensor:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    grid = field.reshape(field.shape[0], height, width, -1)
    resolved_axis = normalize_grid_axis(axis)
    if resolved_axis == 0 and edge == "lower":
        return grid[:, 0, :, :]
    if resolved_axis == 0 and edge == "upper":
        return grid[:, -1, :, :]
    if resolved_axis == 1 and edge == "lower":
        return grid[:, :, 0, :]
    if resolved_axis == 1 and edge == "upper":
        return grid[:, :, -1, :]
    raise ValueError(f"Unsupported edge '{edge}' for axis {axis}")


def paired_edge_values(
    field: torch.Tensor,
    grid_shape: Sequence[int],
    *,
    axis: int,
) -> torch.Tensor:
    lower = edge_values(field, grid_shape, axis=axis, edge="lower")
    upper = edge_values(field, grid_shape, axis=axis, edge="upper")
    return torch.cat([lower.reshape(-1), upper.reshape(-1)], dim=0)


def interior_values(
    field: torch.Tensor,
    grid_shape: Sequence[int],
    *,
    axis: int,
) -> torch.Tensor:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    grid = field.reshape(field.shape[0], height, width, -1)
    resolved_axis = normalize_grid_axis(axis)
    if resolved_axis == 0:
        return grid[:, 1:-1, :, :].reshape(-1)
    return grid[:, :, 1:-1, :].reshape(-1)


def axis_coordinate(
    grid_shape: Sequence[int],
    *,
    axis: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    height, width = int(grid_shape[0]), int(grid_shape[1])
    resolved_axis = normalize_grid_axis(axis)
    size = height if resolved_axis == 0 else width
    coordinate_1d = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device)
    if resolved_axis == 0:
        coordinate = coordinate_1d[:, None].expand(height, width)
    else:
        coordinate = coordinate_1d[None, :].expand(height, width)
    return coordinate.reshape(1, height * width, 1)
