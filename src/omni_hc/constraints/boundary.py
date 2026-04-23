from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintDiagnostic, ConstraintModule


def unit_box_distance(
    coords: torch.Tensor,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    power: float = 1.0,
    reduce: str = "product",
) -> torch.Tensor:
    if coords.ndim < 2:
        raise ValueError("coords must have shape (..., space_dim)")
    scaled = (coords - lower) * (upper - coords)
    scaled = scaled.clamp_min(0.0)
    if power != 1.0:
        scaled = scaled.pow(power)
    if reduce == "product":
        return scaled.prod(dim=-1, keepdim=True)
    if reduce == "min":
        return scaled.min(dim=-1, keepdim=True).values
    raise ValueError(f"Unsupported distance reduction '{reduce}'")


def constant_boundary_value(
    coords: torch.Tensor,
    *,
    value: float = 0.0,
    out_dim: int = 1,
) -> torch.Tensor:
    shape = coords.shape[:-1] + (out_dim,)
    return torch.full(shape, float(value), dtype=coords.dtype, device=coords.device)


class DirichletBoundaryAnsatz(ConstraintModule):
    """
    Enforces Dirichlet boundary values through
    u(x) = g(x) + l(x) * N(x)
    where l(x)=0 on the boundary.
    """

    name = "dirichlet_boundary_ansatz"

    def __init__(
        self,
        *,
        out_dim: int,
        boundary_value: float = 0.0,
        lower: float = 0.0,
        upper: float = 1.0,
        distance_power: float = 1.0,
        distance_reduce: str = "product",
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.boundary_value = float(boundary_value)
        self.lower = float(lower)
        self.upper = float(upper)
        self.distance_power = float(distance_power)
        self.distance_reduce = str(distance_reduce)
        self.target_normalizer = None

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_domain_bounds(self, *, lower: float, upper: float) -> None:
        self.lower = float(lower)
        self.upper = float(upper)

    def _boundary_value_tensor(self, coords: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        g = constant_boundary_value(
            coords,
            value=self.boundary_value,
            out_dim=self.out_dim,
        )
        if self.target_normalizer is None:
            return g
        return self.target_normalizer.encode(g)

    def forward(self, *, pred, coords=None, return_aux=False, **_unused):
        if coords is None:
            raise ValueError("coords are required for DirichletBoundaryAnsatz")
        g = self._boundary_value_tensor(coords, pred)
        distance = unit_box_distance(
            coords,
            lower=self.lower,
            upper=self.upper,
            power=self.distance_power,
            reduce=self.distance_reduce,
        )
        out = g + distance * pred
        if return_aux:
            field = (
                out
                if self.target_normalizer is None
                else self.target_normalizer.decode(out)
            )
            stats = boundary_stats(
                field,
                coords,
                target_value=self.boundary_value,
                lower=self.lower,
                upper=self.upper,
            )
            diagnostics = {
                "constraint/boundary_abs_mean": ConstraintDiagnostic(
                    value=stats["boundary_abs_mean"],
                    reduce="mean",
                ),
                "constraint/boundary_abs_max": ConstraintDiagnostic(
                    value=stats["boundary_abs_max"],
                    reduce="max",
                ),
                "constraint/distance_mean": ConstraintDiagnostic(
                    value=distance.mean(),
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={"pred_base": pred, "distance": distance},
                diagnostics=diagnostics,
            )
        return out


def is_boundary_point(
    coords: torch.Tensor,
    *,
    lower: float = 0.0,
    upper: float = 1.0,
    atol: float = 1e-6,
) -> torch.Tensor:
    lower_hit = torch.isclose(
        coords,
        torch.full_like(coords, float(lower)),
        atol=atol,
        rtol=0.0,
    )
    upper_hit = torch.isclose(
        coords,
        torch.full_like(coords, float(upper)),
        atol=atol,
        rtol=0.0,
    )
    return (lower_hit | upper_hit).any(dim=-1)


def boundary_residual(
    field: torch.Tensor,
    coords: torch.Tensor,
    *,
    target_value: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
    atol: float = 1e-6,
) -> torch.Tensor:
    mask = is_boundary_point(coords, lower=lower, upper=upper, atol=atol)
    target = torch.full_like(field, float(target_value))
    residual = (field - target).abs()
    return residual[mask]


def boundary_stats(
    field: torch.Tensor,
    coords: torch.Tensor,
    *,
    target_value: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
    atol: float = 1e-6,
) -> dict[str, float]:
    residual = boundary_residual(
        field,
        coords,
        target_value=target_value,
        lower=lower,
        upper=upper,
        atol=atol,
    )
    if residual.numel() == 0:
        return {"boundary_abs_mean": 0.0, "boundary_abs_max": 0.0}
    return {
        "boundary_abs_mean": float(residual.mean().item()),
        "boundary_abs_max": float(residual.max().item()),
    }


def structured_wall_distance(
    grid_shape: Sequence[int],
    *,
    transverse_axis: int = 1,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
    power: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be a 2D shape, got {grid_shape}")
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height <= 1 or width <= 1:
        raise ValueError(f"grid_shape entries must be > 1, got {grid_shape}")
    if transverse_axis not in {0, 1, -1, -2}:
        raise ValueError(
            f"transverse_axis must select one grid axis, got {transverse_axis}"
        )

    axis = transverse_axis if transverse_axis >= 0 else 2 + transverse_axis
    size = height if axis == 0 else width
    eta = torch.linspace(0.0, 1.0, size, dtype=dtype, device=device)
    distance_1d = eta * (1.0 - eta)
    if normalize:
        distance_1d = 4.0 * distance_1d
    if power != 1.0:
        distance_1d = distance_1d.pow(power)

    if axis == 0:
        distance = distance_1d[:, None].expand(height, width)
    else:
        distance = distance_1d[None, :].expand(height, width)
    return distance.reshape(1, height * width, 1)


def structured_wall_mask(
    grid_shape: Sequence[int],
    *,
    transverse_axis: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    if len(grid_shape) != 2:
        raise ValueError(f"grid_shape must be a 2D shape, got {grid_shape}")
    height, width = int(grid_shape[0]), int(grid_shape[1])
    if height <= 1 or width <= 1:
        raise ValueError(f"grid_shape entries must be > 1, got {grid_shape}")

    axis = transverse_axis if transverse_axis >= 0 else 2 + transverse_axis
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    if axis == 0:
        mask[0, :] = True
        mask[-1, :] = True
    elif axis == 1:
        mask[:, 0] = True
        mask[:, -1] = True
    else:
        raise ValueError(
            f"transverse_axis must select one grid axis, got {transverse_axis}"
        )
    return mask.reshape(1, height * width)


def structured_wall_stats(
    field: torch.Tensor,
    grid_shape: Sequence[int],
    *,
    target_value: float = 0.0,
    transverse_axis: int = 1,
    channel_indices: Sequence[int] | None = None,
) -> dict[str, float]:
    if field.ndim != 3:
        raise ValueError(f"field must have shape (B, N, C), got {tuple(field.shape)}")
    mask = structured_wall_mask(
        grid_shape,
        transverse_axis=transverse_axis,
        device=field.device,
    ).expand(field.shape[0], -1)
    residual = (field - float(target_value)).abs()
    if channel_indices is not None:
        residual = residual[..., [int(idx) for idx in channel_indices]]
    wall_residual = residual[mask]
    if wall_residual.numel() == 0:
        return {"wall_abs_mean": 0.0, "wall_abs_max": 0.0}
    return {
        "wall_abs_mean": float(wall_residual.mean().item()),
        "wall_abs_max": float(wall_residual.max().item()),
    }


class StructuredWallDirichletAnsatz(ConstraintModule):
    """
    Enforces a constant value on the two walls of a structured 2D mesh.

    For the pipe benchmark, the walls are the transverse-index edges
    j=0 and j=W-1. The ansatz uses index-space distance, not physical
    coordinates, so it is unaffected by coordinate normalization.
    """

    name = "structured_wall_dirichlet_ansatz"

    def __init__(
        self,
        *,
        out_dim: int,
        grid_shape: Sequence[int] | None = None,
        boundary_value: float = 0.0,
        transverse_axis: int = 1,
        distance_power: float = 1.0,
        normalize_distance: bool = True,
        channel_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.grid_shape = None if grid_shape is None else tuple(int(v) for v in grid_shape)
        self.boundary_value = float(boundary_value)
        self.transverse_axis = int(transverse_axis)
        self.distance_power = float(distance_power)
        self.normalize_distance = bool(normalize_distance)
        self.channel_indices = (
            None if channel_indices is None else tuple(int(idx) for idx in channel_indices)
        )
        self.target_normalizer = None

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_grid_shape(self, grid_shape: Sequence[int]) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)

    def _resolve_grid_shape(self, pred: torch.Tensor) -> tuple[int, int]:
        if self.grid_shape is not None:
            return self.grid_shape
        n_points = int(pred.shape[1])
        side = int(n_points**0.5)
        if side * side != n_points:
            raise ValueError(
                "grid_shape is required for non-square structured wall constraints; "
                f"got {n_points} points"
            )
        return (side, side)

    def _boundary_value_tensor(self, pred: torch.Tensor) -> torch.Tensor:
        g = torch.full_like(pred, self.boundary_value)
        if self.target_normalizer is None:
            return g
        return self.target_normalizer.encode(g)

    def _channel_mask(self, pred: torch.Tensor) -> torch.Tensor:
        mask = torch.ones((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)
        if self.channel_indices is None:
            return mask

        mask = torch.zeros((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)
        for channel_idx in self.channel_indices:
            if channel_idx < 0 or channel_idx >= pred.shape[-1]:
                raise ValueError(
                    f"channel index {channel_idx} is out of range for output with "
                    f"{pred.shape[-1]} channel(s)"
                )
            mask[..., channel_idx] = True
        return mask

    def _selected_channels(self, field: torch.Tensor) -> torch.Tensor:
        if self.channel_indices is None:
            return field
        return field[..., list(self.channel_indices)]

    def _wall_edge_values(
        self,
        field: torch.Tensor,
        grid_shape: Sequence[int],
        edge: str,
    ) -> torch.Tensor:
        height, width = int(grid_shape[0]), int(grid_shape[1])
        grid = self._selected_channels(field).reshape(
            field.shape[0], height, width, -1
        )
        axis = self.transverse_axis if self.transverse_axis >= 0 else 2 + self.transverse_axis
        if axis == 0 and edge == "lower":
            return grid[:, 0, :, :]
        if axis == 0 and edge == "upper":
            return grid[:, -1, :, :]
        if axis == 1 and edge == "lower":
            return grid[:, :, 0, :]
        if axis == 1 and edge == "upper":
            return grid[:, :, -1, :]
        raise ValueError(f"Unsupported wall edge '{edge}' for axis {self.transverse_axis}")

    def _wall_values(
        self,
        field: torch.Tensor,
        grid_shape: Sequence[int],
    ) -> torch.Tensor:
        lower = self._wall_edge_values(field, grid_shape, "lower")
        upper = self._wall_edge_values(field, grid_shape, "upper")
        return torch.cat([lower.reshape(-1), upper.reshape(-1)], dim=0)

    def _interior_values(
        self,
        field: torch.Tensor,
        grid_shape: Sequence[int],
    ) -> torch.Tensor:
        height, width = int(grid_shape[0]), int(grid_shape[1])
        grid = self._selected_channels(field).reshape(
            field.shape[0], height, width, -1
        )
        axis = self.transverse_axis if self.transverse_axis >= 0 else 2 + self.transverse_axis
        if axis == 0:
            return grid[:, 1:-1, :, :].reshape(-1)
        if axis == 1:
            return grid[:, :, 1:-1, :].reshape(-1)
        raise ValueError(
            f"transverse_axis must select one grid axis, got {self.transverse_axis}"
        )

    def forward(self, *, pred, return_aux=False, **_unused):
        if pred.ndim != 3:
            raise ValueError(f"pred must have shape (B, N, C), got {tuple(pred.shape)}")
        if pred.shape[-1] != self.out_dim:
            raise ValueError(
                f"Expected pred with out_dim={self.out_dim}, got {pred.shape[-1]}"
            )

        grid_shape = self._resolve_grid_shape(pred)
        if int(grid_shape[0]) * int(grid_shape[1]) != int(pred.shape[1]):
            raise ValueError(
                f"grid_shape={grid_shape} does not match pred point count {pred.shape[1]}"
            )

        g = self._boundary_value_tensor(pred)
        distance = structured_wall_distance(
            grid_shape,
            transverse_axis=self.transverse_axis,
            dtype=pred.dtype,
            device=pred.device,
            power=self.distance_power,
            normalize=self.normalize_distance,
        )
        constrained = g + distance * pred
        channel_mask = self._channel_mask(pred)
        out = torch.where(channel_mask, constrained, pred)

        if return_aux:
            field = (
                out
                if self.target_normalizer is None
                else self.target_normalizer.decode(out)
            )
            base_field = (
                pred
                if self.target_normalizer is None
                else self.target_normalizer.decode(pred)
            )
            stats = structured_wall_stats(
                field,
                grid_shape,
                target_value=self.boundary_value,
                transverse_axis=self.transverse_axis,
                channel_indices=self.channel_indices,
            )
            wall_values = self._wall_values(field, grid_shape)
            wall_base_values = self._wall_values(base_field, grid_shape)
            lower_values = self._wall_edge_values(field, grid_shape, "lower")
            upper_values = self._wall_edge_values(field, grid_shape, "upper")
            lower_base_values = self._wall_edge_values(base_field, grid_shape, "lower")
            upper_base_values = self._wall_edge_values(base_field, grid_shape, "upper")
            interior_delta = self._interior_values(field - base_field, grid_shape).abs()
            interior_delta_mean = (
                interior_delta.mean()
                if interior_delta.numel() > 0
                else pred.new_tensor(0.0)
            )
            wall_target = torch.full_like(wall_values, self.boundary_value)
            diagnostics = {
                "constraint/wall_abs_mean": ConstraintDiagnostic(
                    value=stats["wall_abs_mean"],
                    reduce="mean",
                ),
                "constraint/wall_abs_max": ConstraintDiagnostic(
                    value=stats["wall_abs_max"],
                    reduce="max",
                ),
                "constraint/wall_distance_mean": ConstraintDiagnostic(
                    value=distance.mean(),
                    reduce="mean",
                ),
                "constraint/wall_distance_min": ConstraintDiagnostic(
                    value=distance.min(),
                    reduce="min",
                ),
                "constraint/wall_distance_max": ConstraintDiagnostic(
                    value=distance.max(),
                    reduce="max",
                ),
                "constraint/wall_target_mean": ConstraintDiagnostic(
                    value=wall_target.mean(),
                    reduce="mean",
                ),
                "constraint/wall_pred_mean": ConstraintDiagnostic(
                    value=wall_values.mean(),
                    reduce="mean",
                ),
                "constraint/wall_pred_std": ConstraintDiagnostic(
                    value=wall_values.std(unbiased=False),
                    reduce="mean",
                ),
                "constraint/wall_base_mean": ConstraintDiagnostic(
                    value=wall_base_values.mean(),
                    reduce="mean",
                ),
                "constraint/wall_base_std": ConstraintDiagnostic(
                    value=wall_base_values.std(unbiased=False),
                    reduce="mean",
                ),
                "constraint/wall_base_abs_mean": ConstraintDiagnostic(
                    value=wall_base_values.abs().mean(),
                    reduce="mean",
                ),
                "constraint/wall_base_abs_max": ConstraintDiagnostic(
                    value=wall_base_values.abs().max(),
                    reduce="max",
                ),
                "constraint/wall_lower_abs_mean": ConstraintDiagnostic(
                    value=(lower_values - self.boundary_value).abs().mean(),
                    reduce="mean",
                ),
                "constraint/wall_upper_abs_mean": ConstraintDiagnostic(
                    value=(upper_values - self.boundary_value).abs().mean(),
                    reduce="mean",
                ),
                "constraint/wall_base_lower_abs_mean": ConstraintDiagnostic(
                    value=(lower_base_values - self.boundary_value).abs().mean(),
                    reduce="mean",
                ),
                "constraint/wall_base_upper_abs_mean": ConstraintDiagnostic(
                    value=(upper_base_values - self.boundary_value).abs().mean(),
                    reduce="mean",
                ),
                "constraint/interior_abs_delta_mean": ConstraintDiagnostic(
                    value=interior_delta_mean,
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={"pred_base": pred, "distance": distance},
                diagnostics=diagnostics,
            )
        return out


class PipeInletParabolicAnsatz(ConstraintModule):
    """
    Enforces the pipe inlet ux profile with a smooth extension into the domain.

    The profile is applied through
    u = g + l * N
    where g = alpha(xi) * Umax * 4t(1-t),
    l = 1 - alpha(xi), and alpha(xi) = (1 - xi)^p.
    """

    name = "pipe_inlet_parabolic_ansatz"

    def __init__(
        self,
        *,
        out_dim: int,
        grid_shape: Sequence[int] | None = None,
        amplitude: float = 0.25,
        inlet_axis: int = 0,
        transverse_axis: int = 1,
        decay_power: float = 4.0,
        channel_indices: Sequence[int] | None = None,
        coordinate_channel: int = 1,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.out_dim = int(out_dim)
        self.grid_shape = None if grid_shape is None else tuple(int(v) for v in grid_shape)
        self.amplitude = float(amplitude)
        self.inlet_axis = int(inlet_axis)
        self.transverse_axis = int(transverse_axis)
        self.decay_power = float(decay_power)
        self.channel_indices = (
            None if channel_indices is None else tuple(int(idx) for idx in channel_indices)
        )
        self.coordinate_channel = int(coordinate_channel)
        self.eps = float(eps)
        self.input_normalizer = None
        self.target_normalizer = None

    def set_input_normalizer(self, normalizer) -> None:
        self.input_normalizer = normalizer

    def set_target_normalizer(self, normalizer) -> None:
        self.target_normalizer = normalizer

    def set_grid_shape(self, grid_shape: Sequence[int]) -> None:
        self.grid_shape = tuple(int(v) for v in grid_shape)

    def _resolve_grid_shape(self, pred: torch.Tensor) -> tuple[int, int]:
        if self.grid_shape is not None:
            return self.grid_shape
        n_points = int(pred.shape[1])
        side = int(n_points**0.5)
        if side * side != n_points:
            raise ValueError(
                "grid_shape is required for non-square pipe inlet constraints; "
                f"got {n_points} points"
            )
        return (side, side)

    def _axis(self, axis: int) -> int:
        resolved = axis if axis >= 0 else 2 + axis
        if resolved not in {0, 1}:
            raise ValueError(f"axis must select one grid axis, got {axis}")
        return resolved

    def _channel_mask(self, pred: torch.Tensor) -> torch.Tensor:
        if self.channel_indices is None:
            return torch.ones((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)
        mask = torch.zeros((1, 1, pred.shape[-1]), dtype=torch.bool, device=pred.device)
        for channel_idx in self.channel_indices:
            if channel_idx < 0 or channel_idx >= pred.shape[-1]:
                raise ValueError(
                    f"channel index {channel_idx} is out of range for output with "
                    f"{pred.shape[-1]} channel(s)"
                )
            mask[..., channel_idx] = True
        return mask

    def _selected_channels(self, field: torch.Tensor) -> torch.Tensor:
        if self.channel_indices is None:
            return field
        return field[..., list(self.channel_indices)]

    def _decode_coords(self, coords: torch.Tensor) -> torch.Tensor:
        if self.input_normalizer is None:
            return coords
        return self.input_normalizer.decode(coords)

    def _xi(self, grid_shape: Sequence[int], pred: torch.Tensor) -> torch.Tensor:
        height, width = int(grid_shape[0]), int(grid_shape[1])
        inlet_axis = self._axis(self.inlet_axis)
        size = height if inlet_axis == 0 else width
        xi_1d = torch.linspace(0.0, 1.0, size, dtype=pred.dtype, device=pred.device)
        if inlet_axis == 0:
            xi = xi_1d[:, None].expand(height, width)
        else:
            xi = xi_1d[None, :].expand(height, width)
        return xi.reshape(1, height * width, 1)

    def _alpha(self, grid_shape: Sequence[int], pred: torch.Tensor) -> torch.Tensor:
        xi = self._xi(grid_shape, pred)
        return (1.0 - xi).clamp_min(0.0).pow(self.decay_power)

    def _inlet_profile(
        self,
        coords: torch.Tensor,
        grid_shape: Sequence[int],
        pred: torch.Tensor,
    ) -> torch.Tensor:
        if coords is None:
            raise ValueError("coords are required for PipeInletParabolicAnsatz")
        if coords.ndim != 3 or coords.shape[-1] <= self.coordinate_channel:
            raise ValueError(
                "coords must have shape (B, N, C) with the configured coordinate channel"
            )
        if coords.shape[1] != pred.shape[1]:
            raise ValueError(
                f"coords point count {coords.shape[1]} does not match pred {pred.shape[1]}"
            )

        height, width = int(grid_shape[0]), int(grid_shape[1])
        inlet_axis = self._axis(self.inlet_axis)
        physical_coords = self._decode_coords(coords)
        coord_grid = physical_coords[..., self.coordinate_channel].reshape(
            coords.shape[0], height, width
        )
        if inlet_axis == 0:
            inlet_coord = coord_grid[:, 0, :]
            coord_min = inlet_coord.min(dim=1, keepdim=True).values
            coord_max = inlet_coord.max(dim=1, keepdim=True).values
            t_edge = (inlet_coord - coord_min) / (coord_max - coord_min).clamp_min(self.eps)
            profile_edge = self.amplitude * 4.0 * t_edge * (1.0 - t_edge)
            profile = profile_edge[:, None, :].expand(-1, height, -1)
        else:
            inlet_coord = coord_grid[:, :, 0]
            coord_min = inlet_coord.min(dim=1, keepdim=True).values
            coord_max = inlet_coord.max(dim=1, keepdim=True).values
            t_edge = (inlet_coord - coord_min) / (coord_max - coord_min).clamp_min(self.eps)
            profile_edge = self.amplitude * 4.0 * t_edge * (1.0 - t_edge)
            profile = profile_edge[:, :, None].expand(-1, -1, width)
        return profile.reshape(coords.shape[0], height * width, 1).to(
            dtype=pred.dtype,
            device=pred.device,
        )

    def _encode_target(self, target: torch.Tensor) -> torch.Tensor:
        if self.target_normalizer is None:
            return target
        return self.target_normalizer.encode(target)

    def _inlet_edge_values(
        self,
        field: torch.Tensor,
        grid_shape: Sequence[int],
    ) -> torch.Tensor:
        height, width = int(grid_shape[0]), int(grid_shape[1])
        grid = self._selected_channels(field).reshape(
            field.shape[0], height, width, -1
        )
        inlet_axis = self._axis(self.inlet_axis)
        if inlet_axis == 0:
            return grid[:, 0, :, :]
        return grid[:, :, 0, :]

    def forward(self, *, pred, coords=None, return_aux=False, **_unused):
        if pred.ndim != 3:
            raise ValueError(f"pred must have shape (B, N, C), got {tuple(pred.shape)}")
        if pred.shape[-1] != self.out_dim:
            raise ValueError(
                f"Expected pred with out_dim={self.out_dim}, got {pred.shape[-1]}"
            )

        grid_shape = self._resolve_grid_shape(pred)
        if int(grid_shape[0]) * int(grid_shape[1]) != int(pred.shape[1]):
            raise ValueError(
                f"grid_shape={grid_shape} does not match pred point count {pred.shape[1]}"
            )

        profile_physical = self._inlet_profile(coords, grid_shape, pred)
        alpha = self._alpha(grid_shape, pred)
        g_physical = alpha * profile_physical
        g = self._encode_target(g_physical)
        constrained = g + (1.0 - alpha) * pred
        channel_mask = self._channel_mask(pred)
        out = torch.where(channel_mask, constrained, pred)

        if return_aux:
            field = (
                out
                if self.target_normalizer is None
                else self.target_normalizer.decode(out)
            )
            base_field = (
                pred
                if self.target_normalizer is None
                else self.target_normalizer.decode(pred)
            )
            inlet_values = self._inlet_edge_values(field, grid_shape)
            base_inlet_values = self._inlet_edge_values(base_field, grid_shape)
            target_values = self._inlet_edge_values(profile_physical, grid_shape)
            inlet_residual = (inlet_values - target_values).abs()
            base_inlet_residual = (base_inlet_values - target_values).abs()
            diagnostics = {
                "constraint/inlet_abs_mean": ConstraintDiagnostic(
                    value=inlet_residual.mean(),
                    reduce="mean",
                ),
                "constraint/inlet_abs_max": ConstraintDiagnostic(
                    value=inlet_residual.max(),
                    reduce="max",
                ),
                "constraint/inlet_base_abs_mean": ConstraintDiagnostic(
                    value=base_inlet_residual.mean(),
                    reduce="mean",
                ),
                "constraint/inlet_base_abs_max": ConstraintDiagnostic(
                    value=base_inlet_residual.max(),
                    reduce="max",
                ),
                "constraint/inlet_profile_amplitude": ConstraintDiagnostic(
                    value=pred.new_tensor(self.amplitude),
                    reduce="mean",
                ),
                "constraint/inlet_profile_mean": ConstraintDiagnostic(
                    value=target_values.mean(),
                    reduce="mean",
                ),
                "constraint/inlet_profile_max": ConstraintDiagnostic(
                    value=target_values.max(),
                    reduce="max",
                ),
                "constraint/inlet_alpha_mean": ConstraintDiagnostic(
                    value=alpha.mean(),
                    reduce="mean",
                ),
                "constraint/inlet_alpha_min": ConstraintDiagnostic(
                    value=alpha.min(),
                    reduce="min",
                ),
                "constraint/inlet_alpha_max": ConstraintDiagnostic(
                    value=alpha.max(),
                    reduce="max",
                ),
                "constraint/inlet_decay_power": ConstraintDiagnostic(
                    value=pred.new_tensor(self.decay_power),
                    reduce="mean",
                ),
            }
            return self.as_output(
                out,
                aux={
                    "pred_base": pred,
                    "inlet_profile": profile_physical,
                    "alpha": alpha,
                },
                diagnostics=diagnostics,
            )
        return out
