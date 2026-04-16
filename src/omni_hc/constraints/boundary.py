from __future__ import annotations

from typing import Sequence

import torch

from .base import ConstraintModule


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
            return out, pred, distance
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
