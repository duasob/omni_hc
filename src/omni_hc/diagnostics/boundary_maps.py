from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import numpy as np

from omni_hc.constraints import ConstraintOutput, PipeStreamFunctionBoundaryAnsatz


MapSpace = Literal["output", "stream_function"]


@dataclass(frozen=True)
class BoundaryAnsatzMaps:
    """Maps in an affine hard constraint `f = g + lN`."""

    constraint_name: str
    space: MapSpace
    g: torch.Tensor
    l: torch.Tensor
    grid_shape: tuple[int, int]


def infer_boundary_ansatz_maps(
    constraint,
    *,
    pred_shape: tuple[int, int, int],
    grid_shape: tuple[int, int],
    coords: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> BoundaryAnsatzMaps:
    """
    Infer `g` and `l` from the constraint implementation itself.

    For direct output-space boundary ansatzes, the maps are recovered by probing
    the constraint with `N=0` and `N=1`:

        g = constraint(0)
        l = constraint(1) - constraint(0)

    The pipe stream-function boundary ansatz is different: its affine ansatz is
    applied to latent `psi`, then differentiated into `ux`. For that module we
    use the `stream_psi_bc` and `stream_mask` auxiliary tensors emitted by the
    constraint.
    """
    device = torch.device(device)
    pred_zero = torch.zeros(pred_shape, dtype=dtype, device=device)

    if isinstance(constraint, PipeStreamFunctionBoundaryAnsatz):
        out = _constraint_tensor(
            constraint(pred=pred_zero, coords=coords, return_aux=True)
        )
        aux = getattr(out, "aux", {})
        if "stream_psi_bc" not in aux or "stream_mask" not in aux:
            raise ValueError(
                "PipeStreamFunctionBoundaryAnsatz did not emit stream_psi_bc "
                "and stream_mask. Cannot infer latent ansatz maps."
            )
        return BoundaryAnsatzMaps(
            constraint_name=constraint.name,
            space="stream_function",
            g=_reshape_channels_last(aux["stream_psi_bc"], grid_shape),
            l=_reshape_channels_last(aux["stream_mask"], grid_shape),
            grid_shape=grid_shape,
        )

    g_flat = _constraint_tensor(constraint(pred=pred_zero, coords=coords))
    pred_one = torch.ones_like(pred_zero)
    one_flat = _constraint_tensor(constraint(pred=pred_one, coords=coords))

    if g_flat.shape != one_flat.shape:
        raise ValueError(
            "Cannot infer local affine maps because zero and one probes returned "
            f"different shapes: {tuple(g_flat.shape)} vs {tuple(one_flat.shape)}"
        )

    return BoundaryAnsatzMaps(
        constraint_name=getattr(constraint, "name", constraint.__class__.__name__),
        space="output",
        g=_reshape_channels_last(g_flat, grid_shape),
        l=_reshape_channels_last(one_flat - g_flat, grid_shape),
        grid_shape=grid_shape,
    )


def plot_boundary_ansatz_maps(
    maps: BoundaryAnsatzMaps,
    *,
    out_path: str | Path,
    coords: torch.Tensor | None = None,
    channel: int = 0,
    title: str | None = None,
    show: bool = False,
) -> Path:
    """Write a two-panel figure for the selected `g` and `l` channel."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    g = _channel_numpy(maps.g, channel)
    l = _channel_numpy(maps.l, channel)
    xy = None if coords is None else _coords_numpy(coords, maps.grid_shape)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), dpi=150)
    fig.suptitle(title or f"{maps.constraint_name} ({maps.space})", fontsize=12)

    _plot_map(axes[0], g, xy=xy, title="g: particular field", cmap="coolwarm")
    _plot_map(axes[1], l, xy=xy, title="l: correction mask", cmap="viridis")

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def _constraint_tensor(value) -> torch.Tensor | ConstraintOutput:
    if isinstance(value, ConstraintOutput):
        return value
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Constraint returned unsupported value {type(value)!r}")
    return value


def _reshape_channels_last(
    tensor: torch.Tensor,
    grid_shape: tuple[int, int],
) -> torch.Tensor:
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor shape (B, N, C), got {tuple(tensor.shape)}")
    height, width = grid_shape
    expected_points = height * width
    if tensor.shape[0] != 1:
        raise ValueError("Boundary map diagnostics expect a single sample batch")
    if tensor.shape[1] != expected_points:
        raise ValueError(
            f"Tensor point count {tensor.shape[1]} does not match grid {grid_shape}"
        )
    return tensor[0].reshape(height, width, tensor.shape[-1]).detach().cpu()


def _channel_numpy(tensor: torch.Tensor, channel: int):
    if channel < 0 or channel >= tensor.shape[-1]:
        raise ValueError(f"channel={channel} is out of bounds for shape {tuple(tensor.shape)}")
    return tensor[..., channel].numpy()


def _coords_numpy(coords: torch.Tensor, grid_shape: tuple[int, int]):
    if coords.ndim != 3 or coords.shape[0] != 1 or coords.shape[-1] < 2:
        raise ValueError(f"Expected coords shape (1, N, >=2), got {tuple(coords.shape)}")
    height, width = grid_shape
    grid = coords[0, :, :2].reshape(height, width, 2).detach().cpu().numpy()
    return grid[..., 0], grid[..., 1]


def _plot_map(ax, values, *, xy, title: str, cmap: str) -> None:
    if xy is None:
        image = ax.imshow(values, origin="lower", cmap=cmap)
    else:
        x, y = xy
        x_edges, y_edges = _curvilinear_cell_edges(x, y)
        image = ax.pcolormesh(
            x_edges,
            y_edges,
            values,
            shading="flat",
            cmap=cmap,
            linewidth=0.0,
            antialiased=False,
        )
        ax.plot(x[:, 0], y[:, 0], color="black", linewidth=0.8)
        ax.plot(x[:, -1], y[:, -1], color="black", linewidth=0.8)
        ax.plot(x[0, :], y[0, :], color="black", linewidth=0.8)
        ax.plot(x[-1, :], y[-1, :], color="black", linewidth=0.8)
    ax.set_title(title)
    ax.figure.colorbar(image, ax=ax, shrink=0.82)


def _is_rectilinear_grid(x, y) -> bool:
    return bool(
        ((x - x[:, :1]) == 0.0).all()
        and ((y - y[:1, :]) == 0.0).all()
    )


def _curvilinear_cell_edges(x, y):
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"Expected 2D coordinate arrays with matching shapes, got {x.shape} and {y.shape}")

    height, width = x.shape
    if _is_rectilinear_grid(x, y):
        return _rectilinear_edges(x[:, 0], y[0, :])

    stacked = np.stack([x, y], axis=-1)
    padded = np.pad(stacked, ((1, 1), (1, 1), (0, 0)), mode="edge")
    edges = 0.25 * (
        padded[:-1, :-1]
        + padded[1:, :-1]
        + padded[:-1, 1:]
        + padded[1:, 1:]
    )

    return edges[..., 0], edges[..., 1]


def _rectilinear_edges(x_centers, y_centers):
    return np.meshgrid(
        _axis_edges(np.asarray(x_centers)),
        _axis_edges(np.asarray(y_centers)),
        indexing="ij",
    )


def _axis_edges(centers):
    if centers.ndim != 1 or centers.size < 2:
        raise ValueError("At least two centers are required to infer cell edges")
    edges = np.empty(centers.size + 1, dtype=centers.dtype)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges
