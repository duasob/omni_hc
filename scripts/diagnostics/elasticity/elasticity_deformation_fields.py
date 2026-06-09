from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _summary(name: str, array: np.ndarray) -> None:
    print(
        f"{name}: shape={array.shape} dtype={array.dtype} "
        f"range=[{array.min():.6g}, {array.max():.6g}] "
        f"mean={array.mean():.6g} std={array.std():.6g}"
    )


def _grid_jacobian(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx_di, dx_dj = np.gradient(x)
    dy_di, dy_dj = np.gradient(y)
    return dx_di * dy_dj - dx_dj * dy_di


def _mask_long_triangles(xy: np.ndarray):
    import matplotlib.tri as mtri

    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1])
    triangles = triangulation.triangles
    triangle_xy = xy[triangles]
    edge_lengths = np.stack(
        (
            np.linalg.norm(triangle_xy[:, 0] - triangle_xy[:, 1], axis=1),
            np.linalg.norm(triangle_xy[:, 1] - triangle_xy[:, 2], axis=1),
            np.linalg.norm(triangle_xy[:, 2] - triangle_xy[:, 0], axis=1),
        ),
        axis=1,
    )
    triangulation.set_mask(
        edge_lengths.max(axis=1) > 3.0 * np.median(edge_lengths)
    )
    return triangulation


def _plot_grid(
    axis,
    x: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    stride: int = 2,
) -> None:
    for i in range(0, x.shape[0], stride):
        axis.plot(x[i, :], y[i, :], color="tab:blue", lw=0.45, alpha=0.75)
    for j in range(0, x.shape[1], stride):
        axis.plot(x[:, j], y[:, j], color="tab:orange", lw=0.45, alpha=0.75)
    axis.set_aspect("equal")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_title(title)
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/elasticity"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/elasticity/elasticity_deformation_fields.png"),
    )
    args = parser.parse_args()

    root = args.root
    sample = args.sample
    original_xy = np.load(root / "Meshes/Random_UnitCell_XY_10.npy").transpose(
        2, 0, 1
    )
    original_sigma = np.load(
        root / "Meshes/Random_UnitCell_sigma_10.npy"
    ).T
    omesh_x = np.load(
        root / "Omesh/Random_UnitCell_Deform_X_10_interp.npy"
    ).transpose(2, 0, 1)
    omesh_y = np.load(
        root / "Omesh/Random_UnitCell_Deform_Y_10_interp.npy"
    ).transpose(2, 0, 1)
    omesh_sigma = np.load(
        root / "Omesh/Random_UnitCell_Deform_sigma_10_interp.npy"
    ).transpose(2, 0, 1)
    rmesh_xy = np.load(
        root / "Rmesh/Random_UnitCell_Deform_Grid_XY_10_interp.npy"
    ).transpose(2, 0, 1)
    rmesh_x = np.load(
        root / "Rmesh/Random_UnitCell_Deform_Grid_X_10_interp.npy"
    ).transpose(2, 0, 1)
    rmesh_y = np.load(
        root / "Rmesh/Random_UnitCell_Deform_Grid_Y_10_interp.npy"
    ).transpose(2, 0, 1)
    rmesh_mask = np.load(
        root / "Rmesh/Random_UnitCell_Deform_Grid_mask_10_interp.npy"
    ).transpose(2, 0, 1)
    rmesh_sigma = np.load(
        root / "Rmesh/Random_UnitCell_Deform_Grid_sigma_10_interp.npy"
    ).transpose(2, 0, 1)

    for name, array in (
        ("original_xy", original_xy),
        ("omesh_x", omesh_x),
        ("omesh_y", omesh_y),
        ("omesh_sigma", omesh_sigma),
        ("rmesh_xy", rmesh_xy),
        ("rmesh_x", rmesh_x),
        ("rmesh_y", rmesh_y),
        ("rmesh_mask", rmesh_mask),
        ("rmesh_sigma", rmesh_sigma),
    ):
        _summary(name, array)

    x = omesh_x[sample]
    y = omesh_y[sample]
    jacobian = _grid_jacobian(x, y)
    print("\nOmesh coordinate-map checks")
    print(
        "mean |dX/di|, |dX/dj|:",
        np.abs(np.diff(x, axis=0)).mean(),
        np.abs(np.diff(x, axis=1)).mean(),
    )
    print(
        "mean |dY/di|, |dY/dj|:",
        np.abs(np.diff(y, axis=0)).mean(),
        np.abs(np.diff(y, axis=1)).mean(),
    )
    print(
        "index Jacobian range/non-positive:",
        jacobian.min(),
        jacobian.max(),
        int((jacobian <= 0.0).sum()),
    )
    print(
        "across-sample coordinate std mean/max:",
        omesh_x.std(axis=0).mean(),
        omesh_x.std(axis=0).max(),
        omesh_y.std(axis=0).mean(),
        omesh_y.std(axis=0).max(),
    )

    r_x = rmesh_x[sample]
    r_y = rmesh_y[sample]
    r_valid = rmesh_mask[sample]
    r_jacobian = _grid_jacobian(r_x, r_y)
    print("\nRmesh coordinate-map checks")
    print(
        "index Jacobian range/non-positive on mask:",
        r_jacobian[r_valid].min(),
        r_jacobian[r_valid].max(),
        int((r_jacobian[r_valid] <= 0.0).sum()),
    )

    direct_delta = rmesh_xy - original_xy
    print("\nRmesh point-cloud correspondence")
    print(
        "same-index RMS/max coordinate difference:",
        np.sqrt(np.mean(direct_delta.square()))
        if hasattr(direct_delta, "square")
        else np.sqrt(np.mean(direct_delta**2)),
        np.abs(direct_delta).max(),
    )
    print("\nSample-order stress correlations against original mesh")
    original_mean = original_sigma.mean(axis=1)
    original_max = original_sigma.max(axis=1)
    for name, field in (
        ("Omesh", omesh_sigma),
        ("Rmesh", rmesh_sigma),
    ):
        field_flat = field.reshape(field.shape[0], -1)
        print(
            name,
            "mean correlation=",
            np.corrcoef(original_mean, field_flat.mean(axis=1))[0, 1],
            "max correlation=",
            np.corrcoef(original_max, field_flat.max(axis=1))[0, 1],
        )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].triplot(
        _mask_long_triangles(original_xy[sample]),
        color="0.35",
        lw=0.35,
    )
    axes[0].set_aspect("equal")
    axes[0].set_title("Original unstructured mesh")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    _plot_grid(
        axes[1],
        omesh_x[sample],
        omesh_y[sample],
        title="Omesh coordinate arrays",
    )
    r_x_plot = rmesh_x[sample].copy()
    r_y_plot = rmesh_y[sample].copy()
    r_x_plot[~rmesh_mask[sample]] = np.nan
    r_y_plot[~rmesh_mask[sample]] = np.nan
    _plot_grid(
        axes[2],
        r_x_plot,
        r_y_plot,
        title="Rmesh coordinate arrays",
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
