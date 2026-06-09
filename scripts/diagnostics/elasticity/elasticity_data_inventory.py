from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np


def _load(root: Path, relative: str) -> np.ndarray:
    return np.load(root / relative)


def _triangulation(xy: np.ndarray) -> mtri.Triangulation:
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    points = xy[tri.triangles]
    lengths = np.stack(
        (
            np.linalg.norm(points[:, 0] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 1] - points[:, 2], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 0], axis=1),
        ),
        axis=1,
    )
    tri.set_mask(lengths.max(axis=1) > 3.0 * np.median(lengths))
    return tri


def _inventory(root: Path, output: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*.npy")):
        array = np.load(path, mmap_mode="r")
        rows.append(
            {
                "file": str(path.relative_to(root)),
                "shape": " x ".join(str(value) for value in array.shape),
                "dtype": str(array.dtype),
                "minimum": float(np.min(array)),
                "maximum": float(np.max(array)),
                "mean": float(np.mean(array)),
                "standard_deviation": float(np.std(array)),
                "negative_count": int(np.count_nonzero(array < 0)),
                "zero_count": int(np.count_nonzero(array == 0)),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _plot_meshes(root: Path, sample: int, output: Path) -> None:
    xy = _load(root, "Meshes/Random_UnitCell_XY_10.npy")[:, :, sample]
    sigma = _load(root, "Meshes/Random_UnitCell_sigma_10.npy")[:, sample]
    rr = _load(root, "Meshes/Random_UnitCell_rr_10.npy")[:, sample]
    theta = _load(root, "Meshes/Random_UnitCell_theta_10.npy")[sample]

    angle = np.linspace(0.0, 2.0 * np.pi, len(rr))
    boundary = np.column_stack(
        (0.5 + rr * np.cos(angle), 0.5 + rr * np.sin(angle))
    )

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    scatter = axes[0].scatter(
        xy[:, 0], xy[:, 1], c=sigma, s=9, cmap="viridis"
    )
    axes[0].plot(boundary[:, 0], boundary[:, 1], color="white", lw=1.3)
    axes[0].set_title("XY nodes and scalar stress")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    fig.colorbar(scatter, ax=axes[0], label=r"$\sigma$")

    polar = fig.add_subplot(1, 3, 2, projection="polar")
    axes[1].remove()
    polar.plot(angle, rr, color="tab:blue")
    polar.fill(angle, rr, color="tab:blue", alpha=0.2)
    polar.set_ylim(0.0, 0.45)
    polar.set_title("rr: void radius versus angle")

    axes[2].bar(np.arange(1, 11), theta, color="tab:orange")
    axes[2].axhline(0.0, color="black", lw=0.7)
    axes[2].set_title("theta: 10 geometry coefficients")
    axes[2].set_xlabel("coefficient index")
    axes[2].set_ylabel("value")
    fig.suptitle(f"Meshes representation, sample {sample}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_interp(root: Path, sample: int, output: Path) -> None:
    mask = _load(root, "Interp/Random_UnitCell_mask_10_interp.npy")[
        :, :, sample
    ]
    rr = _load(root, "Interp/Random_UnitCell_rr_10_interp.npy")[:, :, sample]
    sigma = _load(root, "Interp/Random_UnitCell_sigma_10_interp.npy")[
        :, :, sample
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fields = (
        (mask, "mask: solid-domain occupancy", "gray", None),
        (rr, "rr code tiled on grid", "viridis", None),
        (np.ma.masked_where(mask == 0, sigma), "interpolated stress", "viridis", None),
    )
    for axis, (field, title, cmap, limits) in zip(axes, fields):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            aspect="equal",
            vmin=None if limits is None else limits[0],
            vmax=None if limits is None else limits[1],
        )
        axis.set_title(title)
        axis.set_xlabel("grid index / $x$")
        axis.set_ylabel("grid index / $y$")
        fig.colorbar(image, ax=axis)
    fig.suptitle(f"Cartesian interpolation representation, sample {sample}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_omesh(root: Path, sample: int, output: Path) -> None:
    x = _load(root, "Omesh/Random_UnitCell_Deform_X_10_interp.npy")[
        :, :, sample
    ]
    y = _load(root, "Omesh/Random_UnitCell_Deform_Y_10_interp.npy")[
        :, :, sample
    ]
    sigma = _load(root, "Omesh/Random_UnitCell_Deform_sigma_10_interp.npy")[
        :, :, sample
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for i in range(0, x.shape[0], 4):
        axes[0].plot(x[i], y[i], color="tab:blue", lw=0.5)
    for j in range(0, x.shape[1], 2):
        axes[0].plot(x[:, j], y[:, j], color="tab:orange", lw=0.5)
    axes[0].set_title("body-fitted O-grid")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")

    for axis, field, title in (
        (axes[1], x, "physical $x(i,j)$"),
        (axes[2], sigma, "stress on O-grid indices"),
    ):
        image = axis.imshow(field.T, origin="lower", aspect="auto", cmap="viridis")
        axis.set_title(title)
        axis.set_xlabel("circumferential index $i$")
        axis.set_ylabel("radial index $j$")
        fig.colorbar(image, ax=axis)
    fig.suptitle(f"Omesh representation, sample {sample}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_rmesh(root: Path, sample: int, output: Path) -> None:
    source_xy = _load(
        root, "Rmesh/Random_UnitCell_Deform_Grid_XY_10_interp.npy"
    )[:, :, sample]
    x = _load(root, "Rmesh/Random_UnitCell_Deform_Grid_X_10_interp.npy")[
        :, :, sample
    ]
    y = _load(root, "Rmesh/Random_UnitCell_Deform_Grid_Y_10_interp.npy")[
        :, :, sample
    ]
    mask = _load(
        root, "Rmesh/Random_UnitCell_Deform_Grid_mask_10_interp.npy"
    )[:, :, sample]
    sigma = _load(
        root, "Rmesh/Random_UnitCell_Deform_Grid_sigma_10_interp.npy"
    )[:, :, sample]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    axes[0].triplot(_triangulation(source_xy), color="0.3", lw=0.35)
    axes[0].set_title("intermediate 972-point mesh")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("$x$")
    axes[0].set_ylabel("$y$")

    for i in range(0, x.shape[0], 2):
        valid = mask[i]
        axes[1].plot(x[i, valid], y[i, valid], color="tab:blue", lw=0.45)
    for j in range(0, x.shape[1], 2):
        valid = mask[:, j]
        axes[1].plot(x[valid, j], y[valid, j], color="tab:orange", lw=0.45)
    axes[1].set_title("mapped Cartesian grid")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("$x$")
    axes[1].set_ylabel("$y$")

    for axis, field, title, cmap in (
        (axes[2], mask, "mapped-grid mask", "gray"),
        (
            axes[3],
            np.ma.masked_where(~mask, sigma),
            "mapped-grid stress",
            "viridis",
        ),
    ):
        image = axis.imshow(
            field.T, origin="lower", extent=(0, 1, 0, 1), cmap=cmap
        )
        axis.set_title(title)
        axis.set_xlabel("grid index")
        axis.set_ylabel("grid index")
        fig.colorbar(image, ax=axis)
    fig.suptitle(f"Rmesh representation, sample {sample}")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _print_relationships(root: Path) -> None:
    rr = _load(root, "Meshes/Random_UnitCell_rr_10.npy").T
    theta = _load(root, "Meshes/Random_UnitCell_theta_10.npy")
    xy = _load(root, "Meshes/Random_UnitCell_XY_10.npy").transpose(2, 0, 1)
    mask = _load(root, "Interp/Random_UnitCell_mask_10_interp.npy").transpose(
        2, 0, 1
    )
    rr_grid = _load(
        root, "Interp/Random_UnitCell_rr_10_interp.npy"
    ).transpose(2, 0, 1)

    point_radius = np.linalg.norm(xy - np.array([0.5, 0.5]), axis=-1)
    print("Relationship checks")
    print(f"  rr periodic endpoint max error: {np.abs(rr[:, 0] - rr[:, -1]).max():.3e}")
    print(
        "  corr(min point radius, min rr): "
        f"{np.corrcoef(point_radius.min(axis=1), rr.min(axis=1))[0, 1]:.6f}"
    )
    print(
        "  Interp rr variation along tiled axis: "
        f"{np.ptp(rr_grid, axis=2).max():.3e}"
    )
    print(
        "  Interp mask solid fraction: "
        f"{mask.mean():.6f}"
    )

    fourier = np.fft.rfft(rr[:, :-1] - rr[:, :-1].mean(axis=1, keepdims=True))
    features = np.column_stack((fourier[:, 1:6].real, fourier[:, 1:6].imag))
    correlations = np.corrcoef(theta.T, features.T)[:10, 10:]
    best = np.max(np.abs(correlations), axis=1)
    print(
        "  theta/Fourier-mode best absolute correlations: "
        + ", ".join(f"{value:.3f}" for value in best)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory and plot every NPY field in the elasticity dataset."
    )
    parser.add_argument("--root", type=Path, default=Path("data/elasticity"))
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/figures/elasticity/data_inventory"),
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("artifacts/elasticity/elasticity_npy_inventory.csv"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _inventory(args.root, args.csv)
    _print_relationships(args.root)
    _plot_meshes(args.root, args.sample, args.output_dir / "01_meshes.png")
    _plot_interp(args.root, args.sample, args.output_dir / "02_interp.png")
    _plot_omesh(args.root, args.sample, args.output_dir / "03_omesh.png")
    _plot_rmesh(args.root, args.sample, args.output_dir / "04_rmesh.png")
    print(f"Saved inventory to {args.csv}")
    print(f"Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
