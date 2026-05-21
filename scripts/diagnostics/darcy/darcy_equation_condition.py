from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/omni_hc_matplotlib")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from _common import (
    load_darcy_arrays,
    require_matplotlib,
    residual_stats,
    sample_count,
    validate_samples,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check the discrete Darcy condition -div(a grad u)=f on Darcy samples "
            "using harmonic-mean face permeabilities."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/Darcy_421"))
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=1000)
    parser.add_argument("--downsamplex", type=int, default=5)
    parser.add_argument("--downsampley", type=int, default=5)
    parser.add_argument("--force-value", type=float, default=1.0)
    parser.add_argument("--permeability-eps", type=float, default=1e-6)
    parser.add_argument(
        "--interface-steps",
        type=int,
        default=1,
        help=(
            "Number of interior grid steps around permeability jumps to count as "
            "the interface region. Use 1 for immediate neighbours only."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/darcy/darcy_equation_condition"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def darcy_condition_residual(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    force_value: float,
    permeability_eps: float,
    interface_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return interior residual and interface mask for -div(a grad u)-f."""
    if coeff.shape != sol.shape or coeff.ndim != 2:
        raise ValueError(
            "Expected coeff and sol with matching shape (H, W); "
            f"got coeff={coeff.shape}, sol={sol.shape}"
        )
    height, width = sol.shape
    dy = 1.0 / max(height - 1, 1)
    dx = 1.0 / max(width - 1, 1)

    a_face_x = (2.0 * coeff[:, :-1] * coeff[:, 1:]) / (
        coeff[:, :-1] + coeff[:, 1:] + permeability_eps
    )
    a_face_y = (2.0 * coeff[:-1, :] * coeff[1:, :]) / (
        coeff[:-1, :] + coeff[1:, :] + permeability_eps
    )
    flux_x = a_face_x * (sol[:, 1:] - sol[:, :-1]) / dx
    flux_y = a_face_y * (sol[1:, :] - sol[:-1, :]) / dy

    div_x = (flux_x[1:-1, 1:] - flux_x[1:-1, :-1]) / dx
    div_y = (flux_y[1:, 1:-1] - flux_y[:-1, 1:-1]) / dy
    residual = -(div_x + div_y) - float(force_value)

    if interface_steps < 1:
        raise ValueError("interface_steps must be >= 1")

    a_int = coeff[1:-1, 1:-1]
    jump_mask = (
        (a_int != coeff[1:-1, 2:])
        | (a_int != coeff[1:-1, :-2])
        | (a_int != coeff[2:, 1:-1])
        | (a_int != coeff[:-2, 1:-1])
    )
    interface = _dilate_mask_l1(jump_mask, steps=interface_steps - 1)
    return residual, interface


def _dilate_mask_l1(mask: np.ndarray, *, steps: int) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool).copy()
    for _ in range(int(steps)):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return dilated


def summarize(
    coeff: np.ndarray,
    sol: np.ndarray,
    *,
    n_samples: int,
    force_value: float,
    permeability_eps: float,
    interface_steps: int,
) -> list[dict[str, object]]:
    rows = []
    for idx in range(n_samples):
        residual, interface = darcy_condition_residual(
            coeff[idx],
            sol[idx],
            force_value=force_value,
            permeability_eps=permeability_eps,
            interface_steps=interface_steps,
        )
        bulk = ~interface
        row: dict[str, object] = {"sample": idx}
        row.update({f"all_{k}": v for k, v in residual_stats(residual).items()})
        if bulk.any():
            row.update({f"bulk_{k}": v for k, v in residual_stats(residual[bulk]).items()})
        if interface.any():
            row.update(
                {f"interface_{k}": v for k, v in residual_stats(residual[interface]).items()}
            )
        row["interface_fraction"] = float(interface.mean())
        rows.append(row)
    return rows


def print_summary(rows: list[dict[str, object]]) -> None:
    print(f"\nDarcy condition residual summary over {len(rows)} sample(s)")
    keys = [
        "all_mean_abs",
        "bulk_mean_abs",
        "interface_mean_abs",
        "all_p95_abs",
        "all_max_abs",
        "interface_fraction",
    ]
    for key in keys:
        values = np.asarray(
            [float(row[key]) for row in rows if row.get(key) is not None],
            dtype=np.float64,
        )
        if values.size == 0:
            continue
        print(
            f"  {key:<22} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )


def plot_sample(
    sample_idx: int,
    coeff: np.ndarray,
    sol: np.ndarray,
    residual: np.ndarray,
    interface: np.ndarray,
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=150)
    fig.suptitle(f"Darcy sample {sample_idx}: -div(a grad u)-f")

    im0 = axes[0, 0].imshow(coeff, origin="lower", extent=(0, 1, 0, 1), cmap="viridis")
    axes[0, 0].set_title("permeability a")
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(sol, origin="lower", extent=(0, 1, 0, 1), cmap="magma")
    axes[0, 1].set_title("solution u")
    fig.colorbar(im1, ax=axes[0, 1])

    abs_residual = np.abs(residual)
    im2 = axes[1, 0].imshow(
        abs_residual,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="magma",
    )
    axes[1, 0].set_title("|interior residual|")
    fig.colorbar(im2, ax=axes[1, 0])

    im3 = axes[1, 1].imshow(
        interface,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="gray_r",
        vmin=0,
        vmax=1,
    )
    axes[1, 1].set_title("interior interface mask")
    fig.colorbar(im3, ax=axes[1, 1])

    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.tight_layout()

    out_path = out_dir / f"darcy_condition_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used with --no-plot")
    if args.interface_steps < 1:
        raise ValueError("--interface-steps must be >= 1")
    coeff, sol, mat_path = load_darcy_arrays(
        args.data_dir,
        split=args.split,
        downsamplex=args.downsamplex,
        downsampley=args.downsampley,
    )
    validate_samples(args.samples, int(sol.shape[0]))
    n = sample_count(args.summary_samples, int(sol.shape[0]))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Loaded Darcy data: "
        f"path={mat_path}, coeff={coeff.shape}, sol={sol.shape}, split={args.split}"
    )
    print("Condition: -div(a grad u) - f = 0")
    print("Discretization: harmonic-mean face permeability on interior grid cells")
    print(f"Interface region: {args.interface_steps} interior grid step(s)")

    rows = summarize(
        coeff,
        sol,
        n_samples=n,
        force_value=args.force_value,
        permeability_eps=args.permeability_eps,
        interface_steps=args.interface_steps,
    )
    print_summary(rows)
    csv_path = args.out_dir / "darcy_condition_summary.csv"
    write_csv(csv_path, rows)
    print(f"wrote {csv_path}")

    if args.no_plot:
        return
    if plt is None:
        print("matplotlib is not installed; continuing with CSV/statistics only.")
        return
    for sample_idx in args.samples:
        residual, interface = darcy_condition_residual(
            coeff[sample_idx],
            sol[sample_idx],
            force_value=args.force_value,
            permeability_eps=args.permeability_eps,
            interface_steps=args.interface_steps,
        )
        out_path = plot_sample(
            sample_idx,
            coeff[sample_idx],
            sol[sample_idx],
            residual,
            interface,
            out_dir=args.out_dir,
            show=args.show,
        )
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
