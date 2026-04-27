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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect pipe cross-section flux conservation along streamwise index i."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pipe"),
        help="Directory containing Pipe_X.npy, Pipe_Y.npy, and Pipe_Q.npy.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="+",
        default=[0, 10, 100],
        help="Sample indices to print and plot.",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=1000,
        help="Number of leading samples to include in aggregate statistics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/pipe_flux"),
        help="Directory where plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print and write flux statistics; do not write figures.",
    )
    return parser.parse_args()


def load_pipe_arrays(data_dir: Path):
    required = ("Pipe_X.npy", "Pipe_Y.npy", "Pipe_Q.npy")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required pipe files in {data_dir}: {', '.join(missing)}"
        )

    x = np.load(data_dir / "Pipe_X.npy", mmap_mode="r")
    y = np.load(data_dir / "Pipe_Y.npy", mmap_mode="r")
    q = np.load(data_dir / "Pipe_Q.npy", mmap_mode="r")
    if x.shape != y.shape:
        raise ValueError(f"Pipe_X and Pipe_Y shapes differ: {x.shape} vs {y.shape}")
    if q.ndim != 4 or q.shape[0] != x.shape[0] or q.shape[2:] != x.shape[1:]:
        raise ValueError(f"Expected Pipe_Q=(N,C,H,W) matching X/Y; got {q.shape}")
    if q.shape[1] < 2:
        raise ValueError(f"Expected ux and uy channels in Pipe_Q, got {q.shape[1]}")
    return x, y, q


def cross_section_fluxes(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> dict[str, np.ndarray]:
    dx = np.diff(x, axis=1)
    dy = np.diff(y, axis=1)
    ds = np.sqrt(dx**2 + dy**2)
    ux_mid = 0.5 * (ux[:, :-1] + ux[:, 1:])
    uy_mid = 0.5 * (uy[:, :-1] + uy[:, 1:])
    speed_mid = np.sqrt(ux_mid**2 + uy_mid**2)

    ux_dy = np.sum(ux_mid * dy, axis=1)
    normal = np.sum(ux_mid * dy - uy_mid * dx, axis=1)
    speed_ds = np.sum(speed_mid * ds, axis=1)
    uy_ds = np.sum(uy_mid * ds, axis=1)
    return {
        "ux_dy": ux_dy,
        "normal": normal,
        "speed_ds": speed_ds,
        "uy_ds": uy_ds,
    }


def geometry_diagnostics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    dx_di = np.diff(x, axis=0)
    dy_di = np.diff(y, axis=0)
    dx_dj = np.diff(x, axis=1)
    dy_dj = np.diff(y, axis=1)
    tangent_i = np.sqrt(dx_di**2 + dy_di**2)
    tangent_j = np.sqrt(dx_dj**2 + dy_dj**2)
    skew_ratio = np.abs(dx_dj) / np.maximum(np.abs(dy_dj), 1e-12)
    return {
        "max_abs_dx_dj": float(np.max(np.abs(dx_dj))),
        "mean_abs_dx_dj": float(np.mean(np.abs(dx_dj))),
        "max_abs_dy_di": float(np.max(np.abs(dy_di))),
        "mean_abs_dy_di": float(np.mean(np.abs(dy_di))),
        "mean_streamwise_spacing": float(np.mean(tangent_i)),
        "mean_transverse_spacing": float(np.mean(tangent_j)),
        "max_cross_section_skew_ratio": float(np.max(skew_ratio)),
        "mean_cross_section_skew_ratio": float(np.mean(skew_ratio)),
    }


def cell_signed_area(x_cell: np.ndarray, y_cell: np.ndarray) -> float:
    return 0.5 * float(
        np.dot(x_cell, np.roll(y_cell, -1)) - np.dot(y_cell, np.roll(x_cell, -1))
    )


def finite_volume_divergence(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
) -> dict[str, float]:
    h, w = x.shape
    div = np.empty((h - 1, w - 1), dtype=np.float64)
    for i in range(h - 1):
        for j in range(w - 1):
            # Counter-clockwise cell vertices: bottom-left, bottom-right,
            # top-right, top-left in structured index space.
            vertices = (
                (i, j),
                (i + 1, j),
                (i + 1, j + 1),
                (i, j + 1),
            )
            x_cell = np.array([x[ii, jj] for ii, jj in vertices], dtype=np.float64)
            y_cell = np.array([y[ii, jj] for ii, jj in vertices], dtype=np.float64)
            area = abs(cell_signed_area(x_cell, y_cell))
            flux = 0.0
            for start, end in zip(vertices, vertices[1:] + vertices[:1]):
                i0, j0 = start
                i1, j1 = end
                dx = float(x[i1, j1] - x[i0, j0])
                dy = float(y[i1, j1] - y[i0, j0])
                ux_mid = 0.5 * float(ux[i0, j0] + ux[i1, j1])
                uy_mid = 0.5 * float(uy[i0, j0] + uy[i1, j1])
                flux += ux_mid * dy - uy_mid * dx
            div[i, j] = flux / max(area, 1e-12)
    abs_div = np.abs(div)
    return {
        "cell_div_abs_mean": float(abs_div.mean()),
        "cell_div_abs_max": float(abs_div.max()),
        "cell_div_signed_mean": float(div.mean()),
        "cell_div_signed_std": float(div.std()),
    }


def relative_variation(values: np.ndarray) -> float:
    scale = max(float(abs(values[0])), 1e-12)
    return float(np.max(np.abs(values - values[0])) / scale)


def section_summary(values: np.ndarray) -> dict[str, float]:
    inlet = float(values[0])
    outlet = float(values[-1])
    diff = values - inlet
    return {
        "inlet": inlet,
        "outlet": outlet,
        "outlet_minus_inlet": outlet - inlet,
        "max_abs_minus_inlet": float(np.max(np.abs(diff))),
        "mean_abs_minus_inlet": float(np.mean(np.abs(diff))),
        "relative_max_abs_minus_inlet": relative_variation(values),
    }


def print_sample_stats(sample_idx: int, fluxes: dict[str, np.ndarray]) -> None:
    print(f"\nSample {sample_idx} cross-section flux stats")
    for name, values in fluxes.items():
        stats = section_summary(values)
        print(
            f"  {name:<6} "
            f"inlet={stats['inlet']: .8e}, "
            f"outlet={stats['outlet']: .8e}, "
            f"outlet-inlet={stats['outlet_minus_inlet']: .3e}, "
            f"max|section-inlet|={stats['max_abs_minus_inlet']: .3e}, "
            f"rel_max={stats['relative_max_abs_minus_inlet']: .3e}"
        )


def print_sample_extra_stats(
    sample_idx: int,
    geometry: dict[str, float],
    divergence: dict[str, float],
) -> None:
    print(f"  geometry sample {sample_idx}:")
    print(
        f"    max|dx/dj|={geometry['max_abs_dx_dj']: .3e}, "
        f"mean|dx/dj|={geometry['mean_abs_dx_dj']: .3e}, "
        f"max|dy/di|={geometry['max_abs_dy_di']: .3e}, "
        f"mean|dy/di|={geometry['mean_abs_dy_di']: .3e}"
    )
    print(
        f"    mean ds_i={geometry['mean_streamwise_spacing']: .3e}, "
        f"mean ds_j={geometry['mean_transverse_spacing']: .3e}, "
        f"max skew={geometry['max_cross_section_skew_ratio']: .3e}"
    )
    print(f"  finite-volume cell divergence sample {sample_idx}:")
    print(
        f"    abs_mean={divergence['cell_div_abs_mean']: .3e}, "
        f"abs_max={divergence['cell_div_abs_max']: .3e}, "
        f"signed_mean={divergence['cell_div_signed_mean']: .3e}, "
        f"signed_std={divergence['cell_div_signed_std']: .3e}"
    )


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "flux_type",
        "inlet",
        "outlet",
        "outlet_minus_inlet",
        "max_abs_minus_inlet",
        "mean_abs_minus_inlet",
        "relative_max_abs_minus_inlet",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row[name]) for name in fieldnames) + "\n")


def write_extra_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "max_abs_dx_dj",
        "mean_abs_dx_dj",
        "max_abs_dy_di",
        "mean_abs_dy_di",
        "mean_streamwise_spacing",
        "mean_transverse_spacing",
        "max_cross_section_skew_ratio",
        "mean_cross_section_skew_ratio",
        "cell_div_abs_mean",
        "cell_div_abs_max",
        "cell_div_signed_mean",
        "cell_div_signed_std",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(",".join(str(row[name]) for name in fieldnames) + "\n")


def summarize(rows: list[dict[str, object]]) -> None:
    samples = len({int(row["sample"]) for row in rows})
    print(f"Aggregate cross-section flux summary over {samples} sample(s)")
    for flux_type in sorted({str(row["flux_type"]) for row in rows}):
        selected = [row for row in rows if row["flux_type"] == flux_type]
        outlet_diff = np.array([float(row["outlet_minus_inlet"]) for row in selected])
        max_diff = np.array([float(row["max_abs_minus_inlet"]) for row in selected])
        rel_max = np.array([float(row["relative_max_abs_minus_inlet"]) for row in selected])
        print(
            f"  {flux_type:<6} "
            f"mean|outlet-inlet|={np.abs(outlet_diff).mean(): .6e}, "
            f"max|outlet-inlet|={np.abs(outlet_diff).max(): .6e}, "
            f"mean max|section-inlet|={max_diff.mean(): .6e}, "
            f"max rel section drift={rel_max.max(): .6e}"
        )
        percentiles = ", ".join(
            f"p{p}={np.percentile(rel_max, p):.3e}"
            for p in (50, 75, 90, 95, 99, 100)
        )
        print(f"    rel section drift percentiles: {percentiles}")
        worst = sorted(
            selected,
            key=lambda row: float(row["relative_max_abs_minus_inlet"]),
            reverse=True,
        )[:5]
        print("    worst samples:")
        for row in worst:
            print(
                f"      sample={int(row['sample']):<4} "
                f"rel={float(row['relative_max_abs_minus_inlet']):.3e} "
                f"max_abs={float(row['max_abs_minus_inlet']):.3e}"
            )


def summarize_extra(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    print(f"Aggregate geometry/divergence summary over {len(rows)} sample(s)")
    for key in (
        "max_abs_dx_dj",
        "max_abs_dy_di",
        "max_cross_section_skew_ratio",
        "cell_div_abs_mean",
        "cell_div_abs_max",
    ):
        values = np.array([float(row[key]) for row in rows])
        print(
            f"  {key:<30} "
            f"mean={values.mean(): .6e}, "
            f"p95={np.percentile(values, 95): .6e}, "
            f"max={values.max(): .6e}"
        )


def plot_sample(
    out_dir: Path,
    sample_idx: int,
    fluxes: dict[str, np.ndarray],
    *,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots. Rerun with --no-plot.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=130)
    axes = axes.reshape(-1)
    for ax, (name, values) in zip(axes, fluxes.items()):
        ax.plot(values, linewidth=1.8, label=name)
        ax.axhline(values[0], color="0.25", linestyle="--", linewidth=1.0, label="inlet")
        ax.set_xlabel("streamwise index i")
        ax.set_ylabel("cross-section flux")
        ax.set_title(name)
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.suptitle(f"Pipe cross-section flux, sample {sample_idx}")
    fig.tight_layout()

    out_path = out_dir / f"pipe_flux_sample_{sample_idx:04d}.png"
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

    x_all, y_all, q_all = load_pipe_arrays(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    extra_rows: list[dict[str, object]] = []
    n_summary = min(max(int(args.summary_samples), 0), int(q_all.shape[0]))
    for sample_idx in range(n_summary):
        x = np.asarray(x_all[sample_idx], dtype=np.float64)
        y = np.asarray(y_all[sample_idx], dtype=np.float64)
        ux = np.asarray(q_all[sample_idx, 0], dtype=np.float64)
        uy = np.asarray(q_all[sample_idx, 1], dtype=np.float64)
        fluxes = cross_section_fluxes(
            x,
            y,
            ux,
            uy,
        )
        for flux_type, values in fluxes.items():
            rows.append({"sample": sample_idx, "flux_type": flux_type} | section_summary(values))
        extra_rows.append(
            {"sample": sample_idx}
            | geometry_diagnostics(x, y)
            | finite_volume_divergence(x, y, ux, uy)
        )

    if rows:
        summarize(rows)
        summarize_extra(extra_rows)
        csv_path = args.out_dir / "pipe_flux_summary.csv"
        write_summary_csv(csv_path, rows)
        print(f"wrote {csv_path}")
        extra_csv_path = args.out_dir / "pipe_flux_extra_summary.csv"
        write_extra_summary_csv(extra_csv_path, extra_rows)
        print(f"wrote {extra_csv_path}")

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= q_all.shape[0]:
            raise IndexError(f"Sample {sample_idx} is outside [0, {q_all.shape[0]})")
        x = np.asarray(x_all[sample_idx], dtype=np.float64)
        y = np.asarray(y_all[sample_idx], dtype=np.float64)
        ux = np.asarray(q_all[sample_idx, 0], dtype=np.float64)
        uy = np.asarray(q_all[sample_idx, 1], dtype=np.float64)
        fluxes = cross_section_fluxes(
            x,
            y,
            ux,
            uy,
        )
        print_sample_stats(sample_idx, fluxes)
        print_sample_extra_stats(
            sample_idx,
            geometry_diagnostics(x, y),
            finite_volume_divergence(x, y, ux, uy),
        )
        if args.no_plot or plt is None:
            continue
        out_path = plot_sample(args.out_dir, sample_idx, fluxes, show=args.show)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
