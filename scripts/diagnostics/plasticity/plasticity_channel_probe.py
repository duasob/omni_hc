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
    EDGES,
    edge_values,
    load_plasticity_arrays,
    reference_grid,
    require_matplotlib,
    residual_stats,
    safe_corrcoef,
    sample_count,
    scalar_stats,
    select_split,
    validate_samples,
    write_csv,
)


CHANNEL_HYPOTHESIS = (
    "ch0=x_deformed, ch1=y_deformed, ch2=ux_displacement, ch3=uy_displacement"
)


def nan_summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "mean_abs": float("nan"),
            "max_abs": float("nan"),
            "l2": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean_abs": float(np.abs(finite).mean()),
        "max_abs": float(np.abs(finite).max()),
        "l2": float(np.sqrt(np.mean(finite**2))),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Geo-FNO/NSL plasticity output channels. The main hypothesis is "
            "channels 0:2 are deformed coordinates and channels 2:4 are displacement."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/plasticity"),
        help="Directory containing plas_N987_T20.mat, or the .mat file itself.",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="train")
    parser.add_argument("--ntrain", type=int, default=900)
    parser.add_argument("--ntest", type=int, default=80)
    parser.add_argument("--samples", type=int, nargs="+", default=[0, 10, 100])
    parser.add_argument("--summary-samples", type=int, default=64)
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        default=[0, 4, 9, 14, 19],
        help="Timesteps to plot for selected samples.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/plasticity_channel_probe"),
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def channel_stats_rows(output: np.ndarray, *, sample_count_: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(sample_count_)
    for channel in range(4):
        for timestep in range(output.shape[3]):
            stats = scalar_stats(output[:n, :, :, timestep, channel])
            rows.append(
                {
                    "group": "channel_stats",
                    "channel": channel,
                    "timestep": timestep,
                    "metric": "all_values",
                    **stats,
                }
            )
    return rows


def coordinate_displacement_residual_rows(
    output: np.ndarray,
    *,
    sample_count_: int,
) -> list[dict[str, object]]:
    ref = reference_grid((output.shape[1], output.shape[2]))
    rows: list[dict[str, object]] = []
    n = int(sample_count_)
    for timestep in range(output.shape[3]):
        coords = output[:n, :, :, timestep, 0:2].astype(np.float64)
        disp = output[:n, :, :, timestep, 2:4].astype(np.float64)
        residual = coords - ref[None, :, :, :] - disp
        for component, name in enumerate(("x/ux", "y/uy")):
            stats = residual_stats(residual[..., component])
            rows.append(
                {
                    "group": "coords_minus_unit_reference_minus_displacement",
                    "channel": name,
                    "timestep": timestep,
                    **stats,
                }
            )
    return rows


def inferred_material_reference_rows(
    output: np.ndarray,
    *,
    sample_count_: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(sample_count_)
    coords = output[:n, :, :, :, 0:2].astype(np.float64)
    disp = output[:n, :, :, :, 2:4].astype(np.float64)
    material = coords - disp
    inferred_reference = material.mean(axis=(0, 3))
    initial_material = material[:, :, :, 0, :]

    for timestep in range(output.shape[3]):
        residual = material[:, :, :, timestep, :] - inferred_reference[None, :, :, :]
        drift = material[:, :, :, timestep, :] - initial_material
        for component, name in enumerate(("x/ux", "y/uy")):
            ref_stats = residual_stats(residual[..., component])
            rows.append(
                {
                    "group": "coords_minus_displacement_minus_inferred_reference",
                    "channel": name,
                    "timestep": timestep,
                    **ref_stats,
                }
            )
            drift_stats = residual_stats(drift[..., component])
            rows.append(
                {
                    "group": "coords_minus_displacement_temporal_drift_from_t0",
                    "channel": name,
                    "timestep": timestep,
                    **drift_stats,
                }
            )

    i_grid, j_grid = np.meshgrid(
        np.arange(output.shape[1], dtype=np.float64),
        np.arange(output.shape[2], dtype=np.float64),
        indexing="ij",
    )
    design = np.stack(
        [np.ones(i_grid.size), i_grid.reshape(-1), j_grid.reshape(-1)],
        axis=1,
    )
    for component, name in enumerate(("x_reference", "y_reference")):
        target = inferred_reference[..., component].reshape(-1)
        coeff, *_ = np.linalg.lstsq(design, target, rcond=None)
        fitted = coeff[0] + coeff[1] * i_grid + coeff[2] * j_grid
        fit_residual = target.reshape(output.shape[1], output.shape[2]) - fitted
        fit_stats = residual_stats(fit_residual)
        rows.append(
            {
                "group": "inferred_reference_affine_fit",
                "channel": name,
                "timestep": "all",
                "intercept": float(coeff[0]),
                "i_slope": float(coeff[1]),
                "j_slope": float(coeff[2]),
                **fit_stats,
            }
        )
    return rows


def boundary_rows(output: np.ndarray, *, sample_count_: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(sample_count_)
    disp = output[:n, :, :, :, 2:4]
    coords = output[:n, :, :, :, 0:2]
    for edge in EDGES:
        edge_disp = edge_values(disp, edge)
        edge_coords = edge_values(coords, edge)
        for component, name in enumerate(("ux", "uy")):
            stats = scalar_stats(edge_disp[..., component])
            rows.append(
                {
                    "group": f"edge_displacement:{edge}",
                    "channel": name,
                    "timestep": "all",
                    **stats,
                }
            )
        for component, name in enumerate(("x_deformed", "y_deformed")):
            stats = scalar_stats(edge_coords[..., component])
            rows.append(
                {
                    "group": f"edge_coordinates:{edge}",
                    "channel": name,
                    "timestep": "all",
                    **stats,
                }
            )
    return rows


def die_correlation_rows(
    die: np.ndarray,
    output: np.ndarray,
    *,
    sample_count_: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = int(sample_count_)
    timesteps = range(output.shape[3])
    edge_names = ("upper_j0", "lower_jN")
    candidate_channels = (
        (0, "x_deformed"),
        (1, "y_deformed"),
        (2, "ux"),
        (3, "uy"),
    )
    for edge in edge_names:
        for timestep in timesteps:
            for channel, channel_name in candidate_channels:
                values = edge_values(output[:n, :, :, timestep, channel], edge)
                corrs = [
                    safe_corrcoef(die[sample_idx], values[sample_idx])
                    for sample_idx in range(n)
                ]
                corr_stats = nan_summary(np.asarray(corrs, dtype=np.float64))
                rows.append(
                    {
                        "group": f"die_profile_correlation:{edge}",
                        "channel": channel_name,
                        "timestep": timestep,
                        **corr_stats,
                    }
                )
    return rows


def summarize(
    die: np.ndarray,
    output: np.ndarray,
    *,
    sample_count_: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rows.extend(channel_stats_rows(output, sample_count_=sample_count_))
    rows.extend(coordinate_displacement_residual_rows(output, sample_count_=sample_count_))
    rows.extend(inferred_material_reference_rows(output, sample_count_=sample_count_))
    rows.extend(boundary_rows(output, sample_count_=sample_count_))
    rows.extend(die_correlation_rows(die, output, sample_count_=sample_count_))
    return rows


def print_interpretation(rows: list[dict[str, object]], *, n: int, t_final: int) -> None:
    residual_final = [
        row
        for row in rows
        if row["group"] == "coords_minus_unit_reference_minus_displacement"
        and row["timestep"] == t_final
    ]
    inferred_residual_final = [
        row
        for row in rows
        if row["group"] == "coords_minus_displacement_minus_inferred_reference"
        and row["timestep"] == t_final
    ]
    inferred_drift_final = [
        row
        for row in rows
        if row["group"] == "coords_minus_displacement_temporal_drift_from_t0"
        and row["timestep"] == t_final
    ]
    affine_fit = [row for row in rows if row["group"] == "inferred_reference_affine_fit"]
    edge_disp = [
        row
        for row in rows
        if row["group"].startswith("edge_displacement:") and row["timestep"] == "all"
    ]
    die_corr = [
        row
        for row in rows
        if row["group"].startswith("die_profile_correlation:")
        and row["timestep"] == t_final
    ]

    print(f"\nPlasticity channel probe over {n} sample(s)")
    print(f"  hypothesis: {CHANNEL_HYPOTHESIS}")
    print("  evidence from upstream Geo-FNO: visualizes ||channels 2:4|| at points channels 0:2")
    print("  final-step residual for channels 0:2 - unit_reference_grid - channels 2:4:")
    for row in residual_final:
        print(
            f"    {row['channel']:<5} "
            f"mean_abs={row['mean_abs']: .6e}, "
            f"p95_abs={row['p95_abs']: .6e}, "
            f"max_abs={row['max_abs']: .6e}"
        )
    print("  final-step residual for channels 0:2 - channels 2:4 against inferred material grid:")
    for row in inferred_residual_final:
        print(
            f"    {row['channel']:<5} "
            f"mean_abs={row['mean_abs']: .6e}, "
            f"p95_abs={row['p95_abs']: .6e}, "
            f"max_abs={row['max_abs']: .6e}"
        )
    print("  final-step temporal drift of channels 0:2 - channels 2:4 from t=0:")
    for row in inferred_drift_final:
        print(
            f"    {row['channel']:<5} "
            f"mean_abs={row['mean_abs']: .6e}, "
            f"p95_abs={row['p95_abs']: .6e}, "
            f"max_abs={row['max_abs']: .6e}"
        )
    print("  affine fit of inferred material grid versus integer grid indices:")
    for row in affine_fit:
        print(
            f"    {row['channel']:<11} "
            f"intercept={row['intercept']: .6e}, "
            f"i_slope={row['i_slope']: .6e}, "
            f"j_slope={row['j_slope']: .6e}, "
            f"p95_abs={row['p95_abs']: .6e}"
        )
    print("  smallest edge displacement RMS values, useful for identifying the clamped edge:")
    for row in sorted(edge_disp, key=lambda item: float(item["l2"]))[:6]:
        print(
            f"    {row['group'].split(':', 1)[1]:<10} {row['channel']:<2} "
            f"rms={row['l2']: .6e}, max_abs={row['max_abs']: .6e}"
        )
    print("  strongest final-step die profile correlations on upper/lower edges:")
    ranked_corr = sorted(die_corr, key=lambda item: abs(float(item["mean"])), reverse=True)
    for row in ranked_corr[:6]:
        print(
            f"    {row['group'].split(':', 1)[1]:<10} {row['channel']:<10} "
            f"mean_corr={row['mean']: .4f}, std={row['std']: .4f}"
        )


def plot_channel_mosaic(
    sample_idx: int,
    die: np.ndarray,
    output: np.ndarray,
    *,
    timesteps: list[int],
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    channel_names = ("x deformed", "y deformed", "ux", "uy", "||u||")
    fig, axes = plt.subplots(
        len(channel_names),
        len(timesteps),
        figsize=(3.4 * len(timesteps), 13.0),
        dpi=150,
        squeeze=False,
    )
    sample = output[sample_idx]
    disp_mag = np.linalg.norm(sample[..., 2:4], axis=-1)
    fields = [
        sample[..., 0],
        sample[..., 1],
        sample[..., 2],
        sample[..., 3],
        disp_mag,
    ]
    for row_idx, (name, field) in enumerate(zip(channel_names, fields)):
        vmin = float(np.nanmin(field))
        vmax = float(np.nanmax(field))
        for col_idx, timestep in enumerate(timesteps):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(
                field[:, :, timestep].T,
                origin="lower",
                cmap="RdBu_r" if row_idx != 4 else "viridis",
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            ax.set_title(f"{name}, t={timestep}")
            ax.set_xlabel("die x index")
            ax.set_ylabel("through-height index")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    die_ax = axes[0, 0].inset_axes([0.08, 0.08, 0.45, 0.25])
    die_ax.plot(die[sample_idx], color="black", linewidth=1.0)
    die_ax.set_xticks([])
    die_ax.set_yticks([])
    die_ax.set_title("input die", fontsize=8)
    fig.suptitle(f"Plasticity sample {sample_idx}: channel fields")
    fig.tight_layout()
    out_path = out_dir / f"plasticity_channels_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_deformed_scatter(
    sample_idx: int,
    output: np.ndarray,
    *,
    timesteps: list[int],
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    sample = output[sample_idx]
    disp_mag = np.linalg.norm(sample[..., 2:4], axis=-1)
    vmin = float(np.nanmin(disp_mag))
    vmax = float(np.nanmax(disp_mag))
    fig, axes = plt.subplots(
        1,
        len(timesteps),
        figsize=(3.6 * len(timesteps), 3.8),
        dpi=150,
        squeeze=False,
    )
    for ax, timestep in zip(axes.ravel(), timesteps):
        sc = ax.scatter(
            sample[:, :, timestep, 0].reshape(-1),
            sample[:, :, timestep, 1].reshape(-1),
            c=disp_mag[:, :, timestep].reshape(-1),
            s=8,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_title(f"t={timestep}")
        ax.set_xlabel("channel 0")
        ax.set_ylabel("channel 1")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2)
    fig.colorbar(sc, ax=axes.ravel().tolist(), label="||channels 2:4||")
    fig.suptitle(f"Plasticity sample {sample_idx}: Geo-FNO-style deformed scatter")
    out_path = out_dir / f"plasticity_deformed_scatter_sample_{sample_idx:04d}.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def plot_residual_summary(
    rows: list[dict[str, object]],
    *,
    out_dir: Path,
    show: bool,
) -> Path:
    require_matplotlib(plt)
    residual = [
        row
        for row in rows
        if row["group"] == "coords_minus_displacement_minus_inferred_reference"
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8.2, 4.8), dpi=150)
    for component in ("x/ux", "y/uy"):
        part = [row for row in residual if row["channel"] == component]
        timesteps = np.array([int(row["timestep"]) for row in part], dtype=np.int64)
        mean_abs = np.array([float(row["mean_abs"]) for row in part], dtype=np.float64)
        p95_abs = np.array([float(row["p95_abs"]) for row in part], dtype=np.float64)
        ax.plot(timesteps, mean_abs, marker="o", linewidth=1.2, label=f"{component} mean abs")
        ax.plot(timesteps, p95_abs, marker=".", linewidth=1.0, linestyle="--", label=f"{component} p95 abs")
    ax.set_title("Coordinate-displacement consistency against inferred material grid")
    ax.set_xlabel("timestep")
    ax.set_ylabel("|channels 0:2 - channels 2:4 - inferred grid|")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "plasticity_coordinate_displacement_residual.png"
    fig.savefig(out_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

    die, output, mat_path = load_plasticity_arrays(args.data_dir)
    die, output = select_split(
        die,
        output,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    if max(args.timesteps) >= output.shape[3] or min(args.timesteps) < 0:
        raise IndexError(f"timesteps must be in [0, {output.shape[3]})")
    validate_samples(args.samples, output.shape[0])

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    n_summary = sample_count(args.summary_samples, output.shape[0])
    rows = summarize(die, output, sample_count_=n_summary)
    csv_path = out_dir / "plasticity_channel_probe.csv"
    write_csv(csv_path, rows)

    print(f"Loaded {mat_path}")
    print(f"  input shape:  {die.shape}")
    print(f"  output shape: {output.shape}")
    print(f"  wrote CSV:    {csv_path}")
    print_interpretation(rows, n=n_summary, t_final=max(args.timesteps))

    if not args.no_plot:
        plot_paths = [plot_residual_summary(rows, out_dir=out_dir, show=args.show)]
        for sample_idx in args.samples:
            plot_paths.append(
                plot_channel_mosaic(
                    sample_idx,
                    die,
                    output,
                    timesteps=args.timesteps,
                    out_dir=out_dir,
                    show=args.show,
                )
            )
            plot_paths.append(
                plot_deformed_scatter(
                    sample_idx,
                    output,
                    timesteps=args.timesteps,
                    out_dir=out_dir,
                    show=args.show,
                )
            )
        print("  wrote plots:")
        for path in plot_paths:
            print(f"    {path}")


if __name__ == "__main__":
    main()
