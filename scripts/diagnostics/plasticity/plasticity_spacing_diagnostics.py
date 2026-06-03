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

from _common import load_plasticity_arrays, select_split, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure nearest-neighbor coordinate spacings in the raw plasticity "
            "dataset to calibrate hard min-spacing constraints."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/plasticity"),
        help="Directory containing plas_N987_T20.mat, or the .mat file itself.",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--ntrain", type=int, default=900)
    parser.add_argument("--ntest", type=int, default=80)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on samples after split selection. Defaults to the full split.",
    )
    parser.add_argument(
        "--t-out",
        type=int,
        default=None,
        help="Optional cap on timesteps. Defaults to all timesteps in the MAT file.",
    )
    parser.add_argument(
        "--downsamplex",
        type=int,
        default=1,
        help="Match the benchmark x downsampling before measuring spacings.",
    )
    parser.add_argument(
        "--downsampley",
        type=int,
        default=1,
        help="Match the benchmark y downsampling before measuring spacings.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        action="append",
        default=[1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3],
        help=(
            "Spacing thresholds to count. Repeatable. Defaults to "
            "1e-6, 1e-5, 1e-4, 1e-3."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of smallest spacing examples to write for dx and dy.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/plasticity/plasticity_spacing_diagnostics"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def finite_percentiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {key: float("nan") for key in _STAT_KEYS}
    percentiles = np.percentile(
        finite,
        [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9, 100.0],
    )
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(percentiles[0]),
        "p0001": float(percentiles[1]),
        "p001": float(percentiles[2]),
        "p01": float(percentiles[3]),
        "p1": float(percentiles[4]),
        "p5": float(percentiles[5]),
        "median": float(percentiles[6]),
        "p95": float(percentiles[7]),
        "p99": float(percentiles[8]),
        "p999": float(percentiles[9]),
        "max": float(percentiles[10]),
    }


_STAT_KEYS = (
    "count",
    "mean",
    "std",
    "min",
    "p0001",
    "p001",
    "p01",
    "p1",
    "p5",
    "median",
    "p95",
    "p99",
    "p999",
    "max",
)


def summary_row(
    name: str,
    values: np.ndarray,
    *,
    thresholds: list[float],
    expected_positive: bool = True,
) -> dict[str, object]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    row: dict[str, object] = {
        "quantity": name,
        "expected_positive": bool(expected_positive),
        **finite_percentiles(finite),
    }
    if finite.size:
        row["non_positive_count"] = int(np.count_nonzero(finite <= 0.0))
        row["non_positive_fraction"] = float(np.count_nonzero(finite <= 0.0) / finite.size)
        for threshold in thresholds:
            key = f"count_lt_{threshold:g}"
            row[key] = int(np.count_nonzero(finite < threshold))
            row[f"fraction_lt_{threshold:g}"] = float(row[key] / finite.size)
    else:
        row["non_positive_count"] = 0
        row["non_positive_fraction"] = float("nan")
        for threshold in thresholds:
            row[f"count_lt_{threshold:g}"] = 0
            row[f"fraction_lt_{threshold:g}"] = float("nan")
    return row


def smallest_rows(
    name: str,
    values: np.ndarray,
    *,
    axis_names: tuple[str, ...],
    top_k: int,
) -> list[dict[str, object]]:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(-1)
    finite_idx = np.flatnonzero(np.isfinite(flat))
    if finite_idx.size == 0:
        return []
    order = finite_idx[np.argsort(flat[finite_idx])]
    rows: list[dict[str, object]] = []
    for flat_idx in order[: max(int(top_k), 0)]:
        index = np.unravel_index(int(flat_idx), values.shape)
        row: dict[str, object] = {
            "quantity": name,
            "value": float(values[index]),
        }
        row.update({axis: int(index[pos]) for pos, axis in enumerate(axis_names)})
        rows.append(row)
    return rows


def height_envelope_rows(y: np.ndarray, *, top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y = np.asarray(y, dtype=np.float64)
    initial_max = float(np.nanmax(y[:, :, :, 0]))
    top_y = y[:, :, 0, :]
    global_excess = y - initial_max
    top_excess = top_y - initial_max

    rows = []
    for name, values, excess in (
        ("global_y", y, global_excess),
        ("top_j0_y", top_y, top_excess),
    ):
        finite_excess = excess[np.isfinite(excess)]
        exceed_count = int(np.count_nonzero(finite_excess > 0.0))
        rows.append(
            {
                "quantity": name,
                "initial_mass_y_max": initial_max,
                **finite_percentiles(values),
                "exceed_initial_max_count": exceed_count,
                "checked_count": int(finite_excess.size),
                "exceed_initial_max_fraction": float(
                    exceed_count / max(int(finite_excess.size), 1)
                ),
                "max_excess_over_initial_max": float(np.nanmax(excess)),
            }
        )

    excess_rows = []
    flat = global_excess.reshape(-1)
    finite_positive = np.flatnonzero(np.isfinite(flat) & (flat > 0.0))
    if finite_positive.size:
        order = finite_positive[np.argsort(flat[finite_positive])[::-1]]
        for flat_idx in order[: max(int(top_k), 0)]:
            index = np.unravel_index(int(flat_idx), global_excess.shape)
            excess_rows.append(
                {
                    "quantity": "global_y_exceeds_initial_max",
                    "value": float(y[index]),
                    "excess": float(global_excess[index]),
                    "sample": int(index[0]),
                    "i": int(index[1]),
                    "j": int(index[2]),
                    "timestep": int(index[3]),
                    "initial_mass_y_max": initial_max,
                }
            )
    return rows, excess_rows


def make_histogram_plot(
    *,
    dx: np.ndarray,
    dy: np.ndarray,
    height: np.ndarray,
    bottom_y: np.ndarray,
    out_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots. Rerun with --no-plot.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    plot_items = [
        (axes[0, 0], dx, "signed horizontal spacing x[i] - x[i+1]"),
        (axes[0, 1], dy, "signed vertical spacing y[j] - y[j+1]"),
        (axes[1, 0], height, "column height top - bottom"),
        (axes[1, 1], bottom_y, "bottom y"),
    ]
    for ax, values, title in plot_items:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            ax.set_title(title)
            ax.text(0.5, 0.5, "no finite values", ha="center", va="center")
            continue
        ax.hist(finite, bins=80, color="#4C78A8", alpha=0.85)
        ax.axvline(float(finite.min()), color="#E45756", linewidth=1.2, label="min")
        ax.axvline(float(np.percentile(finite, 1.0)), color="#F58518", linewidth=1.2, label="p1")
        ax.axvline(float(np.median(finite)), color="#54A24B", linewidth=1.2, label="median")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.downsamplex <= 0 or args.downsampley <= 0:
        raise ValueError("downsamplex and downsampley must be positive")

    _, output, mat_path = load_plasticity_arrays(args.data_dir)
    _, output = select_split(
        np.empty((output.shape[0], output.shape[1])),
        output,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    if args.max_samples is not None:
        output = output[: max(int(args.max_samples), 0)]
    if args.t_out is not None:
        output = output[:, :, :, : max(int(args.t_out), 0)]
    output = output[:, :: args.downsamplex, :: args.downsampley]
    if output.size == 0:
        raise ValueError("No output values selected by the requested split/caps.")

    coords = output[..., 0:2]
    x = coords[..., 0]
    y = coords[..., 1]

    signed_dx = x[:, :-1, :, :] - x[:, 1:, :, :]
    signed_dy = y[:, :, :-1, :] - y[:, :, 1:, :]
    column_height = y[:, :, 0, :] - y[:, :, -1, :]
    bottom_y = y[:, :, -1, :]
    height_rows, height_excess_rows = height_envelope_rows(y, top_k=args.top_k)

    thresholds = sorted(float(value) for value in args.threshold)
    summary_rows = [
        {
            "source": str(mat_path),
            "split": args.split,
            "shape": tuple(int(v) for v in output.shape),
            **summary_row("signed_dx", signed_dx, thresholds=thresholds),
        },
        {
            "source": str(mat_path),
            "split": args.split,
            "shape": tuple(int(v) for v in output.shape),
            **summary_row("signed_dy", signed_dy, thresholds=thresholds),
        },
        {
            "source": str(mat_path),
            "split": args.split,
            "shape": tuple(int(v) for v in output.shape),
            **summary_row("column_height_top_minus_bottom", column_height, thresholds=thresholds),
        },
        {
            "source": str(mat_path),
            "split": args.split,
            "shape": tuple(int(v) for v in output.shape),
            **summary_row(
                "bottom_y_abs",
                np.abs(bottom_y),
                thresholds=thresholds,
                expected_positive=False,
            ),
        },
    ]

    smallest = []
    smallest.extend(
        smallest_rows(
            "signed_dx",
            signed_dx,
            axis_names=("sample", "edge_i", "j", "timestep"),
            top_k=args.top_k,
        )
    )
    smallest.extend(
        smallest_rows(
            "signed_dy",
            signed_dy,
            axis_names=("sample", "i", "edge_j", "timestep"),
            top_k=args.top_k,
        )
    )
    smallest.extend(
        smallest_rows(
            "column_height_top_minus_bottom",
            column_height,
            axis_names=("sample", "i", "timestep"),
            top_k=args.top_k,
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "spacing_summary.csv", summary_rows)
    write_csv(args.out_dir / "height_envelope_summary.csv", height_rows)
    if smallest:
        write_csv(args.out_dir / "smallest_spacings.csv", smallest)
    if height_excess_rows:
        write_csv(args.out_dir / "height_envelope_exceedances.csv", height_excess_rows)
    if not args.no_plot:
        make_histogram_plot(
            dx=signed_dx,
            dy=signed_dy,
            height=column_height,
            bottom_y=bottom_y,
            out_path=args.out_dir / "spacing_histograms.png",
        )

    print(f"Loaded {mat_path}")
    print(f"Selected output shape: {tuple(int(v) for v in output.shape)}")
    print(f"Wrote {args.out_dir / 'spacing_summary.csv'}")
    print(f"Wrote {args.out_dir / 'height_envelope_summary.csv'}")
    if smallest:
        print(f"Wrote {args.out_dir / 'smallest_spacings.csv'}")
    if height_excess_rows:
        print(f"Wrote {args.out_dir / 'height_envelope_exceedances.csv'}")
    if not args.no_plot:
        print(f"Wrote {args.out_dir / 'spacing_histograms.png'}")
    for row in summary_rows:
        print(
            f"{row['quantity']}: min={row['min']:.6g}, "
            f"p001={row['p001']:.6g}, p1={row['p1']:.6g}, "
            f"median={row['median']:.6g}, non_positive={row['non_positive_count']}"
        )
    for row in height_rows:
        print(
            f"{row['quantity']}: initial_max={row['initial_mass_y_max']:.6g}, "
            f"max={row['max']:.6g}, max_excess={row['max_excess_over_initial_max']:.6g}, "
            f"exceed_count={row['exceed_initial_max_count']}"
        )


if __name__ == "__main__":
    main()
