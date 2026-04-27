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

try:
    from scipy.optimize import curve_fit
except ImportError:
    curve_fit = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit candidate parametric profiles to Pipe inlet ux."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pipe"),
        help="Directory containing Pipe_Y.npy and Pipe_Q.npy.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        nargs="*",
        default=[0, 10, 100],
        help="Specific sample indices to plot and print. Use none with --summary-samples for summary only.",
    )
    parser.add_argument(
        "--summary-samples",
        type=int,
        default=1000,
        help="Number of leading samples to include in aggregate fit statistics.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/pipe_inlet"),
        help="Directory where figures and CSV summaries are written.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively after saving them.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print and write fit statistics; do not write figures.",
    )
    return parser.parse_args()


def load_pipe_arrays(data_dir: Path):
    required = ("Pipe_Y.npy", "Pipe_Q.npy")
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required pipe files in {data_dir}: {', '.join(missing)}"
        )

    y = np.load(data_dir / "Pipe_Y.npy", mmap_mode="r")
    q = np.load(data_dir / "Pipe_Q.npy", mmap_mode="r")
    if q.ndim != 4 or q.shape[1] < 1 or q.shape[0] != y.shape[0] or q.shape[2:] != y.shape[1:]:
        raise ValueError(f"Expected Pipe_Y=(N,H,W), Pipe_Q=(N,C,H,W); got {y.shape}, {q.shape}")
    return y, q


def normalized_inlet_coordinate(y_inlet: np.ndarray) -> np.ndarray:
    y_min = float(y_inlet.min())
    y_max = float(y_inlet.max())
    if y_max <= y_min:
        raise ValueError("Inlet Y coordinate has zero extent.")
    return (y_inlet - y_min) / (y_max - y_min)


def gaussian_profile(t, amplitude, center, sigma, offset):
    sigma = np.maximum(sigma, 1e-12)
    return offset + amplitude * np.exp(-0.5 * ((t - center) / sigma) ** 2)


def wall_zero_gaussian_profile(t, amplitude, center, sigma):
    raw = gaussian_profile(t, 1.0, center, sigma, 0.0)
    left = gaussian_profile(0.0, 1.0, center, sigma, 0.0)
    right = gaussian_profile(1.0, 1.0, center, sigma, 0.0)
    chord = left + (right - left) * t
    shaped = raw - chord
    max_abs = np.max(np.abs(shaped))
    if max_abs <= 1e-12:
        return np.zeros_like(t)
    return amplitude * shaped / max_abs


def parabola_profile(t, amplitude):
    return amplitude * 4.0 * t * (1.0 - t)


def shifted_parabola_profile(t, amplitude, center):
    left = max(float(center), 1e-12)
    right = max(float(1.0 - center), 1e-12)
    z = np.where(t <= center, (t - center) / left, (t - center) / right)
    return amplitude * np.clip(1.0 - z**2, 0.0, None)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    if total <= 1e-20:
        return 1.0 if residual <= 1e-20 else 0.0
    return 1.0 - residual / total


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def fit_inlet_profile(t: np.ndarray, ux: np.ndarray) -> dict[str, dict[str, object]]:
    if curve_fit is None:
        raise RuntimeError("scipy is required for inlet profile fitting.")

    amp0 = max(float(ux.max() - ux.min()), 1e-8)
    center0 = float(t[np.argmax(ux)])
    sigma0 = 0.25
    offset0 = float(min(ux[0], ux[-1]))

    fits: dict[str, dict[str, object]] = {}

    popt, _ = curve_fit(
        gaussian_profile,
        t,
        ux,
        p0=(amp0, center0, sigma0, offset0),
        bounds=([0.0, 0.0, 1e-3, -np.inf], [np.inf, 1.0, 10.0, np.inf]),
        maxfev=20000,
    )
    pred = gaussian_profile(t, *popt)
    fits["gaussian"] = {
        "params": {
            "amplitude": float(popt[0]),
            "center": float(popt[1]),
            "sigma": float(popt[2]),
            "offset": float(popt[3]),
        },
        "pred": pred,
        "rmse": rmse(ux, pred),
        "r2": r2_score(ux, pred),
    }

    popt, _ = curve_fit(
        wall_zero_gaussian_profile,
        t,
        ux,
        p0=(amp0, center0, sigma0),
        bounds=([0.0, 0.0, 1e-3], [np.inf, 1.0, 10.0]),
        maxfev=20000,
    )
    pred = wall_zero_gaussian_profile(t, *popt)
    fits["wall_zero_gaussian"] = {
        "params": {
            "amplitude": float(popt[0]),
            "center": float(popt[1]),
            "sigma": float(popt[2]),
        },
        "pred": pred,
        "rmse": rmse(ux, pred),
        "r2": r2_score(ux, pred),
    }

    popt, _ = curve_fit(
        parabola_profile,
        t,
        ux,
        p0=(max(float(ux.max()), 1e-8),),
        bounds=([0.0], [np.inf]),
        maxfev=20000,
    )
    pred = parabola_profile(t, *popt)
    fits["wall_zero_parabola"] = {
        "params": {"amplitude": float(popt[0]), "center": 0.5},
        "pred": pred,
        "rmse": rmse(ux, pred),
        "r2": r2_score(ux, pred),
    }

    popt, _ = curve_fit(
        shifted_parabola_profile,
        t,
        ux,
        p0=(max(float(ux.max()), 1e-8), center0),
        bounds=([0.0, 0.0], [np.inf, 1.0]),
        maxfev=20000,
    )
    pred = shifted_parabola_profile(t, *popt)
    fits["shifted_wall_zero_parabola"] = {
        "params": {"amplitude": float(popt[0]), "center": float(popt[1])},
        "pred": pred,
        "rmse": rmse(ux, pred),
        "r2": r2_score(ux, pred),
    }

    return fits


def sample_profile(y_all, q_all, sample_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_inlet = np.asarray(y_all[sample_idx, 0, :], dtype=np.float64)
    ux_inlet = np.asarray(q_all[sample_idx, 0, 0, :], dtype=np.float64)
    t = normalized_inlet_coordinate(y_inlet)
    order = np.argsort(t)
    return t[order], y_inlet[order], ux_inlet[order]


def best_model_name(fits: dict[str, dict[str, object]]) -> str:
    return min(fits, key=lambda name: float(fits[name]["rmse"]))


def print_fit(sample_idx: int, fits: dict[str, dict[str, object]]) -> None:
    print(f"\nSample {sample_idx} inlet ux fits")
    for name, fit in sorted(fits.items(), key=lambda item: float(item[1]["rmse"])):
        print(
            f"  {name:<28} "
            f"rmse={float(fit['rmse']): .6e}, "
            f"r2={float(fit['r2']): .8f}, "
            f"params={fit['params']}"
        )


def write_summary_csv(out_path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "model",
        "rmse",
        "r2",
        "amplitude",
        "center",
        "sigma",
        "offset",
    ]
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(",".join(fieldnames) + "\n")
        for row in rows:
            handle.write(
                ",".join(str(row.get(name, "")) for name in fieldnames) + "\n"
            )


def summarize_fits(rows: list[dict[str, object]]) -> None:
    models = sorted({str(row["model"]) for row in rows})
    print(f"\nAggregate inlet ux fit summary over {len({row['sample'] for row in rows})} sample(s)")
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        rmse_values = np.array([float(row["rmse"]) for row in model_rows])
        r2_values = np.array([float(row["r2"]) for row in model_rows])
        print(
            f"  {model:<28} "
            f"mean_rmse={rmse_values.mean(): .6e}, "
            f"max_rmse={rmse_values.max(): .6e}, "
            f"mean_r2={r2_values.mean(): .8f}, "
            f"min_r2={r2_values.min(): .8f}"
        )

    best_counts = {model: 0 for model in models}
    for sample in sorted({int(row["sample"]) for row in rows}):
        sample_rows = [row for row in rows if int(row["sample"]) == sample]
        best = min(sample_rows, key=lambda row: float(row["rmse"]))
        best_counts[str(best["model"])] += 1
    print("Best model counts")
    for model, count in sorted(best_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {model:<28} {count}")


def plot_sample(
    out_dir: Path,
    sample_idx: int,
    t: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    fits: dict[str, dict[str, object]],
    *,
    show: bool,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required for plots. Rerun with --no-plot.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
    axes[0].plot(y, ux, "ko", markersize=3, label="data")
    axes[1].plot(t, ux, "ko", markersize=3, label="data")
    for name, fit in sorted(fits.items(), key=lambda item: float(item[1]["rmse"])):
        label = f"{name} rmse={float(fit['rmse']):.2e}"
        axes[0].plot(y, fit["pred"], linewidth=1.5, label=label)
        axes[1].plot(t, fit["pred"], linewidth=1.5, label=label)
    axes[0].set_xlabel("physical inlet y")
    axes[1].set_xlabel("normalized inlet coordinate")
    for ax in axes:
        ax.set_ylabel("inlet ux")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Pipe inlet ux candidate fits, sample {sample_idx}")
    fig.tight_layout()

    out_path = out_dir / f"pipe_inlet_ux_fit_sample_{sample_idx:04d}.png"
    fig.savefig(out_path)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    if curve_fit is None:
        raise RuntimeError("scipy is required for inlet profile fitting.")
    if args.no_plot and args.show:
        raise ValueError("--show cannot be used together with --no-plot")

    y_all, q_all = load_pipe_arrays(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_summary = min(max(int(args.summary_samples), 0), int(q_all.shape[0]))
    summary_rows: list[dict[str, object]] = []
    for sample_idx in range(n_summary):
        t, _, ux = sample_profile(y_all, q_all, sample_idx)
        fits = fit_inlet_profile(t, ux)
        for model, fit in fits.items():
            params = dict(fit["params"])
            summary_rows.append(
                {
                    "sample": sample_idx,
                    "model": model,
                    "rmse": float(fit["rmse"]),
                    "r2": float(fit["r2"]),
                    "amplitude": params.get("amplitude", ""),
                    "center": params.get("center", ""),
                    "sigma": params.get("sigma", ""),
                    "offset": params.get("offset", ""),
                }
            )

    if summary_rows:
        summarize_fits(summary_rows)
        csv_path = args.out_dir / "pipe_inlet_ux_fit_summary.csv"
        write_summary_csv(csv_path, summary_rows)
        print(f"wrote {csv_path}")

    for sample_idx in args.samples:
        if sample_idx < 0 or sample_idx >= q_all.shape[0]:
            raise IndexError(f"Sample {sample_idx} is outside [0, {q_all.shape[0]})")
        t, y, ux = sample_profile(y_all, q_all, sample_idx)
        fits = fit_inlet_profile(t, ux)
        print_fit(sample_idx, fits)
        if args.no_plot or plt is None:
            continue
        out_path = plot_sample(args.out_dir, sample_idx, t, y, ux, fits, show=args.show)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
