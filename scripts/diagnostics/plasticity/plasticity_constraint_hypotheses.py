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

from _common import load_plasticity_arrays, require_matplotlib, select_split, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test candidate hard-constraint hypotheses for the plasticity forging "
            "dataset over a selected split."
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
        "--tol",
        type=float,
        default=1.0e-7,
        help="Absolute tolerance for zero/sign inequality checks.",
    )
    parser.add_argument(
        "--coord-drift-tol",
        type=float,
        default=1.0e-4,
        help=(
            "Tolerance for coords - displacement temporal drift. This check is "
            "limited by stored dataset precision rather than a physical boundary."
        ),
    )
    parser.add_argument(
        "--area-rel-tol",
        type=float,
        default=1.0e-3,
        help="Relative tolerance for total deformed area preservation.",
    )
    parser.add_argument(
        "--cell-area-tol",
        type=float,
        default=1.0e-12,
        help="Minimum oriented cell area tolerance for no-foldover checks.",
    )
    parser.add_argument(
        "--center-x-tol",
        type=float,
        default=1.0e-3,
        help="Absolute tolerance for center-of-mass x drift from t=0.",
    )
    parser.add_argument(
        "--mean-ux-tol",
        type=float,
        default=1.0e-3,
        help="Absolute tolerance for global mean ux balance.",
    )
    parser.add_argument(
        "--compression-monotone-tol",
        type=float,
        default=1.0e-7,
        help="Tolerance for total vertical compression monotonicity.",
    )
    parser.add_argument(
        "--ux-sign-eps",
        type=float,
        default=1.0e-7,
        help="Deadzone for ux sign persistence. Values within +/- eps are treated as zero.",
    )
    parser.add_argument(
        "--die-speed",
        type=float,
        default=6.0,
        help="Downward die speed used for the raw-coordinate j=0/die envelope check.",
    )
    parser.add_argument(
        "--time-duration",
        type=float,
        default=1.0,
        help=(
            "Physical duration corresponding to one full T_out-step sequence for "
            "the j=0/die envelope check."
        ),
    )
    parser.add_argument(
        "--flip-die-x",
        action="store_true",
        help="Reverse the raw input die profile along x before j=0/die checks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of largest violations to write per hypothesis.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/plasticity/plasticity_constraint_hypotheses"),
    )
    parser.add_argument(
        "--plot-examples",
        action="store_true",
        help="Save visual examples for the largest violation or residual of each check.",
    )
    return parser.parse_args()


def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "max_abs": float("nan"),
        }
    return {
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p95": float(np.percentile(finite, 95)),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "max_abs": float(np.abs(finite).max()),
    }


def violation_summary(
    *,
    name: str,
    values: np.ndarray,
    violation: np.ndarray,
    total_count: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    violation = np.asarray(violation, dtype=np.float64)
    positive = violation > 0.0
    count = int(np.count_nonzero(positive))
    denominator = int(total_count if total_count is not None else violation.size)
    max_violation = float(violation.max()) if violation.size else 0.0
    row: dict[str, object] = {
        "hypothesis": name,
        "status": "pass" if count == 0 else "fail",
        "violation_count": count,
        "checked_count": denominator,
        "violation_fraction": float(count / max(denominator, 1)),
        "max_violation": max_violation,
        **finite_stats(values),
    }
    if extra:
        row.update(extra)
    return row


def top_index_rows(
    *,
    hypothesis: str,
    values: np.ndarray,
    violation: np.ndarray,
    axis_names: tuple[str, ...],
    top_k: int,
    extra: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    values = np.asarray(values, dtype=np.float64)
    violation = np.asarray(violation, dtype=np.float64)
    flat_positive = np.flatnonzero(violation.reshape(-1) > 0.0)
    if flat_positive.size == 0:
        return []
    flat_violation = violation.reshape(-1)
    order = flat_positive[np.argsort(flat_violation[flat_positive])[::-1]]
    rows: list[dict[str, object]] = []
    for flat_idx in order[: max(int(top_k), 0)]:
        index = np.unravel_index(int(flat_idx), violation.shape)
        row: dict[str, object] = {
            "hypothesis": hypothesis,
            "value": float(values[index]),
            "violation": float(violation[index]),
        }
        row.update({name: int(index[pos]) for pos, name in enumerate(axis_names)})
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def material_grid_drift_rows(output: np.ndarray, *, tol: float, top_k: int):
    coords = output[..., 0:2].astype(np.float64)
    disp = output[..., 2:4].astype(np.float64)
    material = coords - disp
    drift = material - material[:, :, :, 0:1, :]
    abs_drift = np.abs(drift)
    violation = np.maximum(abs_drift - tol, 0.0)
    summary = violation_summary(
        name="movement_channels_match_coordinate_change",
        values=abs_drift,
        violation=violation,
        extra={
            "interpretation": (
                "coords - displacement should be time-invariant, supporting "
                "constraints on movement rather than absolute position."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="movement_channels_match_coordinate_change",
        values=abs_drift,
        violation=violation,
        axis_names=("sample", "i", "j", "timestep", "component"),
        top_k=top_k,
    )
    return summary, violations


def lower_edge_uy_zero_rows(output: np.ndarray, *, tol: float, top_k: int):
    values = output[:, :, -1, :, 3].astype(np.float64)
    abs_values = np.abs(values)
    violation = np.maximum(abs_values - tol, 0.0)
    summary = violation_summary(
        name="lower_jN_uy_zero",
        values=values,
        violation=violation,
    )
    violations = top_index_rows(
        hypothesis="lower_jN_uy_zero",
        values=values,
        violation=violation,
        axis_names=("sample", "i", "timestep"),
        top_k=top_k,
    )
    return summary, violations


def uy_non_positive_rows(output: np.ndarray, *, tol: float, top_k: int):
    values = output[..., 3].astype(np.float64)
    violation = np.maximum(values - tol, 0.0)
    summary = violation_summary(
        name="uy_non_positive",
        values=values,
        violation=violation,
    )
    violations = top_index_rows(
        hypothesis="uy_non_positive",
        values=values,
        violation=violation,
        axis_names=("sample", "i", "j", "timestep"),
        top_k=top_k,
    )
    return summary, violations


def final_uy_strictly_negative_rows(output: np.ndarray, *, tol: float, top_k: int):
    final_uy = output[:, :, :, -1, 3].astype(np.float64)
    final_uy_interior = final_uy[:, :, :-1]

    all_violation = np.maximum(final_uy + tol, 0.0)
    all_summary = violation_summary(
        name="final_uy_strictly_negative_all",
        values=final_uy,
        violation=all_violation,
        extra={
            "interpretation": (
                "Checks uy at the final timestep is below -tol everywhere. "
                "This includes the clamped lower edge."
            )
        },
    )
    all_violations = top_index_rows(
        hypothesis="final_uy_strictly_negative_all",
        values=final_uy,
        violation=all_violation,
        axis_names=("sample", "i", "j"),
        top_k=top_k,
    )

    interior_violation = np.maximum(final_uy_interior + tol, 0.0)
    interior_summary = violation_summary(
        name="final_uy_strictly_negative_interior",
        values=final_uy_interior,
        violation=interior_violation,
        extra={
            "interpretation": (
                "Checks final uy < -tol after excluding j=31, which is exactly "
                "zero in the dataset."
            )
        },
    )
    interior_violations = top_index_rows(
        hypothesis="final_uy_strictly_negative_interior",
        values=final_uy_interior,
        violation=interior_violation,
        axis_names=("sample", "i", "j"),
        top_k=top_k,
    )
    return [all_summary, interior_summary], all_violations + interior_violations


def signed_cell_areas(coords: np.ndarray) -> np.ndarray:
    p00 = coords[:, :-1, :-1, :, :]
    p10 = coords[:, 1:, :-1, :, :]
    p11 = coords[:, 1:, 1:, :, :]
    p01 = coords[:, :-1, 1:, :, :]
    x = np.stack([p00[..., 0], p10[..., 0], p11[..., 0], p01[..., 0]], axis=-1)
    y = np.stack([p00[..., 1], p10[..., 1], p11[..., 1], p01[..., 1]], axis=-1)
    return 0.5 * np.sum(
        x * np.roll(y, shift=-1, axis=-1) - y * np.roll(x, shift=-1, axis=-1),
        axis=-1,
    )


def total_area_preservation_rows(
    output: np.ndarray,
    *,
    rel_tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    coords = output[..., 0:2].astype(np.float64)
    area = np.abs(signed_cell_areas(coords)).sum(axis=(1, 2))
    area0 = area[:, 0:1]
    rel_change = (area - area0) / np.maximum(np.abs(area0), 1.0e-12)
    violation = np.maximum(np.abs(rel_change) - rel_tol, 0.0)
    summary = violation_summary(
        name="total_deformed_area_preserved",
        values=rel_change,
        violation=violation,
        extra={
            "area_rel_tol": float(rel_tol),
            "interpretation": (
                "Checks whether total deformed quad area stays constant per sample."
            ),
        },
    )
    violations = top_index_rows(
        hypothesis="total_deformed_area_preserved",
        values=rel_change,
        violation=violation,
        axis_names=("sample", "timestep"),
        top_k=top_k,
        extra={"area_rel_tol": float(rel_tol)},
    )
    return summary, violations


def cell_area_orientation_rows(
    output: np.ndarray,
    *,
    area_tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    coords = output[..., 0:2].astype(np.float64)
    signed_area = signed_cell_areas(coords)
    initial_sign = np.sign(np.median(signed_area[:, :, :, 0], axis=(1, 2)))
    initial_sign[initial_sign == 0.0] = 1.0
    oriented_area = signed_area * initial_sign[:, None, None, None]
    violation = np.maximum(area_tol - oriented_area, 0.0)
    summary = violation_summary(
        name="cell_area_orientation_positive",
        values=oriented_area,
        violation=violation,
        extra={
            "cell_area_tol": float(area_tol),
            "interpretation": (
                "Checks every deformed quad keeps the initial orientation and "
                "has area above tolerance."
            ),
        },
    )
    violations = top_index_rows(
        hypothesis="cell_area_orientation_positive",
        values=oriented_area,
        violation=violation,
        axis_names=("sample", "cell_i", "cell_j", "timestep"),
        top_k=top_k,
        extra={"cell_area_tol": float(area_tol)},
    )
    return summary, violations


def center_x_drift_rows(
    output: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    center_x = output[..., 0].astype(np.float64).mean(axis=(1, 2))
    drift = center_x - center_x[:, 0:1]
    violation = np.maximum(np.abs(drift) - tol, 0.0)
    summary = violation_summary(
        name="center_x_preserved",
        values=drift,
        violation=violation,
        extra={
            "center_x_tol": float(tol),
            "interpretation": "Checks global mean x-coordinate drift from t=0.",
        },
    )
    violations = top_index_rows(
        hypothesis="center_x_preserved",
        values=drift,
        violation=violation,
        axis_names=("sample", "timestep"),
        top_k=top_k,
        extra={"center_x_tol": float(tol)},
    )
    return summary, violations


def mean_ux_balance_rows(
    output: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    mean_ux = output[..., 2].astype(np.float64).mean(axis=(1, 2))
    violation = np.maximum(np.abs(mean_ux) - tol, 0.0)
    summary = violation_summary(
        name="global_mean_ux_zero",
        values=mean_ux,
        violation=violation,
        extra={
            "mean_ux_tol": float(tol),
            "interpretation": "Checks whether horizontal displacement balances globally.",
        },
    )
    violations = top_index_rows(
        hypothesis="global_mean_ux_zero",
        values=mean_ux,
        violation=violation,
        axis_names=("sample", "timestep"),
        top_k=top_k,
        extra={"mean_ux_tol": float(tol)},
    )
    return summary, violations


def vertical_compression_monotone_rows(
    output: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y = output[..., 1].astype(np.float64)
    upper_y = y[:, :, 0, :].mean(axis=1)
    lower_y = y[:, :, -1, :].mean(axis=1)
    height = lower_y - upper_y
    initial_height = height[:, 0:1]
    compression = initial_height - height
    delta_compression = np.diff(compression, axis=1)
    compression_violation = np.maximum(-delta_compression - tol, 0.0)
    compression_summary = violation_summary(
        name="total_vertical_compression_non_decreasing",
        values=delta_compression,
        violation=compression_violation,
        extra={
            "compression_monotone_tol": float(tol),
            "final_mean_compression": float(np.mean(compression[:, -1])),
            "final_min_compression": float(np.min(compression[:, -1])),
            "final_max_compression": float(np.max(compression[:, -1])),
            "interpretation": (
                "Checks total vertical compression based on mean lower-upper "
                "boundary height is monotone over time."
            ),
        },
    )
    compression_violations = top_index_rows(
        hypothesis="total_vertical_compression_non_decreasing",
        values=delta_compression,
        violation=compression_violation,
        axis_names=("sample", "timestep_delta"),
        top_k=top_k,
        extra={"compression_monotone_tol": float(tol)},
    )

    delta_height = np.diff(height, axis=1)
    height_violation = np.maximum(-delta_height - tol, 0.0)
    height_summary = violation_summary(
        name="signed_vertical_span_non_decreasing",
        values=delta_height,
        violation=height_violation,
        extra={
            "compression_monotone_tol": float(tol),
            "final_mean_span": float(np.mean(height[:, -1])),
            "final_min_span": float(np.min(height[:, -1])),
            "final_max_span": float(np.max(height[:, -1])),
            "interpretation": (
                "Checks signed mean lower-upper boundary span is monotone "
                "non-decreasing. This is the opposite sign of the compression "
                "diagnostic under the dataset y convention."
            ),
        },
    )
    height_violations = top_index_rows(
        hypothesis="signed_vertical_span_non_decreasing",
        values=delta_height,
        violation=height_violation,
        axis_names=("sample", "timestep_delta"),
        top_k=top_k,
        extra={"compression_monotone_tol": float(tol)},
    )
    return (
        [compression_summary, height_summary],
        compression_violations + height_violations,
    )


def uy_temporal_monotonic_rows(output: np.ndarray, *, tol: float, top_k: int):
    uy = output[..., 3].astype(np.float64)
    delta = np.diff(uy, axis=3)

    non_increasing_violation = np.maximum(delta - tol, 0.0)
    non_increasing_summary = violation_summary(
        name="uy_temporal_non_increasing",
        values=delta,
        violation=non_increasing_violation,
        total_count=delta.size,
        extra={
            "interpretation": (
                "For downward-negative displacement this is the usual monotone "
                "motion check: uy[t+1] <= uy[t]."
            )
        },
    )
    non_increasing_violations = top_index_rows(
        hypothesis="uy_temporal_non_increasing",
        values=delta,
        violation=non_increasing_violation,
        axis_names=("sample", "i", "j", "timestep_delta"),
        top_k=top_k,
    )

    non_decreasing_violation = np.maximum(-delta - tol, 0.0)
    non_decreasing_summary = violation_summary(
        name="uy_temporal_non_decreasing",
        values=delta,
        violation=non_decreasing_violation,
        total_count=delta.size,
        extra={
            "interpretation": (
                "Included because the original hypothesis said non-decreasing; "
                "this checks uy[t+1] >= uy[t]."
            )
        },
    )
    non_decreasing_violations = top_index_rows(
        hypothesis="uy_temporal_non_decreasing",
        values=delta,
        violation=non_decreasing_violation,
        axis_names=("sample", "i", "j", "timestep_delta"),
        top_k=top_k,
    )
    return (
        [non_increasing_summary, non_decreasing_summary],
        non_increasing_violations + non_decreasing_violations,
    )


def ux_sign_persistence_rows(
    output: np.ndarray,
    *,
    eps: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ux = output[..., 2].astype(np.float64)
    signs = np.zeros_like(ux, dtype=np.int8)
    signs[ux > eps] = 1
    signs[ux < -eps] = -1

    nonzero_count = np.count_nonzero(signs, axis=3)
    sign_sum = np.sum(signs, axis=3)
    has_positive = np.any(signs > 0, axis=3)
    has_negative = np.any(signs < 0, axis=3)
    sign_flip = has_positive & has_negative
    violation = sign_flip.astype(np.float64)
    values = sign_sum.astype(np.float64)
    summary = violation_summary(
        name="ux_sign_persistence",
        values=values,
        violation=violation,
        total_count=violation.size,
        extra={
            "ux_sign_eps": float(eps),
            "active_point_count": int(np.count_nonzero(nonzero_count)),
            "inactive_point_count": int(np.count_nonzero(nonzero_count == 0)),
        },
    )

    rows: list[dict[str, object]] = []
    flat = np.flatnonzero(sign_flip.reshape(-1))
    if flat.size:
        severity = np.minimum(
            np.max(ux, axis=3).reshape(-1),
            -np.min(ux, axis=3).reshape(-1),
        )
        order = flat[np.argsort(severity[flat])[::-1]]
        for flat_idx in order[: max(int(top_k), 0)]:
            sample, i, j = np.unravel_index(int(flat_idx), sign_flip.shape)
            series = ux[sample, i, j, :]
            row = {
                "hypothesis": "ux_sign_persistence",
                "sample": int(sample),
                "i": int(i),
                "j": int(j),
                "violation": float(severity[flat_idx]),
                "min_ux": float(np.min(series)),
                "max_ux": float(np.max(series)),
                "first_nonzero_sign": int(
                    signs[sample, i, j, np.flatnonzero(signs[sample, i, j, :])[0]]
                ),
                "sign_sequence": " ".join(str(int(v)) for v in signs[sample, i, j, :]),
            }
            rows.append(row)
    return summary, rows


def _corrcoef_1d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def die_j0_rows(
    die: np.ndarray,
    output: np.ndarray,
    *,
    die_speed: float,
    time_duration: float,
    tol: float,
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    n_samples, x_count, _, t_count, _ = output.shape
    time = np.arange(t_count, dtype=np.float64) * float(time_duration) / max(t_count, 1)
    material = output[..., 0:2] - output[..., 2:4]
    j0 = output[:, :, 0, :, 0:2].astype(np.float64)

    signed_distance = np.empty((n_samples, x_count, t_count), dtype=np.float64)
    corr = np.empty((n_samples, t_count), dtype=np.float64)
    slope = np.empty((n_samples, t_count), dtype=np.float64)
    fit_rmse = np.empty((n_samples, t_count), dtype=np.float64)

    for sample_idx in range(n_samples):
        die_profile = np.asarray(die[sample_idx], dtype=np.float64)
        x_ref = np.linspace(
            float(np.nanmin(material[sample_idx, :, 0, 0, 0])),
            float(np.nanmax(material[sample_idx, :, 0, 0, 0])),
            x_count,
        )
        die_y = die_profile[:, None] - float(die_speed) * time[None, :]
        for timestep in range(t_count):
            x_t = j0[sample_idx, :, timestep, 0]
            y_t = j0[sample_idx, :, timestep, 1]
            order = np.argsort(x_t)
            j0_y = np.interp(x_ref, x_t[order], y_t[order])
            signed_distance[sample_idx, :, timestep] = die_y[:, timestep] - j0_y
            corr[sample_idx, timestep] = _corrcoef_1d(die_profile, j0_y)

            design = np.stack([np.ones_like(die_profile), die_profile], axis=1)
            coeff, *_ = np.linalg.lstsq(design, j0_y, rcond=None)
            fitted = coeff[0] + coeff[1] * die_profile
            slope[sample_idx, timestep] = float(coeff[1])
            fit_rmse[sample_idx, timestep] = float(
                np.sqrt(np.mean((j0_y - fitted) ** 2))
            )

    envelope_violation = np.maximum(-signed_distance - tol, 0.0)
    envelope_summary = violation_summary(
        name="j0_below_raw_moving_die",
        values=signed_distance,
        violation=envelope_violation,
        extra={
            "die_speed": float(die_speed),
            "time_duration": float(time_duration),
            "interpretation": (
                "Checks die_y - j0_y >= 0 under the raw moving-die convention."
            ),
        },
    )
    envelope_violations = top_index_rows(
        hypothesis="j0_below_raw_moving_die",
        values=signed_distance,
        violation=envelope_violation,
        axis_names=("sample", "i", "timestep"),
        top_k=top_k,
        extra={"die_speed": float(die_speed), "time_duration": float(time_duration)},
    )

    corr_summary = {
        "hypothesis": "j0_shape_matches_raw_die_profile",
        "status": "informational",
        "violation_count": "",
        "checked_count": int(corr.size),
        "violation_fraction": "",
        "max_violation": "",
        **finite_stats(corr),
        "final_mean_corr": float(np.nanmean(corr[:, -1])),
        "final_min_corr": float(np.nanmin(corr[:, -1])),
        "final_mean_linear_slope": float(np.nanmean(slope[:, -1])),
        "final_mean_fit_rmse": float(np.nanmean(fit_rmse[:, -1])),
        "interpretation": (
            "Correlation and one-parameter linear-fit residual between raw die "
            "profile and j=0 y; use this to decide whether the shape relation "
            "is strong enough for a hard constraint."
        ),
    }
    fit_summary = {
        "hypothesis": "j0_shape_linear_fit_rmse",
        "status": "informational",
        "violation_count": "",
        "checked_count": int(fit_rmse.size),
        "violation_fraction": "",
        "max_violation": "",
        **finite_stats(fit_rmse),
        "final_mean_fit_rmse": float(np.nanmean(fit_rmse[:, -1])),
        "final_p95_fit_rmse": float(np.nanpercentile(fit_rmse[:, -1], 95)),
        "interpretation": (
            "Lower RMSE means j=0 is closer to an affine transform of the die profile."
        ),
    }

    return [envelope_summary, corr_summary, fit_summary], envelope_violations


def _save_field_series_plot(
    *,
    field: np.ndarray,
    series: np.ndarray,
    sample: int,
    i: int,
    j: int,
    timestep: int,
    title: str,
    field_label: str,
    series_label: str,
    out_path: Path,
    threshold: float | None = None,
    delta_timestep: int | None = None,
) -> Path:
    require_matplotlib(plt)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), dpi=150)

    ax = axes[0]
    image = ax.imshow(field.T, origin="upper", aspect="auto", cmap="coolwarm")
    ax.scatter([i], [j], marker="x", s=80, linewidths=2.0, color="#111827")
    ax.set_title(f"sample={sample}, timestep={timestep}")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(field_label)

    ax = axes[1]
    time = np.arange(series.shape[0])
    ax.plot(time, series, color="#2563eb", marker="o", linewidth=1.6, markersize=3.5)
    ax.scatter([timestep], [series[timestep]], color="#dc2626", zorder=3)
    if delta_timestep is not None:
        ax.axvspan(delta_timestep, delta_timestep + 1, color="#f97316", alpha=0.18)
    if threshold is not None:
        ax.axhline(threshold, color="#111827", linestyle="--", linewidth=1.0)
    ax.set_title(f"point i={i}, j={j}")
    ax.set_xlabel("timestep")
    ax.set_ylabel(series_label)
    ax.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_line_plot(
    *,
    x: np.ndarray,
    lines: list[tuple[np.ndarray, str, str]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    marker_x: float | None = None,
    marker_y: float | None = None,
) -> Path:
    require_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=150)
    for y, label, color in lines:
        ax.plot(x, y, label=label, color=color, linewidth=1.8)
    if marker_x is not None and marker_y is not None:
        ax.scatter([marker_x], [marker_y], marker="x", s=90, color="#dc2626", zorder=3)
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_die_j0_plot(
    *,
    x: np.ndarray,
    die_y: np.ndarray,
    j0_y: np.ndarray,
    signed_distance: np.ndarray,
    sample: int,
    i: int,
    timestep: int,
    violation: float,
    out_path: Path,
) -> Path:
    require_matplotlib(plt)
    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.2), dpi=150, sharex=True)

    ax = axes[0]
    ax.plot(x, die_y, label="moving raw die", color="#111827", linewidth=1.8)
    ax.plot(x, j0_y, label="j=0 boundary", color="#dc2626", linewidth=1.8)
    ax.scatter([x[i]], [j0_y[i]], marker="x", s=90, color="#dc2626", zorder=3)
    ax.set_title(f"sample={sample}, timestep={timestep}")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(x, signed_distance, color="#2563eb", linewidth=1.8)
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax.scatter(
        [x[i]],
        [signed_distance[i]],
        marker="x",
        s=90,
        color="#dc2626",
        zorder=3,
    )
    ax.set_xlabel("material x")
    ax.set_ylabel("die_y - j0_y")
    ax.grid(True, alpha=0.25)

    fig.suptitle(
        "j0_below_raw_moving_die: largest negative clearance "
        f"(violation={violation:.3e})"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_time_series_plot(
    *,
    series: np.ndarray,
    sample: int,
    title: str,
    ylabel: str,
    out_path: Path,
    threshold: float | None = None,
    marker_timestep: int | None = None,
) -> Path:
    require_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    time = np.arange(series.shape[0])
    ax.plot(time, series, color="#2563eb", marker="o", linewidth=1.8, markersize=3.8)
    if marker_timestep is not None:
        ax.scatter(
            [marker_timestep],
            [series[marker_timestep]],
            color="#dc2626",
            zorder=3,
        )
    if threshold is not None:
        ax.axhline(threshold, color="#111827", linestyle="--", linewidth=1.0)
        ax.axhline(-threshold, color="#111827", linestyle="--", linewidth=1.0)
    ax.axhline(0.0, color="#6b7280", linewidth=0.9, alpha=0.6)
    ax.set_title(f"{title} | sample={sample}")
    ax.set_xlabel("timestep")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _material_x_for_sample(output: np.ndarray, sample: int) -> np.ndarray:
    material = output[sample, :, 0, 0, 0] - output[sample, :, 0, 0, 2]
    return np.asarray(material, dtype=np.float64)


def save_example_plots(
    *,
    die: np.ndarray,
    output: np.ndarray,
    out_dir: Path,
    tol: float,
    coord_drift_tol: float,
    area_rel_tol: float,
    cell_area_tol: float,
    center_x_tol: float,
    mean_ux_tol: float,
    compression_monotone_tol: float,
    ux_sign_eps: float,
    die_speed: float,
    time_duration: float,
) -> list[Path]:
    require_matplotlib(plt)
    example_dir = out_dir / "examples"
    paths: list[Path] = []

    coords = output[..., 0:2].astype(np.float64)
    disp = output[..., 2:4].astype(np.float64)
    material = coords - disp
    drift = material - material[:, :, :, 0:1, :]
    abs_drift = np.abs(drift)
    sample, i, j, timestep, component = np.unravel_index(
        int(np.argmax(abs_drift)),
        abs_drift.shape,
    )
    component_name = ("x", "y")[component]
    paths.append(
        _save_field_series_plot(
            field=abs_drift[sample, :, :, timestep, component],
            series=abs_drift[sample, i, j, :, component],
            sample=sample,
            i=i,
            j=j,
            timestep=timestep,
            title=(
                "movement_channels_match_coordinate_change: largest residual "
                f"({component_name}, max={abs_drift[sample, i, j, timestep, component]:.3e})"
            ),
            field_label=f"|material {component_name} drift|",
            series_label=f"|material {component_name} drift|",
            out_path=example_dir / "movement_channels_match_coordinate_change.png",
            threshold=coord_drift_tol,
        )
    )

    lower = output[:, :, -1, :, 3].astype(np.float64)
    sample, i, timestep = np.unravel_index(int(np.argmax(np.abs(lower))), lower.shape)
    x = _material_x_for_sample(output, sample)
    paths.append(
        _save_line_plot(
            x=x,
            lines=[(lower[sample, :, timestep], "lower edge uy", "#2563eb")],
            title=(
                "lower_jN_uy_zero: largest residual "
                f"(sample={sample}, timestep={timestep}, max_abs={abs(lower[sample, i, timestep]):.3e})"
            ),
            xlabel="material x",
            ylabel="uy at j=31",
            out_path=example_dir / "lower_jN_uy_zero.png",
            marker_x=float(x[i]),
            marker_y=float(lower[sample, i, timestep]),
        )
    )

    uy = output[..., 3].astype(np.float64)
    sample, i, j, timestep = np.unravel_index(int(np.argmax(uy)), uy.shape)
    paths.append(
        _save_field_series_plot(
            field=uy[sample, :, :, timestep],
            series=uy[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=timestep,
            title=(
                "uy_non_positive: largest positive uy "
                f"(value={uy[sample, i, j, timestep]:.3e}, tol={tol:.1e})"
            ),
            field_label="uy",
            series_label="uy",
            out_path=example_dir / "uy_non_positive.png",
            threshold=tol,
        )
    )

    final_uy = uy[:, :, :, -1]
    sample, i, j = np.unravel_index(int(np.argmax(final_uy)), final_uy.shape)
    paths.append(
        _save_field_series_plot(
            field=final_uy[sample],
            series=uy[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=uy.shape[3] - 1,
            title=(
                "final_uy_strictly_negative_all: largest final uy "
                f"(value={final_uy[sample, i, j]:.3e}, tol={tol:.1e})"
            ),
            field_label="final uy",
            series_label="uy",
            out_path=example_dir / "final_uy_strictly_negative_all.png",
            threshold=-tol,
        )
    )

    final_uy_interior = final_uy[:, :, :-1]
    sample, i, j = np.unravel_index(
        int(np.argmax(final_uy_interior)),
        final_uy_interior.shape,
    )
    paths.append(
        _save_field_series_plot(
            field=final_uy[sample],
            series=uy[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=uy.shape[3] - 1,
            title=(
                "final_uy_strictly_negative_interior: largest interior final uy "
                f"(value={final_uy[sample, i, j]:.3e}, tol={tol:.1e})"
            ),
            field_label="final uy",
            series_label="uy",
            out_path=example_dir / "final_uy_strictly_negative_interior.png",
            threshold=-tol,
        )
    )

    delta_uy = np.diff(uy, axis=3)
    sample, i, j, timestep_delta = np.unravel_index(
        int(np.argmax(delta_uy)),
        delta_uy.shape,
    )
    paths.append(
        _save_field_series_plot(
            field=delta_uy[sample, :, :, timestep_delta],
            series=uy[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=timestep_delta + 1,
            title=(
                "uy_temporal_non_increasing: largest upward delta "
                f"(delta={delta_uy[sample, i, j, timestep_delta]:.3e})"
            ),
            field_label="uy[t+1] - uy[t]",
            series_label="uy",
            out_path=example_dir / "uy_temporal_non_increasing.png",
            delta_timestep=timestep_delta,
        )
    )

    sample, i, j, timestep_delta = np.unravel_index(
        int(np.argmax(-delta_uy)),
        delta_uy.shape,
    )
    paths.append(
        _save_field_series_plot(
            field=delta_uy[sample, :, :, timestep_delta],
            series=uy[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=timestep_delta + 1,
            title=(
                "uy_temporal_non_decreasing: largest downward delta "
                f"(delta={delta_uy[sample, i, j, timestep_delta]:.3e})"
            ),
            field_label="uy[t+1] - uy[t]",
            series_label="uy",
            out_path=example_dir / "uy_temporal_non_decreasing.png",
            delta_timestep=timestep_delta,
        )
    )

    ux = output[..., 2].astype(np.float64)
    signs = np.zeros_like(ux, dtype=np.int8)
    signs[ux > ux_sign_eps] = 1
    signs[ux < -ux_sign_eps] = -1
    sign_flip = np.any(signs > 0, axis=3) & np.any(signs < 0, axis=3)
    severity = np.minimum(np.max(ux, axis=3), -np.min(ux, axis=3))
    if np.any(sign_flip):
        sample, i, j = np.unravel_index(
            int(np.argmax(np.where(sign_flip, severity, -np.inf))),
            sign_flip.shape,
        )
    else:
        sample, i, j = np.unravel_index(int(np.argmax(severity)), severity.shape)
    max_abs_timestep = int(np.argmax(np.abs(ux[sample, i, j, :])))
    paths.append(
        _save_field_series_plot(
            field=ux[sample, :, :, max_abs_timestep],
            series=ux[sample, i, j, :],
            sample=sample,
            i=i,
            j=j,
            timestep=max_abs_timestep,
            title=(
                "ux_sign_persistence: strongest sign flip "
                f"(min={np.min(ux[sample, i, j, :]):.3e}, "
                f"max={np.max(ux[sample, i, j, :]):.3e})"
            ),
            field_label="ux",
            series_label="ux",
            out_path=example_dir / "ux_sign_persistence.png",
            threshold=0.0,
        )
    )

    n_samples, x_count, _, t_count, _ = output.shape
    time = np.arange(t_count, dtype=np.float64) * float(time_duration) / max(t_count, 1)
    worst: tuple[float, int, int, int, np.ndarray, np.ndarray, np.ndarray] | None = None
    for sample_idx in range(n_samples):
        die_profile = np.asarray(die[sample_idx], dtype=np.float64)
        x_ref = np.linspace(
            float(np.nanmin(material[sample_idx, :, 0, 0, 0])),
            float(np.nanmax(material[sample_idx, :, 0, 0, 0])),
            x_count,
        )
        die_y = die_profile[:, None] - float(die_speed) * time[None, :]
        for timestep in range(t_count):
            x_t = output[sample_idx, :, 0, timestep, 0]
            y_t = output[sample_idx, :, 0, timestep, 1]
            order = np.argsort(x_t)
            j0_y = np.interp(x_ref, x_t[order], y_t[order])
            signed_distance = die_y[:, timestep] - j0_y
            i_min = int(np.argmin(signed_distance))
            violation = max(float(-signed_distance[i_min] - tol), 0.0)
            if worst is None or violation > worst[0]:
                worst = (
                    violation,
                    sample_idx,
                    i_min,
                    timestep,
                    x_ref.copy(),
                    die_y[:, timestep].copy(),
                    j0_y.copy(),
                )
    if worst is not None:
        violation, sample, i, timestep, x_ref, die_y_t, j0_y = worst
        signed_distance = die_y_t - j0_y
        paths.append(
            _save_die_j0_plot(
                x=x_ref,
                die_y=die_y_t,
                j0_y=j0_y,
                signed_distance=signed_distance,
                sample=sample,
                i=i,
                timestep=timestep,
                violation=violation,
                out_path=example_dir / "j0_below_raw_moving_die.png",
            )
        )

    signed_area = signed_cell_areas(coords)
    total_area = np.abs(signed_area).sum(axis=(1, 2))
    rel_area_change = (total_area - total_area[:, 0:1]) / np.maximum(
        np.abs(total_area[:, 0:1]),
        1.0e-12,
    )
    sample, timestep = np.unravel_index(
        int(np.argmax(np.abs(rel_area_change))),
        rel_area_change.shape,
    )
    paths.append(
        _save_time_series_plot(
            series=rel_area_change[sample],
            sample=sample,
            title=(
                "total_deformed_area_preserved: largest relative area drift "
                f"(max_abs={np.max(np.abs(rel_area_change[sample])):.3e})"
            ),
            ylabel="relative total area change",
            out_path=example_dir / "total_deformed_area_preserved.png",
            threshold=area_rel_tol,
            marker_timestep=timestep,
        )
    )

    initial_sign = np.sign(np.median(signed_area[:, :, :, 0], axis=(1, 2)))
    initial_sign[initial_sign == 0.0] = 1.0
    oriented_area = signed_area * initial_sign[:, None, None, None]
    sample, cell_i, cell_j, timestep = np.unravel_index(
        int(np.argmin(oriented_area)),
        oriented_area.shape,
    )
    paths.append(
        _save_field_series_plot(
            field=oriented_area[sample, :, :, timestep],
            series=oriented_area[sample, cell_i, cell_j, :],
            sample=sample,
            i=cell_i,
            j=cell_j,
            timestep=timestep,
            title=(
                "cell_area_orientation_positive: smallest oriented cell area "
                f"(value={oriented_area[sample, cell_i, cell_j, timestep]:.3e})"
            ),
            field_label="oriented cell area",
            series_label="oriented cell area",
            out_path=example_dir / "cell_area_orientation_positive.png",
            threshold=cell_area_tol,
        )
    )

    center_x = coords[..., 0].mean(axis=(1, 2))
    center_x_drift = center_x - center_x[:, 0:1]
    sample, timestep = np.unravel_index(
        int(np.argmax(np.abs(center_x_drift))),
        center_x_drift.shape,
    )
    paths.append(
        _save_time_series_plot(
            series=center_x_drift[sample],
            sample=sample,
            title=(
                "center_x_preserved: largest center-x drift "
                f"(max_abs={np.max(np.abs(center_x_drift[sample])):.3e})"
            ),
            ylabel="mean x drift from t=0",
            out_path=example_dir / "center_x_preserved.png",
            threshold=center_x_tol,
            marker_timestep=timestep,
        )
    )

    mean_ux = output[..., 2].astype(np.float64).mean(axis=(1, 2))
    sample, timestep = np.unravel_index(int(np.argmax(np.abs(mean_ux))), mean_ux.shape)
    paths.append(
        _save_time_series_plot(
            series=mean_ux[sample],
            sample=sample,
            title=(
                "global_mean_ux_zero: largest absolute mean ux "
                f"(max_abs={np.max(np.abs(mean_ux[sample])):.3e})"
            ),
            ylabel="mean ux",
            out_path=example_dir / "global_mean_ux_zero.png",
            threshold=mean_ux_tol,
            marker_timestep=timestep,
        )
    )

    upper_y = coords[:, :, 0, :, 1].mean(axis=1)
    lower_y = coords[:, :, -1, :, 1].mean(axis=1)
    height = lower_y - upper_y
    compression = height[:, 0:1] - height
    delta_compression = np.diff(compression, axis=1)
    sample, timestep_delta = np.unravel_index(
        int(np.argmin(delta_compression)),
        delta_compression.shape,
    )
    paths.append(
        _save_time_series_plot(
            series=compression[sample],
            sample=sample,
            title=(
                "total_vertical_compression_non_decreasing: largest negative delta "
                f"(delta={delta_compression[sample, timestep_delta]:.3e})"
            ),
            ylabel="total vertical compression",
            out_path=example_dir / "total_vertical_compression_non_decreasing.png",
            threshold=compression_monotone_tol,
            marker_timestep=timestep_delta + 1,
        )
    )
    delta_height = np.diff(height, axis=1)
    sample, timestep_delta = np.unravel_index(
        int(np.argmin(delta_height)),
        delta_height.shape,
    )
    paths.append(
        _save_time_series_plot(
            series=height[sample],
            sample=sample,
            title=(
                "signed_vertical_span_non_decreasing: largest negative delta "
                f"(delta={delta_height[sample, timestep_delta]:.3e})"
            ),
            ylabel="signed mean lower-upper vertical span",
            out_path=example_dir / "signed_vertical_span_non_decreasing.png",
            threshold=compression_monotone_tol,
            marker_timestep=timestep_delta + 1,
        )
    )

    return paths


def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nPlasticity constraint hypothesis summary")
    for row in rows:
        status = str(row["status"])
        name = str(row["hypothesis"])
        if status == "informational":
            print(
                f"  {name:<38} info  "
                f"min={row['min']: .6e} mean={row['mean']: .6e} max={row['max']: .6e}"
            )
        else:
            print(
                f"  {name:<38} {status:<4} "
                f"viol={row['violation_count']}/{row['checked_count']} "
                f"max_viol={row['max_violation']: .6e} "
                f"min={row['min']: .6e} max={row['max']: .6e}"
            )


def main() -> None:
    args = parse_args()
    if args.tol < 0.0:
        raise ValueError("--tol must be non-negative")
    if args.coord_drift_tol < 0.0:
        raise ValueError("--coord-drift-tol must be non-negative")
    if args.area_rel_tol < 0.0:
        raise ValueError("--area-rel-tol must be non-negative")
    if args.cell_area_tol < 0.0:
        raise ValueError("--cell-area-tol must be non-negative")
    if args.center_x_tol < 0.0:
        raise ValueError("--center-x-tol must be non-negative")
    if args.mean_ux_tol < 0.0:
        raise ValueError("--mean-ux-tol must be non-negative")
    if args.compression_monotone_tol < 0.0:
        raise ValueError("--compression-monotone-tol must be non-negative")
    if args.ux_sign_eps < 0.0:
        raise ValueError("--ux-sign-eps must be non-negative")
    if args.time_duration <= 0.0:
        raise ValueError("--time-duration must be positive")

    die, output, mat_path = load_plasticity_arrays(args.data_dir)
    die, output = select_split(
        die,
        output,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise ValueError("--max-samples must be positive")
        die = die[: args.max_samples]
        output = output[: args.max_samples]
    if args.flip_die_x:
        die = die[:, ::-1]

    summaries: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []

    summary, rows = material_grid_drift_rows(
        output,
        tol=float(args.coord_drift_tol),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    summary, rows = lower_edge_uy_zero_rows(output, tol=float(args.tol), top_k=int(args.top_k))
    summaries.append(summary)
    violations.extend(rows)

    summary, rows = uy_non_positive_rows(output, tol=float(args.tol), top_k=int(args.top_k))
    summaries.append(summary)
    violations.extend(rows)

    final_uy_summaries, rows = final_uy_strictly_negative_rows(
        output,
        tol=float(args.tol),
        top_k=int(args.top_k),
    )
    summaries.extend(final_uy_summaries)
    violations.extend(rows)

    uy_summaries, rows = uy_temporal_monotonic_rows(
        output,
        tol=float(args.tol),
        top_k=int(args.top_k),
    )
    summaries.extend(uy_summaries)
    violations.extend(rows)

    summary, rows = ux_sign_persistence_rows(
        output,
        eps=float(args.ux_sign_eps),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    die_summaries, rows = die_j0_rows(
        die,
        output,
        die_speed=float(args.die_speed),
        time_duration=float(args.time_duration),
        tol=float(args.tol),
        top_k=int(args.top_k),
    )
    summaries.extend(die_summaries)
    violations.extend(rows)

    summary, rows = total_area_preservation_rows(
        output,
        rel_tol=float(args.area_rel_tol),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    summary, rows = cell_area_orientation_rows(
        output,
        area_tol=float(args.cell_area_tol),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    summary, rows = center_x_drift_rows(
        output,
        tol=float(args.center_x_tol),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    summary, rows = mean_ux_balance_rows(
        output,
        tol=float(args.mean_ux_tol),
        top_k=int(args.top_k),
    )
    summaries.append(summary)
    violations.extend(rows)

    vertical_summaries, rows = vertical_compression_monotone_rows(
        output,
        tol=float(args.compression_monotone_tol),
        top_k=int(args.top_k),
    )
    summaries.extend(vertical_summaries)
    violations.extend(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "plasticity_constraint_hypotheses_summary.csv"
    write_csv(summary_path, summaries)
    if violations:
        violation_path = args.out_dir / "plasticity_constraint_hypotheses_violations.csv"
        write_csv(violation_path, violations)
    else:
        violation_path = None

    print(f"Loaded: {mat_path}")
    print(f"Split: {args.split}; samples checked: {output.shape[0]}")
    print(f"Output shape: {output.shape}")
    print_summary(summaries)
    print(f"\nWrote summary CSV: {summary_path}")
    if violation_path is not None:
        print(f"Wrote top violations CSV: {violation_path}")
    else:
        print("No violations above tolerance; no violations CSV written.")
    if args.plot_examples:
        paths = save_example_plots(
            die=die,
            output=output,
            out_dir=args.out_dir,
            tol=float(args.tol),
            coord_drift_tol=float(args.coord_drift_tol),
            area_rel_tol=float(args.area_rel_tol),
            cell_area_tol=float(args.cell_area_tol),
            center_x_tol=float(args.center_x_tol),
            mean_ux_tol=float(args.mean_ux_tol),
            compression_monotone_tol=float(args.compression_monotone_tol),
            ux_sign_eps=float(args.ux_sign_eps),
            die_speed=float(args.die_speed),
            time_duration=float(args.time_duration),
        )
        print("\nWrote example plots:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
