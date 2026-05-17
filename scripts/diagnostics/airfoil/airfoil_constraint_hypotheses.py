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
    GAMMA,
    H_INF,
    U_INF,
    load_naca_arrays,
    load_naca_cylinder_arrays,
    load_naca_interp_arrays,
    require_matplotlib,
    select_split,
    wall_normal_velocity,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test candidate hard-constraint hypotheses for the NACA airfoil dataset "
            "over a selected split of the naca/ data."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/airfoil"),
        help="Root directory containing naca/ and naca_interp/ subdirectories.",
    )
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--ntrain", type=int, default=2000)
    parser.add_argument("--ntest", type=int, default=490)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on samples after split selection.",
    )
    parser.add_argument(
        "--far-field-radius",
        type=float,
        default=20.0,
        help="Radius threshold beyond which points are considered far-field.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for exact-zero / positivity checks.",
    )
    parser.add_argument(
        "--far-field-tol",
        type=float,
        default=0.05,
        help="Absolute tolerance for far-field Dirichlet conditions.",
    )
    parser.add_argument(
        "--entropy-tol",
        type=float,
        default=0.05,
        help="Absolute tolerance on |p/rho^gamma - 1| for the isentropic check.",
    )
    parser.add_argument(
        "--enthalpy-tol",
        type=float,
        default=0.1,
        help="Absolute tolerance on |H - H_inf| for total-enthalpy uniformity.",
    )
    parser.add_argument(
        "--wall-normal-tol",
        type=float,
        default=0.05,
        help="Absolute tolerance for the wall no-penetration check.",
    )
    parser.add_argument(
        "--skip-interp",
        action="store_true",
        help="Skip the naca_interp/ checks.",
    )
    parser.add_argument(
        "--skip-cylinder",
        action="store_true",
        help="Skip the cylinder-grid wall checks (these load the ~4 GB cylinder arrays).",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/airfoil/constraint_hypotheses"),
    )
    parser.add_argument(
        "--plot-examples",
        action="store_true",
        help="Save visual examples for the largest violation of each check.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# shared stats helpers
# ---------------------------------------------------------------------------

def finite_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {k: float("nan") for k in ("mean", "median", "p95", "min", "max", "max_abs")}
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


# ---------------------------------------------------------------------------
# hypothesis checks — unstructured naca/ data
# ---------------------------------------------------------------------------

def positive_density_rows(
    Q: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rho = Q[:, 0, :].astype(np.float64)
    violation = np.maximum(tol - rho, 0.0)
    summary = violation_summary(
        name="positive_density",
        values=rho,
        violation=violation,
        extra={
            "interpretation": (
                "rho > 0 everywhere. A softplus or exp reparameterisation of "
                "the density output channel enforces this as a hard constraint."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="positive_density",
        values=rho,
        violation=violation,
        axis_names=("sample", "point"),
        top_k=top_k,
    )
    return summary, violations


def positive_pressure_rows(
    Q: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    p = Q[:, 3, :].astype(np.float64)
    violation = np.maximum(tol - p, 0.0)
    summary = violation_summary(
        name="positive_pressure",
        values=p,
        violation=violation,
        extra={
            "interpretation": (
                "p > 0 everywhere. Same reparameterisation applies as for density."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="positive_pressure",
        values=p,
        violation=violation,
        axis_names=("sample", "point"),
        top_k=top_k,
    )
    return summary, violations


def far_field_rows(
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    *,
    radius: float,
    tol: float,
    top_k: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    r = np.sqrt(X**2 + Y**2)  # (N, P)
    far = r > radius

    rho = Q[:, 0, :].astype(np.float64)
    u   = Q[:, 1, :].astype(np.float64)
    v   = Q[:, 2, :].astype(np.float64)
    p   = Q[:, 3, :].astype(np.float64)

    summaries: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []

    for name, field, target, desc in [
        (
            "far_field_density_unity",
            rho,
            1.0,
            f"rho -> 1 at r > {radius}. Dirichlet decay ansatz zeroes the backbone "
            "correction at large radius.",
        ),
        (
            "far_field_pressure_unity",
            p,
            1.0,
            f"p -> 1 at r > {radius}.",
        ),
        (
            "far_field_transverse_velocity_zero",
            v,
            0.0,
            f"v -> 0 at r > {radius} (AoA = 0, symmetric freestream).",
        ),
        (
            "far_field_streamwise_velocity",
            u,
            U_INF,
            f"u -> M_inf * a_inf = {U_INF:.4f} at r > {radius}.",
        ),
    ]:
        residual = np.where(far, np.abs(field - target), 0.0)
        viol = np.maximum(residual - tol, 0.0)
        summaries.append(
            violation_summary(
                name=name,
                values=np.where(far, field - target, np.nan),
                violation=viol,
                total_count=int(far.sum()),
                extra={"far_field_radius": radius, "target": target, "interpretation": desc},
            )
        )
        violations.extend(
            top_index_rows(
                hypothesis=name,
                values=residual,
                violation=viol,
                axis_names=("sample", "point"),
                top_k=top_k,
                extra={"far_field_radius": radius},
            )
        )
    return summaries, violations


def entropy_rows(
    Q: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rho = Q[:, 0, :].astype(np.float64)
    p   = Q[:, 3, :].astype(np.float64)
    s = p / rho**GAMMA  # normalised entropy; isentropic flow -> s ≈ 1
    residual = np.abs(s - 1.0)
    violation = np.maximum(residual - tol, 0.0)
    summary = violation_summary(
        name="entropy_near_isentropic",
        values=s,
        violation=violation,
        extra={
            "entropy_tol": tol,
            "interpretation": (
                "p / rho^gamma should equal 1 at freestream (inviscid isentropic). "
                "Deviations occur near shock waves. This is informational: the flow "
                "is near-isentropic but not exactly so."
            ),
        },
    )
    violations = top_index_rows(
        hypothesis="entropy_near_isentropic",
        values=residual,
        violation=violation,
        axis_names=("sample", "point"),
        top_k=top_k,
    )
    return summary, violations


def total_enthalpy_rows(
    Q: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rho = Q[:, 0, :].astype(np.float64)
    u   = Q[:, 1, :].astype(np.float64)
    v   = Q[:, 2, :].astype(np.float64)
    p   = Q[:, 3, :].astype(np.float64)
    # H = p/((gamma-1)*rho) + 0.5*(u^2+v^2)
    H = p / ((GAMMA - 1.0) * rho) + 0.5 * (u**2 + v**2)
    residual = np.abs(H - H_INF)
    violation = np.maximum(residual - tol, 0.0)
    summary = violation_summary(
        name="total_enthalpy_near_uniform",
        values=H,
        violation=violation,
        extra={
            "H_inf": H_INF,
            "enthalpy_tol": tol,
            "interpretation": (
                f"For steady adiabatic inviscid flow, H = p/((gamma-1)*rho) + "
                f"0.5*|q|^2 is constant along streamlines. H_inf = {H_INF:.4f}. "
                "Variations arise near shock waves. Informational."
            ),
        },
    )
    violations = top_index_rows(
        hypothesis="total_enthalpy_near_uniform",
        values=residual,
        violation=violation,
        axis_names=("sample", "point"),
        top_k=top_k,
    )
    return summary, violations


def mach_definition_rows(
    Q: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rho = Q[:, 0, :].astype(np.float64)
    u   = Q[:, 1, :].astype(np.float64)
    v   = Q[:, 2, :].astype(np.float64)
    p   = Q[:, 3, :].astype(np.float64)
    M   = Q[:, 4, :].astype(np.float64)
    a = np.sqrt(GAMMA * p / rho)
    M_computed = np.sqrt(u**2 + v**2) / a
    residual = np.abs(M - M_computed)
    violation = np.maximum(residual - tol, 0.0)
    summary = violation_summary(
        name="mach_definition",
        values=residual,
        violation=violation,
        extra={
            "interpretation": (
                "ch4 == sqrt(u^2+v^2) / sqrt(gamma*p/rho). This is exact in the dataset. "
                "If a model predicts [rho, u, v, p], Mach is a derived output at zero cost."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="mach_definition",
        values=residual,
        violation=violation,
        axis_names=("sample", "point"),
        top_k=top_k,
    )
    return summary, violations


# ---------------------------------------------------------------------------
# hypothesis checks — cylinder grid wall
# ---------------------------------------------------------------------------

def wall_no_penetration_rows(
    CX: np.ndarray,
    CY: np.ndarray,
    CQ: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    V_n = wall_normal_velocity(CX, CY, CQ)  # (N, 221)
    residual = np.abs(V_n)
    violation = np.maximum(residual - tol, 0.0)
    summary = violation_summary(
        name="wall_no_penetration",
        values=V_n,
        violation=violation,
        extra={
            "wall_tol": tol,
            "interpretation": (
                "Normal velocity at the airfoil surface (cylinder j=0) should be 0 "
                "(inviscid no-penetration). The normal is approximated via finite "
                "differences along the wall tangent. Residuals reflect numerical "
                "discretisation error rather than a physics violation."
            ),
        },
    )
    violations = top_index_rows(
        hypothesis="wall_no_penetration",
        values=V_n,
        violation=violation,
        axis_names=("sample", "i_wall"),
        top_k=top_k,
    )
    return summary, violations


# ---------------------------------------------------------------------------
# hypothesis checks — naca_interp/ regular grid
# ---------------------------------------------------------------------------

def interp_interior_zero_rows(
    Qi: np.ndarray,
    mask: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    interior = Qi[~mask].astype(np.float64)
    violation = np.maximum(np.abs(interior) - tol, 0.0)
    # Expand back for indexing
    full_violation = np.where(~mask, np.maximum(np.abs(Qi) - tol, 0.0), 0.0)
    summary = violation_summary(
        name="interp_interior_zero",
        values=interior,
        violation=violation,
        total_count=int((~mask).sum()),
        extra={
            "interpretation": (
                "Q_interp is exactly 0 inside the airfoil mask. Enforcing this via "
                "mask multiplication on the model output is a trivially implementable "
                "hard constraint for the naca_interp task."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="interp_interior_zero",
        values=np.where(~mask, np.abs(Qi.astype(np.float64)), 0.0),
        violation=full_violation,
        axis_names=("sample", "i", "j"),
        top_k=top_k,
    )
    return summary, violations


def interp_mach_positive_in_fluid_rows(
    Qi: np.ndarray,
    mask: np.ndarray,
    *,
    tol: float,
    top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    in_fluid = Qi[mask].astype(np.float64)
    violation = np.maximum(tol - in_fluid, 0.0)
    full_violation = np.where(mask, np.maximum(tol - Qi.astype(np.float64), 0.0), 0.0)
    summary = violation_summary(
        name="interp_mach_positive_in_fluid",
        values=in_fluid,
        violation=violation,
        total_count=int(mask.sum()),
        extra={
            "interpretation": (
                "Q_interp (Mach) > 0 in all fluid cells. A softplus backbone output "
                "enforces this as a hard constraint."
            )
        },
    )
    violations = top_index_rows(
        hypothesis="interp_mach_positive_in_fluid",
        values=np.where(mask, Qi.astype(np.float64), 1.0),
        violation=full_violation,
        axis_names=("sample", "i", "j"),
        top_k=top_k,
    )
    return summary, violations


# ---------------------------------------------------------------------------
# optional plots
# ---------------------------------------------------------------------------

def _save_scatter(
    *,
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    require_matplotlib(plt)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    sc = ax.scatter(x, y, c=values, s=2, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_example_plots(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    Q: np.ndarray,
    out_dir: Path,
) -> list[Path]:
    require_matplotlib(plt)
    example_dir = out_dir / "examples"
    paths: list[Path] = []

    rho = Q[:, 0, :].astype(np.float64)
    u   = Q[:, 1, :].astype(np.float64)
    v   = Q[:, 2, :].astype(np.float64)
    p   = Q[:, 3, :].astype(np.float64)
    M   = Q[:, 4, :].astype(np.float64)

    # Worst entropy violation
    s = p / rho**GAMMA
    residual_s = np.abs(s - 1.0)
    sample_idx, point_idx = np.unravel_index(int(np.argmax(residual_s)), residual_s.shape)
    path = example_dir / "entropy_worst_sample.png"
    _save_scatter(
        x=X[sample_idx], y=Y[sample_idx], values=residual_s[sample_idx],
        title=f"entropy |p/rho^gamma - 1| | sample={sample_idx}, max={residual_s[sample_idx].max():.3e}",
        cmap="hot", out_path=path,
    )
    paths.append(path)

    # Worst total enthalpy violation
    H = p / ((GAMMA - 1.0) * rho) + 0.5 * (u**2 + v**2)
    residual_H = np.abs(H - H_INF)
    sample_idx, _ = np.unravel_index(int(np.argmax(residual_H)), residual_H.shape)
    path = example_dir / "enthalpy_worst_sample.png"
    _save_scatter(
        x=X[sample_idx], y=Y[sample_idx], values=residual_H[sample_idx],
        title=f"|H - H_inf| | sample={sample_idx}, max={residual_H[sample_idx].max():.3e}",
        cmap="hot", out_path=path,
    )
    paths.append(path)

    # Mach number field for a representative sample
    path = example_dir / "mach_sample_0.png"
    _save_scatter(
        x=X[0], y=Y[0], values=M[0],
        title="Mach number (sample 0)",
        cmap="RdBu_r", vmin=0.0, vmax=1.5, out_path=path,
    )
    paths.append(path)

    return paths


# ---------------------------------------------------------------------------
# print and main
# ---------------------------------------------------------------------------

def print_summary(rows: list[dict[str, object]]) -> None:
    print("\nAirfoil constraint hypothesis summary")
    for row in rows:
        status = str(row["status"])
        name = str(row["hypothesis"])
        if status == "informational":
            print(
                f"  {name:<40} info  "
                f"min={row['min']: .4e} mean={row['mean']: .4e} max={row['max']: .4e}"
            )
        else:
            print(
                f"  {name:<40} {status:<4} "
                f"viol={row['violation_count']}/{row['checked_count']} "
                f"max_viol={row['max_violation']: .4e} "
                f"min={row['min']: .4e} max={row['max']: .4e}"
            )


def main() -> None:
    args = parse_args()

    print(f"Loading naca/ arrays from {args.data_dir} ...")
    X, Y, Q, theta, res = load_naca_arrays(args.data_dir)
    (X, Y, Q, theta, res) = select_split(
        X, Y, Q, theta, res,
        split=args.split,
        ntrain=args.ntrain,
        ntest=args.ntest,
    )
    if args.max_samples is not None:
        X, Y, Q, theta, res = (
            X[: args.max_samples],
            Y[: args.max_samples],
            Q[: args.max_samples],
            theta[: args.max_samples],
            res[: args.max_samples],
        )
    print(f"  Q shape: {Q.shape}, samples: {Q.shape[0]}")

    summaries: list[dict[str, object]] = []
    violations: list[dict[str, object]] = []

    # --- basic positivity ---
    s, rows = positive_density_rows(Q, tol=args.tol, top_k=args.top_k)
    summaries.append(s); violations.extend(rows)

    s, rows = positive_pressure_rows(Q, tol=args.tol, top_k=args.top_k)
    summaries.append(s); violations.extend(rows)

    # --- far-field Dirichlet ---
    ff_summaries, rows = far_field_rows(
        X, Y, Q,
        radius=args.far_field_radius,
        tol=args.far_field_tol,
        top_k=args.top_k,
    )
    summaries.extend(ff_summaries); violations.extend(rows)

    # --- physics conservation ---
    s, rows = entropy_rows(Q, tol=args.entropy_tol, top_k=args.top_k)
    summaries.append(s); violations.extend(rows)

    s, rows = total_enthalpy_rows(Q, tol=args.enthalpy_tol, top_k=args.top_k)
    summaries.append(s); violations.extend(rows)

    s, rows = mach_definition_rows(Q, tol=args.tol, top_k=args.top_k)
    summaries.append(s); violations.extend(rows)

    # --- cylinder-grid wall ---
    if not args.skip_cylinder:
        print("Loading cylinder arrays ...")
        CX, CY, CQ = load_naca_cylinder_arrays(args.data_dir)
        (CX, CY, CQ) = select_split(
            CX, CY, CQ,
            split=args.split,
            ntrain=args.ntrain,
            ntest=args.ntest,
        )
        if args.max_samples is not None:
            CX = CX[: args.max_samples]
            CY = CY[: args.max_samples]
            CQ = CQ[: args.max_samples]
        s, rows = wall_no_penetration_rows(
            CX, CY, CQ,
            tol=args.wall_normal_tol,
            top_k=args.top_k,
        )
        summaries.append(s); violations.extend(rows)

    # --- naca_interp/ regular-grid checks ---
    if not args.skip_interp:
        print("Loading naca_interp/ arrays ...")
        Xi, Yi, Qi, mask = load_naca_interp_arrays(args.data_dir)
        n_interp = min(Qi.shape[0], Q.shape[0])
        Qi = Qi[:n_interp]; mask = mask[:n_interp]

        s, rows = interp_interior_zero_rows(Qi, mask, tol=args.tol, top_k=args.top_k)
        summaries.append(s); violations.extend(rows)

        s, rows = interp_mach_positive_in_fluid_rows(Qi, mask, tol=args.tol, top_k=args.top_k)
        summaries.append(s); violations.extend(rows)

    # --- write outputs ---
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "airfoil_constraint_hypotheses_summary.csv"
    write_csv(summary_path, summaries)
    if violations:
        violation_path = args.out_dir / "airfoil_constraint_hypotheses_violations.csv"
        write_csv(violation_path, violations)
    else:
        violation_path = None

    print(f"\nLoaded: {args.data_dir}")
    print(f"Split: {args.split}; samples checked: {Q.shape[0]}")
    print_summary(summaries)
    print(f"\nWrote summary CSV: {summary_path}")
    if violation_path is not None:
        print(f"Wrote top violations CSV: {violation_path}")
    else:
        print("No violations above tolerance.")

    if args.plot_examples:
        paths = save_example_plots(X=X, Y=Y, Q=Q, out_dir=args.out_dir)
        print("\nWrote example plots:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
