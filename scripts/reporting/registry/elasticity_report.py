"""Chapter 4 (Implementation) elasticity result declarations.

Binds LaTeX macros to the test-set relative-L2 error of the Transolver
elasticity runs across the four data budgets (plus the full run), for the
unconstrained baseline and the plane-stress von-Mises constraint in its two
decoder variants: the scalar-output constraint and the engineered latent
decoder (which decodes the constraint head from Transolver's internal
representations). Missing runs or metric keys emit ``\\textit{TBD}`` so the
report still compiles.

Run families and their output layouts (relative to repo root):
    Baseline (unconstrained):  outputs/elasticity/none/transolver/<budget>/seed_42
    VM constraint (scalar):    outputs/elasticity/elasticity_plane_stress_vm_constraint/transolver/backbone_out_dim_1/<budget>/seed_42
    VM constraint (latent):    outputs/elasticity/elasticity_plane_stress_vm_latent/transolver/<budget>/seed_42

The scalar VM row uses the ``backbone_out_dim_1`` ablation; the parallel
``backbone_out_dim_32`` sweep is intentionally ignored for this table.

Generate with::

    python -m scripts.reporting.build --name elasticity_rel_l2 --output-dir ../fyp_report
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..core.types import MetricFileRef, ReportArtifact, Row, RunRef

# --- Data budgets: (runfile budget dir, macro token) -------------------------
# Tokens are human-readable since LaTeX macro names cannot contain digits or
# underscores. The budgets correspond to (epochs / train samples):
#   Tiny e5_t50, Small e10_t100, Medium e50_t500, Large e100_t900, Full final(500/1000)
BUDGETS = (
    ("e5_t50", "Tiny"),
    ("e10_t100", "Small"),
    ("e50_t500", "Medium"),
    ("e100_t900", "Large"),
    ("final", "Full"),
)

# --- Run families: (macro token, path template) ------------------------------
# The two constraint variants share the same plane-stress von-Mises constraint
# and differ only in how the constraint head is decoded: ``Vm`` decodes from the
# projected scalar output (backbone_out_dim=1 ablation), while ``VmLatent``
# decodes from Transolver's internal physics-aware representations (the
# engineered latent decoder).
FAMILIES = (
    ("Base", "outputs/elasticity/none/transolver/{budget}/seed_42"),
    (
        "Vm",
        "outputs/elasticity/elasticity_plane_stress_vm_constraint/transolver/backbone_out_dim_1/{budget}/seed_42",
    ),
    (
        "VmLatent",
        "outputs/elasticity/elasticity_plane_stress_vm_latent/transolver/{budget}/seed_42",
    ),
)


def _highlight_row_minimum(results) -> None:
    """Mark the lowest resolved value within each budget (one table row).

    The full-budget (headline) row bolds its winner; every other budget row
    underlines its winner instead, so the table draws the eye to the full-data
    comparison. Mutates ``CellResult.value`` in place. TBD/missing cells and any
    non-numeric value are skipped; ties mark every joint-minimum cell.
    """
    by_macro = {r.macro: r for r in results}
    for _budget, token in BUDGETS:
        cells = []
        for family, _template in FAMILIES:
            cell = by_macro.get(rf"\elasRl{family}{token}")
            if cell is None or not cell.ok:
                continue
            try:
                cells.append((float(cell.value), cell))
            except ValueError:
                continue
        if not cells:
            continue
        command = "textbf" if token == "Full" else "underline"
        best = min(value for value, _cell in cells)
        for value, cell in cells:
            if value == best:
                cell.value = rf"\{command}{{{cell.value}}}"


def _rel_l2_rows() -> list[Row]:
    rows: list[Row] = []
    for family, template in FAMILIES:
        for budget, token in BUDGETS:
            rows.append(
                Row(
                    run=RunRef(template.format(budget=budget)),
                    metric_key="rel_l2",
                    macro=rf"\elasRl{family}{token}",
                    format="{:.4f}",
                )
            )
    return rows


# --- elasticity_rel_l2: relative-L2 error across data budgets -----------------
elasticity_rel_l2 = ReportArtifact(
    name="elasticity_rel_l2",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_elasticity_rel_l2.tex",
    rows=_rel_l2_rows(),
    postprocess=_highlight_row_minimum,
)


# --- Cost diagnostics ---------------------------------------------------------
# Parameter counts and per-forward FLOPs come from each run's diagnostics.yaml
# (scripts/diagnose.py --write-yaml: cost.trainable_params, cost.step_flops).
# Elasticity is a steady-state task, so step_flops is a single forward pass. The
# three variants are profiled at the full (final) budget since cost depends only
# on architecture, not on the data budget. Values are aggregated into a generated
# metrics file so the macros resolve from runs instead of hand-typed literals, and
# so the build can emit the diagnose commands for any run lacking diagnostics.
# Missing diagnostics render as TBD.
ELAS_COST_METRICS = MetricFileRef("ch4_elasticity_cost_metrics.yaml")

# Cost table: variant token -> run profiled for its params/FLOPs. The Vm row uses
# the backbone_out_dim_1 ablation, matching the rel-L2 table.
ELAS_COST_RUNS = {
    "Base": "outputs/elasticity/none/transolver/final/seed_42",
    "Vm": "outputs/elasticity/elasticity_plane_stress_vm_constraint/transolver/backbone_out_dim_1/final/seed_42",
    "VmLatent": "outputs/elasticity/elasticity_plane_stress_vm_latent/transolver/final/seed_42",
}


def _load_run_cost(repo_root: Path, run_path: str) -> tuple[float, float] | None:
    """Return (trainable_params, step_flops) from a run's diagnostics.yaml."""
    path = repo_root / run_path / "diagnostics.yaml"
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text()) or {}
    cost = doc.get("cost") or {}
    params = cost.get("trainable_params")
    flops = cost.get("step_flops")
    if params is None or flops is None:
        return None
    return float(params), float(flops)


def compute_elasticity_cost_metrics(
    repo_root: Path,
) -> tuple[dict[str, float], list[str]]:
    """Aggregate trainable params (M) and per-forward FLOPs (G) per variant.

    Each non-baseline variant also gets its parameter and FLOP overhead as a
    percentage of the ``Base`` (unconstrained) backbone, so the table can show
    how much the constraint decoder costs relative to the backbone it wraps.
    """
    metrics: dict[str, float] = {}
    warnings: list[str] = []
    raw: dict[str, tuple[float, float]] = {}
    for token, run_path in ELAS_COST_RUNS.items():
        cost = _load_run_cost(repo_root, run_path)
        if cost is None:
            warnings.append(f"missing diagnostics: {run_path}/diagnostics.yaml")
            continue
        params, flops = cost
        raw[token] = (params, flops)
        metrics[f"cost/{token}/params_m"] = params / 1e6
        metrics[f"cost/{token}/flops_g"] = flops / 1e9

    base = raw.get("Base")
    if base is not None:
        base_params, base_flops = base
        for token, (params, flops) in raw.items():
            if token == "Base":
                continue
            metrics[f"cost/{token}/params_overhead_pct"] = (
                (params - base_params) / base_params * 100 if base_params else 0.0
            )
            metrics[f"cost/{token}/flops_overhead_pct"] = (
                (flops - base_flops) / base_flops * 100 if base_flops else 0.0
            )
    return metrics, sorted(set(warnings))


def write_elasticity_cost_metrics(repo_root: Path, output_dir: Path) -> Path:
    metrics, warnings = compute_elasticity_cost_metrics(repo_root)
    path = output_dir / "metrics" / "ch4_elasticity_cost_metrics.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": metrics}
    if warnings:
        payload["warnings"] = warnings
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def elasticity_cost_diagnostic_runs() -> tuple[RunRef, ...]:
    """Every run whose diagnostics.yaml feeds the cost table."""
    return tuple(RunRef(path) for path in ELAS_COST_RUNS.values())


def diagnose_command_for_elasticity_run(run: RunRef, repo_root: Path) -> str | None:
    """Return a diagnostics RUNS_FILE command for an elasticity cost source.

    Parameter count and FLOPs do not require trained weights, but the resolved
    config and a checkpoint pin the exact architecture, so we use them when
    present. Returns None when neither exists yet.
    """
    run_dir = run.resolve(repo_root)
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        return None
    config_rel = config_path.relative_to(repo_root).as_posix()
    checkpoint_rel = checkpoint_path.relative_to(repo_root).as_posix()
    return (
        "diagnose: "
        f"--config {config_rel} "
        f"--checkpoint {checkpoint_rel} "
        "--write-yaml"
    )


# --- elasticity_cost: per-variant parameter count and forward FLOPs ------------
# Each variant contributes its absolute trainable parameter count (M) and
# per-forward FLOPs (G). The two constraint variants additionally contribute the
# relative overhead (Delta %) of each over the unconstrained Base backbone; the
# ``+``-signed format makes the sign explicit and ``\%`` is appended by the format
# string. Base has no overhead row (it is the reference; the table prints a dash).
def _cost_rows() -> list[Row]:
    rows: list[Row] = []
    for token in ELAS_COST_RUNS:
        rows.append(
            Row(
                run=ELAS_COST_METRICS,
                metric_key=f"cost/{token}/params_m",
                macro=rf"\elasCostParams{token}",
                format="{:.2f}",
            )
        )
        rows.append(
            Row(
                run=ELAS_COST_METRICS,
                metric_key=f"cost/{token}/flops_g",
                macro=rf"\elasCostFlops{token}",
                format="{:.2f}",
            )
        )
        if token != "Base":
            rows.append(
                Row(
                    run=ELAS_COST_METRICS,
                    metric_key=f"cost/{token}/params_overhead_pct",
                    macro=rf"\elasCostParamsPct{token}",
                    format="{:+.1f}\\%",
                )
            )
            rows.append(
                Row(
                    run=ELAS_COST_METRICS,
                    metric_key=f"cost/{token}/flops_overhead_pct",
                    macro=rf"\elasCostFlopsPct{token}",
                    format="{:+.1f}\\%",
                )
            )
    return rows


elasticity_cost = ReportArtifact(
    name="elasticity_cost",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_elasticity_cost.tex",
    rows=_cost_rows(),
)


ARTIFACTS = [elasticity_rel_l2, elasticity_cost]
