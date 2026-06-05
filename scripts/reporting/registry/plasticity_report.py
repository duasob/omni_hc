"""Plasticity (forging) result declarations.

Binds LaTeX macros to the Transolver plasticity runs for three model families:
the unconstrained baseline, the mesh-consistency constrained model, and the
moving-die envelope constrained model. Missing runs or metric keys emit
``\\textit{TBD}`` so the report still compiles.

The main artifact is budget-aligned: each data-budget row can compare the three
families on relative L2, top-die penetration count, below-floor node count, and
negative-spacing count.

Run families and their output layouts (relative to repo root):
    Baseline (unconstrained):  outputs/plasticity/none/transolver/<budget>/seed_42
    Mesh consistency:          outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/<budget>/seed_42
    Die envelope:              outputs/plasticity/plasticity_envelope_constraint/transolver/<budget>/seed_42

Generate with::

    python -m scripts.reporting.build --name plasticity_budget_table --output-dir ../fyp_report
"""

from __future__ import annotations

from ..core.types import ReportArtifact, Row, RunRef

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
# Base is the unconstrained model; Mesh enforces ordered, non-inverting mesh
# spacings; Env additionally caps the top surface under the moving die profile.
FAMILIES = (
    ("Base", "outputs/plasticity/none/transolver/{budget}/seed_42"),
    (
        "Mesh",
        "outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/{budget}/seed_42",
    ),
    (
        "Env",
        "outputs/plasticity/plasticity_envelope_constraint/transolver/{budget}/seed_42",
    ),
)


def _highlight_row_minimum(results) -> None:
    """Mark the lowest resolved relative-L2 value within each budget row.

    The full-budget (headline) row bolds its winner; every other budget row
    underlines its winner instead, so the table draws the eye to the full-data
    comparison. Mutates ``CellResult.value`` in place. TBD/missing cells and any
    non-numeric value are skipped; ties mark every joint-minimum cell.
    """
    by_macro = {r.macro: r for r in results}
    for _budget, token in BUDGETS:
        cells = []
        for family, _template in FAMILIES:
            cell = by_macro.get(rf"\plasRl{family}{token}")
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


def _budget_rows() -> list[Row]:
    rows: list[Row] = []
    for family, template in FAMILIES:
        for budget, token in BUDGETS:
            run = RunRef(template.format(budget=budget))
            rows.append(
                Row(
                    run=run,
                    metric_key="rel_l2",
                    macro=rf"\plasRl{family}{token}",
                    format="{:.4f}",
                )
            )
            rows.extend(
                [
                    Row(
                        run=run,
                        metric_key=(
                            "constraint/top_envelope_violation_count",
                            "constraint/top_violation_count",
                        ),
                        macro=rf"\plasTopCount{family}{token}",
                        format="{:.0f}",
                    ),
                    Row(
                        run=run,
                        metric_key="constraint/below_y_bottom_violation_count",
                        macro=rf"\plasBottomCount{family}{token}",
                        format="{:.0f}",
                    ),
                    Row(
                        run=run,
                        metric_key=(
                            "constraint/neg_spacing_count",
                            "constraint/axis_order_violation_count",
                        ),
                        macro=rf"\plasNegSpacingCount{family}{token}",
                        format="{:.0f}",
                    ),
                ]
            )
    return rows


# --- plasticity_budget_table: budget rows x metric-family comparisons --------
plasticity_budget_table = ReportArtifact(
    name="plasticity_budget_table",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_plasticity_budget_table.tex",
    rows=_budget_rows(),
    postprocess=_highlight_row_minimum,
)

# Backward-compatible artifact name/path for the existing Chapter 4 table.
plasticity_rel_l2 = ReportArtifact(
    name="plasticity_rel_l2",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_plasticity_rel_l2.tex",
    rows=[
        Row(
            run=RunRef(template.format(budget=budget)),
            metric_key="rel_l2",
            macro=rf"\plasRl{family}{token}",
            format="{:.4f}",
        )
        for family, template in FAMILIES
        for budget, token in BUDGETS
    ],
    postprocess=_highlight_row_minimum,
)


# --- Headline compatibility aliases -----------------------------------------
def _cv_rows() -> list[Row]:
    rows: list[Row] = []
    for family, template in FAMILIES:
        budget, token = ("final", "Full") if family != "Env" else ("e100_t900", "Large")
        run = RunRef(template.format(budget=budget))
        rows.extend(
            [
                Row(
                    run=run,
                    metric_key="rel_l2",
                    macro=rf"\plasCv{family}Rl",
                    format="{:.4f}",
                ),
                Row(
                    run=run,
                    metric_key=(
                        "constraint/neg_spacing_count",
                        "constraint/axis_order_violation_count",
                    ),
                    macro=rf"\plasCv{family}Spacing",
                    format="{:.0f}",
                ),
                Row(
                    run=run,
                    metric_key="constraint/below_y_bottom_violation_count",
                    macro=rf"\plasCv{family}Bottom",
                    format="{:.0f}",
                ),
                Row(
                    run=run,
                    metric_key=(
                        "constraint/top_envelope_violation_count",
                        "constraint/top_violation_count",
                    ),
                    macro=rf"\plasCv{family}Envelope",
                    format="{:.0f}",
                ),
            ]
        )
    return rows


# --- plasticity_cv: compatibility artifact for existing Chapter 5 includes ---
plasticity_cv = ReportArtifact(
    name="plasticity_cv",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_plasticity_cv.tex",
    rows=_cv_rows(),
)


ARTIFACTS = [plasticity_budget_table, plasticity_rel_l2, plasticity_cv]
