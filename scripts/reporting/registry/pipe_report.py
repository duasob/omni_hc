"""Chapter 4 (Implementation) pipe-flow result declarations.

Binds LaTeX macros to the test-set relative-L2 error of the Transolver pipe
runs across the four data budgets (plus the full run), for the unconstrained
baseline and the boundary-ansatz variants. Missing runs or metric keys emit
``\\textit{TBD}`` so the report still compiles.

Run families and their output layouts (relative to repo root):
    Baseline (unconstrained):  outputs/pipe/none/transolver/<budget>/seed_42
    u_x ansatz:                outputs/pipe/pipe_ux_boundary_ansatz/transolver/<budget>/seed_42
    Stream-function ansatz:    outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/<budget>/seed_42
    Stream + u_y supervision:  outputs/pipe/pipe_stream_function_boundary_ansatz_uy/transolver/<budget>/seed_42

Generate with::

    python -m scripts.reporting.build --name pipe_rel_l2 --output-dir ../fyp_report
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
FAMILIES = (
    ("Base", "outputs/pipe/none/transolver/{budget}/seed_42"),
    ("Ux", "outputs/pipe/pipe_ux_boundary_ansatz/transolver/{budget}/seed_42"),
    (
        "Stream",
        "outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/{budget}/seed_42",
    ),
    (
        "StreamUy",
        "outputs/pipe/pipe_stream_function_boundary_ansatz_uy/transolver/{budget}/seed_42",
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
            cell = by_macro.get(rf"\pipeRl{family}{token}")
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
                    macro=rf"\pipeRl{family}{token}",
                    format="{:.4f}",
                )
            )
    return rows


# --- pipe_rel_l2: relative-L2 error across data budgets ----------------------
pipe_rel_l2 = ReportArtifact(
    name="pipe_rel_l2",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_pipe_rel_l2.tex",
    rows=_rel_l2_rows(),
    postprocess=_highlight_row_minimum,
)


ARTIFACTS = [pipe_rel_l2]
