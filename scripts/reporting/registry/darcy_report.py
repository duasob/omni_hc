"""Chapter 4 (Implementation) Darcy-flow result declarations.

Binds LaTeX macros to the test-set relative-L2 error of the Transolver Darcy
runs across the four data budgets (plus the full run), for the unconstrained
baseline and the two flux-constraint differentiation variants (spectral curl
and finite difference). Missing runs or metric keys emit ``\\textit{TBD}`` so
the report still compiles.

Run families and their output layouts (relative to repo root):
    Baseline (unconstrained):  outputs/darcy/none/transolver/<budget>/seed_42
    Flux (spectral curl):      outputs/darcy/darcy_flux_constraint_spectral/transolver/<budget>/seed_42
    Flux (finite difference):  outputs/darcy/darcy_flux_constraint/transolver/<budget>/seed_42

Generate with::

    python -m scripts.reporting.build --name darcy_rel_l2 --output-dir ../fyp_report
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
# The two flux variants differ only in how the stream-function curl is
# differentiated: the spectral curl pads the domain and differentiates in
# Fourier space, while the FD variant uses local finite-difference stencils.
FAMILIES = (
    ("Base", "outputs/darcy/none/transolver/{budget}/seed_42"),
    (
        "FluxSpectral",
        "outputs/darcy/darcy_flux_constraint_spectral/transolver/{budget}/seed_42",
    ),
    ("FluxFd", "outputs/darcy/darcy_flux_constraint/transolver/{budget}/seed_42"),
)


def _highlight_row_minimum(results) -> None:
    """Mark the lower of {Base, Flux} for each budget cell pair.

    The full-budget (headline) row bolds its winner; every other budget row
    underlines its winner instead, so the table draws the eye to the full-data
    comparison.
    """
    by_macro = {r.macro: r for r in results}
    for _budget, token in BUDGETS:
        cells = []
        for family, _template in FAMILIES:
            cell = by_macro.get(rf"\darcyRl{family}{token}")
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
                    macro=rf"\darcyRl{family}{token}",
                    format="{:.4f}",
                )
            )
    return rows


# --- darcy_rel_l2: relative-L2 error across data budgets ---------------------
darcy_rel_l2 = ReportArtifact(
    name="darcy_rel_l2",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_darcy_rel_l2.tex",
    rows=_rel_l2_rows(),
    postprocess=_highlight_row_minimum,
)


ARTIFACTS = [darcy_rel_l2]
