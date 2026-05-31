"""Chapter 4 (Implementation) Navier-Stokes result declarations.

Binds LaTeX macros to the test-set relative-L2 error of the Navier-Stokes runs
across the four data budgets (plus the full run), for the unconstrained baseline
and the global-vorticity (mean) constraint, on each backbone. Missing runs or
metric keys emit ``\\textit{TBD}`` so the report still compiles.

Two tables consume these macros:
    Chapter 4 (ns.tex):        Transolver baseline vs. mean constraint.
    Appendix (appendix_ns.tex): the remaining backbones, same comparison.

Run families and their output layouts (relative to repo root):
    Baseline (unconstrained):  outputs/navier_stokes/none/<model>/<budget>/seed_42
    Mean constraint:           outputs/navier_stokes/mean_constraint/<model>/<budget>/seed_42

Generate with::

    python -m scripts.reporting.build --name ns_rel_l2 --output-dir ../fyp_report
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

# --- Backbones: (macro token, output dir name) -------------------------------
MODELS = (
    ("Transolver", "transolver"),
    ("Galerkin", "galerkin_transformer"),
    ("Factformer", "factformer"),
    ("Ono", "ono"),
)

# --- Run families: (macro token, constraint dir) -----------------------------
FAMILIES = (
    ("Base", "none"),
    ("Mean", "mean_constraint"),
)


def _highlight_row_minimum(results) -> None:
    """Bold the lower of {Base, Mean} for each (model, budget) cell pair.

    Mutates ``CellResult.value`` in place. TBD/missing cells and any
    non-numeric value are skipped; ties bold every joint-minimum cell.
    """
    by_macro = {r.macro: r for r in results}
    for model_token, _model_dir in MODELS:
        for _budget, budget_token in BUDGETS:
            cells = []
            for family, _dir in FAMILIES:
                cell = by_macro.get(rf"\nsRl{family}{model_token}{budget_token}")
                if cell is None or not cell.ok:
                    continue
                try:
                    cells.append((float(cell.value), cell))
                except ValueError:
                    continue
            if not cells:
                continue
            best = min(value for value, _cell in cells)
            for value, cell in cells:
                if value == best:
                    cell.value = rf"\textbf{{{cell.value}}}"


def _rel_l2_rows() -> list[Row]:
    rows: list[Row] = []
    for family, constraint_dir in FAMILIES:
        for model_token, model_dir in MODELS:
            for budget, budget_token in BUDGETS:
                path = (
                    f"outputs/navier_stokes/{constraint_dir}/{model_dir}/"
                    f"{budget}/seed_42"
                )
                rows.append(
                    Row(
                        run=RunRef(path),
                        metric_key="rel_l2",
                        macro=rf"\nsRl{family}{model_token}{budget_token}",
                        format="{:.4f}",
                    )
                )
    return rows


# --- ns_rel_l2: relative-L2 error across data budgets ------------------------
ns_rel_l2 = ReportArtifact(
    name="ns_rel_l2",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_ns_rel_l2.tex",
    rows=_rel_l2_rows(),
    postprocess=_highlight_row_minimum,
)


# --- Headline results table: literature baseline vs. mean constraint ----------
# The MeanConstraint column is pulled from the full-budget constrained runs; the
# Galerkin entry uses the validation-split run, matching the chapter text ("100
# of the 1000 training samples were reserved as a validation set"). The Baseline
# column is the unconstrained error reported in the Transolver paper
# (Wu et al. 2024); it has no local run and is declared as a literal. The
# $\Delta$ column is the relative change, computed in the postprocess below.
#
# (model token, display name, baseline literal, constrained run path)
RESULTS = (
    (
        "Galerkin",
        "0.1401",
        "outputs/navier_stokes/mean_constraint/galerkin_transformer/experiments/validation/seed_42",
    ),
    (
        "Factformer",
        "0.1214",
        "outputs/navier_stokes/mean_constraint/factformer/final/seed_42",
    ),
    (
        "Ono",
        "0.1195",
        "outputs/navier_stokes/mean_constraint/ono/final/seed_42",
    ),
    (
        "Transolver",
        "0.0900",
        "outputs/navier_stokes/mean_constraint/transolver/final/seed_42",
    ),
)


def _results_rows() -> list[Row]:
    rows: list[Row] = []
    for model_token, baseline, run_path in RESULTS:
        rows.append(
            Row(
                run=None,
                metric_key=None,
                macro=rf"\nsResBase{model_token}",
                literal=baseline,
            )
        )
        rows.append(
            Row(
                run=RunRef(run_path),
                metric_key="rel_l2",
                macro=rf"\nsResMean{model_token}",
                format="{:.4f}",
            )
        )
        # Delta is filled in by the postprocess; declared with no source so it
        # renders as TBD if the constrained run is missing.
        rows.append(
            Row(run=None, metric_key=None, macro=rf"\nsResDelta{model_token}")
        )
    return rows


def _fill_delta_and_bold(results) -> None:
    """Compute the relative-change column and bold each MeanConstraint cell.

    The baseline is a literal; the constrained value comes from a run. Delta is
    ``round((mean - base) / base * 100)`` formatted as e.g. ``$-22\\%$``. The
    constrained cell is then wrapped in ``\\textbf{}``. Cells whose constrained
    run is missing are left as TBD.
    """
    by_macro = {r.macro: r for r in results}
    for model_token, _baseline, _run_path in RESULTS:
        base_cell = by_macro.get(rf"\nsResBase{model_token}")
        mean_cell = by_macro.get(rf"\nsResMean{model_token}")
        delta_cell = by_macro.get(rf"\nsResDelta{model_token}")
        if mean_cell is None or not mean_cell.ok:
            continue
        try:
            base = float(base_cell.value)
            mean = float(mean_cell.value)
        except (TypeError, ValueError):
            continue
        if delta_cell is not None and base != 0.0:
            pct = round((mean - base) / base * 100)
            sign = "+" if pct > 0 else "-"
            delta_cell.value = rf"${sign}{abs(pct)}\%$"
            delta_cell.source = f"computed from {base_cell.macro}, {mean_cell.macro}"
            delta_cell.ok = True
        mean_cell.value = rf"\textbf{{{mean_cell.value}}}"


ns_results = ReportArtifact(
    name="ns_results",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_ns_results.tex",
    rows=_results_rows(),
    postprocess=_fill_delta_and_bold,
)


# --- Cost / hook diagnostics ---------------------------------------------------
# Parameter counts and rollout FLOPs come from each run's diagnostics.yaml
# (scripts/diagnose.py --write-yaml: cost.trainable_params, cost.step_flops; the
# profiled step covers the full autoregressive rollout). They are aggregated into
# a generated metrics file so the cost/hook macros resolve from runs instead of
# hand-typed literals, and so the build can emit the diagnose commands needed for
# any run still lacking diagnostics. Missing diagnostics render as TBD.
NS_COST_METRICS = MetricFileRef("ch4_ns_cost_metrics.yaml")

# Cost-overhead table: model token -> (baseline run, constrained run).
NS_COST_RUNS = {
    "Galerkin": (
        "outputs/navier_stokes/none/galerkin_transformer/final/seed_42",
        "outputs/navier_stokes/mean_constraint/galerkin_transformer/final/seed_42",
    ),
    "Factformer": (
        "outputs/navier_stokes/none/factformer/final/seed_42",
        "outputs/navier_stokes/mean_constraint/factformer/final/seed_42",
    ),
    "Ono": (
        "outputs/navier_stokes/none/ono/final/seed_42",
        "outputs/navier_stokes/mean_constraint/ono/final/seed_42",
    ),
    "Transolver": (
        "outputs/navier_stokes/none/transolver/final/seed_42",
        "outputs/navier_stokes/mean_constraint/transolver/final/seed_42",
    ),
}

# Hook-ablation table: variant stem -> run profiled for its params/FLOPs.
NS_HOOK_PROFILE_RUNS = {
    "TransolverBase": "outputs/navier_stokes/none/transolver/final/seed_42",
    "TransolverLast": "outputs/navier_stokes/mean_constraint/transolver/experiments/latent_head_500e/seed_42",
    "TransolverEng": "outputs/navier_stokes/mean_constraint/transolver/final/seed_42",
    "OnoBase": "outputs/navier_stokes/none/ono/final/seed_42",
    "OnoLast": "outputs/navier_stokes/mean_constraint/ono/experiements/final/seed_42",
    "OnoEng": "outputs/navier_stokes/mean_constraint/ono/final/seed_42",
}

_NS_MODEL_BACKBONES = {
    "factformer": "Factformer",
    "galerkin_transformer": "Galerkin_Transformer",
    "ono": "ONO",
    "transolver": "Transolver",
}


def _rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


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


def compute_ns_cost_metrics(repo_root: Path) -> tuple[dict[str, float], list[str]]:
    """Aggregate params (M) and rollout FLOPs (T) for the cost and hook tables."""
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    for stem, run_path in NS_HOOK_PROFILE_RUNS.items():
        cost = _load_run_cost(repo_root, run_path)
        if cost is None:
            warnings.append(f"missing diagnostics: {run_path}/diagnostics.yaml")
            continue
        params, flops = cost
        metrics[f"hook/{stem}/params_m"] = params / 1e6
        metrics[f"hook/{stem}/flops_t"] = flops / 1e12

    for model, (base_path, constr_path) in NS_COST_RUNS.items():
        base = _load_run_cost(repo_root, base_path)
        constr = _load_run_cost(repo_root, constr_path)
        if base is None:
            warnings.append(f"missing diagnostics: {base_path}/diagnostics.yaml")
        if constr is None:
            warnings.append(f"missing diagnostics: {constr_path}/diagnostics.yaml")
        if base is None or constr is None:
            continue
        metrics[f"cost/{model}/params_overhead_m"] = (constr[0] - base[0]) / 1e6
        metrics[f"cost/{model}/flops_overhead_t"] = (constr[1] - base[1]) / 1e12

    return metrics, sorted(set(warnings))


def write_ns_cost_metrics(repo_root: Path, output_dir: Path) -> Path:
    metrics, warnings = compute_ns_cost_metrics(repo_root)
    path = output_dir / "metrics" / "ch4_ns_cost_metrics.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": metrics}
    if warnings:
        payload["warnings"] = warnings
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def ns_cost_diagnostic_runs() -> tuple[RunRef, ...]:
    """Every run whose diagnostics.yaml feeds the cost or hook tables."""
    seen: set[str] = set()
    runs: list[RunRef] = []
    paths = list(NS_HOOK_PROFILE_RUNS.values())
    for base_path, constr_path in NS_COST_RUNS.values():
        paths.extend((base_path, constr_path))
    for path in paths:
        if path not in seen:
            seen.add(path)
            runs.append(RunRef(path))
    return tuple(runs)


def diagnose_command_for_ns_run(run: RunRef, repo_root: Path) -> str | None:
    """Return a diagnostics RUNS_FILE command for an NS cost/hook source.

    Existing profiled runs use their resolved config and checkpoint. For standard
    baseline/constrained NS paths that do not exist locally, fall back to the
    component config because parameter count and FLOPs do not require trained
    checkpoint weights.
    """
    run_dir = run.resolve(repo_root)
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    if config_path.exists() and checkpoint_path.exists():
        return (
            "diagnose: "
            f"--config {_rel(config_path, repo_root)} "
            f"--checkpoint {_rel(checkpoint_path, repo_root)} "
            "--write-yaml"
        )

    parts = Path(run.path).parts
    if (
        len(parts) == 6
        and parts[0] == "outputs"
        and parts[1] == "navier_stokes"
        and parts[5].startswith("seed_")
    ):
        constraint_dir, model_dir, budget_dir, seed_dir = (
            parts[2],
            parts[3],
            parts[4],
            parts[5],
        )
        backbone = _NS_MODEL_BACKBONES.get(model_dir)
        if backbone is not None:
            seed = seed_dir.removeprefix("seed_")
            return (
                "diagnose: "
                "--benchmark navier_stokes "
                f"--backbone {backbone} "
                f"--constraint {constraint_dir} "
                f"--budget {budget_dir} "
                f"--seed {seed} "
                f"--override paths.output_dir={run.path} "
                "--write-yaml"
            )

    return None


# --- ns_cost: ΔParams and ΔFLOPs of the latent-head constraint -----------------
def _cost_rows() -> list[Row]:
    rows: list[Row] = []
    for model_token in NS_COST_RUNS:
        rows.append(
            Row(
                run=NS_COST_METRICS,
                metric_key=f"cost/{model_token}/params_overhead_m",
                macro=rf"\nsCostParams{model_token}",
                format="{:+.3f}",
            )
        )
        rows.append(
            Row(
                run=NS_COST_METRICS,
                metric_key=f"cost/{model_token}/flops_overhead_t",
                macro=rf"\nsCostFlops{model_token}",
                format="{:+.3f}",
            )
        )
    return rows


ns_cost = ReportArtifact(
    name="ns_cost",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_ns_cost.tex",
    rows=_cost_rows(),
)


# --- ns_hook: params, FLOPs, Rel L2 per hook variant (appendix) ----------------
# Rel L2 comes from the runs where available, and from the reported paper baseline
# (literal) for the unconstrained rows. Params/FLOPs resolve from the generated
# diagnostics metrics.
HOOK_RELL2: dict[str, str | RunRef] = {
    "TransolverBase": "0.0900",
    "TransolverLast": RunRef(
        "outputs/navier_stokes/mean_constraint/transolver/experiments/latent_head_500e/seed_42"
    ),
    "TransolverEng": RunRef(
        "outputs/navier_stokes/mean_constraint/transolver/final/seed_42"
    ),
    "OnoBase": "0.1195",
    "OnoLast": RunRef(
        "outputs/navier_stokes/mean_constraint/ono/experiements/final/seed_42"
    ),
    "OnoEng": RunRef("outputs/navier_stokes/mean_constraint/ono/final/seed_42"),
}
HOOK_ORDER = (
    "TransolverBase",
    "TransolverLast",
    "TransolverEng",
    "OnoBase",
    "OnoLast",
    "OnoEng",
)


def _hook_rows() -> list[Row]:
    rows: list[Row] = []
    for stem in HOOK_ORDER:
        if stem in NS_HOOK_PROFILE_RUNS:
            rows.append(
                Row(
                    run=NS_COST_METRICS,
                    metric_key=f"hook/{stem}/params_m",
                    macro=rf"\nsHook{stem}Params",
                    format="{:.3f}",
                )
            )
            rows.append(
                Row(
                    run=NS_COST_METRICS,
                    metric_key=f"hook/{stem}/flops_t",
                    macro=rf"\nsHook{stem}Flops",
                    format="{:.3f}",
                )
            )
        else:
            rows.append(Row(run=None, metric_key=None, macro=rf"\nsHook{stem}Params"))
            rows.append(Row(run=None, metric_key=None, macro=rf"\nsHook{stem}Flops"))
        rel = HOOK_RELL2[stem]
        if isinstance(rel, RunRef):
            rows.append(
                Row(
                    run=rel,
                    metric_key="rel_l2",
                    macro=rf"\nsHook{stem}RelL",
                    format="{:.4f}",
                )
            )
        else:
            rows.append(
                Row(run=None, metric_key=None, macro=rf"\nsHook{stem}RelL", literal=rel)
            )
    return rows


ns_hook = ReportArtifact(
    name="ns_hook",
    chapter=4,
    kind="tex_macros",
    output_subpath="macros/ch4_ns_hook.tex",
    rows=_hook_rows(),
)


ARTIFACTS = [ns_rel_l2, ns_results, ns_cost, ns_hook]
