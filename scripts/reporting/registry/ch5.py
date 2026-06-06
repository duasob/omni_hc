"""Chapter 5 (Constraint Analysis and Validation) declarations.

Each Row binds a LaTeX macro to a (run, metric_key) pair. Missing runs or
metric keys emit ``\textit{TBD}`` so the report still compiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import yaml

from ..core.types import MetricFileRef, ReportArtifact, Row, RunRef

# --- Run paths (relative to repo root) ----------------------------------------
NS_HC = RunRef(
    "outputs/navier_stokes/mean_constraint/galerkin_transformer/final/seed_42"
)
NS_BL = RunRef("outputs/navier_stokes/none/galerkin_transformer/final/seed_42")

DARCY_DIRICHLET_HC = RunRef(
    "outputs/darcy/dirichlet_ansatz_zero/transolver/final/seed_42"
)
DARCY_SINE_HC = RunRef("outputs/darcy/sine_boundary/transolver_latest_coeff_head_test")
DARCY_FLUX_HC = RunRef("outputs/darcy/darcy_flux_constraint/transolver/final/seed_42")
DARCY_BL = RunRef("outputs/darcy/none/transolver/final/seed_42")

PIPE_ANSATZ_HC = RunRef("outputs/pipe/experiments/fno_small_ux_boundary")
PIPE_STREAM_HC = RunRef(
    "outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/final/seed_42"
)
PIPE_BL = RunRef("outputs/pipe/none/transolver/final/seed_42")

ELAS_HC = RunRef(
    "outputs/elasticity/elasticity_deviatoric_stress_constraint/transolver/final/seed_42"
)
ELAS_BL = RunRef("outputs/elasticity/none/transolver/final/seed_42")

PLAS_HC = RunRef(
    "outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/final/seed_42"
)
PLAS_ENV_HC = RunRef(
    "outputs/plasticity/plasticity_envelope_constraint/transolver/final/seed_42"
)
PLAS_BL = RunRef("outputs/plasticity/none/transolver/final/seed_42")
GT = MetricFileRef("ch5_gt_metrics.yaml")
COST = MetricFileRef("ch5_cost_metrics.yaml")


# --- cv_gt: constraint error on the ground-truth test data --------------------
cv_gt = ReportArtifact(
    name="cv_gt",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_gt.tex",
    rows=[
        Row(
            run=GT,
            metric_key="constraint/vorticity_abs_mean",
            macro=r"\cvGtNsVorticity",
        ),
        Row(
            run=GT,
            metric_key="constraint/dirichlet_strict_max",
            macro=r"\cvGtDarcyDirichletStrict",
        ),
        Row(
            run=GT,
            metric_key="constraint/dirichlet_corner_max",
            macro=r"\cvGtDarcyDirichletCorner",
        ),
        Row(run=GT, metric_key="constraint/darcy_res_abs_mean", macro=r"\cvGtDarcyFlux"),
        Row(run=GT, metric_key="constraint/wall_abs_max", macro=r"\cvGtPipeWall"),
        Row(run=GT, metric_key="constraint/inlet_rmse", macro=r"\cvGtPipeInlet"),
        Row(run=GT, metric_key="constraint/div_rel_mean", macro=r"\cvGtPipeDiv"),
        Row(run=None, metric_key=None, macro=r"\cvGtElasticity", literal="/"),
        Row(
            run=GT,
            metric_key=(
                "constraint/neg_spacing_count",
                "constraint/axis_order_violation_count",
            ),
            macro=r"\cvGtPlasticitySpacing",
            format="{:.0f}",
        ),
        Row(
            run=GT,
            metric_key=(
                "constraint/top_envelope_violation_count",
                "constraint/top_violation_count",
            ),
            macro=r"\cvGtPlasticityTopDie",
            format="{:.0f}",
        ),
        Row(
            run=GT,
            metric_key="constraint/below_y_bottom_violation_count",
            macro=r"\cvGtPlasticityBottom",
            format="{:.0f}",
        ),
    ],
)


# --- cv_baseline: unconstrained baselines --------------------------------------
# Baselines were tested without constraint hooks → no constraint/* metrics.
# Rows declared so that, once baselines are re-tested with the relevant
# diagnostic hooks, only the metric_key needs to flip from None to the key.
cv_baseline = ReportArtifact(
    name="cv_baseline",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_baseline.tex",
    rows=[
        Row(
            run=NS_BL,
            metric_key="constraint/vorticity_abs_mean",
            macro=r"\cvBlNsVorticity",
        ),
        Row(
            run=DARCY_BL,
            metric_key=("constraint/dirichlet_strict_max", "boundary_rmse"),
            macro=r"\cvBlDarcyDirichletStrict",
        ),
        Row(
            run=DARCY_BL,
            metric_key="constraint/dirichlet_corner_max",
            macro=r"\cvBlDarcyDirichletCorner",
        ),
        Row(
            run=DARCY_BL,
            metric_key="constraint/darcy_res_abs_mean",
            macro=r"\cvBlDarcyFlux",
        ),
        Row(run=PIPE_BL, metric_key="constraint/wall_abs_max", macro=r"\cvBlPipeWall"),
        Row(run=PIPE_BL, metric_key="constraint/inlet_rmse", macro=r"\cvBlPipeInlet"),
        Row(run=None, metric_key=None, macro=r"\cvBlPipeDiv", literal="/"),
        Row(run=ELAS_BL, metric_key=None, macro=r"\cvBlElasticity", literal="/"),
        Row(
            run=PLAS_BL,
            metric_key=(
                "constraint/neg_spacing_count",
                "constraint/axis_order_violation_count",
            ),
            macro=r"\cvBlPlasticitySpacing",
            format="{:.0f}",
        ),
        Row(
            run=PLAS_BL,
            metric_key=(
                "constraint/top_envelope_violation_count",
                "constraint/top_violation_count",
            ),
            macro=r"\cvBlPlasticityTopDie",
            format="{:.0f}",
        ),
        Row(
            run=PLAS_BL,
            metric_key="constraint/below_y_bottom_violation_count",
            macro=r"\cvBlPlasticityBottom",
            format="{:.0f}",
        ),
    ],
)


# --- cv_constrained: hard-constrained models ----------------------------------
cv_constrained = ReportArtifact(
    name="cv_constrained",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_constrained.tex",
    rows=[
        Row(
            run=NS_HC,
            metric_key="constraint/vorticity_abs_mean",
            macro=r"\cvHcNsVorticity",
        ),
        Row(
            run=DARCY_DIRICHLET_HC,
            metric_key=(
                "constraint/dirichlet_strict_max",
                "constraint/boundary_abs_max",
                "boundary_rmse",
            ),
            macro=r"\cvHcDarcyDirichletStrict",
        ),
        Row(
            run=DARCY_SINE_HC,
            metric_key="constraint/dirichlet_corner_max",
            macro=r"\cvHcDarcyDirichletCorner",
        ),
        Row(
            run=DARCY_DIRICHLET_HC,
            metric_key=(
                "constraint/dirichlet_strict_max",
                "constraint/boundary_abs_max",
                "boundary_rmse",
            ),
            macro=r"\cvHcDarcyDirichletStrictDirichlet",
        ),
        Row(
            run=DARCY_SINE_HC,
            metric_key="constraint/dirichlet_strict_max",
            macro=r"\cvHcDarcyDirichletStrictSine",
        ),
        Row(
            run=DARCY_DIRICHLET_HC,
            metric_key=(
                "constraint/dirichlet_corner_max",
                "constraint/boundary_abs_max",
                "boundary_rmse",
            ),
            macro=r"\cvHcDarcyDirichletCornerDirichlet",
        ),
        Row(
            run=DARCY_SINE_HC,
            metric_key="constraint/dirichlet_corner_max",
            macro=r"\cvHcDarcyDirichletCornerSine",
        ),
        Row(
            run=DARCY_FLUX_HC,
            metric_key="constraint/darcy_res_abs_mean",
            macro=r"\cvHcDarcyFlux",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key=("constraint/wall_abs_max", "constraint/stream_wall_ux_abs_max"),
            macro=r"\cvHcPipeWall",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key=("constraint/inlet_rmse", "constraint/stream_inlet_abs_mean"),
            macro=r"\cvHcPipeInlet",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key=("constraint/div_rel_mean", "constraint/stream_div_abs_mean"),
            macro=r"\cvHcPipeDiv",
        ),
        Row(
            run=PIPE_ANSATZ_HC,
            metric_key="constraint/wall_abs_max",
            macro=r"\cvHcPipeWallAnsatz",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key=("constraint/wall_abs_max", "constraint/stream_wall_ux_abs_max"),
            macro=r"\cvHcPipeWallStream",
        ),
        Row(
            run=PIPE_ANSATZ_HC,
            metric_key="constraint/inlet_rmse",
            macro=r"\cvHcPipeInletAnsatz",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key="constraint/inlet_rmse",
            macro=r"\cvHcPipeInletStream",
        ),
        Row(
            run=PIPE_STREAM_HC,
            metric_key=("constraint/div_rel_mean", "constraint/stream_div_abs_mean"),
            macro=r"\cvHcPipeDivStream",
        ),
        Row(
            run=ELAS_HC,
            metric_key="constraint/det_c_abs_error_max",
            macro=r"\cvHcElasticity",
        ),
        Row(
            run=PLAS_HC,
            metric_key=(
                "constraint/neg_spacing_count",
                "constraint/axis_order_violation_count",
            ),
            macro=r"\cvHcPlasticitySpacing",
            format="{:.0f}",
        ),
        Row(
            run=PLAS_ENV_HC,
            metric_key=(
                "constraint/top_envelope_violation_count",
                "constraint/top_violation_count",
            ),
            macro=r"\cvHcPlasticityTopDie",
            format="{:.0f}",
        ),
        Row(
            run=PLAS_ENV_HC,
            metric_key="constraint/below_y_bottom_violation_count",
            macro=r"\cvHcPlasticityBottom",
            format="{:.0f}",
        ),
    ],
)


# --- cv_cost: baseline params/FLOPs and constraint % overhead -----------------
CV_COST_ROWS = (
    ("Ns", "ns"),
    ("DarcyDirichlet", "darcy_dirichlet"),
    ("DarcyFlux", "darcy_flux"),
    ("PipeWall", "pipe_wall"),
    ("PipeStream", "pipe_stream"),
    ("Elasticity", "elasticity"),
    ("Plasticity", "plasticity"),
)


def _cv_cost_rows() -> list[Row]:
    rows: list[Row] = []
    for macro_token, metric_token in CV_COST_ROWS:
        rows.append(
            Row(
                run=COST,
                metric_key=f"{metric_token}/params_base_m",
                macro=rf"\cvCostParamsBase{macro_token}",
                format="{:.2f}",
            )
        )
        rows.append(
            Row(
                run=COST,
                metric_key=f"{metric_token}/params_overhead_pct",
                macro=rf"\cvCostParamsPct{macro_token}",
                format="{:+.1f}\\%",
            )
        )
        rows.append(
            Row(
                run=COST,
                metric_key=f"{metric_token}/flops_base_t",
                macro=rf"\cvCostFlopsBase{macro_token}",
                format="{:.2f}",
            )
        )
        rows.append(
            Row(
                run=COST,
                metric_key=f"{metric_token}/flops_overhead_pct",
                macro=rf"\cvCostFlopsPct{macro_token}",
                format="{:+.1f}\\%",
            )
        )
    return rows


cv_cost = ReportArtifact(
    name="cv_cost",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_cost.tex",
    rows=_cv_cost_rows(),
)


ARTIFACTS = [cv_gt, cv_baseline, cv_constrained, cv_cost]


class Ch5CostSpec(NamedTuple):
    key: str
    constrained: RunRef
    baseline: RunRef


MODEL_ARTIFACTS = (cv_baseline, cv_constrained)

BENCHMARK_LABELS = {
    "navier_stokes_2d": "Navier-Stokes",
    "darcy_2d": "Darcy",
    "pipe_2d": "Pipe",
    "elasticity_2d": "Elasticity",
    "plasticity_2d": "Plasticity",
}


def effective_training_samples(cfg: dict[str, Any]) -> int:
    ntrain = int((cfg.get("data") or {}).get("ntrain", 1000))
    val_size = int((cfg.get("training") or {}).get("val_size", 0))
    if val_size <= 0:
        return ntrain
    return max(ntrain - min(max(val_size, 1), ntrain - 1), 1)


def resolve_run_config(
    repo_root: Path, run: RunRef
) -> tuple[Path, dict[str, Any] | None]:
    run_dir = run.resolve(repo_root)
    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        return run_dir, None
    return run_dir, yaml.safe_load(cfg_path.read_text()) or {}


def _run_refs_from_artifact(artifact: ReportArtifact) -> tuple[RunRef, ...]:
    seen: set[str] = set()
    runs: list[RunRef] = []
    for row in artifact.rows:
        if not isinstance(row.run, RunRef) or row.run.path in seen:
            continue
        seen.add(row.run.path)
        runs.append(row.run)
    return tuple(runs)


def _registered_model_runs(
    artifacts: tuple[ReportArtifact, ...] = MODEL_ARTIFACTS,
) -> tuple[tuple[str, RunRef], ...]:
    seen: set[str] = set()
    runs: list[tuple[str, RunRef]] = []
    for artifact in artifacts:
        role = "Baseline" if artifact.name == "cv_baseline" else "Constrained"
        for run in _run_refs_from_artifact(artifact):
            key = f"{role}:{run.path}"
            if key in seen:
                continue
            seen.add(key)
            runs.append((role, run))
    return tuple(runs)


def _diagnostics_path(repo_root: Path, run: RunRef) -> Path:
    return run.resolve(repo_root) / "diagnostics.yaml"


def _load_diagnostics(repo_root: Path, run: RunRef) -> dict[str, Any] | None:
    path = _diagnostics_path(repo_root, run)
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text()) or {}


def _run_model_name(cfg: dict[str, Any]) -> str:
    experiment = cfg.get("experiment") or {}
    model = cfg.get("model") or {}
    return str(experiment.get("backbone") or model.get("backbone") or "?")


def _run_constraint_name(cfg: dict[str, Any]) -> str:
    experiment = cfg.get("experiment") or {}
    return str(experiment.get("constraint") or "none")


def _run_benchmark_name(cfg: dict[str, Any]) -> str:
    experiment = cfg.get("experiment") or {}
    benchmark = cfg.get("benchmark") or {}
    return str(benchmark.get("name") or experiment.get("benchmark") or "?")


def _run_benchmark_label(cfg: dict[str, Any]) -> str:
    name = _run_benchmark_name(cfg)
    return BENCHMARK_LABELS.get(name, name.replace("_2d", "").replace("_", " ").title())


def _run_epochs(cfg: dict[str, Any]) -> int:
    return int((cfg.get("training") or {}).get("num_epochs", 0))


def _tex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def render_provenance_table(
    repo_root: Path,
    *,
    artifacts: tuple[ReportArtifact, ...] = MODEL_ARTIFACTS,
) -> str:
    rows: list[str] = []
    for role, run in _registered_model_runs(artifacts):
        _run_dir, cfg = resolve_run_config(repo_root, run)
        if cfg is None:
            label = model = epochs = samples = constraint = r"\textit{TBD}"
        else:
            label = _tex_escape(_run_benchmark_label(cfg))
            model = _tex_escape(_run_model_name(cfg))
            constraint = _tex_escape(_run_constraint_name(cfg))
            epochs = str(_run_epochs(cfg))
            samples = str(effective_training_samples(cfg))
        rows.append(
            "        "
            + " & ".join(
                [
                    label,
                    _tex_escape(role),
                    model,
                    constraint,
                    str(epochs),
                    str(samples),
                ]
            )
            + r" \\"
        )

    body = "\n".join(rows)
    return rf"""% Generated by scripts/reporting/build.py
% Artifact: ch5_run_provenance  (chapter 5)

\begin{{table}}[htbp]
    \centering
    \small
    \caption{{Source runs for the Chapter~5 baseline and constrained model results.}}
    \label{{tab:ch5_run_provenance}}
    \resizebox{{\linewidth}}{{!}}{{%
    \begin{{tabular}}{{llllrr}}
        \toprule
        Result & Role & Model & Constraint & Epochs & Train samples \\
        \midrule
{body}
        \bottomrule
    \end{{tabular}}
    }}
\end{{table}}
"""


def write_provenance_table(repo_root: Path, output_dir: Path) -> Path:
    path = output_dir / "tables" / "ch5_run_provenance.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_provenance_table(repo_root))
    return path


def _run_key(cfg: dict[str, Any]) -> tuple[str, str]:
    return (_run_benchmark_name(cfg), _run_model_name(cfg).lower())


def _cost_keys_for_constrained_run(cfg: dict[str, Any]) -> tuple[str, ...]:
    benchmark = _run_benchmark_name(cfg)
    constraint = _run_constraint_name(cfg)
    if benchmark == "navier_stokes_2d":
        return ("ns",)
    if benchmark == "darcy_2d" and "dirichlet" in constraint:
        return ("darcy_dirichlet",)
    if benchmark == "darcy_2d" and "flux" in constraint:
        return ("darcy_flux",)
    if benchmark == "pipe_2d" and "stream" in constraint:
        return ("pipe_wall", "pipe_stream")
    if benchmark == "elasticity_2d":
        return ("elasticity",)
    if benchmark == "plasticity_2d":
        return ("plasticity",)
    return ()


def cost_specs_from_configs(repo_root: Path) -> tuple[Ch5CostSpec, ...]:
    baselines: dict[tuple[str, str], RunRef] = {}
    constrained: list[tuple[RunRef, dict[str, Any]]] = []

    for role, run in _registered_model_runs():
        _run_dir, cfg = resolve_run_config(repo_root, run)
        if cfg is None:
            continue
        if role == "Baseline":
            baselines[_run_key(cfg)] = run
        else:
            constrained.append((run, cfg))

    specs: list[Ch5CostSpec] = []
    for run, cfg in constrained:
        baseline = baselines.get(_run_key(cfg))
        if baseline is None:
            continue
        for key in _cost_keys_for_constrained_run(cfg):
            specs.append(Ch5CostSpec(key, run, baseline))
    return tuple(specs)


def _cost_number(diagnostics: dict[str, Any], key: str) -> float | None:
    cost = diagnostics.get("cost") or {}
    value = cost.get(key)
    return float(value) if value is not None else None


def _overhead(constrained: float | None, baseline: float | None) -> float | None:
    if constrained is None or baseline is None:
        return None
    return constrained - baseline


def _overhead_pct(overhead: float, baseline: float | None) -> float:
    if not baseline:
        return 0.0
    pct = overhead / baseline * 100
    return 0.0 if abs(pct) < 0.05 else pct


def compute_cost_metrics(repo_root: Path) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    warnings: list[str] = []

    for spec in cost_specs_from_configs(repo_root):
        constrained_diag = _load_diagnostics(repo_root, spec.constrained)
        baseline_diag = _load_diagnostics(repo_root, spec.baseline)
        if constrained_diag is None:
            warnings.append(
                f"missing diagnostics: {spec.constrained.path}/diagnostics.yaml"
            )
        if baseline_diag is None:
            warnings.append(
                f"missing diagnostics: {spec.baseline.path}/diagnostics.yaml"
            )
        if constrained_diag is None or baseline_diag is None:
            continue

        constrained_params = _cost_number(constrained_diag, "trainable_params")
        baseline_params = _cost_number(baseline_diag, "trainable_params")
        constrained_flops = _cost_number(constrained_diag, "step_flops")
        baseline_flops = _cost_number(baseline_diag, "step_flops")

        if baseline_params is not None:
            metrics[f"{spec.key}/params_base_m"] = baseline_params / 1e6
        if baseline_flops is not None:
            metrics[f"{spec.key}/flops_base_t"] = baseline_flops / 1e12

        params_overhead = _overhead(constrained_params, baseline_params)
        flops_overhead = _overhead(constrained_flops, baseline_flops)
        if params_overhead is not None:
            metrics[f"{spec.key}/params_overhead"] = params_overhead
            metrics[f"{spec.key}/params_overhead_pct"] = _overhead_pct(
                params_overhead, baseline_params
            )
        if flops_overhead is not None:
            metrics[f"{spec.key}/flops_overhead"] = flops_overhead
            metrics[f"{spec.key}/flops_overhead_pct"] = _overhead_pct(
                flops_overhead, baseline_flops
            )

    return metrics, warnings


def write_cost_metrics(repo_root: Path, output_dir: Path) -> Path:
    metrics, warnings = compute_cost_metrics(repo_root)
    path = output_dir / "metrics" / "ch5_cost_metrics.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"metrics": metrics}
    if warnings:
        payload["warnings"] = sorted(set(warnings))
    path.write_text(yaml.safe_dump(payload, sort_keys=True))
    return path


def diagnose_command_for_run(run: RunRef, repo_root: Path) -> str | None:
    run_dir = run.resolve(repo_root)
    config_path = run_dir / "resolved_config.yaml"
    checkpoint_path = run_dir / "best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = run_dir / "latest.pt"
    if not config_path.exists() or not checkpoint_path.exists():
        return None
    return (
        "diagnose: "
        f"--config {config_path.relative_to(repo_root).as_posix()} "
        f"--checkpoint {checkpoint_path.relative_to(repo_root).as_posix()} "
        "--write-yaml"
    )


def cost_diagnostic_runs() -> tuple[RunRef, ...]:
    seen: set[str] = set()
    runs: list[RunRef] = []
    for _role, run in _registered_model_runs():
        if run.path not in seen:
            seen.add(run.path)
            runs.append(run)
    return tuple(runs)
