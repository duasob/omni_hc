"""Chapter 5 (Constraint Analysis and Validation) declarations.

Each Row binds a LaTeX macro to a (run, metric_key) pair. Missing runs or
metric keys emit ``\textit{TBD}`` so the report still compiles.
"""

from __future__ import annotations

from ..core.types import MetricFileRef, ReportArtifact, Row, RunRef

# --- Run paths (relative to repo root) ----------------------------------------
NS_HC = RunRef(
    "outputs/navier_stokes/mean_constraint/galerkin_transformer/final/seed_42"
)
NS_BL = RunRef("outputs/navier_stokes/none/galerkin_transformer/final/seed_42")

DARCY_DIRICHLET_HC = RunRef(
    "outputs/darcy/dirichlet_ansatz_zero/transolver/final/seed_42"
)
DARCY_FLUX_HC = RunRef("outputs/darcy/darcy_flux_constraint/transolver/final/seed_42")
DARCY_BL = RunRef("outputs/darcy/none/transolver/final/seed_42")

PIPE_HC = RunRef(
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
PLAS_BL = RunRef("outputs/plasticity/none/transolver/smoke/seed_42")
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
        Row(run=GT, metric_key="constraint/flux_rmse", macro=r"\cvGtDarcyFlux"),
        Row(run=GT, metric_key="constraint/wall_abs_max", macro=r"\cvGtPipeWall"),
        Row(run=GT, metric_key="constraint/inlet_rmse", macro=r"\cvGtPipeInlet"),
        Row(run=GT, metric_key="constraint/div_rmse", macro=r"\cvGtPipeDiv"),
        Row(run=None, metric_key=None, macro=r"\cvGtElasticity", literal="/"),
        Row(
            run=GT,
            metric_key="constraint/neg_spacing_fraction",
            macro=r"\cvGtPlasticity",
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
            macro=r"\cvBlDarcyDirichlet",
        ),
        Row(run=DARCY_BL, metric_key="constraint/flux_rmse", macro=r"\cvBlDarcyFlux"),
        Row(run=PIPE_BL, metric_key="constraint/wall_abs_max", macro=r"\cvBlPipeWall"),
        Row(run=PIPE_BL, metric_key="constraint/inlet_rmse", macro=r"\cvBlPipeInlet"),
        Row(run=None, metric_key=None, macro=r"\cvBlPipeDiv", literal="/"),
        Row(run=None, metric_key=None, macro=r"\cvBlElasticity", literal="/"),
        Row(
            run=PLAS_BL,
            metric_key="constraint/neg_spacing_fraction",
            macro=r"\cvBlPlasticity",
        ),
    ],
)


# --- cv_constrained: hard-constrained models (the main first-cut deliverable) --
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
            macro=r"\cvHcDarcyDirichlet",
        ),
        Row(
            run=DARCY_FLUX_HC,
            metric_key=("constraint/flux_rmse", "constraint/flux_div_abs_mean"),
            macro=r"\cvHcDarcyFlux",
        ),
        Row(
            run=PIPE_HC,
            metric_key=("constraint/wall_abs_max", "constraint/stream_wall_ux_abs_max"),
            macro=r"\cvHcPipeWall",
        ),
        Row(
            run=PIPE_HC,
            metric_key=("constraint/inlet_rmse", "constraint/stream_inlet_abs_mean"),
            macro=r"\cvHcPipeInlet",
        ),
        Row(
            run=PIPE_HC,
            metric_key=("constraint/div_rmse", "constraint/stream_div_abs_mean"),
            macro=r"\cvHcPipeDiv",
        ),
        Row(
            run=ELAS_HC,
            metric_key="constraint/det_c_abs_error_max",
            macro=r"\cvHcElasticity",
        ),
        Row(
            run=PLAS_HC,
            metric_key="constraint/neg_spacing_fraction",
            macro=r"\cvHcPlasticity",
        ),
    ],
)


# --- cv_cost: Δparams, ΔFLOPs -------------------------------------------------
cv_cost = ReportArtifact(
    name="cv_cost",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_cost.tex",
    rows=[
        Row(
            run=COST,
            metric_key="ns/params_overhead",
            macro=r"\cvCostParamsNs",
            format="{:.0f}",
        ),
        Row(
            run=COST,
            metric_key="ns/flops_overhead",
            macro=r"\cvCostFlopsNs",
            format="{:.0f}",
        ),
        Row(
            run=None, metric_key=None, macro=r"\cvCostParamsDarcyDirichlet", literal="0"
        ),
        Row(
            run=COST,
            metric_key="darcy_dirichlet/flops_overhead",
            macro=r"\cvCostFlopsDarcyDirichlet",
            format="{:.0f}",
        ),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsDarcyFlux", literal="0"),
        Row(
            run=COST,
            metric_key="darcy_flux/flops_overhead",
            macro=r"\cvCostFlopsDarcyFlux",
            format="{:.0f}",
        ),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPipeWall", literal="0"),
        Row(
            run=COST,
            metric_key="pipe_wall/flops_overhead",
            macro=r"\cvCostFlopsPipeWall",
            format="{:.0f}",
        ),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPipeStream", literal="0"),
        Row(
            run=COST,
            metric_key="pipe_stream/flops_overhead",
            macro=r"\cvCostFlopsPipeStream",
            format="{:.0f}",
        ),
        Row(
            run=COST,
            metric_key="elasticity/params_overhead",
            macro=r"\cvCostParamsElasticity",
            format="{:.0f}",
        ),
        Row(
            run=COST,
            metric_key="elasticity/flops_overhead",
            macro=r"\cvCostFlopsElasticity",
            format="{:.0f}",
        ),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPlasticity", literal="0"),
        Row(
            run=COST,
            metric_key="plasticity/flops_overhead",
            macro=r"\cvCostFlopsPlasticity",
            format="{:.0f}",
        ),
    ],
)


ARTIFACTS = [cv_gt, cv_baseline, cv_constrained, cv_cost]
