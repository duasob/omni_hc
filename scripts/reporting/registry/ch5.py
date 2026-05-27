"""Chapter 5 (Constraint Analysis and Validation) declarations.

Each Row binds a LaTeX macro to a (run, metric_key) pair. Missing runs or
metric keys emit ``\textit{TBD}`` so the report still compiles.
"""

from __future__ import annotations

from ..core.types import ReportArtifact, Row, RunRef


# --- Run paths (relative to repo root) ----------------------------------------
NS_HC = RunRef("outputs/navier_stokes/mean_constraint/transolver/final/seed_42")
NS_BL = RunRef("outputs/navier_stokes/none/galerkin_transformer/final/seed_42")

DARCY_DIRICHLET_HC = RunRef("outputs/darcy/dirichlet_ansatz_zero/transolver/final/seed_42")
DARCY_FLUX_HC = RunRef("outputs/darcy/darcy_flux_constraint/transolver/final/seed_42")
DARCY_BL = RunRef("outputs/darcy/none/transolver/final/seed_42")

PIPE_HC = RunRef("outputs/pipe/pipe_stream_function_boundary_ansatz/transolver/final/seed_42")
PIPE_BL = RunRef("outputs/pipe/none/transolver/final/seed_42")

ELAS_HC = RunRef("outputs/elasticity/elasticity_deviatoric_stress_constraint/transolver/final/seed_42")
ELAS_BL = RunRef("outputs/elasticity/none/transolver/final/seed_42")

PLAS_HC = RunRef("outputs/plasticity/plasticity_mesh_consistency_constraint/transolver/final/seed_42")
PLAS_BL = None  # no baseline final run yet


# --- cv_gt: constraint error on the ground-truth test data --------------------
# No GT-stats pipeline yet; all rows emit TBD via metric_key=None.
cv_gt = ReportArtifact(
    name="cv_gt",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_gt.tex",
    rows=[
        Row(run=None, metric_key=None, macro=r"\cvGtNsVorticity"),
        Row(run=None, metric_key=None, macro=r"\cvGtDarcyDirichletStrict"),
        Row(run=None, metric_key=None, macro=r"\cvGtDarcyDirichletCorner"),
        Row(run=None, metric_key=None, macro=r"\cvGtDarcyFlux"),
        Row(run=None, metric_key=None, macro=r"\cvGtPipeWall"),
        Row(run=None, metric_key=None, macro=r"\cvGtPipeInlet"),
        Row(run=None, metric_key=None, macro=r"\cvGtPipeDiv"),
        Row(run=None, metric_key=None, macro=r"\cvGtElasticity", literal="/"),
        Row(run=None, metric_key=None, macro=r"\cvGtPlasticity"),
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
        Row(run=NS_BL, metric_key=None, macro=r"\cvBlNsVorticity"),
        Row(run=DARCY_BL, metric_key=None, macro=r"\cvBlDarcyDirichlet"),
        Row(run=DARCY_BL, metric_key=None, macro=r"\cvBlDarcyFlux"),
        Row(run=PIPE_BL, metric_key=None, macro=r"\cvBlPipeWall"),
        Row(run=PIPE_BL, metric_key=None, macro=r"\cvBlPipeInlet"),
        Row(run=PIPE_BL, metric_key=None, macro=r"\cvBlPipeDiv"),
        Row(run=None, metric_key=None, macro=r"\cvBlElasticity", literal="/"),
        Row(run=PLAS_BL, metric_key=None, macro=r"\cvBlPlasticity"),
    ],
)


# --- cv_constrained: hard-constrained models (the main first-cut deliverable) --
cv_constrained = ReportArtifact(
    name="cv_constrained",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_constrained.tex",
    rows=[
        # NS mean constraint enforces \bar\omega = 0 by subtraction; post-residual
        # not currently logged, so TBD until a metric key is added.
        Row(run=NS_HC, metric_key=None, macro=r"\cvHcNsVorticity"),
        Row(run=DARCY_DIRICHLET_HC, metric_key="boundary_rmse",
            macro=r"\cvHcDarcyDirichlet"),
        Row(run=DARCY_FLUX_HC, metric_key="constraint/flux_div_abs_mean",
            macro=r"\cvHcDarcyFlux"),
        Row(run=PIPE_HC, metric_key="constraint/stream_wall_ux_abs_max",
            macro=r"\cvHcPipeWall"),
        Row(run=PIPE_HC, metric_key="constraint/stream_inlet_abs_mean",
            macro=r"\cvHcPipeInlet"),
        Row(run=PIPE_HC, metric_key="constraint/stream_div_abs_mean",
            macro=r"\cvHcPipeDiv"),
        Row(run=ELAS_HC, metric_key="constraint/det_c_abs_error_max",
            macro=r"\cvHcElasticity"),
        Row(run=PLAS_HC, metric_key="constraint/axis_order_margin_min",
            macro=r"\cvHcPlasticity"),
    ],
)


# --- cv_cost: Δparams, ΔFLOPs -------------------------------------------------
# Requires loading the constraint module + a torch.profiler pass; not yet wired.
cv_cost = ReportArtifact(
    name="cv_cost",
    chapter=5,
    kind="tex_macros",
    output_subpath="macros/ch5_cv_cost.tex",
    rows=[
        Row(run=None, metric_key=None, macro=r"\cvCostParamsNs"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsNs"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsDarcyDirichlet", literal="0"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsDarcyDirichlet"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsDarcyFlux", literal="0"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsDarcyFlux"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPipeWall", literal="0"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsPipeWall"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPipeStream", literal="0"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsPipeStream"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsElasticity"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsElasticity"),
        Row(run=None, metric_key=None, macro=r"\cvCostParamsPlasticity", literal="0"),
        Row(run=None, metric_key=None, macro=r"\cvCostFlopsPlasticity"),
    ],
)


ARTIFACTS = [cv_gt, cv_baseline, cv_constrained, cv_cost]
