# Constraint Methods

OmniHC constraints wrap unconstrained backbone predictions and map them onto a
physically restricted output space.

## Families

- [Boundary constraints](boundary/README.md): direct architectural ansatzes of
  the form `f = g + lN`.
- [Stream-function constraints](stream/README.md): latent stream-function
  representations that are converted into physical velocity outputs.
- Darcy constraints:
  - [DarcyFluxConstraint](stream/DarcyFluxConstraint.md): stream-function flux
    construction that hard-builds `div(v) = 1` before recovering pressure.
  - [DarcyDefectCorrectionConstraint](stream/DarcyDefectCorrectionConstraint.md):
    coordinate ansatz for exact zero BCs followed by an iterative Poisson
    defect correction that drives the interior Darcy residual to zero.
- [Elasticity deviatoric stress](elasticity/ElasticityDeviatoricStressConstraint.md):
  2D incompressible hyperelastic von Mises stress from a scalar backbone latent,
  coordinates, and a constrained Right Cauchy-Green tensor.
- [Plasticity mesh consistency](plasticity/PlasticityMeshConsistencyConstraint.md):
  ordered deformation mesh reconstruction from positive learned spacings,
  returning consistent `[x, y, u_x, u_y]` channels.
- [Mean constraints](MeanConstraint.md): global mean-preserving correction.

## Implementation Map

- Public modules live in `src/omni_hc/constraints`.
- Shared implementation helpers live in `src/omni_hc/constraints/utils`.
- Constraint configs live in `configs/constraints`.
- Regression tests live in `tests`.
