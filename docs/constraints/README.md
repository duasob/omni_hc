# Constraint Methods

OmniHC constraints wrap unconstrained backbone predictions and map them onto a
physically restricted output space.

## Families

- [Boundary constraints](boundary/README.md): direct architectural ansatzes of
  the form `f = g + lN`.
- [Stream-function constraints](stream/README.md): latent stream-function
  representations that are converted into physical velocity outputs.
- [Darcy flux projection](DarcyFluxConstraint.md): stream-function flux
  construction followed by pressure recovery.
- [Elasticity deviatoric stress](elasticity/ElasticityDeviatoricStressConstraint.md):
  2D incompressible hyperelastic von Mises stress from a constrained
  Right Cauchy-Green tensor.
- [Mean constraints](MeanConstraint.md): global mean-preserving correction.

## Implementation Map

- Public modules live in `src/omni_hc/constraints`.
- Shared implementation helpers live in `src/omni_hc/constraints/utils`.
- Constraint configs live in `configs/constraints`.
- Regression tests live in `tests`.
