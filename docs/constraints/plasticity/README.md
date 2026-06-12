# Plasticity Constraints

Plasticity constraints in this repo act on the per-time-step output of the
dynamic conditional plasticity task. The benchmark target channels are
`[x, y, u_x, u_y]`, so constraints in this family must return four physical channels for each requested time value.

Implemented constraints (all in `src/omni_hc/constraints/plasticity.py`):

- [PlasticityMeshConsistencyConstraint](PlasticityMeshConsistencyConstraint.md):
  reconstructs ordered deformation meshes from positive learned spacings
  (detailed page).
- `PlasticityEnvelopeConstraint`: reconstructs mesh coordinates confined to the
  moving die envelope; the envelope position is derived from the forcing input
  (`envelope_source: fx`).
- `PlasticityEnvelopeYFreeXConstraint`: envelope variant that constrains only
  the vertical coordinate and leaves the horizontal coordinate free.
- `PlasticityIsotonicRegression`: projects directly predicted coordinates onto
  ordered (monotone) meshes via isotonic regression.
