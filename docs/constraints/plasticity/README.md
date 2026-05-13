# Plasticity Constraints

Plasticity constraints in this repo act on the per-time-step output of the
dynamic conditional plasticity task. The benchmark target channels are
`[x, y, u_x, u_y]`, so constraints in this family must return four physical channels for each requested time value.

More examples:

- [PlasticityMeshConsistencyConstraint](PlasticityMeshConsistencyConstraint.md)
