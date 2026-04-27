# Boundary Constraints

Boundary constraints in this family directly transform a backbone prediction
using an architectural ansatz:

$$
f = g + l \times N
$$

where $N$ is the unconstrained prediction, $g$ is a particular field satisfying
the boundary condition, and $l$ is zero on the constrained boundary.

This guarantees the condition independently of the learned backbone
weights. In code, the shared composition lives in
`src/omni_hc/constraints/utils/boundary_ops.py`.

The diagnostic map script derives `g` and `l` from the actual constraint call,
instead of reimplementing each formula:

```bash
conda run -n omni-hc python scripts/diagnostics/boundary_ansatz_maps.py
```

This makes the figures useful regression artifacts: if an ansatz implementation
changes, the generated maps change with it.

More examples: 
- [DirichletBoundaryAnsatz](DirichletBoundaryAnsatz.md)
- [StructuredWallDirichletAnsatz](StructuredWallDirichletAnsatz.md)
- [PipeInletParabolicAnsatz](PipeInletParabolicAnsatz.md)
- [PipeUxBoundaryAnsatz](PipeUxBoundaryAnsatz.md)

