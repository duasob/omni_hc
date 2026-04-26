# Boundary Constraints

Boundary constraints in this family directly transform a backbone prediction
using an architectural ansatz:

```text
f = g + lN
```

where `N` is the unconstrained prediction, `g` is a particular field satisfying
the boundary condition, and `l` is zero on the constrained boundary.

## Pages

- [BoundaryAnsatz](BoundaryAnsatz.md)
- [DirichletBoundaryAnsatz](DirichletBoundaryAnsatz.md)
- [StructuredWallDirichletAnsatz](StructuredWallDirichletAnsatz.md)
- [PipeInletParabolicAnsatz](PipeInletParabolicAnsatz.md)
- [PipeUxBoundaryAnsatz](PipeUxBoundaryAnsatz.md)

## Figures

Generate boundary ansatz `g` and `l` maps with:

```bash
conda run -n omni-hc python scripts/diagnostics/boundary_ansatz_maps.py
```
