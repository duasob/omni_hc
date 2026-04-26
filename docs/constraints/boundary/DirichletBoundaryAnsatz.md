# DirichletBoundaryAnsatz

`DirichletBoundaryAnsatz` enforces a constant Dirichlet value on an axis-aligned
box domain.

```text
f(x) = g + l(x)N(x)
l(x) = Π_i (x_i - lower)(upper - x_i)
```

## Use Case

This is the simple unit-square Darcy pressure boundary ansatz.

## Config

See `configs/constraints/dirichlet_ansatz_zero.yaml`.

## Tests

Covered by `tests/test_boundary.py`.

