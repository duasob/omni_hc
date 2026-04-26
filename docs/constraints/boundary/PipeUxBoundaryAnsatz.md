# PipeUxBoundaryAnsatz

`PipeUxBoundaryAnsatz` combines the pipe parabolic inlet profile with no-slip
wall enforcement for scalar `ux`.

```text
f = g + lN
g(i,j) = alpha(i) * 0.25 * 4t(j)(1 - t(j))
l(i,j) = (1 - alpha(i)) * wall_distance(j)
alpha(i) = (1 - xi(i))^p
```

This gives exact behavior at the constrained boundaries:

- inlet edge: parabolic `ux` profile
- lower and upper wall edges: zero `ux`

## Config

See `configs/constraints/pipe_ux_boundary.yaml`.

## Tests

Covered by `tests/test_boundary.py` and optional real-data checks in
`tests/test_real_data_constraints.py`.

