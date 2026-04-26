# StructuredWallDirichletAnsatz

`StructuredWallDirichletAnsatz` enforces a constant value on two opposite
structured-grid walls.

```text
f(i,j) = g + wall_distance(j) N(i,j)
```

The wall distance is built in index space, so it remains stable when physical
pipe coordinates vary by sample.

## Use Case

Pipe no-slip walls for scalar velocity channels.

## Config

See `configs/constraints/pipe_wall_no_slip.yaml`.

## Tests

Covered by `tests/test_boundary.py`.

