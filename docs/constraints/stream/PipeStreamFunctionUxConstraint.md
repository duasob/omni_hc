# PipeStreamFunctionUxConstraint

`PipeStreamFunctionUxConstraint` interprets the scalar backbone output as a
stream function `psi` on the curvilinear pipe mesh and returns `ux`.

This tests whether a latent stream-function representation can produce velocity
outputs with better physical structure than a direct scalar prediction.

## Config

See `configs/constraints/pipe_stream_function.yaml`.

## Tests

Covered by `tests/test_stream.py`.

