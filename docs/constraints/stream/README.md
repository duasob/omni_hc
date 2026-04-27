# Stream-Function Constraints

Stream-function constraints treat the backbone output as a latent scalar stream
function. The physical output is recovered by differentiating that stream
function on the mesh.

## Pages

- [PipeStreamFunctionUxConstraint](PipeStreamFunctionUxConstraint.md): recover
  `ux` from a latent stream function, yielding a divergence-free velocity
  construction without direct boundary enforcement.
- [PipeStreamFunctionBoundaryAnsatz](PipeStreamFunctionBoundaryAnsatz.md):
  combine the stream-function construction with hard inlet and wall behavior on
  the pipe benchmark.

Pure stream-function operators live in
`src/omni_hc/constraints/utils/stream_ops.py`.
