# PipeStreamFunctionBoundaryAnsatz

`PipeStreamFunctionBoundaryAnsatz` constrains the latent pipe stream function:

```text
psi = psi_bc(eta) + xi^p eta^2 (1 - eta)^2 N
```

The constrained stream function is converted to velocity in physical
coordinates. This preserves the inlet parabolic profile and keeps the stream
correction from changing wall values.

## Config

See `configs/constraints/pipe_stream_function_boundary.yaml`.

## Diagnostics

```bash
conda run -n omni-hc python scripts/diagnostics/pipe_stream_function.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

