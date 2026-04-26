# DarcyFluxConstraint

`DarcyFluxConstraint` is an operator pipeline rather than a direct boundary
ansatz. The backbone predicts a scalar stream function, then the constraint:

1. pads the stream function,
2. computes a divergence-free stream correction,
3. adds a fixed particular flux,
4. converts flux to a pressure-gradient candidate using permeability,
5. recovers pressure with a Dirichlet sine Poisson solve.

The result is a scalar pressure output with exact Dirichlet boundary values.

This is intentionally not included in the `g/l` boundary-map diagnostic. The
Dirichlet boundary is imposed by the global pressure recovery operator, not by a
local output-space ansatz `f = g + lN`. For Darcy boundary-map figures, use the
direct `DirichletBoundaryAnsatz` config when you want to visualize a local
boundary ansatz.

## Config

See `configs/constraints/darcy_flux_fft_pad.yaml`.

## Tests

Covered by `tests/test_darcy_flux.py` and config wiring tests.
