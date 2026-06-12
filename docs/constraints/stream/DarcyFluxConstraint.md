# DarcyFluxConstraint

`DarcyFluxConstraint` reparameterizes the Darcy prediction through an
intermediate flux field. The benchmark solves

$$
-\nabla \cdot (a \nabla u) = 1
\quad \text{in } \Omega=[0,1]^2,
\qquad
u|_{\partial\Omega}=0,
$$

where `a` is permeability and `u` is pressure. Defining the flux

$$
\mathbf{v}=-a\nabla u
$$

gives the continuity condition

$$
\nabla\cdot\mathbf{v}=1.
$$

## Mechanism

The backbone predicts a scalar stream function $\psi$ rather than pressure.
The constraint constructs

$$
\mathbf{v}_{\mathrm{valid}}
=\mathbf{v}_{\mathrm{part}}+\nabla^\perp\psi,
$$

where the fixed particular field satisfies
$\nabla\cdot\mathbf{v}_{\mathrm{part}}=1$ and the stream correction is
divergence-free under the selected discrete derivative. It then computes

$$
\mathbf{w}=-\frac{\mathbf{v}_{\mathrm{valid}}}{a}
$$

and recovers a scalar pressure by solving

$$
\nabla^2u=\nabla\cdot\mathbf{w}
$$

with a sine-based homogeneous Dirichlet Poisson solver.

The default config differentiates $\psi$ with finite differences. This avoids
the grid-scale striping observed with FFT differentiation on the reflect-padded
field and needs no padding. A separate spectral config retains the FFT
derivative as an experimental ablation.

## Guarantee And Limitation

The construction hard-builds the continuity condition on the intermediate
field $\mathbf{v}_{\mathrm{valid}}$, relative to the selected discrete
operators. The final pressure is a projection of
$\mathbf{w}$ onto a scalar potential. If $\mathbf{w}$ is not curl-free, that
projection does not preserve the intermediate flux exactly. Consequently:

- the intermediate constructed flux satisfies the discrete divergence target;
- the recovered pressure satisfies the homogeneous Dirichlet boundary exactly;
- the pressure-induced flux $-a\nabla u$ is biased toward the Darcy equation,
  but is not guaranteed to have divergence exactly one.

This distinction is reflected in the diagnostics: `flux_*` metrics describe
the constructed intermediate flux, while `darcy_res_*` metrics are computed
from the recovered pressure.

The default config also sets `curl_loss_weight: 0.05`. This adds a soft penalty
on the curl of $\mathbf{w}$ during training, encouraging it to be closer to a
valid pressure gradient. Set the weight to zero to use only the hard
intermediate-flux construction.

## Configs

Default finite-difference config:

[`configs/constraints/darcy_flux_constraint.yaml`](../../../configs/constraints/darcy_flux_constraint.yaml)

```yaml
training:
  max_grad_norm: 1.0

constraint:
  name: "darcy_flux_constraint"
  spectral_backend: "helmholtz_sine"
  stream_derivative: "fd"
  pressure_out_dim: 1
  force_value: 1.0
  permeability_eps: 1.0e-6
  padding: 0
  padding_mode: "reflect"
  particular_field: "y_only"
  curl_loss_weight: 0.05
  enforce_boundary: true
  boundary_value: 0.0
```

Spectral ablation:

[`configs/constraints/darcy_flux_constraint_spectral.yaml`](../../../configs/constraints/darcy_flux_constraint_spectral.yaml)

```yaml
training:
  max_grad_norm: 1.0

constraint:
  name: "darcy_flux_constraint"
  spectral_backend: "helmholtz_sine"
  stream_derivative: "spectral"
  pressure_out_dim: 1
  force_value: 1.0
  permeability_eps: 1.0e-6
  padding: 32
  padding_mode: "reflect"
  particular_field: "y_only"
  enforce_boundary: true
  boundary_value: 0.0
```

Runnable experiment recipes include:

- [`configs/experiments/darcy/fno_small_flux.yaml`](../../../configs/experiments/darcy/fno_small_flux.yaml)
- [`configs/experiments/darcy/fno_small_flux_fft_pad.yaml`](../../../configs/experiments/darcy/fno_small_flux_fft_pad.yaml)
- [`configs/experiments/darcy/gt_small_flux_fft_pad.yaml`](../../../configs/experiments/darcy/gt_small_flux_fft_pad.yaml)

## Diagnostics And Tests

When `return_aux=True`, the constraint exposes the stream correction and
constructed flux and reports:

- `constraint/stream_div_abs_mean`
- `constraint/stream_div_abs_max`
- `constraint/flux_div_abs_mean`
- `constraint/flux_div_abs_max`
- `constraint/flux_rmse`
- `constraint/w_error_abs_mean`
- `constraint/w_error_abs_max`
- `constraint/w_curl_abs_mean`
- `constraint/w_curl_abs_max`
- `constraint/w_curl_mse` when the curl penalty is enabled
- `constraint/darcy_res_abs_mean`
- `constraint/darcy_res_abs_max`
- `constraint/darcy_res_rmse`
- `constraint/boundary_abs_mean`
- `constraint/boundary_abs_max`

Regression coverage in
[`tests/test_darcy_flux.py`](../../../tests/test_darcy_flux.py) checks output
shape, exact Dirichlet recovery, finite-difference mass conservation,
diagnostic emission, scalar-stream validation, and the configurable curl loss.
