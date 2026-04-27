# DarcyFluxConstraint

`DarcyFluxConstraint` is a Darcy-specific hard-constraint pipeline that uses a
latent stream function to build a flux field satisfying the benchmark source
term before recovering the final scalar pressure.

For the current Darcy benchmark, the intended PDE is

$$
-\nabla \cdot (a \nabla u) = 1
\quad \text{in } \Omega = [0,1]^2,
\qquad
u|_{\partial \Omega} = 0
$$

where `a` is the permeability input and `u` is the pressure output.

The key reformulation is to work through the Darcy flux

$$
v = -a \nabla u
$$

which should satisfy

$$
\nabla \cdot v = 1
$$

## Mechanism

The backbone predicts a scalar stream function `psi`, not pressure directly.
The constraint then:

1. pads `psi` on a larger computational box,
2. computes a divergence-free stream correction
   $v_{\mathrm{corr}} = \nabla^\perp \psi$ spectrally,
3. adds a fixed particular flux `v_part` with `div(v_part) = 1`,
4. crops back to the physical domain,
5. converts the valid flux into a pressure-gradient candidate,
6. recovers pressure with a Dirichlet sine Poisson solve.

The constrained flux is

$$
v_{\mathrm{valid}} = v_{\mathrm{part}} + v_{\mathrm{corr}}
$$

Because `v_corr` is divergence-free by construction and `v_part` is chosen to
carry the constant source term, the final flux is built so that

$$
\nabla \cdot v_{\mathrm{valid}} = 1
$$

up to the discrete residual of the numerical operators being used.

The current implementation uses the Cartesian spectral stream-function helper in
[`src/omni_hc/constraints/darcy_flux.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/src/omni_hc/constraints/darcy_flux.py)
with `padding_mode="reflect"` and the `helmholtz_sine` recovery path.

After building the valid flux, the constraint maps it back to pressure through

$$
\nabla u = -\frac{v}{a}
$$

so the recovered gradient candidate is

$$
w = -\frac{v_{\mathrm{valid}}}{a}
$$

The final pressure is then obtained by solving a Dirichlet Poisson problem from
that gradient field with a sine-based solver. This gives a scalar pressure
output that respects the benchmark's zero boundary value.

## Config

Shared constraint config:

[`configs/constraints/darcy_flux_fft_pad.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/darcy_flux_fft_pad.yaml)

```yaml
constraint:
  name: "darcy_flux_projection"
  spectral_backend: "helmholtz_sine"
  pressure_out_dim: 1
  force_value: 1.0
  permeability_eps: 1.0e-6
  padding: 8
  padding_mode: "reflect"
  particular_field: "y_only"
  enforce_boundary: true
  boundary_value: 0.0
```

Darcy experiments using this constraint:

- [`configs/experiments/darcy/fno_small_flux_fft_pad.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/fno_small_flux_fft_pad.yaml)
- [`configs/experiments/darcy/gt_small_flux_fft_pad.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/gt_small_flux_fft_pad.yaml)

For the current loader, `lower` and `upper` do not need to be set manually
because the benchmark metadata already supplies `domain_bounds = (0.0, 1.0)`.

## Diagnostics And Tests

When `return_aux=True`, the constraint reports:

- `constraint/stream_div_abs_mean`
- `constraint/stream_div_abs_max`
- `constraint/flux_div_abs_mean`
- `constraint/flux_div_abs_max`
- `constraint/w_error_abs_mean`
- `constraint/w_error_abs_max`
- `constraint/w_curl_abs_mean`
- `constraint/w_curl_abs_max`
- `constraint/darcy_res_abs_mean`
- `constraint/darcy_res_abs_max`
- `constraint/boundary_abs_mean`
- `constraint/boundary_abs_max`

and auxiliary tensors including:

- `stream_correction`
- `constrained_flux`

Regression coverage in
[`tests/test_darcy_flux.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/tests/test_darcy_flux.py)
checks that:

- the output is a scalar pressure field with the expected shape
- zero Dirichlet boundary values are recovered exactly
- the constraint rejects non-scalar backbone outputs
- the expected diagnostics and auxiliary tensors are emitted
- the stream-divergence residual remains small under the implemented spectral
  construction
