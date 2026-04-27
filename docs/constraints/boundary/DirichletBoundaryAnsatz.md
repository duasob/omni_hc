# DirichletBoundaryAnsatz

![dirichlet_ansatz_zero_g_l](../../figures/darcy/dirichlet_ansatz_zero_g_l.png)

`DirichletBoundaryAnsatz` enforces a constant Dirichlet value on the boundary
of a box domain through the standard architectural ansatz

$$
u(x) = g(x) + l(x)N(x)
$$

where:

- $N(x)$ is the unconstrained backbone output
- $g(x)$ is a particular field satisfying the boundary condition
- $l(x)$ is a distance-like factor that is zero on the boundary

For the current Darcy benchmark, the intended condition is homogeneous:

$$
u|_{\partial \Omega} = 0
$$

so the particular field is just the constant boundary value

$$
g(x) = 0
$$

and the only remaining work is to construct a factor `l(x)` that vanishes on
the full box boundary.

## Mechanism

On a rectangular domain `[lower, upper]^d`, the implementation uses the unit-box
distance helper to build

$$
l(x) = \prod_i (x_i - \mathrm{lower})(\mathrm{upper} - x_i)
$$

up to the configured `distance_power` and reduction rule. This guarantees
`l(x) = 0` whenever any coordinate lies on the boundary, so the ansatz reduces
exactly to `g(x)` there.

In the Darcy benchmark, the adapter provides `domain_bounds = (0.0, 1.0)`, so
the ansatz automatically uses the unit square without needing explicit
`lower` / `upper` overrides in the experiment config.

This is a direct output-space constraint: it does not encode the Darcy PDE
itself, only the pressure boundary values.
## Config

Shared constraint config:

[`configs/constraints/dirichlet_ansatz_zero.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/dirichlet_ansatz_zero.yaml)

```yaml
constraint:
  name: "dirichlet_ansatz"
  boundary_value: 0.0
  distance_power: 1.0
  distance_reduce: "product"
```

Darcy experiment using this constraint:

[`configs/experiments/darcy/fno_small_dirichlet.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/fno_small_dirichlet.yaml)

## Diagnostics And Tests

When `return_aux=True`, the constraint reports:

- `constraint/boundary_abs_mean`
- `constraint/boundary_abs_max`

Regression coverage in
[`tests/test_boundary.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/tests/test_boundary.py)
checks that:

- the boundary mask is detected correctly
- the ansatz enforces exact zero boundary values
- the physical boundary condition is still respected when a target normalizer is
  active
- the boundary diagnostics are emitted
