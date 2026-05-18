# SineBoundaryConstraint


| ![boundary profiles per edge](../../figures/darcy/1000_darcy_boundary_profiles.png) |
| ----------------------------------------------------------------------------------- |
| ![boundary profiles on domain](../../figures/darcy/darcy_boundary_profiles_2d.png)  |

`SineBoundaryConstraint` hard-enforces the boundary values of a 2D prediction by
replacing each edge with a **learned sine-basis expansion**. Unlike the ansatz
family (`DirichletBoundaryAnsatz`), which multiplies the backbone output by a
distance weight, this constraint **directly overwrites** the boundary nodes with
a predicted profile — giving the model explicit control over the boundary shape.

## Motivation

Inspection of the Darcy pressure field reveals that the boundary values are not
exactly zero: they follow a smooth, sample-dependent arch driven by the local
permeability (see figures above). A constant Dirichlet ansatz with `g = 0`
misses this structure. The sine basis is the natural representation because it
satisfies zero-corner conditions by construction.

## Mechanism

For each of the four edges of an H × W grid, the constraint predicts a 1D
profile via a sine basis expansion:

$$
u(x_j) = \sum_{k=1}^{K} c_k \sin\!\left(\frac{k\pi j}{L - 1}\right), \quad j = 0, \ldots, L-1
$$

where $L$ is the edge length and $K$ is `n_modes`. Because $\sin(0) = \sin(k\pi) = 0$
for all integer $k$, the profile is **zero at both corners** by construction —
no special corner handling is needed.

The coefficient vectors $c_k \in \mathbb{R}^{4K}$ (one set per edge) are
predicted by a small MLP that takes as input the **boundary permeability
features**: the permeability values $a(x)$ sampled at all boundary nodes. This
gives the model direct access to the physical driver of the boundary shape.

Optionally, a latent representation from the backbone transformer can be wired
in via `ForwardHookLatentExtractor`. The latent is **mean-pooled over boundary
nodes only** before being concatenated to the permeability features, giving the
MLP a compact, boundary-aware summary of the model's internal state.

The predicted profiles are then **hard-written** into the output tensor,
replacing the backbone's boundary predictions entirely:

```python
out = pred.clone()
out[:, idx_bottom, 0] = coeffs[:, 0] @ basis_h.T   # [B, W]
out[:, idx_top,    0] = coeffs[:, 1] @ basis_h.T
out[:, idx_left,   0] = coeffs[:, 2] @ basis_v.T   # [B, H]
out[:, idx_right,  0] = coeffs[:, 3] @ basis_v.T
```

## Config

```yaml
constraint:
  name: "sine_boundary_constraint"
  n_modes: 16        # number of sine basis functions per edge
  hidden_dim: 128    # MLP hidden width
  n_layers: 2        # number of hidden layers (total depth = n_layers + 2)
  act: "gelu"
```

Config file:
[`configs/constraints/darcy_sine_boundary_constraint.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/darcy_sine_boundary_constraint.yaml)

To wire in a backbone latent (e.g. from a Transolver block), add:

```yaml
constraint:
  name: "sine_boundary_constraint"
  n_modes: 16
  hidden_dim: 128
  n_layers: 2
  latent_module: "blocks.-1"   # path to the module to hook
  latent_dim: 256              # output dimension of that module
```

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `n_modes` | — | Number of sine basis functions per edge |
| `grid_shape` | from `shapelist` | `(H, W)` of the structured grid |
| `hidden_dim` | — | MLP hidden width |
| `n_layers` | `0` | Extra hidden layers (total MLP depth = `n_layers + 2`) |
| `act` | `"gelu"` | Activation function |
| `latent_dim` | `0` | Latent dimension to concatenate; `0` disables latent hook |
| `latent_module` | `None` | Dotted path to the backbone module to hook |

## Diagnostics

When `return_aux=True` the constraint reports:

- `constraint/boundary_correction_mean_abs` — mean absolute difference between
  the predicted boundary profile and the unconstrained backbone output at boundary
  nodes. Useful for monitoring how much the constraint is actively correcting.

The `log_media` classmethod produces a 4-panel figure (one per edge) showing the
ground-truth boundary profile against the model prediction, uploaded to W&B as
`constraint/boundary_profiles`.

## Comparison with DirichletBoundaryAnsatz

| | `DirichletBoundaryAnsatz` | `SineBoundaryConstraint` |
|---|---|---|
| **Mechanism** | Multiplies backbone by distance weight $\ell(x)$ | Overwrites boundary nodes with a learned profile |
| **Boundary value** | Fixed constant $g$ | Predicted per-sample sine expansion |
| **Corner condition** | Exact (distance weight = 0) | Exact (sine basis vanishes at endpoints) |
| **Backbone at boundary** | Suppressed smoothly | Completely replaced |
| **Extra parameters** | None | MLP predicting sine coefficients |
| **Suitable when** | BC is known and constant | BC is unknown / varies per sample |
