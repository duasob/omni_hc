# Constraints
This folder contains reusable hard-constraint modules that can be attached to a backbone without making the backbone itself benchmark-specific.

The current design goal is:

- constraint logic lives here
- backbone construction lives in `integrations/`
- benchmark adapters decide when a constraint is used
- shared task runners only need to know how to pass `coords`, `fx`, and model outputs through the constraint

## Current Pieces

### `base.py`

Defines `ConstraintModule`, the base class for all constraint operators.

### `wrappers.py`

Defines the wrapper machinery used to attach constraints to arbitrary backbones:

- `ConstrainedModel`
- `ForwardHookLatentExtractor`
- `MeanConstraint`

`ConstrainedModel` wraps a backbone, runs the backbone normally, then applies the selected constraint to the backbone prediction. If a latent tensor is needed, `ForwardHookLatentExtractor` captures it from an internal module.

This is the main integration point used by `integrations/nsl/modeling.py`.

### `mean.py`

Contains utilities for mean-preserving constraints:

- `build_mlp`
- `match_mean`
- `MeanCorrection`

These support the original Navier-Stokes-style global mean correction idea.

### `boundary.py`

Contains Dirichlet architectural-ansatz utilities:

- `unit_box_distance(coords)`
- `constant_boundary_value(...)`
- `DirichletBoundaryAnsatz`
- `is_boundary_point(...)`
- `boundary_residual(...)`
- `boundary_stats(...)`

This is the current path for enforcing boundary conditions by construction.

## Supported Constraint Families

## Mean Correction

The mean-correction path is intended for invariants of the form:

```text
reduce(u_pred) = target value
```

Right now the reduction is a mean over spatial dimensions.

The implemented modes are:

- `post_output`
  Directly subtract the predicted mean.
- `post_output_learned`
  Predict a correction from the output, then align its mean before subtraction.
- `latent_head`
  Predict a correction from a latent representation captured by a forward hook.

This is suitable when the target manifold is easiest to enforce by a global correction after the backbone forward pass.

## Dirichlet Architectural Ansatz

The Dirichlet ansatz enforces

```text
u(x) = g(x) + l(x) * N(x)
```

where:

- `N(x)` is the unconstrained model output
- `g(x)` is the desired boundary value
- `l(x)` is a distance-like function that is zero on the boundary

The current implementation assumes a unit box domain and builds `l(x)` from the coordinates:

```text
l(x) = Π_i (x_i - lower) (upper - x_i)
```

or, if configured, the minimum across dimensions instead of the product.

Current assumptions:

- the domain is axis-aligned
- benchmark metadata provides the default box bounds
- coordinates lie inside that box, typically `[0, 1]^d` for the current structured benchmarks
- `g(x)` is currently a constant scalar boundary value

For the current 2D Darcy setup, this reduces to:

```text
l(x, y) = x (1 - x) y (1 - y)
```

because the benchmark adapter provides `domain_bounds = (0.0, 1.0)` and the loader builds coordinates on the unit square.

`axis-aligned` means the domain boundaries line up with the coordinate axes. In 2D, that means a rectangle whose sides are parallel to the `x` and `y` axes. The current ansatz is written for box domains of that form; it is not yet a general signed-distance function for curved or rotated domains.

## Config Interface

Constraints are selected through the top-level `constraint` section in the experiment config.

### Mean Correction Example
See [mean_correction.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/mean_correction.yaml):

```yaml
constraint:
  name: "mean_correction"
  mode: "latent_head"
  correction_hidden: 192
  correction_layers: 2
  correction_act: "tanh"
  channel_dim: -1
```

### Dirichlet Ansatz Example

See [dirichlet_ansatz_zero.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/dirichlet_ansatz_zero.yaml):

```yaml
constraint:
  name: "dirichlet_ansatz"
  boundary_value: 0.0
  distance_power: 1.0
  distance_reduce: "product"
```

For the current Darcy and Navier-Stokes loaders, `lower` and `upper` do not need to be specified in the experiment config because the benchmark metadata already declares `domain_bounds = (0.0, 1.0)`.

If a future benchmark lives on a different box, the preferred pattern is:

- the benchmark adapter sets `meta["domain_bounds"]`
- the constraint uses those bounds automatically
- the config only sets `constraint.lower` / `constraint.upper` when you want to override the benchmark default

## Normalization Caveat

For boundary-value constraints, the physically correct boundary condition must be enforced in physical space, not just in normalized target space.

That is why `DirichletBoundaryAnsatz` supports `set_target_normalizer(...)`.

In the steady-task runner, when a target normalizer exists, the ansatz encodes `g(x)` before combining it with the latent prediction. This ensures the decoded field still satisfies the intended physical boundary condition.

Without that step, a zero boundary value in physical space would incorrectly become a nonzero boundary value after decoding.

## Correctness Checks

Constraint correctness checks should live here, not inside benchmark training code, when they validate the constraint mechanism itself.

Current examples:

- `boundary_residual(...)`
- `boundary_stats(...)`
- tests in [test_boundary.py](/Users/bruno/Documents/Y4/FYP/omni_hc/tests/test_boundary.py)

Good rule:

- if the check asks “does the constraint mathematically enforce what it claims?”, it belongs in `constraints/` or in a generic test
- if the check asks “is this benchmark’s data consistent with the intended physical condition?”, it belongs in benchmark analysis/docs

## How New Constraints Should Be Added

When adding a new constraint:

1. Put the reusable math/operator in this folder.
2. Keep benchmark-specific assumptions out of the constraint where possible.
3. Add a config file under `configs/constraints/`.
4. Extend `integrations/nsl/modeling.py` so the config name maps to the constraint object.
5. Add direct tests for the constraint’s correctness.
6. Add benchmark-level metrics or logging only if needed to monitor satisfaction during training.

## Near-Term Direction

The next likely extensions are:

- non-constant Dirichlet profiles `g(x)`
- more general distance functions for non-rectangular domains
- Neumann or flux-based constraints
- explicit validation hooks that expose constraint violation metrics to W&B in a uniform way

The core principle should remain the same: benchmark adapters choose constraints, but the constraint implementations themselves stay reusable and testable in isolation.
