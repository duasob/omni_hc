# Constraints
This folder contains reusable hard-constraint modules that can be attached to a backbone without making the backbone itself benchmark-specific.

The current design goal is:

- constraint logic lives here
- backbone construction lives in `integrations/`
- benchmark adapters decide when a constraint is used
- shared task runners only need to know how to pass `coords`, `fx`, and model outputs through the constraint

## Current Pieces

### `base.py`

Defines the constraint framework primitives:

- `ConstraintDiagnostic`
- `ConstraintOutput`
- `ConstraintModule`
- `ConstrainedModel`

`ConstrainedModel` wraps a backbone, runs the backbone normally, then applies
the selected constraint to the backbone prediction.

This is the main integration point used by `integrations/nsl/modeling.py`.

### `mean.py`

Contains mean-preserving constraints and their helper functions:

- `build_mlp`
- `match_mean`
- `MeanConstraint`
- `MeanCorrection`

These support the original Navier-Stokes-style global mean correction idea.

### `utils/`

Contains implementation helpers that are not themselves public constraint
families:

- `utils/hooks.py`: `ForwardHookLatentExtractor`
- `utils/spectral.py`: spectral and finite-difference operators
- `utils/boundary_ops.py`: common `f = g + l * N` ansatz helpers
- `utils/structured_grid.py`: structured-grid axis, edge, and shape helpers
- `utils/stream_ops.py`: stream-function velocity and curvilinear divergence helpers

### `boundary.py`

Contains Dirichlet architectural-ansatz utilities:

- `unit_box_distance(coords)`
- `constant_boundary_value(...)`
- `DirichletBoundaryAnsatz`
- `PipeInletParabolicAnsatz`
- `PipeUxBoundaryAnsatz`
- `structured_wall_distance(...)`
- `structured_wall_mask(...)`
- `structured_wall_stats(...)`
- `StructuredWallDirichletAnsatz`
- `is_boundary_point(...)`
- `boundary_residual(...)`
- `boundary_stats(...)`

This is the current path for enforcing boundary conditions by construction.
The direct output-space classes share a common implementation pattern:

```text
f = g + l * N
```

where each class is responsible for constructing its own particular field `g`
and boundary-distance field `l`, then delegates the composition to
`utils/boundary_ops.py`.

### `stream.py`

Contains stream-function constraints:

- `PipeStreamFunctionUxConstraint`
- `PipeStreamFunctionBoundaryAnsatz`

These classes interpret the backbone output as a latent stream function before
recovering a physical output field. Pure stream-function operators live in
`utils/stream_ops.py`.

### `darcy_flux.py`

Contains the Darcy-specific Helmholtz stream-function wrapper constraint:

- `DarcyFluxConstraint`

The current implementation predicts a scalar stream function `psi`:

- the backbone predicts a 1-channel latent stream function
- the wrapper decodes the physical permeability field
- `v_corr = grad_perp(psi)` is built spectrally on a padded grid
- a fixed particular field `v_part` supplies the forcing `div(v_part) = 1`
- the cropped physical flux is converted to a gradient candidate `w = -v / a`
- a Dirichlet-aware sine Poisson solve recovers a scalar pressure field
- the recovered pressure is re-encoded to match the training target space

This keeps the backbone generic while moving the Darcy-specific operator logic
into the constraint layer.

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

## Structured Wall Dirichlet Ansatz

`StructuredWallDirichletAnsatz` enforces a constant value on two opposite
index-space walls of a structured 2D mesh:

```text
u = g + l(j) * N
```

where:

- `N` is the unconstrained model output
- `g` is the constant wall value, encoded through the target normalizer when one exists
- `l(j)` is an index-space distance profile that is zero on the selected wall edges

This is different from `DirichletBoundaryAnsatz`:

- it does not detect boundaries from physical coordinates
- it does not assume a Cartesian unit box in physical space
- it uses the structured grid shape `(H, W)` and a selected mesh axis

For a 2D grid with shape `(H, W)` and `transverse_axis: 1`, the distance profile is:

```text
eta_j = j / (W - 1)
l(j) = 4 eta_j (1 - eta_j)
```

when `normalize_distance: true`. Therefore `l(0) = l(W - 1) = 0`, and the
network output is exactly replaced by the encoded wall value on those edges.

This is the current hard no-slip path for the pipe benchmark. The pipe mesh is
curvilinear, and `Pipe_Y` changes across samples, but the no-slip wall is tied
to the structured mesh edges `j=0` and `j=N`, not to a fixed Cartesian `y`
coordinate. Using index-space distance keeps the constraint valid under
coordinate normalization and sample-varying geometry.

Supported config fields:

- `name`: accepts `structured_wall_dirichlet`, `structured_wall_dirichlet_ansatz`, `pipe_wall_no_slip`, or `pipe_wall_no_slip_ansatz`
- `boundary_value`: constant physical wall value
- `transverse_axis`: the structured-grid axis whose first and last indices are constrained; for pipe this is `1`
- `distance_power`: optional exponent applied to the wall distance
- `normalize_distance`: whether to scale the maximum distance to one
- `channel_indices`: optional list of output channels to constrain when the model predicts multiple channels

Implementation details:

- `integrations/nsl/modeling.py` passes `args.shapelist` into the constraint as `grid_shape`
- the steady task runner calls `set_grid_shape(...)` from benchmark metadata when available
- the steady task runner calls `set_target_normalizer(...)`, so the physical wall value remains correct after target decoding
- diagnostics are emitted through `return_aux=True` and flow into W&B through the existing metric path

Current diagnostics include:

- `constraint/wall_abs_mean`
- `constraint/wall_abs_max`
- `constraint/wall_base_abs_mean`
- `constraint/wall_base_abs_max`
- `constraint/wall_base_lower_abs_mean`
- `constraint/wall_base_upper_abs_mean`
- `constraint/interior_abs_delta_mean`
- `constraint/wall_distance_min`
- `constraint/wall_distance_max`
- `constraint/wall_distance_mean`

For the current scalar pipe loader, this constraint is appropriate for
`data.target_channel: 0` (`ux`) and `data.target_channel: 1` (`uy`). It is not
appropriate unchanged for `data.target_channel: 2` (`p`), because pressure is
not zero on the walls.

## Pipe Inlet Parabolic Ansatz

`PipeInletParabolicAnsatz` enforces the discovered pipe inlet `ux` profile with
a smooth extension into the domain:

```text
u = g + lN
g(i,j) = alpha(i) * Umax * 4t(j)(1-t(j))
l(i,j) = 1 - alpha(i)
alpha(i) = (1 - xi(i))^p
```

At the inlet edge, `xi=0`, so `alpha=1`, `l=0`, and the output is exactly:

```text
u(0,j) = Umax * 4t(j)(1-t(j))
```

Away from the inlet, `alpha` decays smoothly and the backbone output takes over.
For pipe, `t` is computed per sample from the decoded physical inlet `Y`
coordinate:

```text
t = (y - y_min) / (y_max - y_min)
```

This matters because `Pipe_Y` changes across samples and the training loader can
normalize coordinates before the model sees them. The constraint implements
`set_input_normalizer(...)` so it can decode coordinates before computing `t`.

Supported config fields:

- `name`: accepts `pipe_inlet_parabolic`, `pipe_inlet_parabolic_ansatz`, or `structured_inlet_parabolic`
- `amplitude`: inlet peak velocity `Umax`; the benchmark inspection found `0.25`
- `inlet_axis`: structured-grid axis for inlet-to-outlet direction; for pipe this is `0`
- `transverse_axis`: structured-grid transverse axis; for pipe this is `1`
- `coordinate_channel`: coordinate channel used to compute `t`; for pipe `Y` is channel `1`
- `decay_power`: exponent `p` in `alpha=(1-xi)^p`
- `channel_indices`: optional list of output channels to constrain when the model predicts multiple channels

Current diagnostics include:

- `constraint/inlet_abs_mean`
- `constraint/inlet_abs_max`
- `constraint/inlet_base_abs_mean`
- `constraint/inlet_base_abs_max`
- `constraint/inlet_profile_amplitude`
- `constraint/inlet_profile_mean`
- `constraint/inlet_profile_max`
- `constraint/inlet_alpha_mean`
- `constraint/inlet_alpha_min`
- `constraint/inlet_alpha_max`
- `constraint/inlet_decay_power`

This is a standalone inlet constraint. It does not enforce the wall no-slip
condition except at the inlet corners, where the parabolic profile is already
zero.

## Pipe Ux Boundary Ansatz

`PipeUxBoundaryAnsatz` combines the parabolic inlet profile and the no-slip
walls into one scalar `ux` trial function:

```text
u = g + lN
g(i,j) = alpha(i) * Umax * 4t(j)(1-t(j))
l(i,j) = (1 - alpha(i)) * wall_distance(j)
alpha(i) = (1 - xi(i))^p
```

This gives the exact boundary behavior:

- at `i=0`, `alpha=1` and `l=0`, so `u` is the parabolic inlet profile
- at `j=0` and `j=N`, `wall_distance=0`, and the parabolic profile is also zero at the wall corners, so `u=0`

The class inherits the inlet-coordinate handling from `PipeInletParabolicAnsatz`,
including decoded physical `Y` coordinates for computing `t`. The wall distance
uses the same structured-grid index-space distance as `StructuredWallDirichletAnsatz`.

Supported config fields:

- `name`: accepts `pipe_ux_boundary`, `pipe_ux_boundary_ansatz`, `pipe_inlet_wall`, or `pipe_inlet_wall_ansatz`
- `amplitude`: inlet peak velocity `Umax`
- `inlet_axis`: structured-grid inlet-to-outlet axis; for pipe this is `0`
- `transverse_axis`: structured-grid wall-to-wall axis; for pipe this is `1`
- `coordinate_channel`: coordinate channel used to compute `t`; for pipe this is `1`
- `inlet_decay_power`: exponent `p` in `alpha=(1-xi)^p`
- `wall_distance_power`: optional exponent on the wall distance
- `normalize_wall_distance`: whether to scale wall distance to have maximum one
- `channel_indices`: optional list of constrained output channels

Current diagnostics include both inlet and wall satisfaction metrics:

- `constraint/inlet_abs_mean`
- `constraint/inlet_abs_max`
- `constraint/inlet_base_abs_mean`
- `constraint/inlet_base_abs_max`
- `constraint/wall_abs_mean`
- `constraint/wall_abs_max`
- `constraint/wall_base_abs_mean`
- `constraint/wall_base_abs_max`
- `constraint/boundary_distance_min`
- `constraint/boundary_distance_max`
- `constraint/boundary_distance_mean`

## Darcy Flux Projection

For Darcy flow, the PDE is

```text
-div(a grad u) = 1
```

The current implementation reformulates this through the flux

```text
v = -a grad u
```

so that

```text
div(v) = 1
```

Instead of predicting the full flux field directly, the backbone predicts a
scalar stream function `psi`. The wrapper then:

1. pads `psi` on a larger computational box
2. builds a divergence-free correction
   ```text
   v_corr = grad_perp(psi)
   ```
3. adds a fixed particular field `v_part` with `div(v_part) = 1`
4. crops back to the physical Darcy grid and forms
   ```text
   v_valid = v_part + v_corr
   ```
5. computes
   ```text
   w = -v_valid / a
   ```
6. recovers `u` through a sine-transform Dirichlet Poisson solve

What this does enforce well:

- `v_corr` is divergence-free by construction on the padded spectral domain
- the recovered pressure satisfies the constant Dirichlet boundary value exactly
  after the sine solve

What this does not yet hard-enforce:

- `w = -v_valid / a` being exactly curl-free / integrable everywhere
- the full Darcy residual being identically zero on the discrete grid

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

### Darcy Flux Projection Example

See [darcy_flux_fft_pad.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/darcy_flux_fft_pad.yaml):

```yaml
constraint:
  name: "darcy_flux_projection"
  spectral_backend: "helmholtz_sine"
  pressure_out_dim: 1
  force_value: 1.0
  padding: 8
  padding_mode: "reflect"
  particular_field: "y_only"
  enforce_boundary: true
  boundary_value: 0.0
```

For the current Darcy and Navier-Stokes loaders, `lower` and `upper` do not need to be specified in the experiment config because the benchmark metadata already declares `domain_bounds = (0.0, 1.0)`.

If a future benchmark lives on a different box, the preferred pattern is:

- the benchmark adapter sets `meta["domain_bounds"]`
- the constraint uses those bounds automatically
- the config only sets `constraint.lower` / `constraint.upper` when you want to override the benchmark default

### Structured Wall Dirichlet Example

See [pipe_wall_no_slip.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_wall_no_slip.yaml):

```yaml
constraint:
  name: "pipe_wall_no_slip"
  boundary_value: 0.0
  transverse_axis: 1
  distance_power: 1.0
  normalize_distance: true
```

This is used by [fno_small_wall.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/pipe/fno_small_wall.yaml) for the pipe `ux` baseline.

The corresponding benchmark-level inspection scripts are:

```bash
python scripts/inspect_pipe_boundary.py --samples 0 10 100 --summary-samples 1000
python scripts/inspect_pipe_inlet_gaussian.py --samples 0 10 100 --summary-samples 1000
```

The inlet script compares Gaussian-like candidates against wall-zero parabolic
profiles. For the first 1000 pipe samples, the inlet `ux` profile is exactly:

```text
ux(t) = 0.25 * 4 * t * (1 - t)
```

with `t = (y - y_min) / (y_max - y_min)` on the inlet edge. This inlet condition
is not currently hard-enforced by `StructuredWallDirichletAnsatz`; it is a
separate standalone constraint via `PipeInletParabolicAnsatz`.

### Pipe Inlet Parabolic Example

See [pipe_inlet_parabolic.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_inlet_parabolic.yaml):

```yaml
constraint:
  name: "pipe_inlet_parabolic"
  amplitude: 0.25
  inlet_axis: 0
  transverse_axis: 1
  coordinate_channel: 1
  decay_power: 4.0
```

This is used by [fno_small_inlet.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/pipe/fno_small_inlet.yaml) for a standalone pipe `ux` inlet-profile baseline.

### Pipe Ux Boundary Example

See [pipe_ux_boundary.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_ux_boundary.yaml):

```yaml
constraint:
  name: "pipe_ux_boundary"
  amplitude: 0.25
  inlet_axis: 0
  transverse_axis: 1
  coordinate_channel: 1
  inlet_decay_power: 4.0
  wall_distance_power: 1.0
  normalize_wall_distance: true
```

This is used by [fno_small_ux_boundary.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/pipe/fno_small_ux_boundary.yaml) for a scalar pipe `ux` baseline that enforces both the parabolic inlet and no-slip walls.

## Normalization Caveat

For boundary-value constraints, the physically correct boundary condition must be enforced in physical space, not just in normalized target space.

That is why `DirichletBoundaryAnsatz` and `StructuredWallDirichletAnsatz`
support `set_target_normalizer(...)`.

In the steady-task runner, when a target normalizer exists, the ansatz encodes
the boundary value before combining it with the latent prediction. This ensures
the decoded field still satisfies the intended physical boundary condition.

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
- stronger integrability control for `w = -v / a`
- alternative particular fields and padding strategies for Darcy
- explicit validation hooks that expose constraint violation metrics to W&B in a uniform way

The core principle should remain the same: benchmark adapters choose constraints, but the constraint implementations themselves stay reusable and testable in isolation.
