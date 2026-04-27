# Pipe Flow Benchmark

![pipe_boundary_sample_0100](../figures/pipe/pipe_boundary_sample_0100.png)

This benchmark is a steady internal-flow task on a structured curvilinear pipe
mesh. The input is the two-channel coordinate field and the raw dataset target
contains three output channels:

- `Pipe_Q[..., 0]`: $u_x$
- `Pipe_Q[..., 1]`: $u_y$
- `Pipe_Q[..., 2]`: $p$

The default setup from the original benchmark uses `target_channel: 0`, so the main
pipe experiments in this repo focus on $u_x$.

The benchmark defaults live in [`configs/benchmarks/pipe/base.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/benchmarks/pipe/base.yaml).

## Grid And Boundary Layout

![pipe_dataset_boundary_overview](../figures/pipe/pipe_dataset_boundary_overview.png)

The pipe data is a structured curvilinear mesh. The meaningful boundaries are
therefore the structured-grid index edges rather than fixed Cartesian
thresholds:

- $i=0$: inlet
- $i=H-1$: outlet
- $j=0$: lower wall
- $j=W-1$: upper wall

That distinction matters because some constraints should be expressed in index
space, while others should use decoded physical coordinates from the curvilinear
mesh.

## Constraint Variants

The pipe benchmark currently has five documented hard-constraint variants:

- [StructuredWallDirichletAnsatz](../constraints/boundary/StructuredWallDirichletAnsatz.md):
  wall-only direct-output ansatz for zero wall velocity.
- [PipeInletParabolicAnsatz](../constraints/boundary/PipeInletParabolicAnsatz.md):
  inlet-only direct-output ansatz for scalar $u_x$.
- [PipeUxBoundaryAnsatz](../constraints/boundary/PipeUxBoundaryAnsatz.md):
  combined inlet-plus-wall direct-output ansatz for scalar $u_x$.
- [PipeStreamFunctionUxConstraint](../constraints/stream/PipeStreamFunctionUxConstraint.md):
  divergence-free stream-function construction that returns $u_x$.
- [PipeStreamFunctionBoundaryAnsatz](../constraints/stream/PipeStreamFunctionBoundaryAnsatz.md):
  divergence-free stream-function construction with hard inlet and wall
  behavior.

The corresponding experiment configs live under
[`configs/experiments/pipe/`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/pipe),
and they can all be run through the shared commands documented in
[here](../README.md).

## Dataset Checks

Inspect the observed boundary values with:

```bash
python scripts/diagnostics/pipe_boundary.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

Inspect the inlet profile with:

```bash
python scripts/diagnostics/pipe_inlet_profile.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

For a broader dataset summary:

```bash
python scripts/diagnostics/pipe_dataset_summary.py \
  --summary-samples 1000
```

For a discrete finite-volume divergence check on the curvilinear cells:

```bash
python scripts/diagnostics/pipe_divergence.py \
  --summary-samples 1000
```

These checks support the current hard-constraint choices:

- both wall components are zero on $j=0$ and $j=W-1$
- inlet $u_x$ follows the fixed parabola $0.25 \cdot 4t(1-t)$
- inlet $u_y$ is zero in the inspected data slice
- outlet values remain sample-dependent
- the dataset is not uniformly pointwise divergence-free under the current
  cell-based finite-volume check: many samples are close, but some show
  localized spikes
