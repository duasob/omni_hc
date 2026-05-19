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

The physical pipe-flow target is still a two-component incompressible velocity
field plus pressure. For the full velocity field, incompressibility requires the
divergence-free constraint

$$
\nabla \cdot \mathbf{u}
= \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}
= 0.
$$

This matters when interpreting scalar $u_x$ experiments: a model trained only
against $u_x$ does not directly observe or penalize the dataset's $u_y$ channel.
Stream-function constraints can provide a divergence-free velocity completion
internally, but the training objective still supervises only the returned
$u_x$ unless the benchmark is changed to predict both velocity components or add
an auxiliary loss on $u_y$.

The benchmark defaults live in [`configs/benchmarks/pipe/base.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/benchmarks/pipe/base.yaml).

## Dataset Diagnostics

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

The velocity boundary profiles are therefore shown only for the non-trivial
inlet and outlet edges:

![pipe_ux_boundary_edge_profiles](../figures/pipe/pipe_ux_boundary_edge_profiles.png)

![pipe_uy_boundary_edge_profiles](../figures/pipe/pipe_uy_boundary_edge_profiles.png)

The top and bottom wall profiles are omitted from the plots because the no-slip
condition makes the edge rows flat zero lines. The tables also report the first
interior row next to each wall to show the near-wall scale:

| Boundary | Mean $u_x$ | Mean $|u_x|$ | Max $|u_x|$ | Min $u_x$ | Max $u_x$ | Std $u_x$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom wall (j=0) | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| Bottom first interior (j=1) | 7.6649e-03 | 7.6663e-03 | 2.9095e-02 | -2.8398e-03 | 2.9095e-02 | 3.4262e-03 |
| Top first interior (j=W-2) | 8.0617e-03 | 8.2772e-03 | 1.8101e+00 | -1.8101e+00 | 3.8942e-01 | 1.0773e-02 |
| Top wall (j=W-1) | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |

| Boundary | Mean $u_y$ | Mean $|u_y|$ | Max $|u_y|$ | Min $u_y$ | Max $u_y$ | Std $u_y$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottom wall (j=0) | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |
| Bottom first interior (j=1) | 5.7941e-04 | 2.4670e-03 | 1.6355e-02 | -1.1699e-02 | 1.6355e-02 | 3.2257e-03 |
| Top first interior (j=W-2) | 5.8771e-04 | 4.1196e-03 | 3.7753e+00 | -8.6606e-01 | 3.7753e+00 | 3.8234e-02 |
| Top wall (j=W-1) | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 |

## Hard Constraints

The pipe benchmark currently has five documented hard-constraint variants:

- [StructuredWallDirichletAnsatz](../constraints/boundary/StructuredWallDirichletAnsatz.md):
  wall-only direct-output ansatz for zero wall velocity.
- [PipeInletParabolicAnsatz](../constraints/boundary/PipeInletParabolicAnsatz.md):
  inlet-only direct-output ansatz for scalar $u_x$.
- [PipeUxBoundaryAnsatz](../constraints/boundary/PipeUxBoundaryAnsatz.md):
  combined inlet-plus-wall direct-output ansatz for scalar $u_x$.
- [PipeStreamFunctionUxConstraint](../constraints/stream/PipeStreamFunctionUxConstraint.md):
  stream-function construction that returns $u_x$ from an internally recovered
  divergence-free velocity field. Because the benchmark supervises only $u_x$,
  the recovered $u_y$ is not directly trained against the dataset.
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
