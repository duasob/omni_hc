# Pipe Benchmark

This is the first `pipe` benchmark slice added to the shared OmniHC training flow.

## Scope

- benchmark metadata in `src/omni_hc/benchmarks`
- structured-grid pipe loader in `src/omni_hc/benchmarks/pipe/data.py`
- benchmark adapter in `src/omni_hc/benchmarks/pipe/adapter.py`
- shared steady-state task runner in `src/omni_hc/training/tasks/steady.py`
- benchmark defaults in `configs/benchmarks/pipe`
- experiment config in `configs/experiments/pipe`

## Current Commands

Unconstrained FNO baseline:

```bash
conda run -n omni-hc python scripts/train.py \
  --config configs/experiments/pipe/fno_small.yaml \
  --device cpu
```

```bash
conda run -n omni-hc python scripts/test.py \
  --config configs/experiments/pipe/fno_small.yaml \
  --device cpu
```

Wall-constrained FNO baseline:

```bash
conda run -n omni-hc python scripts/train.py \
  --config configs/experiments/pipe/fno_small_wall.yaml \
  --device cpu
```

```bash
conda run -n omni-hc python scripts/test.py \
  --config configs/experiments/pipe/fno_small_wall.yaml \
  --device cpu
```

Standalone inlet-constrained FNO baseline:

```bash
conda run -n omni-hc python scripts/train.py \
  --config configs/experiments/pipe/fno_small_inlet.yaml \
  --device cpu
```

```bash
conda run -n omni-hc python scripts/test.py \
  --config configs/experiments/pipe/fno_small_inlet.yaml \
  --device cpu
```

Combined inlet-and-wall `ux` boundary baseline:

```bash
conda run -n omni-hc python scripts/train.py \
  --config configs/experiments/pipe/fno_small_ux_boundary.yaml \
  --device cpu
```

```bash
conda run -n omni-hc python scripts/test.py \
  --config configs/experiments/pipe/fno_small_ux_boundary.yaml \
  --device cpu
```

## Boundary Checks

The pipe data is a structured curvilinear mesh. The physical boundary is best
identified by structured-grid index edges rather than by Cartesian coordinate
thresholds:

- `i=0`: inlet
- `i=N`: outlet
- `j=0`: lower wall
- `j=N`: upper wall

The current inspection script verifies the observed boundary values:

```bash
conda run -n omni-hc python scripts/diagnostics/pipe_boundary.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

For `Pipe_Q` channel `0` (`ux`) and channel `1` (`uy`), the wall edges
`j=0` and `j=N` are exactly zero in the benchmark data. The inlet and outlet are
not zero, and pressure is not a zero-wall target.

The inlet `ux` profile can be checked with:

```bash
conda run -n omni-hc python scripts/diagnostics/pipe_inlet_profile.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

Despite the script name, this check showed that the inlet `ux` profile is
exactly a centered wall-zero parabola over the normalized inlet coordinate:

```text
ux(t) = 0.25 * 4 * t * (1 - t)
```

where `t = (y - y_min) / (y_max - y_min)` on the inlet edge.

## Structured Wall Constraint

The wall-constrained experiment uses `StructuredWallDirichletAnsatz` through
[pipe_wall_no_slip.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_wall_no_slip.yaml):

```yaml
constraint:
  name: "pipe_wall_no_slip"
  boundary_value: 0.0
  transverse_axis: 1
  distance_power: 1.0
  normalize_distance: true
```

This applies the architectural ansatz only along the transverse mesh axis,
which is the `j` direction for the pipe arrays. For the current scalar loader,
this is appropriate for:

- `data.target_channel: 0` (`ux`)
- `data.target_channel: 1` (`uy`)

It should not be used unchanged for `data.target_channel: 2` (`p`), because the
pressure channel is not zero on the pipe walls.

## Parabolic Inlet Constraint

The standalone inlet-constrained experiment uses `PipeInletParabolicAnsatz`
through [pipe_inlet_parabolic.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_inlet_parabolic.yaml):

```yaml
constraint:
  name: "pipe_inlet_parabolic"
  amplitude: 0.25
  inlet_axis: 0
  transverse_axis: 1
  coordinate_channel: 1
  decay_power: 4.0
```

It enforces the discovered inlet profile with a smooth ansatz:

```text
u = g + lN
g(i,j) = alpha(i) * 0.25 * 4t(j)(1-t(j))
l(i,j) = 1 - alpha(i)
alpha(i) = (1 - xi(i))^p
```

At the inlet edge `i=0`, `alpha=1`, so the prediction is exactly the parabolic
profile. Away from the inlet, `alpha` decays and the backbone output takes over.
The coordinate `t` is computed from decoded physical inlet `Y`, so it remains
valid when `Pipe_Y` changes across samples and when model inputs are normalized.

## Combined Ux Boundary Constraint

The combined `ux` boundary experiment uses `PipeUxBoundaryAnsatz` through
[pipe_ux_boundary.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_ux_boundary.yaml):

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

It uses the same parabolic inlet extension for `g`, but makes the multiplicative
distance zero on both the inlet and the walls:

```text
u = g + lN
g(i,j) = alpha(i) * 0.25 * 4t(j)(1-t(j))
l(i,j) = (1 - alpha(i)) * wall_distance(j)
alpha(i) = (1 - xi(i))^p
```

Therefore the scalar `ux` output satisfies:

- `i=0`: parabolic inlet profile exactly
- `j=0` and `j=N`: no-slip wall value exactly

## Notes

- `fno_small.yaml` is the plain FNO baseline under the common OmniHC harness.
- `fno_small_wall.yaml` adds the hard wall no-slip constraint for scalar velocity targets.
- `fno_small_inlet.yaml` adds the standalone hard parabolic inlet constraint for scalar `ux`.
- `fno_small_ux_boundary.yaml` adds the combined hard inlet and wall constraint for scalar `ux`.
- The upstream NSL pipe baseline feeds the normalized `Pipe_X` and `Pipe_Y` tensors as both position input and feature input. This loader keeps that behavior for compatibility.
- `data.target_channel` selects which `Pipe_Q` channel becomes the scalar target field. The default is `0`, matching the upstream pipe loader.
