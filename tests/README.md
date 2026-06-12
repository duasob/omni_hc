# Tests

Run the suite from the repo root:

```bash
python -m pytest tests/
```

The tests are CPU-only and need no datasets; they exercise constraint math on
synthetic tensors and config/IO plumbing on temporary files.

## Constraint correctness

- `test_boundary.py` — Dirichlet, structured-wall, inlet-parabolic, and
  combined pipe boundary ansatzes: exact boundary values, `g + l·N` structure,
  and emitted diagnostics.
- `test_sine_boundary.py` — sine-basis boundary prediction and hard
  enforcement on interior predictions.
- `test_darcy_flux.py` — flux-based pressure recovery and the hard-built
  continuity equation `div(v) = 1`.
- `test_stream.py` — stream-function constructions: divergence-free outputs
  and hard inlet/wall behaviour.
- `test_mean.py` — global mean preservation and the mean-matching correction.
- `test_elasticity.py` — plane-stress von Mises construction
  (`ElasticityPlaneStressVMConstraint`), including incompressibility of the
  recovered kinematics.
- `test_plasticity_mesh_consistency.py` — ordered mesh reconstruction from
  positive spacings and channel layout `[x, y, u_x, u_y]`.

## Configuration plumbing

- `test_config_composition.py` — component-name resolution, merge order,
  output-dir layout, and failure on unknown components.
- `test_constraint_config_wiring.py` — constraint YAML blocks reach the
  constraint constructors through `ConstrainedModel` building.

## Training utilities and runtime

- `test_benchmark_runtime.py` — every registered benchmark adapter resolves
  and exposes train/test/tune callables.
- `test_checkpoint_summary.py` — checkpoint bundle save/load round-trip.
- `test_logging_utils.py` — metric logging helpers.
- `test_reproducibility.py` — seeding determinism.

## Metrics, diagnostics, and reporting

- `test_metrics.py` — per-benchmark constraint metric functions.
- `test_dynamic_component_metrics.py` — per-component metrics of the dynamic
  conditional (plasticity) task.
- `test_boundary_maps.py` — inference of `g`/`l` boundary ansatz maps for
  diagnostics.
- `test_reporting.py` — the report table/macro generation pipeline under
  `scripts/reporting/`.
