# Architecture

OmniHC wraps generic neural-PDE backbones with hard-constraint modules so that
predictions satisfy chosen physical invariants by construction. The repository
is organised so that benchmarks, backbones, constraints, and training budgets
are independent components that compose into a single run config.

## Package layout

```
src/omni_hc/
├── core/          # YAML loading, deep-merge, run-config composition
├── benchmarks/    # one adapter per benchmark + BENCHMARKS registry
├── constraints/   # ConstrainedModel, ConstraintModule, all constraint families
│   ├── metrics/   # per-benchmark constraint diagnostics for evaluation
│   └── utils/     # boundary ops, stream ops, spectral helpers, hooks
├── training/      # train/test/tune runners and task loops
│   └── tasks/     # steady, autoregressive, dynamic_conditional
├── integrations/
│   └── nsl/       # bridge to the vendored Neural-Solver-Library backbones
└── diagnostics/   # boundary ansatz map inference
```

Supporting trees: `configs/` (component YAML files), `scripts/` (entry points,
diagnostics, reporting), `external/Neural-Solver-Library/` (vendored backbone
library), `tests/`, `docs/`, `scripts/notebooks/` (jupytext figure notebooks).

## Config composition (`omni_hc.core`)

Every run is described by four named components, each a YAML file:

| Component | Resolves to | Provides |
| :-- | :-- | :-- |
| benchmark | `configs/benchmarks/<name>/base.yaml` | dataset, geometry, task |
| backbone | `configs/backbones/<benchmark>/<name>.yaml` | model and optimiser defaults |
| constraint | `configs/constraints/[<benchmark>/]<name>.yaml` | hard-constraint spec (or `none`) |
| budget | `configs/budgets/<name>.yaml` | epochs, batch size, seed, logging |

`compose_run_config` (`src/omni_hc/core/composition.py`) resolves each name by
convention, deep-merges the files in order, then applies experiment-recipe
overrides (`configs/experiments/.../*.yaml`) and CLI `--override` values last.
The resolved config records its provenance under `experiment.source_configs`
and derives the output directory
`outputs/<benchmark>/<constraint>/<backbone>/<budget>/seed_<n>/`. If that
directory already holds a `resolved_config.yaml` from a *different* experiment,
a timestamped subfolder is used instead of overwriting the run. Tune mode adds
an `optuna` component (`configs/optuna/`).

## Benchmarks (`omni_hc.benchmarks`)

Each benchmark subpackage (`darcy`, `navier_stokes`, `pipe`, `elasticity`,
`plasticity`) provides an `adapter.py` (train/test/tune/log functions) and a
`data.py` (dataset loading). Adapters are registered in the `BENCHMARKS` dict
(`src/omni_hc/benchmarks/__init__.py`) as `BenchmarkAdapter` instances
(`src/omni_hc/benchmarks/base.py`); the training runner looks up the adapter
named by `benchmark.name` in the config, so adding a benchmark means writing an
adapter and registering it.

## Constraints (`omni_hc.constraints`)

`ConstrainedModel` (`src/omni_hc/constraints/base.py`) wraps any backbone and
applies a `ConstraintModule` to its raw output. Constraint modules return a
`ConstraintOutput` carrying the projected prediction plus optional auxiliary
tensors, named diagnostics, and an optional extra loss term. Constraints are
constructed from their YAML block via the `ConstraintModule.build` classmethod
and matched by snake-case name in `_CONSTRAINT_CLASSES`
(`src/omni_hc/integrations/nsl/modeling.py`). The implemented families are
documented under [docs/constraints](constraints/README.md).

## Training (`omni_hc.training`)

`runner.py` exposes `train_benchmark` / `test_benchmark` / `tune_benchmark`,
which dispatch through the benchmark adapter into a task loop under
`training/tasks/`: `steady` (single-shot prediction), `autoregressive`
(rollout in time), and `dynamic_conditional` (time-conditioned prediction).
Shared utilities cover checkpointing (`common.py`), W&B logging
(`logging_utils.py`), seeding (`reproducibility.py`), and Optuna search
(`search.py`, `optuna_utils.py`).

## Backbone integration (`omni_hc.integrations.nsl`)

Backbones come from the vendored [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library)
at `external/Neural-Solver-Library` (override with `--nsl-root` or
`OMNI_HC_NSL_ROOT`). `build_model_args` merges NSL defaults with the config's
`model.args`, and `create_model` instantiates the backbone and wraps it in a
`ConstrainedModel` when a constraint is configured.

## Entry points

`scripts/train.py`, `scripts/test.py`, and `scripts/tune.py` parse the shared
CLI flags, call `compose_run_config`, and dispatch to the runner.
`scripts/diagnose.py` reports parameter and FLOP counts. Dataset- and
constraint-level checks live in `scripts/diagnostics/`, batch dispatch in
`scripts/run_batch.sh` + `scripts/hpc/`, and the report table/macro pipeline in
`scripts/reporting/` (`python -m scripts.reporting.build`).
