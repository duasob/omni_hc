# Navier-Stokes Example

This is the first benchmark slice to migrate from `hc_fluid`.

## Scope

The initial goal is not to port the entire old repo in one shot. The goal is to establish the pattern for all future benchmarks:

- benchmark metadata in `src/omni_hc/benchmarks`
- benchmark adapter in `src/omni_hc/benchmarks/navier_stokes/adapter.py`
- autoregressive task runner in `src/omni_hc/training/tasks/autoregressive.py`
- reusable constraints in `src/omni_hc/constraints`
- backend glue in `src/omni_hc/integrations/nsl`
- benchmark-specific run config in `configs/experiments/navier_stokes`

## Physics

For the periodic 2D Navier-Stokes setup, the first hard constraint is preservation of global vorticity. The old `hc_fluid` demo used a modular correction head attached to a backbone by a forward hook. That mechanism is the first reusable component already moved into the new core.

## Migration plan

1. Keep Neural-Solver-Library external and let OmniHC resolve it from config, env, or `external/Neural-Solver-Library`.
2. Use OmniHC scripts as the experiment harness, while NSL provides the backbone implementations.
3. Compare at least two backbones under the same Navier-Stokes harness.

The runtime is selected from `benchmark.name` in the config, so these same scripts should remain stable as other datasets are added.

## Current Commands

Train:

```bash
python scripts/train.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu
```

Test:

```bash
python scripts/test.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu
```

Optuna:

```bash
python scripts/tune.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu
```

Comparison configs:

- [fno_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/fno_small_mean.yaml)
- [gt_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/gt_small_mean.yaml)

Note:

- the current `gt_small_mean` smoke config forces `unified_pos: 0` because the upstream NSL positional embedding helper is not CPU-safe in its current form.
