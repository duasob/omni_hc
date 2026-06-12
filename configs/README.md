# Config Layout

Runs are composed from four source components:

- `benchmarks/`: dataset, geometry, normalization, and canonical split metadata
- `backbones/<benchmark>/`: benchmark-specific model and optimizer defaults
- `constraints/`: hard-constraint defaults
- `budgets/`: runtime scale, seeds, batch sizes, epochs, Optuna trial counts, and logging policy

`scripts/train.py`, `scripts/tune.py`, and `scripts/test.py` compose those
components directly. The common path is to pass component names:

```bash
python scripts/train.py \
  --benchmark darcy \
  --backbone FNO \
  --constraint darcy_flux_constraint \
  --budget debug
```

Names resolve to config files by convention:

```text
--benchmark darcy
  -> configs/benchmarks/darcy/base.yaml

--backbone FNO
  -> configs/backbones/darcy/FNO.yaml

--constraint darcy_flux_constraint
  -> configs/constraints/darcy_flux_constraint.yaml

--budget debug
  -> configs/budgets/debug.yaml
```

Use `--constraint none` or omit `--constraint` for unconstrained runs.

## Ownership

Benchmark configs should not define training schedules, optimizer settings,
W&B logging, or Optuna search spaces. They identify the problem and data.

Backbone configs own model defaults and any optimizer defaults that belong to a
specific benchmark/backbone pair.

Budget configs make a run cheap or expensive. They can override sample counts,
validation size, epochs, batch sizes, seeds, W&B behavior, and Optuna trial
counts.

Optuna search spaces live under `configs/optuna/`. `tune.py` composes them in
addition to the train-time components:

```bash
python scripts/tune.py \
  --benchmark darcy \
  --backbone Galerkin_Transformer \
  --constraint darcy_flux_constraint \
  --budget debug \
  --optuna darcy_flux_constraint
```

If `--optuna` is omitted, `tune.py` looks for a search-space config matching
the constraint first, then the backbone.

## Experiments

Experiment files are optional named recipes. They can specify the same
components as flags and then apply final overrides:

```yaml
name: "darcy_fno_flux"
benchmark: "darcy"
backbone: "FNO"
constraint: "darcy_flux_constraint"
budget: "debug"
optuna: "darcy_flux_constraint"

overrides:
  paths:
    output_dir: "outputs/darcy/fno_flux"
  wandb_logging:
    run_name: "darcy_fno_flux"
```

Run an experiment recipe with:

```bash
python scripts/train.py --config configs/experiments/darcy/fno_small_flux.yaml
python scripts/tune.py --config configs/experiments/darcy/fno_small_flux.yaml --budget debug
python scripts/test.py --config configs/experiments/darcy/fno_small_flux.yaml
```

CLI flags override the experiment component names when provided.

## Batch Runs

Batches of runs are dispatched from run-list text files under `experiments/`
with `scripts/run_batch.sh` (locally) or `scripts/hpc/runner.sh` (PBS cluster):

```bash
RUNS_FILE=experiments/darcy/transolver_unconstrained_data.txt scripts/run_batch.sh
```

For Colab, set `OMNI_HC_OUTPUT_ROOT` to a Drive-backed directory if you want
durable outputs.

Training and tuning use validation metrics by default. If a benchmark supports
`training.val_size: 0`, the full training subset is used for optimization and
`best.pt` is selected by the training loss instead. The canonical held-out test
split is `data.ntest` in the resolved config and should only be evaluated with
`scripts/test.py` after model selection.
