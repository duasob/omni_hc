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
  --constraint darcy_flux_fft_pad \
  --budget debug
```

Names resolve to config files by convention:

```text
--benchmark darcy
  -> configs/benchmarks/darcy/base.yaml

--backbone FNO
  -> configs/backbones/darcy/FNO.yaml

--constraint darcy_flux_fft_pad
  -> configs/constraints/darcy_flux_fft_pad.yaml

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
  --constraint darcy_flux_fft_pad \
  --budget tune_debug \
  --optuna darcy_flux_fft_pad
```

If `--optuna` is omitted, `tune.py` looks for a search-space config matching
the constraint first, then the backbone.

## Experiments

Experiment files are optional named recipes. They can specify the same
components as flags and then apply final overrides:

```yaml
name: "darcy_fno_flux_fft_pad"
benchmark: "darcy"
backbone: "FNO"
constraint: "darcy_flux_fft_pad"
budget: "debug"
optuna: "darcy_flux_fft_pad"

overrides:
  paths:
    output_dir: "outputs/darcy/fno_flux_fft_pad"
  wandb_logging:
    run_name: "darcy_fno_flux_fft_pad"
```

Run an experiment recipe with:

```bash
python scripts/train.py --config configs/experiments/darcy/fno_small_flux_fft_pad.yaml
python scripts/tune.py --config configs/experiments/darcy/fno_small_flux_fft_pad.yaml --budget tune_debug
python scripts/test.py --config configs/experiments/darcy/fno_small_flux_fft_pad.yaml
```

CLI flags override the experiment component names when provided.

## Sweep Scripts

Batch launch is handled by shell scripts under `scripts/sweeps/`. They are
plain loops over component names:

```bash
scripts/sweeps/darcy_transformers_train.sh
scripts/sweeps/darcy_transformers_tune.sh
scripts/sweeps/plasticity_backbones_train.sh
```

Override sweep settings with environment variables:

```bash
BUDGET=smoke SEEDS="1 2 3" DEVICE=cuda scripts/sweeps/darcy_transformers_train.sh
```

For Colab tuning, set `OMNI_HC_OUTPUT_ROOT` to a Drive-backed directory if you
want durable outputs:

```bash
OMNI_HC_OUTPUT_ROOT=/content/drive/MyDrive/omni_hc \
  BUDGET=tune_colab \
  scripts/sweeps/darcy_transformers_tune.sh
```

Training and tuning use validation metrics only. The canonical held-out test
split is `data.ntest` in the resolved config and should only be evaluated with
`scripts/test.py` after model selection.
