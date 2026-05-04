# Config Layout

Configs are split by responsibility:

- `benchmarks/`: dataset and benchmark defaults
- `backbones/`: model-family defaults
- `constraints/`: hard-constraint defaults
- `experiments/<benchmark>/optuna/`: benchmark and constraint-specific search spaces

A run config can compose these layers with `extends`.

Backbone configs copied from Neural-Solver-Library StandardBench scripts live
under:

```text
configs/backbones/<benchmark>/<model>.yaml
```

This keeps benchmark-specific baseline hyperparameters separate. The same model
can have different trusted defaults for Darcy, Pipe, Elasticity, and the other
StandardBench datasets without overwriting another benchmark's config.

The intended entrypoints are:

- `python scripts/train.py --config ...`
- `python scripts/test.py --config ...`
- `python scripts/tune.py --config ...`

The runtime is selected from `benchmark.name` inside the resolved config.

## Batch Debug Workflow

Budget configs under `configs/budgets/` keep debug, smoke, and search runs
small and repeatable. Sweep configs under `configs/sweeps/` reference trusted
experiment configs, usually initialized from the known-good baseline settings.
Each sweep run can either reference one existing `config` or compose multiple
config layers with `extends`.

Budgets own compute size: sample counts, epochs, batch sizes, and Optuna trial
counts. They should not define Optuna search spaces. Search spaces belong in
benchmark experiment configs, for example:

```text
configs/experiments/darcy/optuna/darcy_flux_fft_pad.yaml
```

Training and tuning runs must use validation metrics only. The canonical test
split is the held-out 200 samples defined by `data.ntest: 200`, and it should
only be evaluated with `scripts/test.py` after final model selection. Budget
configs may reduce `data.ntrain` for cheaper debug/search runs, but they should
not reduce `data.ntest`.

For final selected runs, use the `final` budget: it sets `data.ntrain: 1000`
and `training.val_size: 100`, so the training task fits on 900 samples and
selects checkpoints on 100 validation samples. The last 200 samples remain
reserved for the final test pass.

Preview a sweep without launching training:

```bash
python scripts/batch_train.py \
  --sweep configs/sweeps/darcy_flux.yaml \
  --budget debug \
  --dry-run
```

Preview the Darcy transformer baseline sweep:

```bash
python scripts/batch_train.py \
  --sweep configs/sweeps/darcy_transformers.yaml \
  --budget debug_transformer \
  --dry-run
```

Run the same sweep:

```bash
python scripts/batch_train.py \
  --sweep configs/sweeps/darcy_flux.yaml \
  --budget debug
```

For Colab tuning, prefer Drive-backed outputs. `batch_train.py` and
`batch_tune.py` default to `/content/drive/MyDrive/omni_hc/...` when running in
Colab, or you can set `OMNI_HC_OUTPUT_ROOT` explicitly.

Run a tiny Optuna wiring check:

```bash
python scripts/batch_tune.py \
  --sweep configs/sweeps/darcy_transformers.yaml \
  --budget tune_debug \
  --only galerkin_transformer \
  --dry-run
```

Run a Colab-sized batch tune:

```bash
python scripts/batch_tune.py \
  --sweep configs/sweeps/darcy_transformers.yaml \
  --budget tune_colab \
  --continue-on-failure
```

Generated resolved configs are written under `artifacts/generated_configs/`.
Run outputs are written under `outputs/batch/` locally or Drive on Colab.
