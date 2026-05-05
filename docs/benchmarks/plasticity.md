# Plasticity

The plasticity benchmark uses the Geo-FNO/NSL forging dataset:

- `plas_N987_T20.mat`
- `input`: `(987, 101)`
- `output`: `(987, 101, 31, 20, 4)`

The input is a one-dimensional die profile sampled at 101 points. The NSL
loader broadcasts that profile across the second spatial axis, so each model
receives:

- coordinates: `(N, 3131, 2)`
- function input: `(N, 3131, 1)`
- time input: `(N, 20)`
- target: `(N, 3131, 80)`, where `80 = 20 * 4`

The runtime task is `dynamic_conditional`: the model is called once per time
step with the same die field and a scalar normalized time value.

## Splits

The upstream NSL StandardBench scripts use:

- `ntrain: 900`
- `ntest: 80`

OmniHC still creates validation data from the training subset. The test subset
is held out as the last 80 samples and should only be used after final model
selection.

## Channel Semantics

The four output channels are:

- channel `0`: deformed physical `x` coordinate
- channel `1`: deformed physical `y` coordinate
- channel `2`: displacement component `u_x`
- channel `3`: displacement component `u_y`

The upstream Geo-FNO plasticity visualizer scatters points at channels `0:2` and
colors them by `||channels 2:4||`. Local diagnostics also verify that
`channels 0:2 - channels 2:4` is a nearly fixed material grid over time. For the
first 64 training samples, the final-step mean absolute residual against the
inferred material grid is approximately `6.4e-6` in `x` and `1.7e-6` in `y`.

The material grid is not normalized to `[0, 1]`. It is approximately:

- `x_ref = 0.35 - 0.5 * i`
- `y_ref = 14.9 - 0.5 * j`

With this orientation, `j=0` is the upper die side and `j=30` is the lower
clamped side. The diagnostic reports `u_y = 0` on `lower_jN`.

Use the channel probe to reproduce the check:

```bash
python scripts/diagnostics/plasticity/plasticity_channel_probe.py \
  --data-dir data/plasticity \
  --summary-samples 64 \
  --samples 0
```

## Runnable Configs

- [`configs/experiments/plasticity/fno.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/plasticity/fno.yaml)
- [`scripts/sweeps/plasticity_backbones_train.sh`](/Users/bruno/Documents/Y4/FYP/omni_hc/scripts/sweeps/plasticity_backbones_train.sh)

Example:

```bash
python scripts/train.py --benchmark plasticity --backbone FNO --budget debug
```
