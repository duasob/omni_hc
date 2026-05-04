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

The four output channels are not yet fully verified. NSL's visualization logic
uses the final timestep as:

- channels `0:2`: plotting coordinates
- channels `2:4`: a two-component vector visualized by magnitude

Before adding plasticity hard constraints, we should verify the channel names
against the dataset generation source or the paper. A useful diagnostic is to
check whether channels `2:4` equal channels `0:2` minus the reference grid.

## Runnable Configs

- [`configs/experiments/plasticity/fno.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/plasticity/fno.yaml)
- [`configs/sweeps/plasticity_backbones.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/sweeps/plasticity_backbones.yaml)
