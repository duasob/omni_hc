# Pipe Example

This is the first `pipe` benchmark slice added to the shared OmniHC training flow.

## Scope

- benchmark metadata in `src/omni_hc/benchmarks`
- structured-grid pipe loader in `src/omni_hc/benchmarks/pipe/data.py`
- benchmark adapter in `src/omni_hc/benchmarks/pipe/adapter.py`
- shared steady-state task runner in `src/omni_hc/training/tasks/steady.py`
- benchmark defaults in `configs/benchmarks/pipe`
- experiment config in `configs/experiments/pipe`

## Current Command

```bash
python scripts/train.py \
  --config configs/experiments/pipe/fno_small.yaml \
  --device cpu
```

```bash
python scripts/test.py \
  --config configs/experiments/pipe/fno_small.yaml \
  --device cpu
```

## Notes

- This v0 benchmark is intentionally unconstrained. It is just a plain FNO baseline under the common OmniHC harness.
- The upstream NSL pipe baseline feeds the normalized `Pipe_X` and `Pipe_Y` tensors as both position input and feature input. This loader keeps that behavior for compatibility.
- `data.target_channel` selects which `Pipe_Q` channel becomes the scalar target field. The default is `0`, matching the upstream pipe loader.
