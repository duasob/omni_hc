# Darcy Example

This is the first steady-state benchmark slice added after Navier-Stokes.

## Scope

- benchmark metadata in `src/omni_hc/benchmarks`
- structured-grid Darcy loader in `src/omni_hc/benchmarks/darcy/data.py`
- benchmark adapter in `src/omni_hc/benchmarks/darcy/adapter.py`
- shared steady-state task runner in `src/omni_hc/training/tasks/steady.py`
- benchmark defaults in `configs/benchmarks/darcy`
- experiment config in `configs/experiments/darcy`

## Current Command

```bash
python scripts/train.py \
  --config configs/experiments/darcy/fno_small.yaml \
  --device cpu
```

```bash
python scripts/test.py \
  --config configs/experiments/darcy/fno_small.yaml \
  --device cpu
```

Dirichlet ansatz variant:

```bash
python scripts/train.py \
  --config configs/experiments/darcy/fno_small_dirichlet.yaml \
  --device cpu
```

The first pass is unconstrained. That verifies the shared `train.py` path for a second benchmark before adding a Darcy-specific hard constraint.

Boundary inspection on the shipped Darcy targets shows the intended boundary profile is effectively zero:

- boundary mean absolute value is about `5e-5`
- worst-case boundary magnitude is about `2.2e-3`

That is consistent with using `g(x)=0` in the architectural ansatz.

Note:

- the current CPU smoke config forces `model.args.unified_pos: 0` because the upstream NSL positional embedding helper still assumes CUDA.
