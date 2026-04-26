# PipeInletParabolicAnsatz

`PipeInletParabolicAnsatz` enforces the discovered parabolic pipe inlet profile
for scalar `ux`.

```text
f = g + lN
g(i,j) = alpha(i) * 0.25 * 4t(j)(1 - t(j))
l(i,j) = 1 - alpha(i)
alpha(i) = (1 - xi(i))^p
```

The transverse coordinate `t` is computed from decoded physical inlet
coordinates, so the profile follows the real pipe geometry.

## Config

See `configs/constraints/pipe_inlet_parabolic.yaml`.

## Diagnostics

```bash
conda run -n omni-hc python scripts/diagnostics/pipe_inlet_profile.py \
  --samples 0 10 100 \
  --summary-samples 1000
```

