# Plasticity Backbones

These configs encode the valid Neural-Solver-Library StandardBench plasticity
defaults from:

```text
external/Neural-Solver-Library/scripts/StandardBench/plasticity
```

All valid plasticity scripts use `loader=plas`, `task=dynamic_conditional`,
`space_dim=2`, `fun_dim=1`, `out_dim=4`, and `T_out=20`.

`Factformer.sh` is intentionally not copied here because the upstream file
appears to be a pipe benchmark copy (`loader=pipe`, `out_dim=1`,
`save_name=pipe_Factformer`) rather than a plasticity run.
