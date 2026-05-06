# Plasticity Backbones

These configs encode the valid Neural-Solver-Library StandardBench plasticity
defaults from:

```text
external/Neural-Solver-Library/scripts/StandardBench/plasticity
```

All valid plasticity scripts use `loader=plas`, `task=dynamic_conditional`,
`space_dim=2`, `fun_dim=1`, `out_dim=4`, and `T_out=20`.

`Factformer.yaml` keeps the plasticity benchmark fields above and uses the
local NSL model-factory spelling `Factformer`. The upstream `Factformer.sh`
appears to be a pipe benchmark copy (`loader=pipe`, `out_dim=1`,
`save_name=pipe_Factformer`), so only the model hyperparameters are reused.
