# Darcy StandardBench Backbones

These configs encode the unconstrained Neural-Solver-Library Darcy defaults from
`external/Neural-Solver-Library/scripts/StandardBench/darcy`.

They are benchmark-specific because the StandardBench hyperparameters vary by
benchmark. Use this layout for future benchmarks:

```text
configs/backbones/<benchmark>/<model>.yaml
```

The current local NSL checkout does not provide Darcy scripts or model-factory
entries for HT-NET or OFormer, so they are not represented here yet.
