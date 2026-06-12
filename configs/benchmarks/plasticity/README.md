# Plasticity Benchmark Config

`base.yaml` owns the dataset and tensor-layout assumptions for the plasticity
benchmark. It deliberately does not choose a backbone, optimizer schedule, hard
constraint, or W&B policy.

The benchmark follows the NSL StandardBench plasticity setup:

- `loader: plas`
- `task: dynamic_conditional`
- `T_out: 20`
- `out_dim: 4`
- target channels per time value: `[x, y, u_x, u_y]`

The target tensor is stored as:

```text
(batch, points, T_out * out_dim)
```

so the default shape is `(batch, 3131, 80)`.

Backbone configs under `configs/backbones/plasticity/` keep the NSL
model-specific defaults. The mesh-consistency hard constraint is configured
separately in:

[`configs/constraints/plasticity_mesh_consistency_constraint.yaml`](../../../configs/constraints/plasticity_mesh_consistency_constraint.yaml)

The corresponding benchmark documentation is:

[`docs/benchmarks/plasticity.md`](../../../docs/benchmarks/plasticity.md)
