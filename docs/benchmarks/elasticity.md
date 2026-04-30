# Elasticity Benchmark



This benchmark is a steady scalar-stress prediction task on an unstructured 2D
point cloud. The upstream Neural-Solver-Library setup uses only:

- `Meshes/Random_UnitCell_XY_10.npy`: point coordinates, interpreted as
  `(samples, points, 2)` after orientation.
- `Meshes/Random_UnitCell_sigma_10.npy`: scalar stress target, interpreted as
  `(samples, points, 1)` after orientation.

The default point-cloud setup therefore predicts one scalar `sigma` per material
point from the 2D point coordinates.

The benchmark defaults live in
[`configs/benchmarks/elasticity/base.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/benchmarks/elasticity/base.yaml).

## Hard Constraints

The first documented hard-constraint variant is:

- [ElasticityDeviatoricStressConstraint](../constraints/elasticity/ElasticityDeviatoricStressConstraint.md):
  maps two latent channels, $\theta$ and $\log\lambda$, to a 2D
  incompressible Right Cauchy-Green tensor and returns deviatoric von Mises
  stress.

The corresponding experiment config is
[`configs/experiments/elasticity/fno_small_deviatoric_stress.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/elasticity/fno_small_deviatoric_stress.yaml).

## Dataset Checks

Inspect the NSL-style tensor layout with:

```bash
python scripts/diagnostics/elasticity/elasticity_loader_check.py
```

For sample plots and scalar stress summaries:

```bash
python scripts/diagnostics/elasticity/elasticity_dataset_summary.py \
  --samples 0 10 100 \
  --summary-samples 1000
```
