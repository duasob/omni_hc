# OmniHC

OmniHC is the framework repo for hard-constrained scientific machine learning experiments. The design goal is simple: keep reusable constraint machinery, benchmark metadata, and config composition in one place, while keeping each benchmark implementation isolated enough that the project can grow beyond a single Navier-Stokes demo.

This project shifts from soft encouragement of physics to hard guarantees. By introducing modular, plug-and-play hard constraints, OmniHC restricts model outputs to physically valid manifolds, enforcing target invariants up to machine precision regardless of learned weights.

A demo benchmark on 2D incompressible Navier-Stokes flow can be found here: [duasob/physics_preserving_ns_flow](https://github.com/duasob/physics_preserving_ns_flow)

The idea is that any standard backbone architecture can be wrapped with a hard constraint module to yield physically consistent predictions. 

```python
from omni_hc.constraints import ConstrainedModel, DirichletBoundaryAnsatz

# any standard architecture
backbone = TransformerBackbone() 

# any constraint module implementing the desired physical invariants
constraint = DirichletBoundaryAnsatz() 

# Model will is guaranteed to satisfy the constraint by construction, regardless of backbone weights
model = ConstrainedModel(
    backbone=backbone,
    constraint=constraint,
)
```


| Benchmark              | Physical Domain                                  | Dataset Source                                                                      | Hard Constraint Implemented                                              | Implementation Method                                | Status             |
| :--------------------- | :----------------------------------------------- | :---------------------------------------------------------------------------------- | :----------------------------------------------------------------------- | :--------------------------------------------------- | :----------------- |
| **Navier-Stokes (2D)** | Incompressible Viscous Fluid Dynamics (Periodic) | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)     | Divergence-Free Field ($\nabla \cdot \mathbf{u} = 0$) and global vorticity | Leray Projection (FFT) / zero-mode correction        | Implemented |
| **Darcy Flow**         | Steady-state Porous Media                        | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)     | Dirichlet boundary conditions and Divergence of Velocity $= 1$                      | Architectural ansatz / Stream Function             | Implemented           |
| **Airfoil**            | Transonic Aerodynamics                           | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | No-penetration boundary                                                  | Normal-component projection                          | Drafting           |
| **Pipe Flow**          | Internal Fluid Dynamics                          | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | No-slip wall condition and mass conservation                             | Distance masking / global projection                 | Drafting           |
| **Elasticity**         | Solid Mechanics (Point Cloud)                    | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Symmetric stress tensor                                                  | Restricted tensor assembly                           | Drafting           |
| **Plasticity**         | Time-dependent Deformation                       | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Thermodynamic irreversibility                                            | Monotonic cumulative integration                     | Drafting           |


Some example configs for navier-stokes flow are:

- [fno_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/fno_small_mean.yaml)
- [gt_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/gt_small_mean.yaml)

```bash
python scripts/train.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu

python scripts/test.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu

python scripts/tune.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu
```

W&B is optional and controlled by the `wandb_logging` section in each experiment config.

The benchmark adapter is selected from `benchmark.name`, so the same entrypoints can be reused as additional datasets are added.
