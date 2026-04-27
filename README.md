# OmniHC
Omni-model Hard Constraints: A Modular Framework for Physics-Preserving Transformers Across Standard Benchmarks

This project shifts from "encouraging" physics (soft constraints) to guaranteeing them. The idea is that any standard backbone architecture can be wrapped with a hard constraint module to yield physically consistent predictions. 

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

I have done my best to document things extensively but this is still a work in progress. The current benchmarks and constraints are summarized below, and the documentation is organized under [docs](docs/README.md).

| Benchmark | Physical Domain | Dataset Source | Hard Constraint Implemented | Status |
| :-- | :-- | :-- | :-- | :-- |
| [**Navier-Stokes**](docs/benchmarks/navier_stokes.md) | Incompressible Viscous Fluid Dynamics (Periodic) | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) | [MeanConstraint](docs/constraints/mean/MeanConstraint.md): global vorticity mean preservation | Implemented |
| [**Darcy Flow**](docs/benchmarks/darcy.md) | Steady-state Porous Media | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) | [DirichletBoundaryAnsatz](docs/constraints/boundary/DirichletBoundaryAnsatz.md): direct architectural ansatz for exact zero pressure on the box boundary.<br>[DarcyFluxConstraint](docs/constraints/stream/DarcyFluxConstraint.md): flux-based pressure recovery that hard-builds the continuity equation `div(v) = 1` before solving back for pressure. | Implemented |
| **Airfoil** | Transonic Aerodynamics | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | No-penetration boundary | Drafting |
| [**Pipe Flow**](docs/benchmarks/pipe.md) | Internal Fluid Dynamics | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [StructuredWallDirichletAnsatz](docs/constraints/boundary/StructuredWallDirichletAnsatz.md): wall-only direct-output ansatz for zero wall velocity.<br>[PipeInletParabolicAnsatz](docs/constraints/boundary/PipeInletParabolicAnsatz.md): inlet-only direct-output ansatz for scalar `u_x`.<br>[PipeUxBoundaryAnsatz](docs/constraints/boundary/PipeUxBoundaryAnsatz.md): combined inlet-plus-wall direct-output ansatz for scalar `u_x`.<br>[PipeStreamFunctionUxConstraint](docs/constraints/stream/PipeStreamFunctionUxConstraint.md): divergence-free stream-function construction that returns `u_x`.<br>[PipeStreamFunctionBoundaryAnsatz](docs/constraints/stream/PipeStreamFunctionBoundaryAnsatz.md): divergence-free stream-function construction with hard inlet and wall behavior. | Implemented |
| **Elasticity** | Solid Mechanics (Point Cloud) | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Symmetric stress tensor | Drafting |
| **Plasticity** | Time-dependent Deformation | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Thermodynamic irreversibility | Drafting |


Project documentation is organized under [docs](docs/README.md). Some example
configs for Navier-Stokes flow are:

- [fno_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/fno_small_mean.yaml)
- [gt_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/gt_small_mean.yaml)

```bash
conda run -n omni-hc python scripts/train.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu

conda run -n omni-hc python scripts/test.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu

conda run -n omni-hc python scripts/tune.py \
  --config configs/experiments/navier_stokes/fno_small_mean.yaml \
  --device cpu
```

Both Optuna and W&B are optional and controlled by the `wandb_logging` and `optuna` sections in each experiment config.
The benchmark adapter is selected from `benchmark.name`, so the same entrypoints can be reused as additional datasets are added.
