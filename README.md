# OmniHC
Omni-model Hard Constraints: A Modular Framework for Physics-Preserving Transformers Across Standard Benchmarks

This project shifts from "encouraging" physics (soft constraints) to guaranteeing them. The idea is that any standard backbone architecture can be wrapped with a hard constraint module to return physically consistent predictions. 

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
| [**Pipe Flow**](docs/benchmarks/pipe.md) | Internal Fluid Dynamics | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [StructuredWallDirichletAnsatz](docs/constraints/boundary/StructuredWallDirichletAnsatz.md): wall-only direct-output ansatz for zero wall velocity.<br>[PipeInletParabolicAnsatz](docs/constraints/boundary/PipeInletParabolicAnsatz.md): inlet-only direct-output ansatz for scalar `u_x`.<br>[PipeUxBoundaryAnsatz](docs/constraints/boundary/PipeUxBoundaryAnsatz.md): combined inlet-plus-wall direct-output ansatz for scalar `u_x`.<br>[PipeStreamFunctionUxConstraint](docs/constraints/stream/PipeStreamFunctionUxConstraint.md): divergence-free stream-function construction that returns `u_x`.<br>[PipeStreamFunctionBoundaryAnsatz](docs/constraints/stream/PipeStreamFunctionBoundaryAnsatz.md): divergence-free stream-function construction with hard inlet and wall behavior. | Implemented |
| [**Elasticity**](docs/benchmarks/elasticity.md) | Solid Mechanics (Point Cloud) | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [ElasticityDeviatoricStressConstraint](docs/constraints/elasticity/ElasticityDeviatoricStressConstraint.md): deviatoric von Mises stress from a 2D incompressible Right Cauchy-Green tensor | Implemented |
| [**Plasticity**](docs/benchmarks/plasticity.md) | Time-dependent Deformation | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [PlasticityMeshConsistencyConstraint](docs/constraints/plasticity/PlasticityMeshConsistencyConstraint.md): ordered deformation mesh reconstruction from positive learned spacings with consistent `[x, y, u_x, u_y]` channels. | Implemented |

All benchmarks share the same entrypoints for training, testing, and tuning. Experiments can be specified in two equivalent ways.

```bash
# Option 1: compose from individual components
python scripts/train.py \
  --benchmark navier_stokes \
  --backbone Galerkin_Transformer \
  --constraint mean_constraint \
  --budget final

# Option 2: use a pre-composed config for a specific experiment
python scripts/train.py --config configs/experiments/navier_stokes/fno_small_mean.yaml
```
More about launching from config: [configs/README.md](configs/README.md)
