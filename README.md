# OmniHC
Omni-model Hard Constraints: A Modular Framework for Physics-Preserving Transformers Across Standard Benchmarks

This project shifts from "encouraging" physics (soft constraints) to guaranteeing them. The idea is that any standard backbone architecture can be wrapped with a hard constraint module to return physically consistent predictions. 

```python
from omni_hc.constraints import ConstrainedModel, DirichletBoundaryAnsatz

# any standard architecture
backbone = TransformerBackbone() 

# any constraint module implementing the desired physical invariants
constraint = DirichletBoundaryAnsatz() 

# Model is guaranteed to satisfy the constraint by construction, regardless of backbone weights
model = ConstrainedModel(
    backbone=backbone,
    constraint=constraint,
)
```

This framework was developed as my Final Year Project at Imperial College London. The benchmarks and constraints are summarized below, and the documentation is organized under [docs](docs/README.md).

| Benchmark | Physical Domain | Dataset Source | Hard Constraints Implemented |
| :-- | :-- | :-- | :-- |
| [**Navier-Stokes**](docs/benchmarks/navier_stokes.md) | Incompressible Viscous Fluid Dynamics (Periodic) | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) | [MeanConstraint](docs/constraints/mean/MeanConstraint.md): global vorticity mean preservation |
| [**Darcy Flow**](docs/benchmarks/darcy.md) | Steady-state Porous Media | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) | [DirichletBoundaryAnsatz](docs/constraints/boundary/DirichletBoundaryAnsatz.md): direct architectural ansatz for exact zero pressure on the box boundary.<br>[SineBoundaryConstraint](docs/constraints/boundary/SineBoundaryConstraint.md): per-edge sine-basis boundary prediction hard-enforced on the interior prediction.<br>[DarcyFluxConstraint](docs/constraints/stream/DarcyFluxConstraint.md): flux-based pressure recovery that hard-builds the continuity equation `div(v) = 1` before solving back for pressure. |
| [**Pipe Flow**](docs/benchmarks/pipe.md) | Internal Fluid Dynamics | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [StructuredWallDirichletAnsatz](docs/constraints/boundary/StructuredWallDirichletAnsatz.md): wall-only direct-output ansatz for zero wall velocity.<br>[PipeInletParabolicAnsatz](docs/constraints/boundary/PipeInletParabolicAnsatz.md): inlet-only direct-output ansatz for scalar `u_x`.<br>[PipeUxBoundaryAnsatz](docs/constraints/boundary/PipeUxBoundaryAnsatz.md): combined inlet-plus-wall direct-output ansatz for scalar `u_x`.<br>[PipeStreamFunctionUxConstraint](docs/constraints/stream/PipeStreamFunctionUxConstraint.md): divergence-free stream-function construction that returns `u_x`.<br>[PipeStreamFunctionBoundaryAnsatz](docs/constraints/stream/PipeStreamFunctionBoundaryAnsatz.md): divergence-free stream-function construction with hard inlet and wall behavior. |
| [**Elasticity**](docs/benchmarks/elasticity.md) | Solid Mechanics (Point Cloud) | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [ElasticityPlaneStressVMConstraint](docs/constraints/elasticity/ElasticityPlaneStressVMConstraint.md): von Mises stress recovered from an incompressible plane-stress Right Cauchy-Green tensor parameterization. |
| [**Plasticity**](docs/benchmarks/plasticity.md) | Time-dependent Deformation | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | [PlasticityMeshConsistencyConstraint](docs/constraints/plasticity/PlasticityMeshConsistencyConstraint.md): ordered deformation mesh reconstruction from positive learned spacings with consistent `[x, y, u_x, u_y]` channels.<br>[PlasticityEnvelopeConstraint](docs/constraints/plasticity/README.md): mesh reconstruction confined to the moving die envelope.<br>[PlasticityEnvelopeYFreeXConstraint](docs/constraints/plasticity/README.md): envelope variant constraining only the vertical coordinate.<br>[PlasticityIsotonicRegression](docs/constraints/plasticity/README.md): isotonic projection of predicted coordinates onto ordered meshes. |

## Installation

The project uses Python 3.10 and depends on PyTorch. Backbones are provided by a vendored copy of the [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library) under `external/`.

```bash
git clone --recursive https://github.com/duasob/omni_hc.git
cd omni_hc

conda create -n omni-hc python=3.10
conda activate omni-hc
pip install -e ".[dev,experiments]"
```

Core dependencies are installed with the package; experiment-only dependencies (Optuna, Weights & Biases) come with the `experiments` extra. Colab setup lives in [`scripts/colab.ipynb`](scripts/colab.ipynb), and HPC batch tooling in [`scripts/hpc/`](scripts/hpc).

### Data and outputs

Datasets are the standard FNO / Geo-FNO releases linked in the benchmark table above. Trained weights and resolved configs for every reported run are available from [Google Drive](https://drive.google.com/drive/folders/18phhSBtTSYlRZqOuzO6Qja_A0dGUG8mL?usp=sharing).

The code reads datasets from `data/` and writes run outputs (checkpoints `best.pt`/`latest.pt`, `resolved_config.yaml`, metrics) to `outputs/` at the repo root. Both can be plain directories or symlinks to bulk storage:

```bash
ln -s /path/to/datasets data
ln -s /path/to/run_outputs outputs
```

## Running experiments

All benchmarks share the same entrypoints for training (`scripts/train.py`), evaluation (`scripts/test.py`), and tuning (`scripts/tune.py`). Experiments can be specified in two equivalent ways.

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
More about launching from config: [configs/README.md](configs/README.md). Batch dispatch from a run-list file is handled by `scripts/run_batch.sh` (see `experiments/` for run lists).

## Tests

```bash
python -m pytest tests/
```

See [tests/README.md](tests/README.md) for what each test file covers.
