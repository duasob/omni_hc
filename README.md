# OmniHC

OmniHC is the framework repo for hard-constrained scientific machine learning experiments. The design goal is simple: keep reusable constraint machinery, benchmark metadata, and config composition in one place, while keeping each benchmark implementation isolated enough that the project can grow beyond a single Navier-Stokes demo.

This project shifts from soft encouragement of physics to hard guarantees. By introducing modular, plug-and-play hard constraints, OmniHC restricts model outputs to physically valid manifolds, enforcing target invariants up to machine precision regardless of learned weights.

The old `hc_fluid` layout is still the best working reference for a single benchmark experiment. `omni_hc` should absorb the reusable parts of that repo without inheriting its experiment-specific clutter.

## Current Direction

`hc_fluid` currently mixes:

- reusable wrappers and training utilities
- a single benchmark implementation
- datasets, checkpoints, and generated images
- a vendored upstream solver library inside `src/`

For `omni_hc`, the cleaner boundary is:

- `src/omni_hc/`: reusable framework code
- `external/`: third-party backends such as Neural-Solver-Library
- `configs/`: composable run, benchmark, backbone, and constraint configs
- `examples/`: benchmark-specific demos and migration targets
- `docs/`: architecture and migration notes
- `data/`, `checkpoints/`, `outputs/`: local, untracked experiment state

See [docs/architecture.md](docs/architecture.md) for the concrete layout.

## Benchmarks

| Benchmark              | Physical Domain                                  | Dataset Source                                                                      | Hard Constraint Implemented                                              | Implementation Method                                | Status             |
| :--------------------- | :----------------------------------------------- | :---------------------------------------------------------------------------------- | :----------------------------------------------------------------------- | :--------------------------------------------------- | :----------------- |
| **Navier-Stokes (2D)** | Incompressible Viscous Fluid Dynamics (Periodic) | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)     | Divergence-Free Field ($\nabla \cdot \mathbf{u} = 0$) and global vorticity | Leray Projection (FFT) / zero-mode correction        | First migration target |
| **Darcy Flow**         | Steady-state Porous Media                        | [fno](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-)     | Dirichlet boundary conditions and strict positivity                      | Architectural ansatz / activation design             | Drafting           |
| **Airfoil**            | Transonic Aerodynamics                           | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | No-penetration boundary                                                  | Normal-component projection                          | Drafting           |
| **Pipe Flow**          | Internal Fluid Dynamics                          | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | No-slip wall condition and mass conservation                             | Distance masking / global projection                 | Drafting           |
| **Elasticity**         | Solid Mechanics (Point Cloud)                    | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Symmetric stress tensor                                                  | Restricted tensor assembly                           | Drafting           |
| **Plasticity**         | Time-dependent Deformation                       | [geo-fno](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) | Thermodynamic irreversibility                                            | Monotonic cumulative integration                     | Drafting           |

## Initial Repo Layout

```text
omni_hc/
├── external/
├── configs/
├── docs/
├── examples/
├── scripts/
├── src/omni_hc/
└── tests/
```

The first benchmark slice is described in [examples/navier_stokes/README.md](examples/navier_stokes/README.md).

## Navier-Stokes Demo

The current smoke-test configs are:

- [fno_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/fno_small_mean.yaml)
- [gt_small_mean.yaml](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/navier_stokes/gt_small_mean.yaml)

The backend resolver looks for Neural-Solver-Library in this order:

1. `--nsl-root`
2. `backend.nsl_root` in the config
3. `OMNI_HC_NSL_ROOT`
4. `external/Neural-Solver-Library`

Example commands:

```bash
conda activate omni-hc
cd /Users/bruno/Documents/Y4/FYP/omni_hc

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
