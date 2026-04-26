# Architecture

The repository structure should keep the constraint code reusable while making
benchmark assumptions explicit. The main design rules are:

- reusable code should not live next to benchmark artifacts
- benchmark logic should not be hidden inside generic training scripts
- vendored upstream libraries should not be mixed into the package namespace
- configs should compose cleanly instead of becoming benchmark-specific monoliths

## Target layout

```text
omni_hc/
├── README.md
├── pyproject.toml
├── external/
│   └── Neural-Solver-Library/
├── configs/
│   ├── backbones/
│   ├── benchmarks/
│   │   └── navier_stokes/
│   ├── constraints/
│   └── experiments/
├── docs/
│   ├── benchmarks/
│   └── constraints/
├── scripts/
│   └── diagnostics/
├── src/
│   └── omni_hc/
│       ├── benchmarks/
│       ├── constraints/
│       ├── core/
│       ├── integrations/
│       └── training/
└── tests/
```

## Boundaries

### `src/omni_hc/core`

Framework utilities that are benchmark-agnostic:

- config loading and composition
- registries
- future runner abstractions
- future checkpoint metadata helpers

This layer should not import a specific benchmark dataset.

### `src/omni_hc/constraints`

Reusable hard-constraint implementations and wrappers:

- projection utilities
- latent-hook wrappers
- correction heads
- future benchmark-agnostic interfaces for post-processing and architectural constraints

Constraint code can depend on tensor shapes and model outputs, but not on a specific dataset path convention.

### `src/omni_hc/benchmarks`

Benchmark-specific metadata and adapters:

- dataset conventions
- canonical benchmark names
- benchmark registry entries
- future loader builders and evaluation adapters

This is where Navier-Stokes-specific logic belongs instead of the core package.

### `src/omni_hc/integrations`

Backend adapters live here.

- NSL path resolution
- mapping OmniHC configs to backend model args
- backend-specific model construction

The backend is a dependency of OmniHC, not part of its package namespace.

### `src/omni_hc/training`

Shared experiment runtime:

- training loops
- evaluation loops
- checkpoint saving/loading
- optional W&B and Optuna helpers

### `configs/`

Configs should be layered, not duplicated. The intended pattern is:

1. benchmark defaults
2. backbone defaults
3. constraint defaults
4. run-specific overrides

For actual runs, the important layer is `configs/experiments/`.

The framework should compose these files into one resolved runtime config.

### `docs/benchmarks/`

Benchmark-facing documentation should explain:

- the physical invariant
- the constraint mechanism
- how it maps to the framework
- which config files reproduce the experiment

These pages replace the former standalone examples entrypoint and keep narrative
examples next to the rest of the project writeup.

### `scripts/diagnostics/`

Diagnostic scripts should generate the figures and benchmark checks referenced
from the docs. They are executable support material, not prose modules.

### `external/`

Third-party backends should live here or be referenced through explicit config/env paths.
That keeps upstream code separate from `src/omni_hc` while still making it usable.

## What should stay out of git

These should be local runtime state, not part of the framework structure:

- raw datasets
- checkpoints
- generated rollout videos
- W&B run artifacts
- exploratory notebooks unless they are curated references

## Migration order

The cleanest first migration from `hc_fluid` is:

1. move generic mean-constraint code into `src/omni_hc/constraints`
2. move config composition into `src/omni_hc/core`
3. describe Navier-Stokes as the first benchmark slice
4. add an explicit backend integration layer for Neural-Solver-Library
5. migrate loaders, trainers, testing, and tuning into OmniHC around that backend boundary

That keeps the new repo opinionated without prematurely freezing the full training stack.
