# OmniHC Documentation

This directory is the project write-up workspace. It should contain enough
technical narrative to support both repository users and the final report.

## Map

- [Architecture](architecture.md): package boundaries and design principles.
- [Constraints](constraints/README.md): hard-constraint methods and module-level notes.
- [Benchmarks](benchmarks/README.md): benchmark-specific assumptions, configs, and commands.
- [Testing](../tests/README.md): what the automated tests cover and current gaps.
- [Figures](figures/README.md): curated figures for docs and report reuse.

## Running Configs

Benchmark and constraint pages link to concrete experiment configs under
`configs/experiments/`. Use the common entrypoints below with the config path
from the page you are reading:

```bash
python scripts/train.py --config path/to/config.yaml --device cpu
python scripts/test.py --config path/to/config.yaml --device cpu
python scripts/tune.py --config path/to/config.yaml --device cpu
```

If `Neural-Solver-Library` is not installed at
`external/Neural-Solver-Library`, pass `--nsl-root /path/to/Neural-Solver-Library`
or set `OMNI_HC_NSL_ROOT`.

## Report-Oriented Structure

The intended flow for final-report material is:

1. Define the benchmark and physical invariant.
2. State what the dataset empirically satisfies.
3. Describe the hard constraint and its mathematical form.
4. Point to configs, tests, and diagnostics that reproduce the claim.
5. Record limitations and assumptions.
