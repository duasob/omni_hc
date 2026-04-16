# Config Layout

Configs are split by responsibility:

- `benchmarks/`: dataset and benchmark defaults
- `backbones/`: model-family defaults
- `constraints/`: hard-constraint defaults

A run config can compose these layers with `extends`.

The intended entrypoints are:

- `python scripts/train.py --config ...`
- `python scripts/test.py --config ...`
- `python scripts/tune.py --config ...`

The runtime is selected from `benchmark.name` inside the resolved config.
