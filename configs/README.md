# Config Layout

Configs are split by responsibility:

- `benchmarks/`: dataset and benchmark defaults
- `backbones/`: model-family defaults
- `constraints/`: hard-constraint defaults

A run config can compose these layers with `extends`.

