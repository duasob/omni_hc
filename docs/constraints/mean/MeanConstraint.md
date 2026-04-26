# MeanConstraint

`MeanConstraint` enforces global mean constraints by subtracting a correction
from the backbone output.

Modes:

- `post_output`: subtract the output mean directly.
- `post_output_learned`: learn a correction from the output.
- `latent_head`: learn a correction from a captured latent representation.

## Config

See `configs/constraints/mean_correction.yaml`.

## Tests

Covered by `tests/test_mean.py`.

