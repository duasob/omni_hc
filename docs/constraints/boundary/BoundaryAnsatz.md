# Boundary Ansatz

The common direct boundary mechanism is:

```text
f = g + lN
```

- `N`: unconstrained backbone output.
- `g`: particular field matching the target boundary values.
- `l`: distance-like field that is zero where the constraint must hold.

This guarantees the boundary condition independently of the learned backbone
weights. In code, the shared composition lives in
`src/omni_hc/constraints/utils/boundary_ops.py`.

The diagnostic map script derives `g` and `l` from the actual constraint call,
instead of reimplementing each formula:

```bash
conda run -n omni-hc python scripts/diagnostics/boundary_ansatz_maps.py
```

This makes the figures useful regression artifacts: if an ansatz implementation
changes, the generated maps change with it.

## Design Notes

- The boundary value must be constructed in physical target space, then encoded
  if a target normalizer is active.
- The residual diagnostics should decode outputs before measuring physical
  boundary errors.
- Structured pipe boundaries are index-space boundaries, not Cartesian
  coordinate-threshold boundaries.
