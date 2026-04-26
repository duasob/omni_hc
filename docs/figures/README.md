# Figures

Generated figures should normally be written to ignored runtime directories
under `artifacts/`.

Curated figures that are useful for docs or the final report can be copied into
this directory and referenced from Markdown pages.

Diagnostic scripts live under `scripts/diagnostics`.

## Boundary Ansatz Maps

Use this script to generate `g` and `l` maps for the boundary ansatz modules:

```bash
conda run -n omni-hc python scripts/diagnostics/boundary_ansatz_maps.py
```

By default it writes figures to `artifacts/boundary_ansatz_maps/` for:

- `DirichletBoundaryAnsatz`
- `StructuredWallDirichletAnsatz`
- `PipeInletParabolicAnsatz`
- `PipeUxBoundaryAnsatz`
- `PipeStreamFunctionBoundaryAnsatz`

For direct output-space constraints, the script infers:

```text
g = constraint(N=0)
l = constraint(N=1) - constraint(N=0)
```

For `PipeStreamFunctionBoundaryAnsatz`, the affine ansatz is on the latent
stream function, so the script plots the emitted `stream_psi_bc` and
`stream_mask` auxiliary maps.
