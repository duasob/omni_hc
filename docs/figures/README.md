# Figures

Generated figures should normally be written to ignored runtime directories
under `artifacts/`.

Curated figures that are useful for docs or the final report can be copied into
this directory and referenced from Markdown pages.

Diagnostic scripts live under `scripts/diagnostics`.

## Boundary Ansatz Maps

Use this script to generate `g` and `l` maps for the boundary ansatz modules:

```bash
python scripts/diagnostics/boundary_ansatz_maps.py
```

By default it writes figures to `artifacts/boundary_ansatz_maps/` for:

- `DirichletBoundaryAnsatz`
- `StructuredWallDirichletAnsatz`
- `PipeInletParabolicAnsatz`
- `PipeUxBoundaryAnsatz`
- `PipeStreamFunctionBoundaryAnsatz`

For direct output-space constraints, the script infers:

$$g = \operatorname{constraint}(N=0)$$

$$l = \operatorname{constraint}(N=1) - \operatorname{constraint}(N=0)$$

For `PipeStreamFunctionBoundaryAnsatz`, the affine ansatz is on the latent
stream function, so the script plots the emitted `stream_psi_bc` and
`stream_mask` auxiliary maps.

## Pipe Figures

Curated pipe figures currently used by the benchmark and constraint docs:

- [`pipe_boundary_sample_0100.png`](pipe/pipe_boundary_sample_0100.png): sample
  pipe field used on the pipe benchmark page.
- [`pipe_dataset_boundary_overview.png`](pipe/pipe_dataset_boundary_overview.png):
  structured-grid inlet, outlet, and wall layout.
- [`pipe_wall_no_slip_g_l.png`](pipe/pipe_wall_no_slip_g_l.png): wall-only
  ansatz maps for `StructuredWallDirichletAnsatz`.
- [`pipe_inlet_parabolic_g_l.png`](pipe/pipe_inlet_parabolic_g_l.png):
  inlet-only ansatz maps for `PipeInletParabolicAnsatz`.
- [`pipe_dataset_inlet_profiles.png`](pipe/pipe_dataset_inlet_profiles.png):
  observed inlet profile diagnostics used by `PipeInletParabolicAnsatz`.
- [`pipe_ux_boundary_g_l.png`](pipe/pipe_ux_boundary_g_l.png): combined inlet
  and wall ansatz maps for `PipeUxBoundaryAnsatz`.
- [`pipe_stream_function_boundary_g_l.png`](pipe/pipe_stream_function_boundary_g_l.png):
  stream-function boundary ansatz maps for `PipeStreamFunctionBoundaryAnsatz`.
- [`pipe_divergence_sample_0000.png`](pipe/pipe_divergence_sample_0000.png):
  sample finite-volume divergence diagnostic.
- [`pipe_divergence_dataset_summary.png`](pipe/pipe_divergence_dataset_summary.png):
  dataset-level divergence diagnostic summary.
