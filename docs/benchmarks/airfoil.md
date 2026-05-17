# Airfoil Benchmark

This benchmark predicts steady compressible inviscid (Euler) flow fields around
2D NACA-style airfoils parameterised by 8 shape coefficients. The dataset was
generated with a structured C-grid solver at Mach 0.8 and AoA 0°, and comes
in two representations.

## Dataset layout

### `data/airfoil/naca/` — unstructured C-grid (2490 samples)

| File | Shape | Description |
|---|---|---|
| `NACA_X.npy` | `(2490, 11220)` | Physical x-coordinates of mesh points |
| `NACA_Y.npy` | `(2490, 11220)` | Physical y-coordinates |
| `NACA_Q.npy` | `(2490, 5, 11220)` | Flow quantities (channels-first) |
| `NACA_Cylinder_X.npy` | `(2490, 221, 51)` | x-coordinates on C-grid |
| `NACA_Cylinder_Y.npy` | `(2490, 221, 51)` | y-coordinates on C-grid |
| `NACA_Cylinder_Q.npy` | `(2490, 5, 221, 51)` | Flow quantities on C-grid |
| `NACA_theta.npy` | `(2490, 8)` | Airfoil shape parameters |
| `NACA_res.npy` | `(2490,)` | Solver residuals (all < 5×10⁻⁷) |

The 11220 unstructured points are the flattened 220×51 cylinder grid (the
221st row wraps back to the first, so 220 unique circumferential stations ×
51 radial layers). The C-grid is laid out so that `j=0` is the airfoil
wall and `j=50` is the far-field boundary at radius R=40.

**Q channels** (both `NACA_Q` and `NACA_Cylinder_Q`):

| Index | Symbol | Description |
|---|---|---|
| 0 | ρ | Density |
| 1 | u | x-velocity |
| 2 | v | y-velocity |
| 3 | p | Static pressure |
| 4 | M | Local Mach number = \|q\|/√(γp/ρ) |

Note: Mach (ch4) is a derived quantity. Given [ρ, u, v, p] the model can
compute ch4 exactly; it need not be an independent output.

**Geometry (`theta`):** 8 CST-style shape perturbation coefficients. Column 0
is always 0. Columns 1–7 are i.i.d. uniform on [-0.05, 0.05].

### `data/airfoil/naca_interp/` — regular Cartesian grid (1200 samples)

| File | Shape | Description |
|---|---|---|
| `NACA_X_interp.npy` | `(1200, 101, 101)` | x-coordinates, x ∈ [-0.5, 1.5] |
| `NACA_Y_interp.npy` | `(1200, 101, 101)` | y-coordinates, y ∈ [-1, 1] |
| `NACA_Q_interp.npy` | `(1200, 101, 101)` | Mach number interpolated to grid |
| `NACA_mask_interp.npy` | `(1200, 101, 101)` | bool: True = fluid, False = airfoil |

`Q_interp` is exactly 0 inside the airfoil (`mask=False`). This subset
corresponds to the first 1200 samples of `naca/`.

## Reference conditions

| Quantity | Value |
|---|---|
| γ | 1.4 |
| ρ∞ | 1 |
| p∞ | 1 |
| a∞ = √(γ p∞/ρ∞) | ≈ 1.1832 |
| M∞ | 0.8 |
| u∞ = M∞ a∞ | ≈ 0.9466 |
| v∞ | 0 |
| H∞ = p/((γ-1)ρ) + ½u² | ≈ 2.948 |

## Verified physics

| Property | Result |
|---|---|
| ρ > 0, p > 0 | exact |
| Far-field ρ → 1, p → 1 (r > 20) | mean abs error < 0.005 |
| Far-field v → 0 | mean abs error ~ 1×10⁻⁵ |
| Far-field u → 0.9466 | mean abs error < 0.02 |
| Entropy p/ρ^γ ≈ 1 | per-sample std ≈ 0.01 |
| Total enthalpy H ≈ H∞ | per-sample std ≈ 0.04 |
| Mach = \|q\|/a | exact (ch4 is derived, not independent) |
| Wall normal velocity (j=0) | mean abs ≈ 0.009, p99 ≈ 0.07 |
| Supersonic pockets (M > 1) | ~5% of points |

## Hard Constraint Candidates

No constraints are implemented yet. The hypothesis script below quantifies how
well each property holds in the dataset; use those numbers to decide which to
promote to architectural constraints.

**Exact / trivially implementable:**

- **Mask ansatz** (`naca_interp`): multiply backbone output by `mask` to enforce
  zero Mach inside the airfoil. Zero cost.
- **Positive ρ, p**: apply softplus or exp to the density and pressure output
  heads. Currently exact in the dataset.
- **Derived Mach**: compute ch4 = \|q\|/√(γp/ρ) from [ρ, u, v, p] instead of
  predicting it as an independent channel.

**Approximate / architecture-design required:**

- **Far-field Dirichlet decay**: output = Q∞ + f(r) × backbone, where f(r) → 0
  as r → R. Similar in spirit to `DirichletBoundaryAnsatz`. The far-field
  conditions hold tightly (< 0.005 error), making this a strong candidate.
- **No-penetration at the airfoil wall**: the normal velocity at the wall should
  be zero. Requires sample-dependent surface normals (computed from theta or
  from the cylinder grid geometry). Residuals are small but not negligible
  (~0.009 mean abs) — may be numerical noise or a tractable constraint.
- **Isentropic pressure-density coupling**: hard-couple p = ρ^γ. The dataset is
  near-isentropic (std ~0.01) but shock waves cause systematic deviations up to
  ~0.08, so this would be an approximation rather than an exact constraint.

## Dataset Checks

Run the constraint hypothesis checker:

```bash
python scripts/diagnostics/airfoil/airfoil_constraint_hypotheses.py \
  --data-dir data/airfoil \
  --split all \
  --plot-examples
```

Pass `--skip-cylinder` to skip loading the ~4 GB cylinder arrays (needed only
for the wall no-penetration check).

Key options:

| Flag | Default | Purpose |
|---|---|---|
| `--far-field-radius` | 20.0 | Radius threshold for far-field checks |
| `--far-field-tol` | 0.05 | Abs tolerance on far-field BC residuals |
| `--entropy-tol` | 0.05 | Abs tolerance on \|p/ρ^γ - 1\| |
| `--enthalpy-tol` | 0.1 | Abs tolerance on \|H - H∞\| |
| `--wall-normal-tol` | 0.05 | Abs tolerance on wall normal velocity |
