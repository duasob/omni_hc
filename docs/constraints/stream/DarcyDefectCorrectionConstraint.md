# DarcyDefectCorrectionConstraint

`DarcyDefectCorrectionConstraint` is a hard-constraint module for the Darcy flow
benchmark that combines a **coordinate-based Dirichlet ansatz** with an iterative
**defect correction** step. It enforces both the zero-pressure boundary condition
and (approximately) the interior Darcy PDE by construction.

The benchmark PDE is

$$
-\nabla \cdot (a \nabla u) = 1
\quad \text{in } \Omega = [0,1]^2,
\qquad
u|_{\partial \Omega} = 0
$$

where $a$ is the (binary) permeability field and $u$ is the scalar pressure.

---

## Mechanism

The constraint applies three successive transformations to the backbone output.

### 1. Dirichlet Ansatz

The backbone predicts a raw pressure field $\hat u$ in normalized space. The
constraint first decodes it to physical units, then multiplies by a distance
function that vanishes on the domain boundary:

$$
u_{\mathrm{ansatz}}(x,y) = \varphi(x,y) \cdot \hat{u}_{\mathrm{phys}}(x,y),
\qquad
\varphi(x,y) = x(1-x)\,y(1-y)
$$

This guarantees $u_{\mathrm{ansatz}}|_{\partial\Omega} = 0$ exactly, regardless
of what the backbone predicts.

### 2. Discrete Darcy Residual

The residual of the Darcy PDE at each interior grid point is computed using a
**harmonic-mean face permeability**:

$$
a_{i+1/2,j} = \frac{2\,a_{i,j}\,a_{i+1,j}}{a_{i,j} + a_{i+1,j}}
$$

and analogously in the $y$-direction. The harmonic mean is the physically correct
choice for the discrete divergence operator with a variable-coefficient diffusion
tensor — it corresponds to resistors in series for piecewise-constant $a$.

The discrete residual at interior node $(i,j)$ is then

$$
r_{i,j} = -\bigl[D_x(a_{\mathrm{hm}}\,D_x u) + D_y(a_{\mathrm{hm}}\,D_y u)\bigr]_{i,j} - 1
$$

where $D_x$, $D_y$ are standard second-order finite difference operators.

### 3. Defect Correction via Poisson Solve

Given $r$, a correction $\delta u$ is found by solving a **homogeneous Dirichlet
Poisson problem**:

$$
\Delta(\delta u) = \frac{r}{a}, \qquad \delta u|_{\partial\Omega} = 0
$$

and the pressure is updated as

$$
u_{\mathrm{hard}} = u_{\mathrm{ansatz}} + \delta u
$$

The solve is performed by a DST-I (discrete sine transform) spectral Poisson
solver, giving an exact solution to the constant-coefficient equation in $O(N^2 \log N)$.

This can be applied for `n_correction_steps` iterations, re-computing $r$ on
the updated $u$ each time.

---

## Correctness Properties

**Inside each uniform-$a$ region (bulk):** The Poisson solve with RHS $r/a$ is
exact when $a$ is constant, because the equation reduces to $a_0 \Delta(\delta u)
= r$ and the spectral solver is exact for this case. After one step the Darcy
residual in the bulk is identically zero.

**At permeability interfaces:** The pointwise approximation $\Delta(\delta u)
\approx r/a$ loses accuracy because $a$ changes across the interface face. The
correction is still applied but the interface residual reduces by a factor

$$
\left|1 - \frac{a_{\mathrm{hm}}}{a_{\mathrm{local}}}\right|
$$

per iteration. For the binary $\{3, 12\}$ dataset:
$a_{\mathrm{hm}} = 2 \cdot 3 \cdot 12 / (3 + 12) = 4.8$,
giving a convergence factor of $|1 - 4.8/3| = 0.6$ or $|1 - 4.8/12| = 0.6$
— both the same by symmetry. Two or three steps reduce the interface residual
to ~14% of its initial value.

**Boundary condition:** The correction $\delta u$ is solved with zero Dirichlet
BCs, so $u_{\mathrm{hard}}|_{\partial\Omega} = u_{\mathrm{ansatz}}|_{\partial\Omega}
= 0$ exactly after every step.

---

## Config

```yaml
constraint:
  name: "darcy_defect_correction"
  force_value: 1.0         # RHS source term f in -div(a∇u) = f
  n_correction_steps: 1    # number of Poisson correction iterations
  lower: 0.0               # domain lower bound (set by benchmark metadata)
  upper: 1.0               # domain upper bound (set by benchmark metadata)
  permeability_eps: 1.0e-6 # numerical floor for harmonic-mean denominator
```

Config file: [`configs/constraints/darcy_defect_correction.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/darcy_defect_correction.yaml)

Experiment configs for direct comparison:

| Experiment | Config |
|---|---|
| Baseline FNO | [`comparison_fno_baseline.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/comparison_fno_baseline.yaml) |
| Dirichlet ansatz | [`comparison_fno_dirichlet.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/comparison_fno_dirichlet.yaml) |
| Defect correction | [`comparison_fno_defect_correction.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/darcy/comparison_fno_defect_correction.yaml) |

For `lower` and `upper`, the benchmark adapter sets these automatically via
`set_domain_bounds`, so they do not need to be overridden in most configs.

---

## Diagnostics

When `return_aux=True`, the following metrics are reported:

| Key | Description |
|---|---|
| `constraint/darcy_res_abs_mean` | Mean absolute Darcy residual over all interior points |
| `constraint/darcy_res_abs_max` | Max absolute Darcy residual |
| `constraint/darcy_res_bulk_abs_mean` | Mean residual at **bulk** points (uniform-$a$ neighbourhood) — should be ~0 after one step |
| `constraint/darcy_res_intf_abs_mean` | Mean residual at **interface** points (adjacent to a permeability jump) — decays at factor 0.6 per step |
| `constraint/boundary_abs_mean` | Mean absolute boundary value of $u$ — always 0 by ansatz |
| `constraint/boundary_abs_max` | Max absolute boundary value — always 0 by ansatz |
| `constraint/correction_norm_mean` | Mean absolute magnitude of the last $\delta u$ correction |

The bulk/interface split is the key diagnostic for understanding constraint
satisfaction: a well-trained model will show near-zero bulk residual from the
first epoch, while the interface residual depends on how many correction steps
are applied.

---

## Implementation

- Constraint: [`src/omni_hc/constraints/darcy_correction.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/src/omni_hc/constraints/darcy_correction.py)
- Spectral Poisson solver: [`src/omni_hc/constraints/utils/spectral.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/src/omni_hc/constraints/utils/spectral.py) — `sine_poisson_solve_dirichlet_2d`
- Tests: [`tests/test_darcy_correction.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/tests/test_darcy_correction.py)

Tests verify:

- Dirichlet BCs are exactly zero on all four boundary edges after the ansatz
- One correction step reduces the Darcy residual by >99% for constant permeability
- Diagnostics are emitted with the expected keys under `return_aux=True`
- Normalizer encode/decode is applied correctly so physical BCs are zero
- Missing `fx` or `coords` raises `ValueError`

---

## Comparison with DarcyFluxConstraint

| Property | [`DarcyFluxConstraint`](DarcyFluxConstraint.md) | `DarcyDefectCorrectionConstraint` |
|---|---|---|
| Approach | Stream-function → flux → Poisson | Ansatz → residual → Poisson correction |
| Backbone output | Stream function $\psi$ | Pressure $u$ directly |
| Boundary | Via Poisson solve (not exact at discrete level) | Exact by ansatz construction |
| Bulk residual | Not guaranteed zero | Exactly zero after one step |
| Interface residual | Non-zero (ill-conditioned $\psi \to u$ path) | Decays at ~0.6 per iteration |
| Requires `coords` | No | Yes |
| Gradient chain | Long (spectral grad → crop → ÷a → div → Poisson) | Short (ansatz multiply → Poisson) |
