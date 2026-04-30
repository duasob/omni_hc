# ElasticityDeviatoricStressConstraint

`ElasticityDeviatoricStressConstraint` maps a two-channel backbone output to the
single scalar stress target used by the 2D elasticity point-cloud benchmark.

The backbone output is interpreted as:

- $\theta$: orientation of the principal stretch basis.
- $\log \lambda$: logarithmic principal stretch.

The constraint directly constructs the Right Cauchy-Green tensor:

$$
\mathbf{C}
=
\mathbf{R}(\theta)
\begin{bmatrix}
\lambda^2 & 0 \\
0 & \lambda^{-2}
\end{bmatrix}
\mathbf{R}(\theta)^\top.
$$

This guarantees:

$$
\mathbf{C} = \mathbf{C}^\top,\qquad
\mathbf{C} \succ 0,\qquad
\det(\mathbf{C}) = 1.
$$

Since $\det(\mathbf{C})=\det(\mathbf{F})^2$, this enforces the 2D
incompressibility condition $\det(\mathbf{F})=1$ at the level used by the stress
law.

## Stress Law

The scalar invariants are:

$$
I_1 = \operatorname{tr}(\mathbf{C}),
\qquad
I_2 =
\frac{1}{2}
\left(
\operatorname{tr}(\mathbf{C})^2
-
\operatorname{tr}(\mathbf{C}^2)
\right).
$$

With material parameters $C_1 = 1.863 \times 10^5$ and
$C_2 = 9.79 \times 10^3$, the Second Piola-Kirchhoff stress without the pressure
term is:

$$
\boldsymbol{\sigma}
=
2 C_1 \mathbf{I}
+
2 C_2 (I_1 \mathbf{I} - \mathbf{C}).
$$

For incompressible hyperelasticity, the pressure term contributes only to the
hydrostatic part. Because the benchmark target is a scalar von Mises stress, the
constraint uses the 2D deviatoric stress:

$$
\boldsymbol{\sigma}_{dev}
=
\boldsymbol{\sigma}
-
\frac{1}{2}
\operatorname{tr}(\boldsymbol{\sigma})\mathbf{I}.
$$

The returned scalar is:

$$
\sigma_{VM}
=
\sqrt{
\frac{3}{2}
\boldsymbol{\sigma}_{dev}:\boldsymbol{\sigma}_{dev}
}.
$$

At $\mathbf{C}=\mathbf{I}$, the deviatoric stress is zero and therefore
$\sigma_{VM}=0$.

## Config

```yaml
constraint:
  name: "elasticity_deviatoric_stress"
  backbone_out_dim: 2
  target_out_dim: 1
  c1: 1.863e5
  c2: 9.79e3
  max_log_lambda: 8.0
```

`backbone_out_dim` deliberately differs from the benchmark target dimension:
the backbone emits two latent kinematic quantities, and the constraint returns
the scalar $\sigma_{VM}$ used for the loss.

`max_log_lambda` is a numerical guard on the stretch range. It does not rescale
the stress law.
