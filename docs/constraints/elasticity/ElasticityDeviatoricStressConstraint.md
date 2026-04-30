# ElasticityDeviatoricStressConstraint

`ElasticityDeviatoricStressConstraint` maps a backbone latent field to the
single scalar stress target used by the 2D elasticity point-cloud benchmark.

The default configuration keeps the backbone at one output channel. That scalar
latent value is concatenated with point coordinates `[z, x, y]` and passed
through a small constraint-owned MLP that predicts:

- $\theta$: orientation of the principal stretch basis.
- $\log \lambda$: logarithmic principal stretch.

This keeps the backbone interface identical to the scalar elasticity benchmark:
the backbone still predicts one channel per point. The extra 
parameterization to go from the right Cauchy-Green tensor $C$ to the output stress lives inside the constraint, where its scale and initialization
can be controlled without changing the external model. This was key to avoid instability in the gradients of $\log \lambda$.

## Parameter Head

The computation is:

```text
backbone(coords) -> z
head([z, x, y]) -> theta_raw, log_lambda_raw
log_lambda = max_log_lambda * tanh(log_lambda_raw)
```

The head is a pointwise MLP. It does not mix information across points: spatial
context is still the responsibility of the backbone. The coordinate channels
give the head enough local geometric information to map the scalar latent field
onto stretch parameters around the hole.

The final layer of the head is initialized conservatively:

- `head_init_scale` scales the final layer weights.
- `theta_bias` initializes the raw orientation channel.
- `log_lambda_bias` initializes the raw stretch channel.

With the default `theta_bias: 0.0` and `log_lambda_bias: 0.0`, the initial
stretch is close to:

```text
log_lambda = 0
lambda = 1
C = I
sigma_VM = 0
```

That avoids starting the model in a saturated high-stress state.

> For debugging and ablations, `backbone_out_dim: 2` is still supported. In that mode the backbone output is interpreted directly as `(theta_raw, log_lambda_raw)` and the internal head is skipped.


The MLP predicts the right Cauchy-Green tensor $C$ for each point. We know that $C$ is formed as $C=F^{T}F$, and for the  Incompressible Hyperelasticity case there is no volume change, adn therefore $\det(F)=1$. This means that $C$ is a positive semi-definitie matrix that satisfies:
$$
\mathbf{C} = \mathbf{C}^\top,\qquad
\mathbf{C} \succ 0,\qquad
\det(\mathbf{C}) = 1.
$$
We can obtain that directly from $\theta$ and $\lambda$, which in physical terms are interpreted as the orientation of the principal stretch basis and the principal stretch:
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
To ensure non-negativity of the eigenvalues, we instead predict $\log(\lambda)$.

Since $\det(\mathbf{C})=\det(\mathbf{F})^2$, this enforces the 2D
incompressibility condition $\det(\mathbf{F})=1$ at the level used by the stress
law.

## Stress Law

The scalar invariants are (as detailed in the original [Geo-FNO paper](https://arxiv.org/abs/2207.05209)):

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

And the final returned scalar is:

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
  backbone_out_dim: 1
  target_out_dim: 1
  c1: 1.863e5
  c2: 9.79e3
  max_log_lambda: 0.03
  head_hidden_dim: 32
  head_layers: 2
  head_init_scale: 1.0e-3
  theta_bias: 0.0
  log_lambda_bias: 0.0
```

`backbone_out_dim` deliberately matches the benchmark target dimension in the
default configuration. The backbone emits one scalar latent channel, while the
constraint-owned head maps `[z, x, y]` to the two kinematic quantities used to
construct $\mathbf{C}$.

`max_log_lambda` bounds the predicted logarithmic stretch with a smooth `tanh` map. The small default value keeps the maximum representable stress close to the scale of the elasticity benchmark target.

Because the stress is proportional to:

$$
\left|\lambda^2 - \lambda^{-2}\right|
=
\left|2\sinh(2\log \lambda)\right|,
$$

large `max_log_lambda` values can make the stress scale explode. 

## Diagnostics

When `return_aux=True`, the constraint reports the constructed kinematic and
stress fields, including:

- `theta_raw`, `theta`
- `log_lambda_raw`, `log_lambda`, `lambda`
- `right_cauchy_green_c11`, `right_cauchy_green_c12`, `right_cauchy_green_c22`
- `det_c`
- `stress_11`, `stress_22`, `stress_12`, `stress_trace`
- `stress_dev_11`, `stress_dev_22`, `stress_dev_12`, `stress_dev_inner`
- `sigma_physical`

In the default head mode, the auxiliary output also includes:

- `param_head_input_z`
- `param_head_input_x`
- `param_head_input_y`

Useful aggregate diagnostics include:

- `constraint/det_c_abs_error_max`: should stay near numerical zero.
- `constraint/lambda_min`, `constraint/lambda_mean`, `constraint/lambda_max`:
  show whether stretch is using the allowed range.
- `constraint/log_lambda_raw_mean` and `constraint/log_lambda_raw_std`: help
  detect saturation of the bounded `tanh` map.
- `constraint/sigma_max`: should be comparable to the physical target scale
  implied by the chosen `max_log_lambda`.
