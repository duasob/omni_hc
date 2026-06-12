# ElasticityPlaneStressVMConstraint

`ElasticityPlaneStressVMConstraint` maps a latent field to a scalar von Mises
stress using an incompressible Mooney-Rivlin plane-stress membrane model.

## Latent Kinematics

By default, the backbone returns a 32-component latent vector at each point.
A constraint-owned MLP maps `[z, x, y]` to bounded mean and deviatoric
in-plane log stretches:

$$
m=m_{\max}\tanh(m_{\mathrm{raw}}),\qquad
d=-d_{\max}\tanh^2(d_{\mathrm{raw}}).
$$

The principal stretches are

$$
\lambda_1=e^{m+d},\qquad
\lambda_2=e^{m-d},\qquad
\lambda_3=e^{-2m}.
$$

Therefore $\lambda_1\lambda_2\lambda_3=1$ exactly. The third stretch is the
through-thickness response required by 3D incompressibility; it is not fixed to
one as it would be under plane strain.

The non-positive $d$ convention also fixes the otherwise arbitrary labeling
of the two in-plane principal stretches: $\lambda_1\leq\lambda_2$.

## Plane-Stress Recovery

For incompressible Mooney-Rivlin elasticity, the principal Cauchy stresses are

$$
\sigma_i=-p+2C_1\lambda_i^2-2C_2\lambda_i^{-2}.
$$

The pressure is eliminated by imposing the membrane condition
$\sigma_3=0$:

$$
p=2C_1\lambda_3^2-2C_2\lambda_3^{-2}.
$$

The scalar output is then

$$
\sigma_{\mathrm{VM}}
=\sqrt{\sigma_1^2-\sigma_1\sigma_2+\sigma_2^2},
$$

which is the 3D von Mises expression after setting $\sigma_3=0$.

## Training Stability

The vector-latent path uses a small pointwise decoder over `[z, x, y]`.
The implementation retains:

- smooth bounded maps for both latent log stretches;
- small final-layer weights in the decoder;
- a small nonzero initial bias for $d$, avoiding initialization exactly at
  the von Mises cusp;
- gradient clipping through `training.max_grad_norm`.

The default bounds are much smaller than the previous 2D reparameterization
because the physically complete plane-stress law is substantially stiffer.

## Engineered Decoder (Transolver latent hook)

The decoder can instead consume the backbone's internal physics-aware
representations rather than the projected output `z`. When the constraint config
declares a `latent_module` list, `build` attaches a `ForwardHookLatentExtractor`
over those module paths and feeds the concatenated latent (optionally with
coordinates, via `decoder_include_coords`) to the head. For Transolver the
default hooks are `blocks.-2`, `blocks.-1.Attn`, and `blocks.-1.ln_3`, giving the
decoder the desliced physics-attention features that a single projected scalar
discards. This mirrors the Navier-Stokes `latent_engineered` constraint and is
shipped as a separate experiment
(`configs/experiments/elasticity/transolver_plane_stress_vm_latent.yaml`); the
plane-stress kinematics and hard guarantees above are unchanged.

## Diagnostics

The auxiliary output includes $m,d,\lambda_1,\lambda_2,\lambda_3$, the
pressure, all three principal Cauchy stresses, $C_{33}=\lambda_3^2$, and the
in-plane and full determinants of $F$. Key diagnostics are:

- `constraint/full_det_f_abs_error_max`;
- `constraint/plane_stress_abs_error_max`;
- min/max values for all three principal stretches;
- means and standard deviations of the raw $m,d$ channels.

The stretches describe local principal stretch magnitudes. They do not by
themselves recover a globally compatible deformed shape because the principal
directions, rigid rotation, and displacement compatibility are not represented.
