steady Darcy operator problem

We want to learn a field-to-field map in [[Neural Operator Learning]]:

- input field: permeability / coefficient field $a(x, y)$
- output field: solution / pressure field $u(x, y)$

The PDE is

$$
-\nabla \cdot (a(x, y)\nabla u(x, y)) = f(x, y)
$$

For the standard 2D Darcy benchmark used here:

- the forcing is fixed across the dataset
- the forcing is explicitly $f(x, y) = 1$
- the boundary condition is fixed zero [[Dirichlet boundary condition]]

So the problem is

$$
-\nabla \cdot (a\nabla u) = 1
\quad\text{in } \Omega=[0,1]^2,
\qquad
u|_{\partial \Omega}=0
$$

The spatial gradient of the pressure is

$$
\nabla u = \left(\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}\right)
$$

Using [[Finite Difference Methods (FDM)]] to build these derivatives is possible, but the derivative error is tied to the grid resolution. A spectral route is attractive because differentiation becomes multiplication in frequency space.

## Why the naive Fourier route breaks

Standard Fourier differentiation assumes periodic structure. Darcy on the unit box is not periodic; it has zero Dirichlet boundaries. If we directly differentiate a non-periodic pressure field with FFTs, we get strong boundary artifacts and [[Gibbs phenomenon]].

So the first implementation does not treat the physical domain as periodic. Instead, it uses an `fft_pad` backend:

- pad the spatial fields beyond the physical domain
- do the spectral operations on the padded grid
- crop the result back to the original domain

This is an MVP that reduces boundary artifacts, but it is still only an approximation to the true Dirichlet spectral treatment.

## Why we move from pressure to flux

Trying to impose the hard constraint directly on

$$
-\nabla \cdot (a\nabla u)=1
$$

is difficult because the coefficient field $a(x,y)$ varies spatially. That prevents a simple closed-form Fourier projection directly on $u$.

The workaround is to introduce the physical Darcy flux

$$
v(x,y) = -a(x,y)\nabla u(x,y)
$$

Then the PDE becomes

$$
\nabla \cdot v = 1
$$

This is the key simplification: the differential constraint is now constant-coefficient.

## Important subtlety: periodic FFTs cannot represent `div(v)=1`

On a periodic domain, the spatial mean of a divergence must be zero. So a pure periodic Fourier projection cannot enforce

$$
\nabla \cdot v = 1
$$

exactly by itself.

That is why the implementation uses an affine decomposition:

$$
v = v_p + v_c
$$

where:

- $v_p$ is a simple particular field with $\nabla\cdot v_p = 1$
- $v_c$ is a correction field constrained to satisfy $\nabla\cdot v_c = 0$

For the current MVP, we use a simple particular field that varies linearly in one direction. The network predicts a latent flux field, we subtract the particular field, and only project the residual correction to the divergence-free subspace.

## Final `fft_pad` MVP

The current architecture-level constraint is:

1. The backbone takes the permeability field $a(x,y)$ as input.
2. The backbone predicts a 2-channel latent flux field

$$
v_{\text{pred}}(x,y) \in \mathbb{R}^2
$$

3. The wrapper pads the predicted flux and the permeability field.
4. On the padded grid, build a particular field $v_p$ such that

$$
\nabla\cdot v_p = 1
$$

5. Form the correction field

$$
v_{\text{corr,raw}} = v_{\text{pred}} - v_p
$$

6. Project that correction with a Fourier [[Leray projection]] so that

$$
\nabla\cdot v_{\text{corr}} = 0
$$

7. Reconstruct the constrained flux

$$
v = v_p + v_{\text{corr}}
$$

This gives a flux field whose divergence is handled through the affine decomposition above, but only approximately because the backend still relies on FFTs on a padded finite grid.

## Recovering the pressure

From Darcy's law,

$$
v = -a\nabla u
$$

so the pressure gradient is estimated as

$$
w = \nabla u = -\frac{v}{a}
$$

where the division is pointwise.

Then take the divergence:

$$
\nabla \cdot w = \Delta u
$$

So we recover a scalar pressure field by solving a Poisson problem for $u$:

$$
\Delta u = \nabla\cdot w
$$

On the padded grid, the `fft_pad` backend computes this spectrally:

$$
\mathcal{F}(\nabla\cdot w)
=
ik_x \mathcal{F}(w_x) + ik_y \mathcal{F}(w_y)
$$

and

$$
\mathcal{F}(\Delta u)
=
-(k_x^2 + k_y^2)\mathcal{F}(u)
$$

so the Poisson solve becomes

$$
\mathcal{F}(u)
=
-\frac{\mathcal{F}(\nabla\cdot w)}{k_x^2 + k_y^2}
$$

with the zero-frequency mode handled separately to avoid division by zero.

Finally:

- apply the inverse FFT
- crop back to the physical domain
- explicitly clamp the boundary values to the Dirichlet target
- compute the regression loss on the recovered pressure field

## What this constraint enforces today

The current `fft_pad` backend is the first practical version, not the final mathematically clean one.

It gives:

- a modular constraint wrapper that is independent of the backbone
- a latent 2-channel flux representation
- padded Fourier projection of the correction field
- padded Poisson recovery of pressure
- exact boundary overwrite on the final cropped pressure field

It does **not** give a fully exact Dirichlet spectral solve, because the Fourier backend is still periodic under the hood.

## Future step: sine backend

The next spectral backend should be a `sine` backend specialized to zero-Dirichlet pressure fields.

That future backend would replace the padded periodic FFT machinery with a boundary-compatible spectral basis, so that:

- the Poisson solve is naturally aligned with $u=0$ on the boundary
- spectral differentiation is better matched to the physical domain
- the current padding/cropping workaround becomes unnecessary or much smaller

So the final design is:

- current implementation: `fft_pad` backend for an MVP
- future implementation: `sine` backend for a more faithful Dirichlet-aware spectral constraint
