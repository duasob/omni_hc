# DarcyFluxConstraint

`DarcyFluxConstraint` is an operator pipeline that bakes the information that we have about the Darcy Equation 

$$ - \nabla \cdot(a (x)\nabla u (x)) = f(x) $$ into our model. Taking into account the boundary conditions, and the we have a constant source $f (x)=1$ we know that this benchmark should satisfy

$$ -\nabla \cdot (a\nabla u) = 1
\quad\text{in } \Omega=[0,1]^2,
\qquad
u|_{\partial \Omega}=0 $$

The intuition is to reformulate this through the flux: 
$$
v(x) = -a(x) \nabla u(x)
$$
which has to satisfy the divergence constraint
$$
\nabla \cdot v = 1
$$

The pipeline, in general terms becomes:
- Obtain a hard constrained version of $v$ that satisfies $\nabla \cdot v=1$
- Move from $v$ back to $u$

### Enforcing Flux Divergence 
The backbone predicts a scalar stream function $\psi$ such that 

$$ v_{corr} = \left( \frac{\partial \psi}{\partial x}, \frac{\partial \psi}{\partial y}\right) $$

Computing $v_{corr}$ is done via spectral curl. To avoid the Gibbs phenomenon $\psi$ must be padded (and therefore creating a periodic extension of the domain).

By definition, this ensures that the divergence of $v_{corr}$ is zero. Then, we design an artificial $v_{part}$ such that $\nabla \cdot v_{part}=1$.

The constrained flux is simply 
$$ v_{valid} = v_{corr} + v_{part}$$
where the divergence of $v_{valid}$ has to be 1.

## Flux to Pressure
The next step is to move back from the valid flux $v_{valid}$ back to the pressure field $u$.  From the Darcy Equation we know that 
$$
	w (x) = \nabla u(x) = -\frac{a(x)}{v(x)}
$$
where $a$ is our permeability field input and $w$ is the pressure gradient. From there, we can recover pressure using a Dirichlet sine Poisson solver, [more info on this]

In summary, given a stream function by the unconstrained backbone:
1. pads the stream function (for periodicity)
2. computes a divergence-free stream correction,
3. adds a fixed particular flux,
4. converts flux to a pressure-gradient candidate using permeability,
5. recovers pressure with a Dirichlet sine Poisson solve.

The result is a scalar pressure output with exact Dirichlet boundary values.

## Config

See `configs/constraints/darcy_flux_fft_pad.yaml`.

## Tests

Covered by `tests/test_darcy_flux.py` and config wiring tests.
