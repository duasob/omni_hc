Instead of predicting the full 2D flux field $v$, we predict a scalar stream
function $\psi$ and define the correction field through the rotated gradient
$$
v_{corr} = \nabla^\perp \psi
= \left(\frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x}\right).
$$
This guarantees
$$
\nabla \cdot v_{corr} = 0
$$
identically, so the constrained flux is written as
$$
v_{valid} = v_{part} + v_{corr},
$$
where $v_{part}$ is chosen so that
$$
\nabla \cdot v_{part} = 1.
$$

The key point is that we do **not** numerically project $v_{corr}$ onto a
divergence-free subspace after it is built. The stream-function ansatz enforces
that property by construction.

To compute $v_{corr}$ spectrally without triggering Gibbs artifacts, $\psi$ must
be padded first. The model predicts $\psi$ on the physical Darcy grid, then we
embed it in a larger computational box and pad it smoothly (for the first MVP:
reflect padding). Spectral derivatives are taken only on this padded $\psi$
field, yielding a padded $v_{corr}$.

For now, the focus is only on the continuity constraint:
$$
\nabla \cdot v_{valid} = 1.
$$
The remaining Darcy recovery steps are handled separately:
1. compute
$$
w = \nabla u = -\frac{v_{valid}}{a}
$$
2. recover $u$ with a Dirichlet-aware sine Poisson solver rather than an FFT
Poisson solve.

The main open issue is not divergence, but whether
$$
w = -\frac{v_{valid}}{a}
$$
is sufficiently close to a true gradient field. For the current Helmholtz debug
path we explicitly ignore that curl/integrability question and only validate the
divergence constraint first.
