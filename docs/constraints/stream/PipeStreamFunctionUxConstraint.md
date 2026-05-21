# PipeStreamFunctionUxConstraint

`PipeStreamFunctionUxConstraint` interprets the scalar backbone output as a
stream function $\psi$ on the curvilinear pipe mesh and recovers the physical
velocity field from that latent representation.

## Mechanism

![pipe_divergence_sample_0000](../../figures/pipe/pipe_divergence_sample_0000.png)

The backbone predicts a scalar stream function $\psi$. The constraint reshapes
it onto the pipe grid, differentiates it in physical coordinates, and returns
only the $u_x$ component:

$$\psi \mapsto \mathbf{u} = (u_x, u_y)$$

$$u_x = \frac{\partial \psi}{\partial y}, \qquad u_y = -\frac{\partial \psi}{\partial x}$$

This construction makes the recovered velocity divergence-free in the continuous
setting:

$$\nabla \cdot \mathbf{u} = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} = \frac{\partial^2 \psi}{\partial x \partial y} - \frac{\partial^2 \psi}{\partial y \partial x} = 0$$

## Interpretation For A Scalar $u_x$ Target

This constraint should be interpreted carefully in the current pipe setup. The
benchmark uses `target_channel: 0`, so the training loss is applied only to the
returned $u_x$. Although the constraint internally recovers

$$
(u_x, u_y) = \left(\frac{\partial \psi}{\partial y}, -\frac{\partial \psi}{\partial x}\right),
$$

the recovered $u_y$ is auxiliary. It is logged through diagnostics and media, but
it is not compared with the dataset's $u_y$ channel and does not receive a direct
boundary loss.

So the stream-function parameterization does impose a structural restriction on
the class of admissible $u_x$ fields: the model can only output $u_x$ fields that
can be written as the transverse derivative of some scalar $\psi$ on the
curvilinear mesh. It also provides a divergence-free velocity completion for any
predicted $\psi$. However, because $u_y$ is latent in the objective, there are
many possible $\psi$ fields that can match a given $u_x$ while inducing different
$u_y$ fields. In the plain `PipeStreamFunctionUxConstraint`, the model is free to
use those latent degrees of freedom if they help fit $u_x$.

That means this constraint is weaker than a true incompressible-velocity
constraint for the pipe benchmark. It is most defensible as a representational
bias for scalar $u_x$ prediction and as a diagnostic way to inspect the
divergence-free completion implied by the learned stream function. It is not, by
itself, evidence that the learned full velocity field matches the dataset's
$u_y$.

The implementation computes the derivatives on the curvilinear pipe mesh using
[`src/omni_hc/constraints/utils/stream_ops.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/src/omni_hc/constraints/utils/stream_ops.py).

This is not a boundary ansatz. It enforces incompressibility structurally by
recovering velocity from $\psi$, but it does not directly force the pipe inlet
profile or wall values.

The dataset-level divergence diagnostic is summarized in:

![pipe_divergence_dataset_summary](../../figures/pipe/pipe_divergence_dataset_summary.png)

The nonzero values in that plot are to be expected. These come from the fact that the zero-divergence is a continuous statement:
$$\nabla \cdot (\partial_y \psi, -\partial_x \psi) = 0$$
and in the implementation, we only have $\psi$ on a discrete curvilinear grid.
The constraint recovers velocity with discrete finite-difference derivatives over
the sampled field, while divergence is measured via a separate finite-volume
operator. On a mesh with finite spacing, a sampled $\psi$ is only an
approximation to a continuous field, and the finite-difference and finite-volume
operations are not exact. That leaves a small numerical divergence, especially
around curved areas or where the field changes rapidly relative to the grid
resolution.

## Boundary Implications

The plain stream-function constraint does not enforce boundary conditions on the
latent $u_y$. For example, it does not require $u_y=0$ at the inlet or walls, and
the scalar $u_x$ loss cannot correct that unless the induced $u_y$ also changes
$u_x$ through the shared $\psi$ representation. The only hard restriction is that
the returned $u_x$ must come from the same $\psi$ that defines the auxiliary
$u_y$.

For stronger scalar-boundary behavior, use
[`PipeStreamFunctionBoundaryAnsatz`](PipeStreamFunctionBoundaryAnsatz.md), which
builds the inlet $u_x$ profile and wall $u_x$ behavior into the stream function.
That is still not the same as supervising or hard-constraining the dataset's
$u_y$ channel everywhere. For a true full-velocity incompressibility constraint,
the benchmark/objective would need to return both $u_x$ and $u_y$, train against
`Pipe_Q[..., 0:2]`, and optionally add explicit boundary checks for both velocity
components. In that setup, the stream-function construction would be a direct
hard incompressibility constraint rather than only a latent completion behind a
scalar $u_x$ target.

## Config

Shared constraint config:

[`configs/constraints/pipe_stream_function.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/constraints/pipe_stream_function.yaml)

```yaml
constraint:
  name: "pipe_stream_function_ux"
  eps: 1.0e-12
```

Pipe experiment using this constraint:

[`configs/experiments/pipe/fno_small_stream.yaml`](/Users/bruno/Documents/Y4/FYP/omni_hc/configs/experiments/pipe/fno_small_stream.yaml)

## Diagnostics And Tests

When `return_aux=True`, the constraint reports:

- `constraint/stream_div_abs_mean`
- `constraint/stream_div_abs_max`
- `constraint/stream_uy_abs_mean`
- `constraint/stream_jac_min`
- `constraint/stream_psi_std`

and auxiliary tensors including `stream_psi`, `stream_uy`, and `stream_div`.

Regression coverage in
[`tests/test_stream.py`](/Users/bruno/Documents/Y4/FYP/omni_hc/tests/test_stream.py)
checks that:

- $u_x$ is recovered correctly from a known stream function
- normalized inputs and targets are handled correctly
- the emitted divergence diagnostics are present
