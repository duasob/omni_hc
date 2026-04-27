# Darcy Benchmark
This is a steady PDE operator problem where the input is the permeability field  $a(x)$ and the output is the pressure field $u(x)$. These are related through [darcy's law](https://en.wikipedia.org/wiki/Darcy%27s_law), which describes how fluid moves through porous media:
$$-\nabla  \cdot  (a (x)\nabla u (x)) = f (x)$$
where $f(x)$ is a fixed forcing field.

> later, want a plot that shows what the forcing field is in the dataset


## Hard Constraints

- [DirichletBoundaryAnsatz](../constraints/boundary/DirichletBoundaryAnsatz.md)
- [DarcyFluxConstraint](../constraints/stream/DarcyFluxConstraint.md)
