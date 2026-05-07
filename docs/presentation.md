## idea

- [Neural Solver-Library](https://github.com/thuml/Neural-Solver-Library/tree/main) :: library of Sci-ML models and benchmarks. 
- [Transolver](https://github.com/thuml/Transolver) 































```python
from omni_hc.constraints import ConstrainedModel, DirichletBoundaryAnsatz

# any standard architecture
backbone = TransformerBackbone() 

# any constraint module implementing the desired physical invariants
constraint = DirichletBoundaryAnsatz() 

# Model will is guaranteed to satisfy the constraint by construction, regardless of backbone weights
model = ConstrainedModel(
    backbone=backbone,
    constraint=constraint,
)
```



## benchmarks
![image](https://github.com/thuml/Transolver/raw/main/PDE-Solving-StandardBenchmark/fig/standard_benchmark.png)


## hard constraints

| Benchmark                                             | Physical Domain                                  | Hard Constraint Implemented                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Status              |
| :---------------------------------------------------- | :----------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------ |
| [**Navier-Stokes**](docs/benchmarks/navier_stokes.md) | Incompressible Viscous Fluid Dynamics (Periodic) | [MeanConstraint](docs/constraints/mean/MeanConstraint.md): global vorticity mean preservation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  | Implemented         |
| [**Darcy Flow**](docs/benchmarks/darcy.md)            | Steady-state Porous Media                        | [DirichletBoundaryAnsatz](docs/constraints/boundary/DirichletBoundaryAnsatz.md): direct architectural ansatz for exact zero pressure on the box boundary.<br>[DarcyFluxConstraint](docs/constraints/stream/DarcyFluxConstraint.md): flux-based pressure recovery that hard-builds the continuity equation `div(v) = 1` before solving back for pressure.                                                                                                                                                                                                                                                                                                                                                                                                                                       | Implemented         |
| **Airfoil**                                           | Transonic Aerodynamics                           | No-penetration boundary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Drafting            |
| [**Pipe Flow**](docs/benchmarks/pipe.md)              | Internal Fluid Dynamics                          | [StructuredWallDirichletAnsatz](docs/constraints/boundary/StructuredWallDirichletAnsatz.md): wall-only direct-output ansatz for zero wall velocity.<br>[PipeInletParabolicAnsatz](docs/constraints/boundary/PipeInletParabolicAnsatz.md): inlet-only direct-output ansatz for scalar `u_x`.<br>[PipeUxBoundaryAnsatz](docs/constraints/boundary/PipeUxBoundaryAnsatz.md): combined inlet-plus-wall direct-output ansatz for scalar `u_x`.<br>[PipeStreamFunctionUxConstraint](docs/constraints/stream/PipeStreamFunctionUxConstraint.md): divergence-free stream-function construction that returns `u_x`.<br>[PipeStreamFunctionBoundaryAnsatz](docs/constraints/stream/PipeStreamFunctionBoundaryAnsatz.md): divergence-free stream-function construction with hard inlet and wall behavior. | Implemented         |
| [**Elasticity**](docs/benchmarks/elasticity.md)       | Solid Mechanics (Point Cloud)                    | [ElasticityDeviatoricStressConstraint](docs/constraints/elasticity/ElasticityDeviatoricStressConstraint.md): deviatoric von Mises stress from a 2D incompressible Right Cauchy-Green tensor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | Implemented         |
| [**Plasticity**](docs/benchmarks/plasticity.md)       | Time-dependent Deformation                       | Channel semantics under verification before hard-constraint design                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | Baseline integrated |

## experiments
