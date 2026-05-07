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






















from [Transolver](https://github.com/thuml/Transolver) 
## benchmarks
- 6 benchmarks
- 1000 training samples / 200 test (largest is 3GB)
- divererse: 
	- pointcloud / structured mesh / regular grid
	- steady-state / autoregressive / zero-shot time series
![image](https://github.com/thuml/Transolver/raw/main/PDE-Solving-StandardBenchmark/fig/standard_benchmark.png)

![benchmarks](benchmarks/Readme.md)


















## hard constraints

| Benchmark                                        | Hard Constraint Implemented                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| :----------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**Navier-Stokes**](benchmarks/navier_stokes.md) | [MeanConstraint](constraints/mean/MeanConstraint.md): global vorticity mean preservation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [**Darcy Flow**](benchmarks/darcy.md)            | [DirichletBoundaryAnsatz](constraints/boundary/DirichletBoundaryAnsatz.md): direct architectural ansatz for exact zero pressure on the box boundary.<br>[DarcyFluxConstraint](constraints/stream/DarcyFluxConstraint.md): flux-based pressure recovery that hard-builds the continuity equation `div(v) = 1` before solving back for pressure.                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **Airfoil**                                      | No-penetration boundary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [**Pipe Flow**](benchmarks/pipe.md)              | [StructuredWallDirichletAnsatz](constraints/boundary/StructuredWallDirichletAnsatz.md): wall-only direct-output ansatz for zero wall velocity.<br>[PipeInletParabolicAnsatz](constraints/boundary/PipeInletParabolicAnsatz.md): inlet-only direct-output ansatz for scalar `u_x`.<br>[PipeUxBoundaryAnsatz](constraints/boundary/PipeUxBoundaryAnsatz.md): combined inlet-plus-wall direct-output ansatz for scalar `u_x`.<br>[PipeStreamFunctionUxConstraint](constraints/stream/PipeStreamFunctionUxConstraint.md): divergence-free stream-function construction that returns `u_x`.<br>[PipeStreamFunctionBoundaryAnsatz](constraints/stream/PipeStreamFunctionBoundaryAnsatz.md): divergence-free stream-function construction with hard inlet and wall behavior. |
| [**Elasticity**](benchmarks/elasticity.md)       | [ElasticityDeviatoricStressConstraint](constraints/elasticity/ElasticityDeviatoricStressConstraint.md): deviatoric von Mises stress from a 2D incompressible Right Cauchy-Green tensor                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| [**Plasticity**](benchmarks/plasticity.md)       | Channel semantics under verification before hard-constraint design                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
