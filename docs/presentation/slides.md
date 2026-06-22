---
marp: true
theme: peril
class: light
paginate: true
html: true
math: mathjax
title: OmniHC
description: Hard constraints for neural PDE surrogates
style: |
  section {
    --slide-state-progress-enabled: 0;
  }
---

<!--
Deck switches:
- Theme: change `light` to `dark` in the `class:` line above.
- Slide-state progress: set `--slide-state-progress-enabled` above to `0` to hide it.
-->

<span class="title-marker"></span>

# Towards physics-preserving transformer architectures

<!-- ## Hard constraints for neural PDE surrogates -->

**Bruno Duaso**

<!-- Final Year Project -->

---

```python
from omni_hc.constraints import ConstrainedModel, DirichletBoundaryAnsatz

backbone = TransformerBackbone()
constraint = DirichletBoundaryAnsatz()

model = ConstrainedModel(
    backbone=backbone,
    constraint=constraint,
)
```

---

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="ns-problem-card">
  <img class="ns-problem-gif" src="assets/ns/ns_problem_rollout.gif" alt="Autoregressive Navier-Stokes prediction loop">
</div>  

##


$$
\int_\Omega \omega\,dA = 0
$$

<!-- <p class="takeaway">The issue is not only one-step error: any invariant violation becomes part of the next input.</p> -->

<div class="slide-bench-progress">
  <span class="active-pill">Navier–Stokes</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<img src="assets/ns/ns_mean_correction_diagram.png" alt="Global vorticity projection diagram">


<div class="slide-bench-progress">
  <span class="active-pill">Navier–Stokes</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>



---

<!-- _class: ns-table-slide -->

<!-- # Navier–Stokes: improves across backbones -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<table class="ns-results-table">
  <thead>
    <tr>
      <th></th>
      <th>Galerkin</th>
      <th>FACTFORMER</th>
      <th>ONO</th>
      <th>Transolver</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="model-name">Base</td>
      <td>0.0983</td>
      <td>0.0659</td>
      <td>0.1195</td>
      <td>0.0900</td>
    </tr>
    <tr>
      <td class="model-name">Mean constrained</td>
      <td class="best">0.0836</td>
      <td class="best">0.0648</td>
      <td class="best">0.1148</td>
      <td class="best">0.0869</td>
    </tr>
    <tr>
      <td class="model-name">Relative change</td>
      <td class="delta positive">+15.0%</td>
      <td class="delta positive">+1.7%</td>
      <td class="delta positive">+3.9%</td>
      <td class="delta positive">+3.4%</td>
    </tr>
  </tbody>
</table>

<!-- <div class="result-grid three compact-results">
  <div class="metric">
    <strong>4 / 4</strong>
    <span>backbones improve</span>
  </div>
  <div class="metric">
    <strong>+6%</strong>
    <span>average reduction</span>
  </div>
  <div class="metric">
    <strong>500 / 1000</strong>
    <span>epochs / train samples</span>
  </div>
</div> -->

<!-- <p class="table-note">Relative L<sub>2</sub> test error at the full-data budget. Lower is better.</p> -->

<div class="slide-bench-progress">
  <span class="active-pill">Navier–Stokes</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

<div class="ns-problem-card">
  <img class="ns-problem-gif" src="assets/pipe/pipe_flow_tracers.gif" alt="Pipe Flow prediction Loop">
</div>  

<div class="pipe-equations">



$$
\begin{array}{ll@{\qquad}r}
\mathbf{Wall}& \mathbf u|_{\Gamma_w} = 0 &(1) \\[0.7em]
\mathbf{Inlet}& u_x|_{\Gamma_{\mathrm{in}}} = 4U\eta(1-\eta) &(2) \\[0.7em]
\mathbf{Divergence}& \nabla\cdot \mathbf u = 0 &(3)
\end{array}
$$

</div>

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

---
<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

# Wall Constraint
$$
\begin{array}{cl@{\qquad}r}
\mathbf u|_{\Gamma_w} = 0 &(1) \\[0.7em]
\end{array}
$$

<div class="ns-problem-card">
  <img class="ns-problem-gif" src="assets/pipe/pipe_boundary_constraint_construction.png" alt="Pipe Flow prediction Loop">
</div>  

##
$$
l(\eta) \times \mathcal{N}(\theta) = \mathbf u,
\qquad
l(\eta)=\eta(1-\eta)
$$



<div class="slide-state-progress">
  <span >Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

---
<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

# Wall + Inlet Constraint

$$
\begin{array}{cl@{\qquad}r}
\mathbf u|_{\Gamma_w} = 0 &(1),
\qquad
u_x|_{\Gamma_{\mathrm{in}}} = 4U\eta(1-\eta) &(2)
\end{array}
$$

<div class="ns-problem-card">
  <img class="ns-problem-gif" src="assets/pipe/pipe_inlet_wall_constraint_construction.png" alt="Pipe Flow prediction Loop">
</div>  

<!-- $$
g(\xi,\eta) + l(\xi,\eta) \times \mathcal{N}(\theta) = \mathbf u
$$ -->


$$
\begin{aligned}
g(\xi,\eta) = \alpha(\xi)\,4U\eta(1-\eta),
\qquad
l(\xi,\eta) = \bigl(1-\alpha(\xi)\bigr)\eta(1-\eta),
\qquad
\alpha(\xi)=(1-\xi)
\end{aligned}
$$
##
##
##


<div class="slide-state-progress">
  <span >Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

---
<div class="slide-state-progress">
  <span >Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>


# Divergence Constraint 

<!-- $$
\nabla\cdot \mathbf u = 0 \qquad (3)
$$ -->
##
$$

\nabla  = \begin{bmatrix}
\frac{\partial}{\partial x} \\ 
\frac{\partial}{\partial y} 
\end{bmatrix} 
\qquad
\mathbf u = \begin{bmatrix}
u_x \\ 
u_y 
\end{bmatrix}
\qquad 
\nabla\cdot \mathbf u = \frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y} 

$$

##

$$
\text{predict }
\psi \in \mathbb{R},
\qquad
\mathbf u = \nabla^\perp \psi =
\begin{bmatrix}
\frac{\partial \psi}{\partial y} \\
-\frac{\partial \psi}{\partial x}
\end{bmatrix}
$$


##

$$
\nabla\cdot\mathbf u
=
\frac{\partial}{\partial x}
\left(
\frac{\partial \psi}{\partial y}
\right)
+
\frac{\partial}{\partial y}
\left(
-\frac{\partial \psi}{\partial x}
\right)
=
\frac{\partial^2\psi}{\partial x\partial y}
-
\frac{\partial^2\psi}{\partial y\partial x}
=
0
\quad (3)
$$

##
##
##

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---
<div class="slide-state-progress">
  <span >Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

# Wall + Inlet + Divergence Constraint
##

$$
\psi_\theta = g_\psi + l_\psi N_\theta,
\qquad
\mathbf u_\theta = \nabla^\perp \psi_\theta
=
\begin{bmatrix}
\partial_y \psi_\theta \\
-\partial_x \psi_\theta
\end{bmatrix}
$$
##
##
$$
\begin{array}{lll}
\mathbf{Divergence}
&
\nabla\cdot\mathbf u_\theta=0
&
\Longrightarrow\quad \mathbf u_\theta=\nabla^\perp\psi_\theta
\\[0.9em]
\mathbf{Inlet}
&
u_x|_{\Gamma_{\mathrm{in}}}=4U\eta(1-\eta)
&
\Longrightarrow\quad
\psi_\theta(0,\eta)=H\displaystyle\int_0^\eta 4Us(1-s)\,ds
\\[1.1em]
\mathbf{Walls}
&
\mathbf u|_{\Gamma_w}=0
&
\Longrightarrow\quad
\psi_\theta=\mathrm{const},
\qquad
\partial_n\psi_\theta=0
\quad\text{on }\Gamma_w
\end{array}
$$
##
##
##
##


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---


<!-- _class: pipe-learning-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<!-- # What the hard constraint changes during learning -->

<img class="pipe-learning-gif" src="assets/pipe/pipe_fast_learning_trace_epoch1.gif" alt="One-epoch pipe learning trace comparing unconstrained and stream-constrained models">

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: pipe-results-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<!-- # Pipe flow: boundary and divergence constraints -->

<table class="pipe-results-table">
  <thead>
    <tr>
      <th>Budget<br><span>epoch / train</span></th>
      <th>Base</th>
      <th>$u_x$</th>
      <th>Stream</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5 / 50</td>
      <td>0.4437</td>
      <td>0.1642</td>
      <td class="best">0.1507</td>
    </tr>
    <tr>
      <td>10 / 100</td>
      <td>0.3498</td>
      <td>0.1648</td>
      <td class="best">0.1451</td>
    </tr>
    <tr>
      <td>50 / 500</td>
      <td class="best">0.0130</td>
      <td>0.0163</td>
      <td>0.0159</td>
    </tr>
    <tr>
      <td>100 / 900</td>
      <td>0.0080</td>
      <td class="best">0.0074</td>
      <td>0.0141</td>
    </tr>
    <tr>
      <td>500 / 1000</td>
      <td>0.0056</td>
      <td class="best">0.0054</td>
      <td>0.0114</td>
    </tr>
  </tbody>
</table>

<p class="table-note">Relative L<sub>2</sub> test error. Lower is better.</p>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="active-pill">Pipe flow</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---


<!-- # Darcy flow: pressure through a porous medium -->
##
##
$$-\nabla\cdot(a\nabla u)=f=1,\qquad u|_{\partial\Omega}=0$$

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="darcy-problem-layout">
  <!-- <div class="darcy-figure-card concept"> -->
    <img src="assets/darcy/darcy_problem_3d.png" alt="3D Darcy flow concept diagram">
    <!-- <span>Forcing through heterogeneous permeability produces pressure.</span> -->
  <!-- </div> -->
  <!-- <div class="darcy-figure-card sample"> -->
    <img src="assets/darcy/darcy_dataset_sample.png" alt="Darcy dataset sample showing permeability and pressure">
    <!-- <span>Learning task: map permeability <strong>a(x,y)</strong> to pressure <strong>u(x,y)</strong>.</span> -->
  <!-- </div> -->
</div>


<!-- </div> -->

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: darcy-constraint-slide -->

# Dirichlet Ansatz

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="darcy-constraint-layout">
  <div class="darcy-constraint-card distance-card">
    <img src="assets/darcy/darcy_distance_p1.png" alt="Darcy distance function for p equals one">
  </div>
  <div class="darcy-constraint-card performance-card">
    <img src="assets/darcy/darcy_distance_power_bars.png" alt="Darcy final validation performance across distance powers">
  </div>
</div>

##
$$
u^*(x,y)=\ell_p(x,y)\,\mathcal{N}_\theta(x,y)
$$
##
<!-- $$
\ell_p(x,y)=\left[x(1-x)y(1-y)\right]^p
$$ -->

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: darcy-boundary-slide -->

# The data boundary is not exactly zero

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="darcy-boundary-slide">
  <img src="assets/darcy/darcy_boundary_profiles_combined.png" alt="Darcy boundary pressure profiles across dataset">
</div>

<!-- <div class="darcy-boundary-layout">
  <div class="darcy-boundary-card profile-grid">
    <img src="assets/darcy/darcy_boundary_profiles_grid.png" alt="Darcy boundary pressure profiles across dataset">
  </div>
  <div class="darcy-boundary-card profile-field">
    <img src="assets/darcy/darcy_boundary_profiles_2d.png" alt="Darcy pressure field with boundary profiles">
  </div>
</div> -->

<!-- <p class="takeaway">The physical condition is \(u=0\), but the benchmark fields contain a small systematic boundary profile.</p> -->

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: darcy-sine-boundary-slide -->

# Sine boundary constraint

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="darcy-sine-layout">
  <div class="darcy-sine-boundary-arch">
    <img src="assets/darcy/darcy_sine_boundary_architecture.png" alt="Sine boundary constraint architecture">
  </div>
  <div class="darcy-sine-results">
    <table class="darcy-boundary-heads-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Bnd. rel-L₂ ↓</th>
          <th>RMSE [10⁻⁵] ↓</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Transolver</th>
          <td>0.285</td>
          <td>1.70</td>
        </tr>
        <tr>
          <th>+ Dirichlet</th>
          <td>1.000</td>
          <td>5.92</td>
        </tr>
        <tr>
          <th>+ Sine</th>
          <td class="best">0.126</td>
          <td class="best">0.81</td>
        </tr>
      </tbody>
    </table>
    <div class="result-note">
      <strong>Boundary prediction</strong>
      <span>Darcy test set</span>
    </div>
  </div>
</div>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: darcy-boundary-learning-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<img class="darcy-boundary-learning-gif" src="assets/darcy/darcy_boundary_learning_trace_epoch1.gif" alt="One-epoch Darcy boundary learning trace comparing base and sine boundary models">

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: darcy-flux-slide -->

# Flux constraint

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="darcy-flux-layout">
  <div class="darcy-flux-arch">
    <img src="assets/darcy/darcy_flux_pipeline.png" alt="Darcy flux constraint pipeline">
  </div>
  <div class="darcy-flux-equations">

$$
\mathbf{v}_{\mathrm{valid}}
= \mathbf{v}_{\mathrm{part}} + \nabla^\perp \psi
$$

$$
\nabla\!\cdot\mathbf{v}_{\mathrm{valid}} = 1
$$

$$
\mathbf{w}=-\mathbf{v}_{\mathrm{valid}}/a
$$

$$
\nabla^2 u = \nabla\!\cdot\mathbf{w}
$$

  </div>
</div>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Darcy</span>
  <span class="dot future"></span>
  <span class="dot future"></span>
</div>

---

<!-- _class: elasticity-problem-slide -->

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="elasticity-setup-frame">
  <img class="elasticity-setup-gif" src="assets/elasticity/elasticity_setup_schematic_samples.gif" alt="Elasticity plane-stress samples">
</div>

$$
\mathbf{F}=\frac{\partial \mathbf{x}}{\partial \mathbf{ X}},
\qquad
\mathbf{\sigma (F)} =
\begin{bmatrix}
\sigma_{11} & \sigma _{12} & \sigma _{13} \\
\sigma _{12}  & \sigma _{22}  & \sigma_{23} \\
\sigma_{13} & \sigma _{23}  & \sigma _{33}
\end{bmatrix},
\qquad
\sigma _{\text{VM}}(\mathbf{\sigma (F)}) \in \mathbb{R} 
$$
##


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---


<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---


<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>

---


<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>

---

---

# Conclusions

---


# Questions
