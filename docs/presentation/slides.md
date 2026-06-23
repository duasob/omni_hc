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
    --slide-state-progress-enabled: 1;
  }
---

<!--
Deck switches:
- Theme: change `light` to `dark` in the `class:` line above.
- Slide-state progress: set `--slide-state-progress-enabled` above to `0` to hide it.
-->

<span class="title-marker"></span>

# Towards physics-preserving transformer architectures

**Bruno Duaso**

## Supervised by Prof. Giordano Scarciotti

<!-- Final Year Project -->



---

<!-- _class: bench-overview-slide -->


<div class="bench-overview-grid">
  <div class="bench-tile">
    <img src="assets/ns/ns_benchmark_rollout_square.gif" alt="Navier-Stokes vorticity rollout">
    <strong>Navier-Stokes</strong>
  </div>
  <div class="bench-tile">
    <img src="assets/pipe/pipe_flow_tracers.gif" alt="Pipe flow tracer animation">
    <strong>Pipe flow</strong>
  </div>
  <div class="bench-tile">
    <img src="assets/darcy/darcy_problem_3d.png" alt="Darcy flow pressure through porous medium">
    <strong>Darcy flow</strong>
  </div>
  <div class="bench-tile">
    <img class="blend-bg" src="assets/elasticity/elasticity_diagram_output.png" alt="Elasticity setup samples">
    <strong>Elasticity</strong>
  </div>
  <div class="bench-tile">
    <img class="blend-bg" src="assets/plasticity/plasticity_forging_cell_grid_sample_0100_focus.gif" alt="Plasticity forging samples">
    <strong>Plasticity</strong>
  </div>
</div>

<!-- <p class="bench-thesis">For each domain, the architecture guarantees one physical property by construction.</p> -->

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
<span class="title-marker"></span>

# Navier-Stokes

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
<span class="title-marker"></span>

# Pipe Flow

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
##
##

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
\nabla\cdot \mathbf u 
=
\frac{\partial u_x}{\partial x} + \frac{\partial u_y}{\partial y}
= 0 
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
      <th>(1) + (2)</th>
      <th>(1) + (2) + (3)</th>
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
<span class="title-marker"></span>

# Darcy Flow

---



<!-- # Darcy flow: pressure through a porous medium -->
##
##
$$
u|_{\partial\Omega}=0,
\qquad 
-\nabla\cdot(a\nabla u)=f=1
$$

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

<!-- _class: darcy-results-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<div class="darcy-results-layout">
  <img src="assets/darcy/darcy_problem_3d.png" alt="3D Darcy flow concept diagram">

  <div class="darcy-results-panel">
    <table class="darcy-results-table">
      <thead>
        <tr>
          <th>Budget<br><span>epoch / train</span></th>
          <th>Baseline</th>
          <th>Flux Constraint</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>5 / 50</td>
          <td class="best">0.2522</td>
          <td>0.3195</td>
        </tr>
        <tr>
          <td>10 / 100</td>
          <td class="best">0.2226</td>
          <td>0.3271</td>
        </tr>
        <tr>
          <td>50 / 500</td>
          <td class="best">0.0232</td>
          <td>0.0286</td>
        </tr>
        <tr>
          <td>100 / 900</td>
          <td>0.0268</td>
          <td class="best">0.0136</td>
        </tr>
        <tr>
          <td>500 / 1000</td>
          <td class="best">0.0057</td>
          <td>0.0088</td>
        </tr>
      </tbody>
    </table>

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
<span class="title-marker"></span>

# Elasticity

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



<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---

<!-- _class: elasticity-constraint-math-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="elasticity-constraint-math">


$$
\det \mathbf{F} =1,
\qquad
\mathbf{F}(\lambda)
=
\begin{bmatrix}
\lambda_1 & 0 & 0 \\
0 & \lambda_2 & 0 \\
0 & 0 & \lambda_3
\end{bmatrix}
$$


##

$$
(\lambda_1,\lambda_2,\lambda_3)
=
\left(e^{m+d},\,e^{m-d},\,e^{-2m}\right)
$$


##
##

$$
\det \mathbf{F}
=
\lambda_1\lambda_2\lambda_3
=
e^{m+d}e^{m-d}e^{-2m}
=1
$$
##
##

</div>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---

<!-- _class: elasticity-constraint-visual-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active">Constraint</span>
  <span>Results</span>
</div>

<div class="elasticity-constraint-visual-layout">
  <img src="assets/elasticity/elasticity_constraint_pipeline.png" alt="Elasticity plane-stress constraint pipeline">
  <img src="assets/elasticity/elasticity_reparameterization_real_output.png" alt="Elasticity reparameterization real output">
</div>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---

<!-- _class: elasticity-results-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span >Constraint</span>
  <span class="active" >Results</span>
</div>

<table class="elasticity-results-table">
  <thead>
    <tr>
      <th>Budget<br><span>epoch / train</span></th>
      <th>Base</th>
      <th>PS<br>latent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5 / 50</td>
      <td>0.5552</td>
      <td class="best">0.5161</td>
    </tr>
    <tr>
      <td>10 / 100</td>
      <td>0.5498</td>
      <td class="best">0.4894</td>
    </tr>
    <tr>
      <td>50 / 500</td>
      <td class="best">0.0527</td>
      <td>0.0622</td>
    </tr>
    <tr>
      <td>100 / 900</td>
      <td class="best">0.0208</td>
      <td>0.0281</td>
    </tr>
    <tr>
      <td>500 / 1000</td>
      <td>0.0059</td>
      <td class="best">0.0053</td>
    </tr>
  </tbody>
</table>

<p class="table-note">Relative L<sub>2</sub> test error. Lower is better.</p>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Elasticity</span>
  <span class="dot future"></span>
</div>

---
<span class="title-marker"></span>

# Plasticity

---

![hydraulic_press](assets/plasticity/hydraulic_press.png)

--- 

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="ns-problem-card plasticity-problem-card">
  <img class="plasticity-gif" src="assets/plasticity/plasticity_forging_samples_0000_0010_0100.gif" alt="Plasticity forging rollouts across multiple samples">
</div>

<!-- $$
\mathcal{N_{\theta}(d, t)} \to (x_{i},y_{i})
$$ -->




<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>

---

<div class="slide-state-progress">
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="ns-problem-card plasticity-problem-card">
    <img class="plasticity-cell-focus-gif" src="assets/plasticity/plasticity_forging_cell_grid_sample_0100_focus.gif" alt="Plasticity forging cell grid focused rollout">
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
  <span class="active">Problem</span>
  <span>Constraint</span>
  <span>Results</span>
</div>

<div class="ns-problem-card plasticity-problem-card">
  <img class="plasticity-gif" src="assets/plasticity/plasticity_valid_invalid_cell_evolution.png" alt="Valid and invalid plasticity cell evolution">
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
  <span class="active" >Constraint</span>
  <span >Results</span>
</div>

<div class="ns-problem-card plasticity-problem-card">
  <img class="plasticity-gif" src="assets/plasticity/plasticity_constraint_channel_cell_mapping.png" alt="Plasticity forging rollouts across multiple samples">
</div>


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>

---

<!-- _class: plasticity-envelope-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span class="active" >Constraint</span>
  <span >Results</span>
</div>

<div class="plasticity-envelope-layout">
  <img src="assets/plasticity/plasticity_envelope_constraint_schematic.png" alt="Plasticity envelope constraint schematic">

  <div class="plasticity-envelope-equations">

$$
{\color{#0f766e}{c_i}} = e_i(t)-{\color{#7c3aed}{g_i}}
$$
##

$$
w_{i,j}
=
\frac{\mathrm{softplus}(\hat{w}_{i,j})}
{\sum_k \mathrm{softplus}(\hat{w}_{i,k})}
$$
##

$$
{\color{#dc2626}{\Delta y_{i,j}}}
=
w_{i,j}\left({\color{#0f766e}{c_i}}-y_{\text{bottom}}\right)
$$

  </div>
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
  <span >Constraint</span>
  <span  class="active" >Results</span>
</div>

<div class="ns-problem-card plasticity-problem-card">
  <img class="plasticity-gif" src="assets/plasticity/plasticity_fast_learning_trace_epoch1.gif" alt="Plasticity forging rollouts across multiple samples">
</div>


<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>


---

<!-- _class: plasticity-results-slide -->

<div class="slide-state-progress">
  <span>Problem</span>
  <span>Constraint</span>
  <span class="active">Results</span>
</div>

<table class="plasticity-results-table">
  <thead>
    <tr>
      <th>Budget<br><span>epoch / train</span></th>
      <th>Base</th>
      <th>Envelope</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>5 / 50</td>
      <td>0.1446</td>
      <td class="best">0.0365</td>
    </tr>
    <tr>
      <td>10 / 100</td>
      <td>0.0367</td>
      <td class="best">0.0357</td>
    </tr>
    <tr>
      <td>50 / 500</td>
      <td class="best">0.0095</td>
      <td>0.0149</td>
    </tr>
    <tr>
      <td>100 / 900</td>
      <td class="best">0.0067</td>
      <td>0.0119</td>
    </tr>
    <tr>
      <td>500 / 1000</td>
      <td class="best">0.0020</td>
      <td>0.0022</td>
    </tr>
  </tbody>
</table>

<p class="table-note">Relative L<sub>2</sub> test error. Lower is better.</p>

<div class="slide-bench-progress">
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="dot done"></span>
  <span class="active-pill">Plasticity</span>
</div>

---

<span class="title-marker"></span>

# Conclusions & Future Work

---
