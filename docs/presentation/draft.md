---
marp: true
theme: peril
paginate: false
class: light
---

<div class="variant-label">Full benchmark pills / bottom</div>

<div class="mock-slide">
  <h1>Navier-Stokes</h1>
  <h2>Autoregressive vorticity prediction</h2>

  <div class="mock-grid">
    <div>Diagram</div>
    <div>Constraint</div>
    <div>Result</div>
  </div>

  <div class="bench-nav">
    <span class="active">Navier-Stokes</span>
    <span>Darcy</span>
    <span>Pipe flow</span>
    <span>Elasticity</span>
    <span>Plasticity</span>
  </div>
</div>

---

<div class="variant-label">Full benchmark pills / top</div>

<div class="mock-slide">
  <div class="bench-nav top">
    <span class="active">Navier-Stokes</span>
    <span>Darcy</span>
    <span>Pipe flow</span>
    <span>Elasticity</span>
    <span>Plasticity</span>
  </div>

  <h1>Navier-Stokes</h1>
  <h2>Autoregressive vorticity prediction</h2>

  <div class="mock-grid lower">
    <div>Diagram</div>
    <div>Constraint</div>
    <div>Result</div>
  </div>
</div>

---

<div class="variant-label">Active benchmark pill + dots</div>

<div class="mock-slide">
  <h1>Navier-Stokes</h1>
  <h2>Autoregressive vorticity prediction</h2>

  <div class="mock-grid">
    <div>Diagram</div>
    <div>Constraint</div>
    <div>Result</div>
  </div>

  <div class="bench-dot-nav">
    <span class="active-pill">Navier-Stokes</span>
    <span class="dot future"></span>
    <span class="dot future"></span>
    <span class="dot future"></span>
    <span class="dot future"></span>
  </div>
</div>

---

<div class="variant-label">Middle benchmark / dot progress</div>

<div class="mock-slide">
  <h1>Pipe flow</h1>
  <h2>Velocity from a stream function</h2>

  <div class="mock-grid">
    <div>Diagram</div>
    <div>Constraint</div>
    <div>Result</div>
  </div>

  <div class="bench-dot-nav">
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="active-pill">Pipe flow</span>
    <span class="dot future"></span>
    <span class="dot future"></span>
  </div>
</div>

---

<div class="variant-label">Benchmark pills + state pills</div>

<div class="mock-slide">
  <h1>Darcy flow</h1>
  <h2>Problem</h2>

  <div class="state-nav">
    <span class="active">Problem</span>
    <span>Constraint</span>
    <span>Results</span>
  </div>

  <div class="mock-grid state-space">
    <div>Boundary condition</div>
    <div>Flux mismatch</div>
    <div>What fails?</div>
  </div>

  <div class="bench-nav">
    <span>Navier-Stokes</span>
    <span class="active">Darcy</span>
    <span>Pipe flow</span>
    <span>Elasticity</span>
    <span>Plasticity</span>
  </div>
</div>

---

<div class="variant-label">Dot benchmarks + state pills</div>

<div class="mock-slide">
  <h1>Darcy flow</h1>
  <h2>Constraint</h2>

  <div class="state-nav">
    <span>Problem</span>
    <span class="active">Constraint</span>
    <span>Results</span>
  </div>

  <div class="mock-grid state-space">
    <div>Architecture</div>
    <div>Projection</div>
    <div>Guarantee</div>
  </div>

  <div class="bench-dot-nav">
    <span class="dot done"></span>
    <span class="active-pill">Darcy</span>
    <span class="dot future"></span>
    <span class="dot future"></span>
    <span class="dot future"></span>
  </div>
</div>

---

<div class="variant-label">State pills top + dot progress bottom</div>

<div class="mock-slide">
  <div class="state-nav top-centered">
    <span>Problem</span>
    <span class="active">Constraint</span>
    <span>Results</span>
  </div>

  <h1>Darcy flow</h1>
  <h2>Constraint</h2>

  <div class="mock-grid top-state-space">
    <div>Architecture</div>
    <div>Projection</div>
    <div>Guarantee</div>
  </div>

  <div class="bench-dot-nav">
    <span class="dot done"></span>
    <span class="active-pill">Darcy</span>
    <span class="dot future"></span>
    <span class="dot future"></span>
    <span class="dot future"></span>
  </div>
</div>

---

<div class="variant-label">State pills as footer, benchmarks as dots</div>

<div class="mock-slide">
  <h1>Darcy flow</h1>
  <h2>Results</h2>

  <div class="mock-grid">
    <div>Property error</div>
    <div>Rel. L2</div>
    <div>Tradeoff</div>
  </div>

  <div class="footer-stack">
    <div class="state-nav compact">
      <span>Problem</span>
      <span>Constraint</span>
      <span class="active">Results</span>
    </div>
    <div class="bench-dot-nav inline">
      <span class="dot done"></span>
      <span class="active-pill">Darcy</span>
      <span class="dot future"></span>
      <span class="dot future"></span>
      <span class="dot future"></span>
    </div>
  </div>
</div>

---

<div class="variant-label">Compact trail + dot progress</div>

<div class="mock-slide">
  <div class="trail-nav">
    <span class="bench">Elasticity</span>
    <span class="divider">/</span>
    <span class="state">Results</span>
  </div>

  <h1>Elasticity</h1>
  <h2>Latent features remove the bottleneck</h2>

  <div class="mock-grid trail-space">
    <div>Latent input</div>
    <div>Stress field</div>
    <div>Error</div>
  </div>

  <div class="bench-dot-nav">
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="active-pill">Elasticity</span>
    <span class="dot future"></span>
  </div>
</div>

---

<div class="variant-label">Left rail states + bottom benchmark progress</div>

<div class="mock-slide rail-layout">
  <div class="state-rail">
    <span>Problem</span>
    <span class="active">Constraint</span>
    <span>Results</span>
  </div>

  <div class="rail-content">
    <h1>Plasticity</h1>
    <h2>Constraint</h2>

    <div class="mock-grid rail-space">
      <div>Envelope</div>
      <div>Clamp</div>
      <div>Validity</div>
    </div>
  </div>

  <div class="bench-dot-nav">
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="dot done"></span>
    <span class="active-pill">Plasticity</span>
  </div>
</div>

---

<div class="variant-label">Minimal: active benchmark + state only</div>

<div class="mock-slide">
  <div class="minimal-nav">
    <span>Pipe flow</span>
    <b>Constraint</b>
  </div>

  <h1>Pipe flow</h1>
  <h2>Velocity from a stream function</h2>

  <div class="mock-grid minimal-space">
    <div>Stream function</div>
    <div>Velocity field</div>
    <div>Divergence</div>
  </div>
</div>
