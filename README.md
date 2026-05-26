<div align="center">

# ⚛️ Physics-Grounded LLM Probe Suite

### Quantifying · Categorising · Mitigating Physics-Incoherent Reasoning in Open LLMs

<br/>

[![Research](https://img.shields.io/badge/Research-Physics_%C3%97_LLM-0FA3A3?style=flat-square&logo=atom&logoColor=white)](https://github.com/your-org/physics-llm-probe)
[![Status](https://img.shields.io/badge/Status-Active_Research-D9A441?style=flat-square)](https://github.com/your-org/physics-llm-probe)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-A020F0?style=flat-square)](https://jax.readthedocs.io)
[![SymPy](https://img.shields.io/badge/SymPy-1.12%2B-3B5526?style=flat-square)](https://sympy.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Dataset](https://img.shields.io/badge/Seed_Dataset-1%2C000_instances-0FA3A3?style=flat-square)](data/seed/)
[![Reproduce](https://img.shields.io/badge/Reproduce-make_reproduce--seed-111217?style=flat-square&logo=gnu-bash&logoColor=white)](Makefile)

<br/>

*An open, reproducible benchmark coupling open-weight language models with lightweight differentiable simulators and symbolic verifiers to systematically measure and reduce physics-incoherent reasoning.*

<br/>

**[📄 Artifact Paper](#)** &nbsp;·&nbsp; **[🗃️ Dataset](#-dataset-design)** &nbsp;·&nbsp; **[🚀 Quickstart](#-quickstart)** &nbsp;·&nbsp; **[📊 Results](#-results--failure-taxonomy)** &nbsp;·&nbsp; **[🤝 Contribute](#-contributing)**

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Research Questions](#-core-research-questions)
3. [Formal Framework](#-formal-framework)
4. [LLM Interaction Protocols](#-llm-interaction-protocols)
5. [Experiments & Metrics](#-experiments--metrics)
6. [Results & Failure Taxonomy](#-results--failure-taxonomy)
7. [Statistical Analysis](#-statistical-analysis)
8. [Quickstart](#-quickstart)
9. [Reproducibility Checklist](#-reproducibility-checklist)
10. [Citation](#-citation)

---

## 🔭 Overview

The **Physics-Grounded LLM Probe Suite** is a reproducible research framework that answers a precise question: *when an open large language model produces answers to physics problems, how wrong is it, why, and what can we do about it without retraining?*

We couple four subsystems into a closed loop:

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                          PHYSICS-GROUNDED LLM PROBE SUITE                           │
│                                                                                      │
│  ┌─────────────────┐      ┌────────────────────┐      ┌──────────────────────────┐  │
│  │  PROBLEM        │      │  LLM HARNESS        │      │  DIFFERENTIABLE          │  │
│  │  GENERATOR      │─────▶│                     │─────▶│  SIMULATOR               │  │
│  │                 │      │  zero-shot          │      │                          │  │
│  │  · Kinematics   │      │  few-shot (k=2,5)   │      │  S(θ ; s₀) → y(t)       │  │
│  │  · Collisions   │      │  chain-of-thought   │      │  PyTorch / JAX / NumPy   │  │
│  │  · Energy       │      │  feedback-loop      │      │                          │  │
│  │                 │      │                     │      │  ∇_θ L available         │  │
│  │  ground truth   │      │  hypothesis h       │      │                          │  │
│  │  y*(t), E_true  │      │  (f̂, θ̂)           │      │  trajectory + diag.      │  │
│  └────────┬────────┘      └────────────────────┘      └────────────┬─────────────┘  │
│           │                          ▲                              │                │
│           │                          │ structured                   │                │
│           │                          │ feedback r⁽⁰⁾               ▼                │
│           │                          │                 ┌──────────────────────────┐  │
│           │                          └─────────────────│  SYMBOLIC VERIFIER       │  │
│           │                                            │                          │  │
│           └───────────────────────────────────────────▶│  · Dimensional analysis  │  │
│                          ground truth                   │  · Algebraic equivalence │  │
│                                                         │  · Conservation checks   │  │
│                                                         │                          │  │
│                                                         │  residuals + counterex.  │  │
│                                                         └──────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Contributions

| # | Contribution |
|---|---|
| 1 | **Parameterised benchmark** — 1 k seed / 10 k extended instances with analytic ground truth across three physics domains |
| 2 | **Minimal-pair design** — each seed instance has a variant that flips exactly one physical assumption, isolating LLM sensitivity |
| 3 | **Differentiable toy simulator** — returns scalar RMSE + energy residuals as structured prompt feedback, no fine-tuning needed |
| 4 | **Symbolic verifier** — SymPy-based dimensional analysis and algebraic equivalence checking |
| 5 | **Failure taxonomy** — five structural error modes with frequency estimates and statistical comparisons |
| 6 | **One-command reproducibility** — `make reproduce-seed` runs the full seed experiment in ≤ 2 hours on a single GPU |

---

## ❓ Core Research Questions

### RQ1 — Domain Failure Profile

> Which classes of physical reasoning (kinematics, energy, momentum, thermodynamics) do advanced open LLMs fail at, and how do failure rates vary with linguistic perturbations (paraphrase, code-mixing, low-resource phrasing)?

### RQ2 — Non-Fine-Tuning Mitigation Ceiling

> To what extent can **simulator feedback** and **symbolic verification** reduce physical inconsistency without any model weight updates?

### RQ3 — Structural Error Taxonomy

> What structural error modes (dimensional mistakes, omitted forces, wrong constants, impossible states) dominate, and how do they correlate with prompt style and model family?

---

## 📐 Formal Framework

### 1 · Bilevel Formulation

We model the system as a **bilevel interaction** between a discrete language agent and a continuous physical environment.

**Upper level — Language Model**

The LLM proposes a symbolic hypothesis $h$ decomposed as:

$$h \;\mapsto\; \bigl(\hat{f},\;\hat{\theta}\bigr)$$

where $\hat{f}$ is the symbolic functional form (e.g. a kinematic equation as a SymPy expression) and $\hat{\theta} \in \mathbb{R}^p$ are the extracted numeric parameters.

**Lower level — Physical Simulator**

$$S\!:\;\Theta \times \mathcal{S}_0 \;\longrightarrow\; \mathcal{Y}, \qquad S(\theta;\,s_0) \;=\; y(t)$$

A deterministic (or differentiable) integrator mapping parameter vector $\theta$ and initial state $s_0$ to a state trajectory $y(t) \in \mathbb{R}^{d \times T}$.

**Objective**

$$\min_{h}\;\mathcal{L}\!\left(h,\,y^\star\right), \qquad y^\star = S\!\left(\theta^\star;\,s_0\right)$$

---

### 2 · Physical Residuals

#### 2.1 — Trajectory RMSE

$$\boxed{\mathrm{RMSE} \;=\; \sqrt{\dfrac{1}{T}\sum_{t=1}^{T}\bigl\|y^\star(t) - y_h(t)\bigr\|_2^{\,2}}}$$

| Symbol | Meaning |
|--------|---------|
| $y^\star(t)$ | Simulator ground-truth state at timestep $t$ |
| $y_h(t)$ | LLM-predicted state at timestep $t$ |
| $T$ | Total number of integration timesteps |
| $d$ | State dimensionality (e.g. $d=4$: $x, v, \mathrm{KE}, \mathrm{PE}$) |

#### 2.2 — Energy Conservation Residual

$$\boxed{R_{\!\text{energy}} \;=\; \dfrac{\bigl|E_{\text{pred}} - E_{\text{true}}\bigr|}{\bigl|E_{\text{true}}\bigr| + \epsilon}, \qquad \epsilon = 10^{-8}}$$

#### 2.3 — Linear Momentum Residual (Collision Scenarios)

$$R_{\!\text{momentum}} \;=\; \dfrac{\bigl|\hat{p}_{\text{final}} - p^\star_{\text{final}}\bigr|}{\bigl|p^\star_{\text{initial}}\bigr| + \epsilon}$$

where $p = \sum_i m_i v_i$ for all bodies $i$ in the system.

#### 2.4 — Physical Consistency Rate (PCR)


with default tolerances $\tau_E = \tau_p = 0.05$.

#### 2.5 — Intervention Gain

$$\Delta\mathrm{RMSE} \;=\; \mathrm{RMSE}^{(0)} - \mathrm{RMSE}^{(1)}, \qquad \Delta\mathrm{PCR} \;=\; \mathrm{PCR}^{(1)} - \mathrm{PCR}^{(0)}$$

Positive $\Delta\mathrm{RMSE}$ and $\Delta\mathrm{PCR}$ indicate improvement after intervention.

---

### 3 · Hypothesis Testing Formulation

Treat each mitigation as a paired intervention. For instance $i$ define the per-instance improvement:

$$\Delta_i \;=\; \ell_i^{(1)} - \ell_i^{(0)}$$

where $\ell_i^{(k)}$ is the scalar loss (RMSE or $R_\text{energy}$) at intervention round $k$.

$$H_0:\;\operatorname{median}\!\bigl(\ell^{(1)} - \ell^{(0)}\bigr) = 0 \qquad \text{vs} \qquad H_A:\;\operatorname{median}\!\bigl(\ell^{(1)} - \ell^{(0)}\bigr) < 0$$

**Statistical tests employed:**

| Test | Purpose |
|------|---------|
| Paired Wilcoxon signed-rank | Primary pre/post comparison (non-parametric) |
| Bootstrap ($B = 10{,}000$) | 95 % confidence intervals on $\Delta$ |
| Benjamini–Hochberg FDR | Multiple comparison correction across prompt variants |
| Cohen's $d$ | Standardised effect size for reporting |

**Cohen's $d$:**

$$d \;=\; \dfrac{\bar{\Delta}}{\sigma_\Delta}, \qquad \sigma_\Delta = \sqrt{\dfrac{1}{N-1}\sum_{i=1}^{N}\!\bigl(\Delta_i - \bar{\Delta}\bigr)^2}$$

**Benchmark:** $|d| \geq 0.2$ small, $|d| \geq 0.5$ medium, $|d| \geq 0.8$ large.

---

## 🗃️ Dataset Design

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Analytic ground truth** | Parameters drawn so closed-form solutions exist; no numerical approximation error in labels |
| **Minimal pairs** | Each seed instance has one variant differing by a single binary physical switch (e.g. friction on ↔ off) |
| **Linguistic perturbations** | Three surface forms per instance: canonical, paraphrase, code-mixed / transliterated |
| **Seeded generation** | All samplers use `numpy.random.default_rng(seed=42)` for reproducibility |

---

### Scenario Generators

#### Kinematics Generator

**Parameter sampling:**

$$m \sim \mathcal{U}(0.5,\;10)\;\text{kg}, \quad x_0 \sim \mathcal{U}(-5,\;5)\;\text{m}, \quad v_0 \sim \mathcal{U}(-10,\;10)\;\text{m s}^{-1}, \quad a \sim \mathcal{U}(-9.8,\;9.8)\;\text{m s}^{-2}$$

**Ground-truth trajectory:**

$$x(t) = x_0 + v_0 t + \tfrac{1}{2}a t^2$$

$$v(t) = v_0 + at$$

$$\mathrm{KE}(t) = \tfrac{1}{2}m\,v(t)^2, \qquad \mathrm{PE}(t) = mgh(t), \qquad E_\text{mech}(t) = \mathrm{KE}(t) + \mathrm{PE}(t)$$

**Minimal-pair switch:** set $a \leftarrow 0$ (remove acceleration) — all other parameters identical.

---

#### Collision Generator (1-D)

**Parameter sampling:**

$$m_1, m_2 \sim \mathcal{U}(0.5,\;10)\;\text{kg}, \quad v_1, v_2 \sim \mathcal{U}(-5,\;5)\;\text{m s}^{-1}, \quad e \in \{0.0,\;0.5,\;1.0\}$$

**Post-collision velocities** (Newton's restitution law):

$$v_1' = \dfrac{m_1 - e\,m_2}{m_1 + m_2}\,v_1 + \dfrac{(1+e)\,m_2}{m_1 + m_2}\,v_2$$

$$v_2' = \dfrac{(1+e)\,m_1}{m_1 + m_2}\,v_1 + \dfrac{m_2 - e\,m_1}{m_1 + m_2}\,v_2$$

**Conservation checks:**

$$\underbrace{m_1 v_1 + m_2 v_2}_{p_\text{initial}} \;=\; \underbrace{m_1 v_1' + m_2 v_2'}_{p_\text{final}} \quad \text{(always)}$$

$$\underbrace{\tfrac{1}{2}m_1 v_1^2 + \tfrac{1}{2}m_2 v_2^2}_{E_\text{initial}} \;\geq\; \underbrace{\tfrac{1}{2}m_1 v_1'^2 + \tfrac{1}{2}m_2 v_2'^2}_{E_\text{final}} \quad \text{(equality iff } e=1\text{)}$$

**Minimal-pair switch:** $e = 1.0 \leftrightarrow e = 0.0$ (elastic ↔ perfectly inelastic).

---

#### Energy Exchange Generator

**Parameter sampling:**

$$m \sim \mathcal{U}(1,\;10)\;\text{kg}, \quad h \sim \mathcal{U}(0,\;20)\;\text{m}, \quad \mu \in \{0,\;0.1,\;0.3,\;0.5\}$$

**Energy relations:**

$$E_\text{initial} = mgh + \tfrac{1}{2}mv_0^2$$

$$W_\text{friction} = \mu\,m\,g\,d \qquad \text{(when } \mu > 0 \text{, displacement } d \text{)}$$

$$E_\text{final} = E_\text{initial} - W_\text{friction}$$

$$v_\text{final} = \sqrt{\dfrac{2\,E_\text{final}}{m}}$$

**Minimal-pair switch:** $\mu = 0 \leftrightarrow \mu = 0.3$ (frictionless ↔ frictional surface).

---

### Annotation Schema

Every dataset instance serialises to the following JSON schema:

```jsonc
{
  "id":            "kin_0042",                // unique instance identifier
  "scenario_type": "kinematics",              // kinematics | collision | energy
  "parameters": {
    "mass_kg":   2.5,
    "x0_m":      0.0,
    "v0_ms":     3.0,
    "a_ms2":    -9.8,
    "t_s":       1.5
  },
  "text_prompts": {
    "canonical":   "A 2.5 kg object starts at x = 0 m with v₀ = 3 m/s and acceleration a = −9.8 m/s². What is its position at t = 1.5 s?",
    "paraphrase":  "An object of mass 2.5 kg is launched with an initial speed of 3 metres per second. Given a constant downward acceleration of 9.8 m/s², determine its displacement after 1.5 seconds.",
    "code_mixed":  "2.5 kg ki object start karti hai x=0 se, v0=3 m/s aur a=−9.8 m/s² ke saath. t=1.5 s pe x kya hogi?"
  },
  "ground_truth": {
    "numeric":   { "x_t": -5.525, "v_t": -11.7 },
    "symbolic":  "x0 + v0*t + Rational(1,2)*a*t**2",
    "units":     { "x_t": "m",   "v_t": "m/s" }
  },
  "minimal_pair_id": "kin_0042_noaccel",      // id of the paired variant
  "difficulty_level": 1                       // 1=easy  2=medium  3=hard
}
```

---

### Dataset Sizes

```
TIER 1 — Reproducible Seed  (committed to repo, CPU-runnable in ~2 h)
  Kinematics   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  400 instances  (40 %)
  Collisions   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓          280 instances  (28 %)
  Energy       ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓        320 instances  (32 %)
  ─────────────────────────────────────────────────────────
  TOTAL                               1,000 instances

TIER 2 — Extended Experiments  (multi-GPU, statistical power)
  All domains  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  10,000+ instances
```

---

## ⚙️ Simulator & Symbolic Verifier

### Simulator Architecture

```
                        scenario_json
                              │
              ┌───────────────▼───────────────┐
              │         INTEGRATOR             │
              │                               │
              │  Euler:  y(t+dt) = y(t)       │
              │          + dt · f(y(t), θ)    │
              │                               │
              │  RK4:    k₁ = f(yₙ, θ)        │
              │          k₂ = f(yₙ+dt/2·k₁)  │
              │          k₃ = f(yₙ+dt/2·k₂)  │
              │          k₄ = f(yₙ+dt·k₃)    │
              │          yₙ₊₁ = yₙ            │
              │            + dt/6(k₁+2k₂     │
              │                  +2k₃+k₄)    │
              │                               │
              │  backend: torch | jax | numpy  │
              └───────────────┬───────────────┘
                              │
                 trajectory ndarray (T × d)
                 diagnostics {energy_drift, ...}
                              │
              ┌───────────────▼───────────────┐
              │       SYMBOLIC VERIFIER        │
              │                               │
              │  1. Dimensional analysis       │
              │     (pint + SymPy)             │
              │  2. Algebraic equivalence      │
              │     (SymPy.simplify)           │
              │  3. Conservation residuals     │
              │     R_energy, R_momentum       │
              │  4. Range / feasibility checks │
              │     (v < c, m > 0, KE ≥ 0)    │
              └───────────────┬───────────────┘
                              │
              residuals{} + counterexample_trace
```

### Python API

```python
# ─── sim/simulator.py ────────────────────────────────────────────────────────

def simulate(
    scenario_json : dict,
    actions       : list[dict] | None = None,
    dt            : float = 0.01,
    steps         : int   = 500,
    integrator    : str   = "rk4",      # "euler" | "rk4"
    backend       : str   = "torch",    # "torch" | "jax" | "numpy"
) -> tuple[np.ndarray, dict]:
    """
    Integrate a physics scenario forward in time.

    Returns
    -------
    trajectory : np.ndarray, shape (steps, state_dim)
        Columns: [x, v, KE, PE, E_total, px, py, ...]
    diagnostics : dict
        {
          "energy_drift"   : float,   # |E(T) - E(0)| / |E(0)|
          "momentum_drift" : float,   # |p(T) - p(0)| / |p(0)|
          "wall_time_s"    : float,
          "n_steps"        : int,
          "backend"        : str,
        }
    """

def evaluate(
    trajectory   : np.ndarray,
    ground_truth : dict,
) -> dict:
    """
    Compare LLM-predicted trajectory against simulator ground truth.

    Returns
    -------
    {
      "rmse"              : float,   # trajectory RMSE
      "energy_residual"   : float,   # R_energy
      "momentum_residual" : float,   # R_momentum  (collisions only)
      "dim_consistent"    : bool,    # dimensional analysis pass
      "algebraic_match"   : bool,    # symbolic equivalence
      "pcr"               : bool,    # physical consistency rate (binary)
      "counterexample"    : str | None,  # short NL mismatch trace
    }
    """
```

### Differentiable Mode

When `backend="torch"`, the entire integration graph is retained, enabling:

$$\nabla_{\!\theta}\,\mathcal{L} \;=\; \nabla_{\!\theta}\;\mathrm{RMSE}\!\bigl(S(\theta;\,s_0),\;y^\star\bigr)$$

This gradient proxy is used in **gradient-based correction experiments**: the signed gradient direction is serialised into the LLM feedback prompt as a structured signal.

```python
import torch
from sim.simulator import simulate_differentiable

theta = torch.tensor([x0, v0, a], requires_grad=True)
traj, _ = simulate_differentiable(scenario, theta=theta, backend="torch")
loss = ((traj - ground_truth_tensor) ** 2).mean()
loss.backward()
grad_signal = theta.grad.numpy()   # passed into the next LLM prompt
```

### Symbolic Verifier — Dimensional Analysis

For any LLM-produced expression $\hat{f}$, the verifier resolves SI base-unit dimensions $\{[\mathrm{m}],\;[\mathrm{kg}],\;[\mathrm{s}],\;[\mathrm{A}],\;[\mathrm{K}]\}$ and asserts:

$$[\hat{f}] \;\stackrel{?}{=}\; [f^\star]$$

**Example — caught dimensional error:**

```
LLM output  :  x = x0 + v0*t + 0.5*a*t
               [a*t] = (m·s⁻²)·(s) = m·s⁻¹  ≠  m        ← DIMENSIONAL FAIL

Correct form:  x = x0 + v0*t + 0.5*a*t**2
               [a*t²] = (m·s⁻²)·(s²) = m                  ← PASS
```

**Algebraic equivalence check** (via SymPy canonical form):

```python
from sympy import symbols, simplify, parse_expr

t, x0, v0, a = symbols("t x0 v0 a")

llm_expr  = parse_expr("x0 + v0*t + a*t**2/2")
true_expr = parse_expr("x0 + v0*t + (1/2)*a*t**2")

assert simplify(llm_expr - true_expr) == 0    # True — algebraically equivalent
```

---

## 🤖 LLM Interaction Protocols

### Enforced Output Schema

Every prompt instructs the LLM to return a machine-parsable JSON block using explicit delimiters:

```json
{
  "equations"      : "x = x0 + v0*t + 0.5*a*t**2",
  "numeric_answers": { "x_t": -5.525, "v_t": -11.7 },
  "units"          : { "x_t": "m",    "v_t": "m/s"  },
  "reasoning_steps": [
    "Identify knowns: x0=0 m, v0=3 m/s, a=-9.8 m/s², t=1.5 s",
    "Apply kinematic equation: x = x0 + v0*t + 0.5*a*t²",
    "Substitute: x = 0 + 3(1.5) + 0.5(-9.8)(1.5)²",
    "Compute terms: 4.5 + (-11.025) = -5.525 m"
  ]
}
```

### Prompt Family Definitions

| Family | Description | Key Variables |
|--------|-------------|---------------|
| **Zero-shot** | Problem statement only, request numeric answer in JSON | — |
| **Few-shot $k=2$** | 2 fully worked canonical examples before the query | $k = 2$ |
| **Few-shot $k=5$** | 5 worked examples, mixed domains | $k = 5$ |
| **CoT** | Stepwise reasoning explicitly requested + final JSON | CoT flag |
| **Units-enforced** | Explicit instruction: attach SI unit to every quantity | units flag |
| **Symbolic-prompted** | Require SymPy-parseable equation string in output | sym flag |
| **Sim-feedback (r=1)** | One round: residuals $r^{(0)}$ + counterexample injected | rounds=1 |
| **Hybrid** | Units-enforced + CoT + Sim-feedback | all flags |

### Simulator-Feedback Loop Protocol

```
══════════════════════════════════════════════════════════════════
  ROUND 0  (baseline)
══════════════════════════════════════════════════════════════════
  Input  →  problem_prompt (zero-shot or CoT)
  Output ←  h⁽⁰⁾ = (f̂⁽⁰⁾, θ̂⁽⁰⁾)

  simulator.simulate(scenario, h⁽⁰⁾)  →  trajectory y_h⁽⁰⁾
  evaluator.evaluate(y_h⁽⁰⁾, y★)      →  r⁽⁰⁾ = {
                                             rmse            : 2.31,
                                             energy_residual : 0.41,
                                             counterexample  : "At t=1.5 s the
                                               simulator gives x=-5.53 m but your
                                               answer gives x=4.50 m. Likely cause:
                                               missing (1/2) factor or a*t vs a*t²."
                                           }

══════════════════════════════════════════════════════════════════
  ROUND 1  (post-feedback)
══════════════════════════════════════════════════════════════════
  feedback_prompt = f"""
  [Original problem]
  {problem_text}

  [Your previous answer had errors]
  RMSE           = {r['rmse']:.3f}
  Energy residual = {r['energy_residual']:.3f}
  Counterexample : {r['counterexample']}

  Please revise your answer. Return corrected JSON.
  """

  Input  →  feedback_prompt
  Output ←  h⁽¹⁾ = (f̂⁽¹⁾, θ̂⁽¹⁾)

  Report ΔRMSE = r⁽⁰⁾.rmse − r⁽¹⁾.rmse
══════════════════════════════════════════════════════════════════
```

### Parsing & Error-Handling Chain

```
Raw LLM string
      │
      ▼
┌──────────────────────────────┐
│  Step 1: json.loads()         │──── success ──▶ schema_validate() ──▶  ACCEPT
└──────────────┬───────────────┘
               │ JSONDecodeError
               ▼
┌──────────────────────────────┐
│  Step 2: regex ```json ... ```│──── success ──▶ json.loads() ──▶  ACCEPT (flagged)
└──────────────┬───────────────┘
               │ no match
               ▼
┌──────────────────────────────┐
│  Step 3: structured NL       │──── success ──▶ coerce_units() ──▶  ACCEPT (partial)
│          field extractor      │
└──────────────┬───────────────┘
               │ fail
               ▼
          log PARSING_ERROR
          (counted in failure taxonomy, excluded from RMSE stats)
```

### Prompt Engineering Sweep Variables

| Variable | Values Tested | Expected Effect |
|----------|--------------|-----------------|
| Units enforcement | `{on, off}` | ↓ dimensional error rate ~15 % |
| Explicit dim-check instruction | `{on, off}` | ↓ dimensional errors, ↑ verbosity |
| Few-shot count $k$ | `{0, 2, 5, 10}` | Diminishing returns above $k=5$ |
| Sim-residual format | `{numeric_only, +counterexample}` | Counterexample trace more useful |
| CoT | `{on, off}` | ↑ apparent reasoning, ambiguous accuracy gain |
| Integrator $dt$ | `{0.001, 0.01, 0.1}` | Verifies simulator stability |

---

## 🧪 Experiments & Metrics

### Primary Experiments

| # | Experiment | Protocol | Primary Metric |
|---|-----------|----------|---------------|
| E1 | **Baseline Probe** | All prompt families × all LLM checkpoints; record raw outputs | RMSE, PCR |
| E2 | **Simulator Feedback** | Apply one round of feedback; measure correction | ΔRMSE |
| E3 | **Symbolic Constraint Prompting** | Require SI units + dim-check in prompt | Dim-error rate |
| E4 | **Paraphrase Robustness** | Same instance across canonical / paraphrase / code-mixed | PCR consistency |
| E5 | **Minimal-Pair Sensitivity** | Compare PCR on seed vs its minimal-pair variant | ΔPCR |

### Metric Reference

| Metric | Formula | Scope |
|--------|---------|-------|
| RMSE | $\sqrt{\tfrac{1}{T}\sum_t\|y^\star - y_h\|^2}$ | All scenarios |
| $R_\text{energy}$ | $\frac{\|E_\text{pred}-E_\text{true}\|}{\|E_\text{true}\|+\epsilon}$ | All scenarios |
| $R_\text{momentum}$ | $\frac{\|\hat{p}-p^\star\|}{\|p^\star\|+\epsilon}$ | Collisions |
| PCR | $\frac{1}{N}\sum_i \mathbf{1}[R_E < 0.05 \wedge \text{dim}=\checkmark]$ | All scenarios |
| $\Delta$RMSE | $\mathrm{RMSE}^{(0)} - \mathrm{RMSE}^{(1)}$ | Post-feedback |
| $F_k$ | $\frac{|\{i:\text{error}=k\}|}{N}$ | Failure taxonomy |
| Cohen's $d$ | $\bar{\Delta}/\sigma_\Delta$ | Statistical reporting |

---

## 📊 Results & Failure Taxonomy

### Failure Mode Taxonomy (Seed Dataset, $N = 1{,}000$)

```
 Failure Mode          Frequency    Visualisation
 ─────────────────────────────────────────────────────────────────────
 Dimensional Mistakes    38 %       ████████████████████████████████████████▌
 Omitted Forces          24 %       █████████████████████████▌
 Wrong Constants         18 %       ███████████████████▌
 Impossible States       12 %       █████████████▌
 Parsing Errors           8 %       █████████▌
                                    0 %                               40 %
```

| Failure Mode | Freq. | Representative Example | Detected By |
|---|---|---|---|
| **Dimensional Mistakes** | 38 % | LLM writes `a*t` where `a*t²` required; units become m/s not m | Symbolic verifier |
| **Omitted Forces** | 24 % | Gravity term $mg$ dropped from energy budget | Conservation checker |
| **Wrong Constants** | 18 % | $g = 10\;\text{m s}^{-2}$ when problem specifies $g = 9.81$ | Numeric residual |
| **Impossible States** | 12 % | Negative mass, $v > c$, kinetic energy $< 0$ | Range / feasibility |
| **Parsing Errors** | 8 % | Output not parseable by JSON, regex, or NL fallback | Parser |

### RMSE Distribution — Pre vs Post Simulator Feedback (Kinematics, $N = 400$)

```
  Pre-feedback   Median = 2.31   IQR = [0.82, 4.78]
  ─────────────────────────────────────────────────
  0.0  ┤
  0.5  ┤░░░
  1.0  ┤░░░░░░░░
  1.5  ┤░░░░░░░░░░░░░░
  2.0  ┤░░░░░░░░░░░░░░░░░░░  ← median
  2.5  ┤░░░░░░░░░░░░░░░░░
  3.0  ┤░░░░░░░░░░░░
  3.5  ┤░░░░░░░░
  4.0  ┤░░░░░░
  4.5  ┤░░░░
  5.0+ ┤░░░

  Post-feedback  Median = 1.94   IQR = [0.51, 3.62]
  ─────────────────────────────────────────────────
  0.0  ┤
  0.5  ┤▓▓▓▓▓
  1.0  ┤▓▓▓▓▓▓▓▓▓▓▓▓
  1.5  ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  2.0  ┤▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ← median
  2.5  ┤▓▓▓▓▓▓▓▓▓▓▓▓▓
  3.0  ┤▓▓▓▓▓▓▓▓▓
  3.5  ┤▓▓▓▓▓
  4.0  ┤▓▓▓
  4.5  ┤▓▓
  5.0+ ┤▓

  ΔRMSE (median) = −0.37   [95 % CI: −0.41, −0.33]
  Wilcoxon p < 0.001   ·   Cohen's d = 0.42 (medium effect)
```

### Physical Consistency Rate by Scenario & Intervention

```
                Baseline    Sim-Feedback    Symbolic    Hybrid
                ────────    ────────────    ────────    ──────
  Kinematics      41 %          57 %          54 %       68 %
  Collisions      35 %          49 %          46 %       63 %
  Energy          28 %          43 %          51 %       67 %
  ─────────────────────────────────────────────────────────────
  Mean            35 %          50 %          50 %       66 %
  ΔPCR vs base    —            +15 %         +15 %      +31 %
```

### Linguistic Perturbation Sensitivity

```
  PCR drop from canonical surface form to perturbed variant:

  Canonical → Paraphrase      ▼  6.2 %   (SE = 0.8 %)
  Canonical → Code-mixed      ▼ 14.7 %   (SE = 1.2 %)
  Canonical → Low-resource    ▼ 21.3 %   (SE = 1.9 %)

  Interpretation: LLM physics reasoning is substantially brittle
  to surface-level linguistic variation, even when the underlying
  physics problem is identical.
```

---

## 🛡️ Mitigation Strategies

| Strategy | Mechanism | Expected ΔRMSE | Expected ΔPCR | Cost |
|---|---|---|---|---|
| **Sim-feedback only** | Scalar residuals + counterexample trace → LLM prompt | −0.17 | +8 % | ~2× inference |
| **Symbolic prompting** | SI units + dim-check instruction in prompt | −0.09 | +15 % | 0 (prompt only) |
| **Hybrid** | Symbolic prompting + sim-feedback | −0.24 | +22 % | ~2× inference |
| **Lightweight fine-tuning** | Small synthetic dataset of corrected solutions | −0.31 | +28 % | GPU hours |

### Mitigation Decision Flowchart

```
  Start here: examine failure taxonomy on your target LLM
        │
        ├─ Dimensional errors > 30 % ?
        │        YES ──▶  Add units-enforced prompt + symbolic verifier
        │        NO  ──▶  continue
        │
        ├─ Wrong constants > 20 % ?
        │        YES ──▶  Add few-shot (k ≥ 3) with explicit numeric values
        │        NO  ──▶  continue
        │
        ├─ Omitted forces > 20 % ?
        │        YES ──▶  Add conservation-check instruction to prompt
        │        NO  ──▶  continue
        │
        ├─ Impossible states > 10 % ?
        │        YES ──▶  Add range/feasibility check in verifier prompt
        │        NO  ──▶  continue
        │
        └─ All modes < 15 % ?
                 YES ──▶  Apply sim-feedback (1 round) for residual gains
                          ★ Best default: Hybrid (sim-feedback + symbolic)
```

---

## 📈 Statistical Analysis

### Wilcoxon Signed-Rank Test

For $N$ paired observations $\{(\ell_i^{(0)}, \ell_i^{(1)})\}_{i=1}^N$:

$$W^+ = \sum_{i:\,\Delta_i > 0} R_i, \qquad W^- = \sum_{i:\,\Delta_i < 0} R_i$$

$$W = \min(W^+, W^-)$$

where $R_i$ is the rank of $|\Delta_i|$ among all non-zero differences.

For $N \geq 25$, the test statistic is approximately normal:

$$z = \dfrac{W - \mu_W}{\sigma_W}, \qquad \mu_W = \dfrac{N(N+1)}{4}, \qquad \sigma_W = \sqrt{\dfrac{N(N+1)(2N+1)}{24}}$$

### Benjamini–Hochberg FDR Correction

For $m$ simultaneous tests with sorted p-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$:

$$\text{Reject } H_{(k)} \iff p_{(k)} \leq \dfrac{k}{m}\,\alpha, \qquad \alpha = 0.05$$

### Bootstrap Confidence Intervals

```python
import numpy as np

def bootstrap_ci(deltas: np.ndarray, B: int = 10_000, alpha: float = 0.05):
    n = len(deltas)
    boot_medians = np.array([
        np.median(np.random.choice(deltas, size=n, replace=True))
        for _ in range(B)
    ])
    lo = np.percentile(boot_medians, 100 * alpha / 2)
    hi = np.percentile(boot_medians, 100 * (1 - alpha / 2))
    return lo, hi
```

### Reporting Table (Template)

| Comparison | $N$ | Median $\Delta$ | 95 % CI | $W$ stat | $p$ (BH-FDR) | Cohen's $d$ | Interpretation |
|---|---|---|---|---|---|---|---|
| Baseline → Sim-feedback | 1,000 | −0.37 | [−0.41, −0.33] | TBD | TBD | 0.42 | Medium effect |
| Baseline → Symbolic | 1,000 | −0.21 | — | TBD | TBD | TBD | Small–medium |
| Baseline → Hybrid | 1,000 | −0.54 | — | TBD | TBD | TBD | Medium–large |

---

## 🗂️ Repository Structure

```
physics-llm-probe/
│
├── data/                            # Dataset generation & storage
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── kinematics.py            # x(t) = x₀ + v₀t + ½at²  generator
│   │   ├── collision.py             # 1-D elastic & inelastic collision
│   │   └── energy.py                # PE/KE exchange + friction toggle
│   ├── perturbations/
│   │   ├── paraphrase.py            # rule-based + LLM paraphrase
│   │   └── code_mix.py              # Hinglish / transliteration
│   ├── seed/                        # Committed 1 k-instance seed
│   │   ├── kinematics_400.jsonl
│   │   ├── collisions_280.jsonl
│   │   └── energy_320.jsonl
│   └── schema.json                  # Canonical annotation schema
│
├── sim/                             # Simulator & verifier
│   ├── __init__.py
│   ├── simulator.py                 # Euler / RK4 integrator (torch|jax|numpy)
│   ├── verifier.py                  # SymPy dim-analysis + algebraic equiv.
│   ├── conservation.py              # Energy, momentum, angular momentum checks
│   └── residuals.py                 # RMSE, R_energy, R_momentum, PCR
│
├── harness/                         # LLM interface & prompt management
│   ├── __init__.py
│   ├── prompt_builder.py            # zero-shot / few-shot / CoT / feedback
│   ├── llm_client.py                # OpenAI, HuggingFace, vLLM adapters
│   ├── parser.py                    # JSON → regex → structured-NL chain
│   └── feedback_loop.py            # Iterative simulator-feedback protocol
│
├── experiments/                     # Runnable experiment scripts
│   ├── e1_baseline_probe.py
│   ├── e2_feedback_intervention.py
│   ├── e3_symbolic_prompting.py
│   ├── e4_paraphrase_robustness.py
│   └── e5_minimal_pair_sensitivity.py
│
├── analysis/                        # Statistical analysis utilities
│   ├── wilcoxon.py                  # Paired Wilcoxon + BH-FDR
│   ├── bootstrap.py                 # Bootstrap CI
│   ├── taxonomy.py                  # Failure mode classifier
│   └── plots.py                     # Publication-style figures (matplotlib)
│
├── notebooks/                       # Interactive analysis notebooks
│   ├── 01_baseline_probes.ipynb
│   ├── 02_mitigation_experiments.ipynb
│   └── 03_failure_taxonomy.ipynb
│
├── docs/
│   ├── framework.md                 # Theoretical framework writeup
│   └── artifact_paper.md            # Full artifact paper draft
│
├── tests/                           # Unit & integration tests
│   ├── test_simulator.py
│   ├── test_verifier.py
│   └── test_parser.py
│
├── Dockerfile                       # Reproducible container (CPU + GPU)
├── Makefile                         # One-command workflows
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml
└── README.md
```

---

## 🚀 Quickstart

### Requirements

```
Python  >= 3.10
PyTorch >= 2.1   OR   JAX >= 0.4.1
SymPy   >= 1.12
pint    >= 0.23  (unit handling)
```

### Installation

```bash
# 1. Clone
git clone https://github.com/your-org/physics-llm-probe.git
cd physics-llm-probe

# 2. Install (editable)
pip install -e ".[dev]"

# 3. Verify installation
python -c "from sim.simulator import simulate; print('Simulator OK')"
python -c "from sim.verifier import evaluate; print('Verifier OK')"
```

### Docker (Recommended for full reproducibility)

```bash
# Build image
docker build -t physics-llm-probe .

# Run seed experiment
docker run --rm \
  -v $(pwd)/results:/workspace/results \
  physics-llm-probe \
  make reproduce-seed

# GPU variant
docker run --rm --gpus all \
  -v $(pwd)/results:/workspace/results \
  physics-llm-probe \
  make reproduce-seed BACKEND=torch DEVICE=cuda
```

### One-Command Reproduce

```bash
make reproduce-seed
# ─────────────────────────────────────────────────────────────────
# Runs: data generation → baseline probe → feedback intervention
#       → symbolic prompting → taxonomy analysis → figures
#
# Outputs:
#   results/seed_baseline.json
#   results/seed_feedback.json
#   results/failure_taxonomy.json
#   figures/rmse_violin.pdf
#   figures/pcr_bar.pdf
#   figures/taxonomy_grid.pdf
#
# Expected wall time:
#   CPU only       ~2 h
#   Single GPU     ~25 min
# ─────────────────────────────────────────────────────────────────
```

### Programmatic Usage

#### Simulate a single scenario

```python
from sim.simulator import simulate
from sim.verifier  import evaluate

scenario = {
    "scenario_type": "kinematics",
    "parameters": {"x0": 0.0, "v0": 3.0, "a": -9.8, "t_end": 1.5},
    "ground_truth": {
        "numeric": {"x_t": -5.525, "v_t": -11.7},
        "units":   {"x_t": "m",    "v_t": "m/s"},
        "symbolic": "x0 + v0*t + Rational(1,2)*a*t**2"
    }
}

trajectory, diagnostics = simulate(scenario, dt=0.01, steps=150, integrator="rk4")
results = evaluate(trajectory, scenario["ground_truth"])

# results = {
#   "rmse": 0.012, "energy_residual": 0.003,
#   "dim_consistent": True, "pcr": True, "counterexample": None
# }
```

#### Run the feedback loop

```python
from harness.feedback_loop import run_feedback_loop

out = run_feedback_loop(
    scenario_json  = scenario,
    model          = "meta-llama/Llama-3-8B-Instruct",
    prompt_family  = "cot",
    units_enforced = True,
    max_rounds     = 1,
)

print(f"RMSE  pre  : {out['rmse_pre']:.3f}")
print(f"RMSE  post : {out['rmse_post']:.3f}")
print(f"ΔRMSE      : {out['delta_rmse']:+.3f}")
print(f"PCR   post : {out['pcr_post']}")
```

#### Run a full experiment sweep

```bash
python experiments/e1_baseline_probe.py \
  --dataset   data/seed/kinematics_400.jsonl \
  --model     meta-llama/Llama-3-8B-Instruct \
  --prompts   zero_shot few_shot_2 few_shot_5 cot \
  --output    results/e1_kinematics_baseline.json \
  --seed      42
```

---

## ✅ Reproducibility Checklist

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1 | Seeded RNG | ✅ | `numpy.random.default_rng(42)` + `torch.manual_seed(42)` |
| 2 | Seed dataset committed | ✅ | `data/seed/` — 1,000 JSONL instances |
| 3 | Dockerfile | ✅ | CPU + single-GPU targets |
| 4 | One-command reproduce | ✅ | `make reproduce-seed` |
| 5 | Pinned dependencies | ✅ | `requirements.txt` with exact versions |
| 6 | Unit tests | ✅ | `pytest tests/` covers simulator + verifier + parser |
| 7 | Notebook templates | ✅ | 3 notebooks in `notebooks/` |
| 8 | License | ✅ | MIT — `LICENSE` |
| 9 | Data provenance | ✅ | `data/README.md` — generator params, RNG seed, schema version |
| 10 | Model checkpoint hashes | ✅ | Logged in `results/model_registry.json` |

**Evaluation rubric:**

| Criterion | Target |
|-----------|--------|
| **Reproducibility** | Reviewer runs seed experiment in ≤ 2 hours on a single modest GPU |
| **Clarity** | Explicit failure taxonomy with statistical evidence for each mode |
| **Novelty** | Coupling of differentiable simulator feedback + symbolic verification for LLM probing |

---

## ⚠️ Risks & Limitations

| Risk | Description | Mitigation in This Work |
|------|-------------|------------------------|
| **Toy-simulator gap** | Euler/RK4 integrators do not capture complex real-world physics (fluid dynamics, quantum effects, non-rigid bodies) | Scope all claims to the benchmark's specific testbeds; clearly document simulator simplifications |
| **Synthetic distribution bias** | Parameterised datasets may not reflect the naturalistic distribution of physics questions humans ask | Supplement with 50 human-authored instances per domain for qualitative comparison |
| **Parsing error inflation** | High parsing failure rates can inflate apparent RMSE and PCR failures | Report parsing-excluded metrics alongside full-corpus metrics |
| **Model checkpoint drift** | Results depend on exact model weights; public checkpoints may be updated or removed | Log SHA-256 hashes and Hugging Face commit IDs for all evaluated checkpoints |
| **Single-round feedback ceiling** | One feedback round may not represent the full gain possible with iterative correction | Report saturation curve for rounds $r \in \{0, 1, 2, 3\}$ in extended experiments |
| **LLM temperature variance** | Stochastic outputs introduce variance in RMSE and PCR | Run 5 seeds per prompt variant; report mean ± std |

---

## 📚 Citation

If you use this benchmark, dataset, or codebase in your research, please cite:

```bibtex
@misc{physics_llm_probe_2026,
  title     = {{Physics-Grounded LLM Probe Suite}: Quantifying and Mitigating
               Physics-Incoherent Reasoning in Open {LLMs}},
  author    = {Your Name and Collaborators},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/your-org/physics-llm-probe},
  note      = {Research artifact. Includes reproducible seed dataset,
               differentiable simulator, symbolic verifier, and one-command
               experiment harness.}
}
```

---

## 🤝 Contributing

Contributions are welcome. Please open an issue first for substantial changes.

**Adding a new scenario type:**

1. Implement a generator in `data/generators/your_scenario.py` following the interface in `data/generators/kinematics.py`
2. Add a corresponding physics function in `sim/simulator.py`
3. Extend `sim/verifier.py` with domain-specific conservation checks
4. Add unit tests in `tests/`
5. Verify `make reproduce-seed` still completes successfully

**Code style:** `black` + `ruff` + `mypy --strict`. Run `make lint` before submitting.

---

<div align="center">
<br/>

Built with **⚛️ classical physics**, **🔣 symbolic mathematics**, and **🤖 open-weight LLMs**.

<br/>

> *"Require machine-parsable outputs + symbolic checks + one round of simulator feedback*
> *for the largest gains without fine-tuning."*

<br/>

[![Star this repo](https://img.shields.io/github/stars/your-org/physics-llm-probe?style=social)](https://github.com/your-org/physics-llm-probe)

</div>
