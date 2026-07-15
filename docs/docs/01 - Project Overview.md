---
tags: [theory, nadir]
---

# 01 - Project Overview

## What Nadir is

Nadir is a **geometric and predictive Decision Support System (DSS)** for multi-period KPI trajectory optimization. Instead of static benchmarking ("where do we rank today?"), it answers a *planning* question: given a per-period budget and levers on specific KPIs, **how should investments be scheduled over a horizon $t = 1 \dots T$** so that the focus subject beats competitors or enters the Pareto-efficient frontier at the lowest cost?

Conceptually it behaves like a **Geometric Model Predictive Control (MPC)** engine: at each period it observes the state (all subjects' KPI vectors), evaluates candidate budget allocations, applies the best one, and lets the world evolve (competitor drift, passive decay) before re-planning.

## The three ingredients

1. **Order theory** — subjects are compared with Pareto dominance, forming a [[02 - Posets and Pareto Dominance|poset]]. Maximal elements = the Pareto frontier.
2. **A scalar objective** — the [[03 - Signed Distance Metric|signed distance]] $D(s) \le 0$ measures how far the focus subject is from dominating everyone; $D = 0$ means it sits on the frontier.
3. **Economics** — [[04 - Cost Functions|cost curves]] translate a desired % improvement on a KPI into money, modeling diminishing marginal returns.

The [[05 - Optimization and Simulation|optimizer]] closes the loop: split the per-period budget across KPIs to maximize next-period $D$.

## State vs. flow variables

A key modeling choice: KPI levels are **cumulative state variables** (progress persists across periods), while investments are **per-period flow variables**. Improvements are not "rented" — you don't pay maintenance to keep a gained level, but *passive decay* can erode it if you stop investing (see [[Configuration]]).

## Phases

### Phase 1 — Geometric-deterministic layer ✅
- Poset-based dominance with transitively reduced **Hasse diagrams**.
- User-defined **cost curves** (quadratic $\alpha u^2$, linear) with inverse mapping budget → max achievable Δ%.
- **Greedy per-period optimizer** over the budget simplex (myopic, single step lookahead).
- **Counterfactual simulation**: focus invests, competitors drift/jitter stochastically.

### Phase 2 — Predictive & causal layer 🔮 (planned)
- **Dynamic cost estimation**: marginal costs that respond to saturation and history.
- **Neural interdependency modeling**: a multi-output NN inside the loop simulates domino effects across KPIs when a budget allocation is applied.
- **Physics-informed constraints**: budget conservation via Softmax, non-negativity via Softplus — hard boundaries the optimizer cannot exploit.
- Full multi-period MPC planner (PyTorch/JAX, end-to-end differentiable trajectories).

## Tech stack

| Concern | Tools |
|---|---|
| Data | Polars (Pandas at the edges) |
| Graphs | NetworkX |
| Numerics | NumPy |
| Visualisation | Plotly |
| Notebooks | marimo |
| Phase 2 (target) | PyTorch / JAX, SciPy, Pyomo, IPOPT |

→ Next: [[02 - Posets and Pareto Dominance]]
