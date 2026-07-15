---
tags: [theory, nadir, optimization, simulation]
---

# 05 - Optimization and Simulation

## The per-period allocation problem

At each period the focus subject holds a period budget $B_t$ and must split it across $n$ KPIs with weights $w$ on the **simplex** ($w_i \ge 0$, $\sum w_i = 1$). Each slice $w_i B_t$ buys a KPI improvement via the inverse [[04 - Cost Functions|cost function]], producing a candidate next state. The optimizer picks:

$$w^\star = \arg\max_{w \in \Delta^{n-1}} D\big(\text{next\_state}(w)\big)$$

where $D$ is the [[03 - Signed Distance Metric]] evaluated against the current competitor states.

## How it's solved (Phase 1)

`optimize_allocation` in [`core/optimizer.py`](../../core/optimizer.py) does a **greedy grid search**: it enumerates a lattice of weight vectors on the simplex (`_simplex_grid`, resolution adapted to dimensionality by `_grid_resolution`) and evaluates $D$ for each candidate next state.

> [!warning] Myopic by design
> This is a *single-step lookahead* — it maximizes next-period $D$ only. It can be short-sighted when a KPI needs sustained multi-period investment to pay off. It also optimizes each KPI independently — see [[Future Developments]] for the open question of KPIs that are in opposition to each other.

## Counterfactual simulation

`simulate_history` in [`core/simulation.py`](../../core/simulation.py) runs the whole scenario over the horizon:

```
for t in 1..T:
    1. Focus subject spends B_t:
       - adaptive mode: re-run optimize_allocation on the current state
       - static mode:   use fixed user-supplied weights
    2. All other subjects drift (passive_decay_rate ± drift_noise_std)
    3. Snapshot the poset state
```

It returns two DataFrames: **history** (every subject's KPI values per period) and **allocations** (the focus subject's budget split per period). `resolve_budget_schedule` lets the period budget be a constant or a per-period schedule.

## Reaching the frontier

`time_to_frontier(history, ...)` scans the simulated history and returns the **first period where the focus subject is a maximal element** of the poset (Pareto frontier, see [[02 - Posets and Pareto Dominance]]) — the headline KPI of a scenario: *"with this budget and these cost curves, we enter the frontier in period k."*

## Visual outputs

[[Core Modules#viz py|viz.py]] renders three Plotly views: the Hasse diagram at a chosen period, subject trajectories on a 2-KPI plane, and a stacked bar chart of budget allocations over time.

→ Technical deep-dive: [[Core Modules]] · [[Architecture]]
