<p align="center">
  <img src="docs/assets/nadir_mark.svg" alt="Nadir logo" width="200"/>
</p>

# Nadir

**Nadir** is a Decision Support System (DSS) designed to solve multi-period trajectory optimization problems. By mapping performance profiles using **Partially Ordered Sets (Posets)** and leveraging **counterfactual simulation**, Nadir helps analysts and systems answer a core strategic question: 

> *"If we control the levers of specific KPIs, how should we schedule investments over time to beat competitors or reach the Pareto efficient frontier at the lowest possible cost?"*

Unlike traditional static benchmarking tools, Nadir acts as a **Geometric Model Predictive Control (MPC)** engine, evaluating cumulative state transitions and dynamic resource allocations across a discrete-time horizon ($t = 1 \dots T$).

---

## 🎯 Project Scope

This repository has a dual purpose:
1. **A ready-to-use engine** for simple, self-contained decision-support applications — computing Pareto frontiers, cost-optimal investment trajectories, and counterfactual simulations directly from tabular KPI data.
2. **A theoretical exercise**, exploring geometric and order-theoretic modeling as a foundation that other, more advanced decision-support products can build upon.

Ideas that go beyond this scope are tracked separately in [Future Developments](docs/docs/Future%20Developments.md) rather than implemented here.

---

## 🚀 Key Architectural Pillars
* **Poset-Based Dominance:** Subjects are evaluated across $n$-dimensional quantitative KPIs. The system builds transitively reduced Hasse diagrams to isolate Pareto-optimal maximal elements.
* **Multi-Period State Variables:** Differentiates between cumulative state variables (KPI levels) and per-period flow variables (investment costs) to model continuous progress without repetitive maintenance penalties.
* **Custom Cost Curves:** Supports user-defined smooth quadratic ($\alpha u^2$) and asymmetric non-smooth cost functions to simulate diminishing marginal returns and realistic asset decay.
* **Mathematical Optimization:** Formulates target-tracking or frontier-entry trajectories through non-linear mathematical programming (including MILP/MINLP formulations for disjunctive Pareto constraints).

---

## 🛠️ Tech Stack
* **Optimization & Modeling:** Python (SciPy, Pyomo, IPOPT)
* **Data & Graph Structures:** NetworkX, Polars / Pandas