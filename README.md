# Nadir 📐🔮

**Nadir** is a geometric and predictive Decision Support System (DSS) designed to solve multi-period trajectory optimization problems. By mapping performance profiles using **Partially Ordered Sets (Posets)** and leveraging **counterfactual simulation**, Nadir helps analysts and systems answer a core strategic question: 

> *"If we control the levers of specific KPIs, how should we schedule investments over time to beat competitors or reach the Pareto efficient frontier at the lowest possible cost?"*

Unlike traditional static benchmarking tools, Nadir acts as a **Geometric Model Predictive Control (MPC)** engine, evaluating cumulative state transitions and dynamic resource allocations across a discrete-time horizon ($t = 1 \dots T$).

---

## 🚀 Key Architectural Pillars

### 1. Geometric-Deterministic Layer (Phase 1)
* **Poset-Based Dominance:** Subjects are evaluated across $n$-dimensional quantitative KPIs. The system builds transitively reduced Hasse diagrams to isolate Pareto-optimal maximal elements.
* **Multi-Period State Variables:** Differentiates between cumulative state variables (KPI levels) and per-period flow variables (investment costs) to model continuous progress without repetitive maintenance penalties.
* **Custom Cost Curves:** Supports user-defined smooth quadratic ($\alpha u^2$) and asymmetric non-smooth cost functions to simulate diminishing marginal returns and realistic asset decay.
* **Mathematical Optimization:** Formulates target-tracking or frontier-entry trajectories through non-linear mathematical programming (including MILP/MINLP formulations for disjunctive Pareto constraints).

### 2. Predictive & Causal Layer (Phase 2)
* **Dynamic Cost Estimation:** Replaces static coefficients with a predictive model that dynamically adjusts marginal costs based on current system saturation and historical constraints.
* **Neural Interdependency Modeling:** Embeds a multi-output Neural Network into the optimization loop. Instead of brute-forcing raw KPI shifts, the optimizer refines *budget allocations*. The network then simulates the interconnected, real-world domino effects across all KPIs simultaneously.
* **Physics-Informed Constraints:** Guarantees hard physical boundaries (e.g., budget budget conservation via Softmax layers, non-negativity via Softplus) to avoid optimization exploitation of model blind spots.

---

## 🛠️ Tech Stack (Target)
* **Optimization & Modeling:** Python (SciPy, Pyomo, IPOPT)
* **Deep Learning & Autograd:** PyTorch / JAX (for end-to-end differentiable simulation trajectories)
* **Data & Graph Structures:** NetworkX, Polars / Pandas