---
tags: [theory, nadir, posets]
---

# 02 - Posets and Pareto Dominance

## Why a poset?

With $n \ge 2$ KPIs there is no single "best" ranking: subject A can be better on market share and worse on production cost than subject B. Multi-dimensional performance is therefore only **partially ordered**. Nadir formalizes this with a poset over subjects.

## Orientation

Every KPI has a **polarity** (`higher_is_better` in [[Configuration]]). Internally, all values are *oriented* so that "more is always better": a lower-is-better KPI is negated. This lets dominance be checked with simple component-wise comparisons.

## Dominance relation

For oriented vectors $a, b \in \mathbb{R}^n$:

$$a \succ b \iff a_i \ge b_i \ \forall i \ \text{and} \ \exists i: a_i > b_i$$

Possible outcomes between two subjects (`DominanceRelation` in [`core/poset.py`](../../core/poset.py)):
- **Dominates** / **is dominated** — strict Pareto dominance one way.
- **Equal** — identical oriented vectors.
- **Incomparable** — each wins on at least one dimension. This is what makes the order *partial*.

## Hasse diagram

`PosetEngine.compute_dominance_graph()` builds the full dominance DiGraph; `compute_hasse_diagram()` applies **transitive reduction** so only *covering* relations remain (if A ≻ B ≻ C, the edge A→C is dropped). The Hasse diagram is the minimal visual representation of the ordering — rendered with Plotly in [[Core Modules#viz py|viz.py]].

## Pareto frontier

The **maximal elements** of the poset — subjects not dominated by anyone — form the Pareto frontier (`get_maximal_elements()`). Reaching the frontier means $D(s) = 0$ in the [[03 - Signed Distance Metric]], and it is the natural terminal goal for the focus subject (`time_to_frontier` in [[Core Modules#simulation py|simulation.py]]).

Symmetrically, `get_most_dominated_element()` identifies the weakest subject — useful for diagnostics.

## Drift

Real competitors are not static. `PosetEngine.apply_drift()` evolves every subject's KPIs passively:

$$x_{t+1} = x_t \cdot (1 + \mu + \varepsilon), \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

where $\mu$ = `passive_decay_rate` (mean drift when nobody invests) and $\sigma$ = `drift_noise_std`, both per-KPI from [[Configuration]]. This makes the simulation a **stochastic counterfactual**, not a fixed-target chase.

→ Next: [[03 - Signed Distance Metric]]
