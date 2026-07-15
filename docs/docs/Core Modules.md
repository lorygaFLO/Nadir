---
tags: [technical, nadir, api]
---

# Core Modules

API reference for [`core/`](../../core/). Theory background in [[01 - Project Overview]] and siblings.

## poset.py

The order-theoretic foundation → [[02 - Posets and Pareto Dominance]].

- **`KpiSpec`** (dataclass): `name`, polarity (`higher_is_better`), drift/decay parameters. `KpiSpec.from_config(name, cfg)` builds one from a YAML `kpi_definitions` entry.
- **`DominanceRelation`** (enum): `DOMINATES`, `IS_DOMINATED`, `EQUAL`, `INCOMPARABLE`.
- **`PosetEngine`** (dataclass over a list of `KpiSpec`):
  - `set_subject(name, values)` — register/update a subject's KPI vector (validated against specs).
  - `evaluate_dominance(a, b)` — pairwise relation on oriented vectors.
  - `compute_dominance_graph()` — full dominance `nx.DiGraph`.
  - `compute_hasse_diagram()` — transitive reduction of the above.
  - `get_maximal_elements()` — the Pareto frontier.
  - `get_most_dominated_element()` — weakest subject.
  - `apply_drift(rng, ...)` — stochastic passive evolution of all subjects.

## dataset.py

Ingestion & validation → [[Data Format]].

- `required_columns(kpi_specs)` — canonical uppercase column list.
- `load_subject_dataset(path, kpi_specs)` — read CSV/Parquet, validate columns, cast types, forward-fill gaps, sort by (TIME, SUBJECT). Raises `ValueError` on missing columns or unfillable gaps.
- `extract_snapshot(df, t, kpi_specs)` — slice one period into the `{subject: values}` dict `PosetEngine` consumes.

## cost_functions.py

Investment economics → [[04 - Cost Functions]].

- Registry pairs: `quadratic_cost` / `quadratic_max_delta_pct`, `linear_cost` / `linear_max_delta_pct`.
- **`CostFunctionSpec`**: `from_config(name, cfg)`, `cost(delta_pct)` (Δ% → money), `max_delta_pct(budget)` (money → Δ%).

## metrics.py

Objective function → [[03 - Signed Distance Metric]].

- `oriented_bounds(...)` — min/max per oriented dimension for normalization.
- `signed_distances(focus, competitors, ...)` — per-competitor projected-gap distances (≤ 0).
- `aggregate_signed_distance(...)` — the scalar $D(s) = \sum_j d(s, C_j)$.

> [!warning] No `positive_cap`
> Older revisions had a `positive_cap` parameter; it was removed with the projected-gap refactor. Neither `metrics`, `optimizer`, nor `simulation` accept it.

## optimizer.py

Per-period budget split → [[05 - Optimization and Simulation]].

- `optimize_allocation(...)` — greedy grid search over the budget simplex maximizing next-period $D$. Internals: `_simplex_grid(n_dims, steps)` enumerates weight lattices; `_grid_resolution(n_dims)` adapts density to dimensionality.

## simulation.py

Scenario runner → [[05 - Optimization and Simulation]].

- `resolve_budget_schedule(...)` — constant or per-period budget list over the horizon.
- `simulate_history(...)` — the main loop (focus invests adaptively or with static weights; competitors drift). Returns `(history, allocations)` Polars DataFrames.
- `time_to_frontier(...)` — first period the focus subject is Pareto-maximal, or none.

## viz.py

Plotly figures:

- `plot_hasse(engine, focus_subject, t)` — Hasse diagram at period *t*, focus highlighted.
- `plot_kpi_plane(...)` — subject trajectories on a 2-KPI scatter plane.
- `plot_allocations(...)` — stacked bars of the focus budget split per period.
