---
tags: [technical, nadir, architecture]
---

# Architecture

## Repository layout

```
Nadir/
├── config/
│   └── cost_settings.yaml      # simulation settings + per-KPI definitions → [[Configuration]]
├── core/                       # the engine → [[Core Modules]]
│   ├── poset.py                # KpiSpec, PosetEngine (dominance, Hasse, drift)
│   ├── dataset.py              # CSV/Parquet ingestion & validation → [[Data Format]]
│   ├── cost_functions.py       # CostFunctionSpec + registry
│   ├── metrics.py              # signed distance D(s)
│   ├── optimizer.py            # greedy simplex-grid allocator
│   ├── simulation.py           # multi-period counterfactual runner
│   └── viz.py                  # Plotly figures
├── data/
│   └── example_subjects.csv    # toy dataset (Us + 3 competitors)
├── notebooks/                  # marimo notebooks (see [[Recipes#Run a marimo notebook]])
│   ├── master_notebook.py
│   ├── phase1_sandbox.py
│   └── cost_functions_sandbox.py
├── tests/                      # pytest suite
└── docs/
    ├── agents/                 # design docs for AI agents
    └── docs/                   # ← this vault
```

## Data flow

```mermaid
flowchart TD
    YAML[cost_settings.yaml] -->|KpiSpec.from_config| SPECS[KpiSpec list]
    YAML -->|CostFunctionSpec.from_config| COSTS[CostFunctionSpec per KPI]
    CSV[CSV / Parquet dataset] -->|load_subject_dataset| DF[Validated Polars DataFrame]
    DF -->|extract_snapshot t=0| SNAP[Subject → KPI vector dict]
    SPECS --> ENGINE[PosetEngine]
    SNAP --> ENGINE
    ENGINE --> D[metrics: signed distance D]
    COSTS --> OPT[optimizer: optimize_allocation]
    D --> OPT
    OPT --> SIM[simulation: simulate_history]
    ENGINE --> SIM
    SIM --> HIST[history + allocations DataFrames]
    HIST --> VIZ[viz: Hasse / trajectories / allocation bars]
    HIST --> TTF[time_to_frontier]
```

## Layering rules

- `poset.py` is the foundation: no imports from other core modules.
- `dataset.py`, `metrics.py`, `cost_functions.py` depend only on `poset.py` (specs/orientation).
- `optimizer.py` composes metrics + cost functions; `simulation.py` composes everything; `viz.py` is a pure consumer.
- Notebooks and tests sit on top and never get imported by `core/`.

See [[Future Developments]] for an open modeling question (KPIs in opposition to each other) not yet addressed by this layout.
