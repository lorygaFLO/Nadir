---
tags: [technical, nadir, config]
---

# Configuration

Everything is driven by [`config/cost_settings.yaml`](../../config/cost_settings.yaml). Two top-level sections.

## `simulation_settings`

| Key | Example | Meaning |
|---|---|---|
| `currency` | `"EUR"` | Display currency for costs/budgets |
| `time_horizon_periods` | `100` | Number of simulated periods $T$ |
| `random_seed` | `42` | Seed for the drift RNG — fixes the stochastic scenario for reproducibility |
| `period_budget` | `50000` | Default per-period investment for the focus subject (can be overridden with a per-period schedule, see `resolve_budget_schedule` in [[Core Modules#simulation py|simulation.py]]) |

## `kpi_definitions`

Fully generic: **add as many KPI entries as needed** — the engine handles any number (Phase 1 examples use two). Each key becomes a `KpiSpec` + `CostFunctionSpec` and must match an (uppercased) column in the dataset → [[Data Format]].

```yaml
market_share_pct:
  higher_is_better: true      # polarity
  cost_function_type: "quadratic"
  alpha: 1500000              # cost scale coefficient
  drift_noise_std: 0.02       # σ of per-period random perturbation (fraction of state)
  passive_decay_rate: -0.01   # mean per-period drift when no investment is made
```

| Key | Consumed by | Notes |
|---|---|---|
| `higher_is_better` | poset orientation, metrics, cost direction | See [[02 - Posets and Pareto Dominance#Orientation]] |
| `cost_function_type` | `CostFunctionSpec` registry | `"quadratic"` or `"linear"` → [[04 - Cost Functions]] |
| `alpha` | cost curves | Bigger α ⇒ each Δ% costs more |
| `drift_noise_std` | `PosetEngine.apply_drift` | Std-dev of the stochastic jitter, as fraction of current value |
| `passive_decay_rate` | `PosetEngine.apply_drift` | Mean drift with **no** investment. Sign is in *raw value space*: `-0.01` on a higher-is-better KPI = erosion; `+0.01` on unit cost = costs creep up 1%/period |

> [!tip] Interpreting `passive_decay_rate`
> It models "the world doesn't stand still": market share slowly bleeds away, costs inflate. It's what makes *doing nothing* an actively losing strategy in the simulation.

## Adding a new KPI — checklist

1. Add the entry under `kpi_definitions` with all six keys.
2. Add the matching **UPPERCASE** column to your dataset ([[Data Format]]).
3. Re-run tests: `python -m pytest -q` ([[Recipes#Run the test suite]]).
