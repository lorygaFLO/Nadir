---
tags: [theory, nadir, economics]
---

# 04 - Cost Functions

## Idea

Improving a KPI is not free. A **cost function** per KPI maps a desired relative change $u$ (Δ% of the current level) to an investment cost in currency:

$$\text{cost} = f_\alpha(u)$$

Cost here is an **effort proxy and budget constraint** — it is *not* the optimization objective (that is the [[03 - Signed Distance Metric|signed distance]]). The optimizer asks the *inverse* question far more often: *"given a budget slice $b$, what is the maximum Δ% I can buy?"* → $f_\alpha^{-1}(b)$.

## Registered shapes (see [`core/cost_functions.py`](../../core/cost_functions.py))

| Type | Cost $f_\alpha(u)$ | Inverse $f_\alpha^{-1}(b)$ | Models |
|---|---|---|---|
| `quadratic` | $\alpha u^2$ | $\sqrt{b/\alpha}$ | Diminishing marginal returns — each extra % costs more than the last |
| `linear` | $\alpha u$ | $b/\alpha$ | Constant marginal cost |

$\alpha$ is a per-KPI scale coefficient set in [[Configuration]] (e.g. `alpha: 1500000` means moving market share by 10% quadratically costs $1{,}500{,}000 \cdot 0.1^2 = 15{,}000$ EUR).

## `CostFunctionSpec`

Each KPI gets a `CostFunctionSpec` (built via `from_config` from the YAML), exposing:
- `cost(delta_pct)` — forward map, Δ% → money.
- `max_delta_pct(budget)` — inverse map, money → best achievable Δ%.

The registry pattern makes adding new shapes (e.g. asymmetric non-smooth curves for improvement vs. worsening) a matter of registering a pair of forward/inverse functions.

## Direction of improvement

The Δ% is always applied in the KPI's *improving* direction given its polarity: for `higher_is_better: false` KPIs (e.g. unit production cost), a positive Δ% *reduces* the raw value. `allow_worsening` in the config guards against allocations that would move a KPI the wrong way.

## Phase 2 outlook

Static $\alpha$ coefficients are a Phase 1 simplification. Phase 2 plans **dynamic cost estimation**: a predictive model adjusting marginal costs based on saturation (the closer to a physical ceiling, the more each % costs) and historical constraints — see [[01 - Project Overview#Phases]].

→ Next: [[05 - Optimization and Simulation]]
