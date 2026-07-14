---
tags: [theory, nadir, metrics]
---

# 03 - Signed Distance Metric

## Purpose

The poset tells you *whether* the focus subject is dominated, but optimization needs a **smooth scalar objective**. Nadir uses a signed, aggregated distance $D(s)$ that quantifies "how far from dominating everyone" the focus subject $s$ is.

## Construction (see [`core/metrics.py`](../../core/metrics.py))

1. **Orient** all KPI values by polarity (higher = better), as in [[02 - Posets and Pareto Dominance]].
2. **Min-max normalize** each KPI dimension across all subjects, so no KPI dominates the geometry by scale.
3. For each competitor $C_j$, compute a **projected gap**: only the dimensions where the focus *lags* contribute.

$$d(s, C_j) = -\left\lVert \max\!\big(0,\ \tilde{C_j} - \tilde{s}\big) \right\rVert_2$$

where $\tilde{\cdot}$ denotes oriented, normalized vectors and the $\max$ is component-wise. If the focus is ahead (or equal) on every dimension vs. $C_j$, the distance is exactly **0** — there is no bonus for over-winning.

4. **Aggregate** over all competitors:

$$D(s) = \sum_j d(s, C_j) \le 0$$

## Properties

- $D(s) \le 0$ always; $D(s) = 0 \iff$ the focus subject is on the **Pareto frontier** (no competitor beats it on any dimension it hasn't matched).
- **No overshoot incentive**: because gaps are projected (clipped at 0 per dimension), investing further in a KPI already "won" against a competitor yields no reward. This prevents the optimizer from wastefully piling budget into a single strong KPI.
- Monotone: improving any lagging KPI weakly increases $D$.

> [!note] Historical note
> An earlier design used a symmetric signed distance with a `positive_cap` parameter to cap the reward for overshooting. The projected-gap formulation replaced it entirely — `positive_cap` no longer exists anywhere in the API.

## Role in the pipeline

$D$ is the objective maximized by [[05 - Optimization and Simulation|the optimizer]] each period, and the natural progress indicator to plot over a simulated history.

→ Next: [[04 - Cost Functions]]
