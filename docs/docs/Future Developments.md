---
tags: [theory, nadir, roadmap]
---

# Future Developments

Nadir's current implementation is deliberately self-contained: poset dominance, signed-distance objective, static cost curves, and a greedy per-period optimizer. Each KPI is treated as independently improvable — the optimizer assigns a Δ% to a KPI without accounting for how that change might affect other KPIs.

## Open problem: KPIs in opposition to each other

It is still unclear how to model KPIs that move in opposite directions when one of them is improved — i.e. where a gain on one KPI necessarily comes with a worsening of another, coupled one.

**Example:** you can't increase a country's GDP without also increasing its public debt (when growth is deficit-financed) — pushing GDP up drags debt up with it, even though debt going up is "bad". Nadir has no mechanism yet to express this kind of cross-KPI coupling, so the optimizer could currently recommend allocations that ignore such trade-offs.

This is not implemented and is left as an open question for future work.
