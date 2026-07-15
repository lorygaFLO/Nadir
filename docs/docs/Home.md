---
tags: [moc, nadir]
aliases: [Index, Map of Content]
---

# 🏠 Nadir Vault — Home

**Nadir** is a geometric Decision Support System (DSS): it maps competitor performance profiles with **posets** (partially ordered sets), measures how far the *focus subject* is from the Pareto frontier, and simulates **multi-period budget allocations** to close that gap at minimum cost.

> [!question] The core question
> *"If we control the levers of specific KPIs, how should we schedule investments over time to beat competitors or reach the Pareto frontier at the lowest possible cost?"*

## 🧭 Map of Content

### Theory
- [[01 - Project Overview]] — what Nadir is, phases, and design philosophy
- [[02 - Posets and Pareto Dominance]] — dominance relations, Hasse diagrams, frontier
- [[03 - Signed Distance Metric]] — the projected-gap objective $D(s)$
- [[04 - Cost Functions]] — mapping KPI improvements to investment cost
- [[05 - Optimization and Simulation]] — greedy allocator + counterfactual simulation

### Technical
- [[Architecture]] — repository layout & data flow between modules
- [[Core Modules]] — API reference for everything in `core/`
- [[Configuration]] — `cost_settings.yaml` explained key by key
- [[Data Format]] — expected dataset schema and validation rules

### Practical
- [[Recipes]] — 🍳 ready-made commands: venvs (conda & vanilla), running notebooks, tests, etc.
- [[Future Developments]] — 🔮 open modeling questions not yet addressed

## 🚦 Quick start

1. Set up an environment → [[Recipes#Create a virtual environment]]
2. Understand the data you need → [[Data Format]]
3. Tune KPIs and costs → [[Configuration]]
4. Run a notebook → [[Recipes#Run a marimo notebook]]

## Status

The geometric-deterministic core (poset, cost curves, greedy optimizer, simulation) is built. How to handle KPIs that are in opposition to each other is still an open question — see [[Future Developments]].
