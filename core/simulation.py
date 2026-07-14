"""Scenario simulator: focus subject invests a budget, competitors drift.

Produces a tidy history DataFrame with the same wide schema used by
``core.dataset``: SUBJECT | TIME | <KPI_1> | ... | <KPI_n> (uppercase).

The focus subject's annual KPI step is **not** chosen directly: each
period an annual budget is split across KPIs (via allocation weights)
and converted into the largest affordable Delta% through the inverse
cost functions (see ``core.cost_functions``).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from core.cost_functions import CostFunctionSpec
from core.dataset import SUBJECT_COL, TIME_COL, extract_snapshot
from core.optimizer import optimize_allocation
from core.poset import KpiSpec, PosetEngine


def resolve_budget_schedule(
    annual_budget: float | Sequence[float], horizon: int
) -> list[float]:
    """Normalize a fixed or per-year budget into one value per period.

    A scalar is repeated for every period. A sequence shorter than the
    horizon is extended with its last value; a longer one is truncated.
    """
    if isinstance(annual_budget, (int, float)):
        return [float(annual_budget)] * horizon
    schedule = [float(b) for b in annual_budget]
    if not schedule:
        raise ValueError("annual_budget sequence is empty.")
    if len(schedule) < horizon:
        schedule += [schedule[-1]] * (horizon - len(schedule))
    return schedule[:horizon]


def _normalized_weights(
    kpi_specs: Sequence[KpiSpec],
    allocation_weights: Mapping[str, float] | None,
) -> dict[str, float]:
    """Per-KPI budget shares summing to 1 (default: equal split)."""
    if allocation_weights is None:
        return {spec.name: 1.0 / len(kpi_specs) for spec in kpi_specs}
    raw = {
        spec.name: float(allocation_weights.get(spec.name, 0.0))
        for spec in kpi_specs
    }
    total = sum(raw.values())
    if total <= 0:
        raise ValueError(
            "allocation_weights must contain at least one positive weight."
        )
    return {name: w / total for name, w in raw.items()}


def simulate_history(
    kpi_specs: Sequence[KpiSpec],
    cost_specs: Sequence[CostFunctionSpec],
    initial_states: Mapping[str, np.ndarray],
    focus_subject: str,
    annual_budget: float | Sequence[float],
    horizon: int,
    seed: int,
    jitter_std: float | None = None,
    allocation_weights: Mapping[str, float] | None = None,
    adaptive: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Simulate ``horizon`` periods; return ``(history, allocations)``.

    Each period the focus subject spends that year's budget: the amount
    allocated to each KPI is converted into the largest affordable
    Delta% through the KPI's inverse cost function and applied in the
    favourable direction.

    With ``adaptive=True`` the split is re-optimized **every period**
    against the current field state (projected-gap signed distance
    metric D), so budget flows to the KPIs where the focus still lags.
    Otherwise the static ``allocation_weights`` (default equal split)
    are used throughout.

    Every other subject receives seeded Gaussian jitter with std
    ``jitter_std`` and no passive decay; ``jitter_std=None`` (or ``0``)
    keeps competitors constant.

    ``allocations`` has one row per period: TIME plus the budget amount
    spent on each (uppercase) KPI column.
    """
    cost_by_name = {spec.kpi_name: spec for spec in cost_specs}
    missing = [spec.name for spec in kpi_specs if spec.name not in cost_by_name]
    if missing:
        raise ValueError(f"No cost function spec for KPI(s): {missing}.")

    weights = _normalized_weights(kpi_specs, allocation_weights)
    budgets = resolve_budget_schedule(annual_budget, horizon)

    rng = np.random.default_rng(seed)
    sim_specs = [
        replace(spec, drift_noise_std=jitter_std or None, passive_decay_rate=None)
        for spec in kpi_specs
    ]
    engine = PosetEngine(kpi_specs=sim_specs, subjects=dict(initial_states))

    rows: list[dict[str, object]] = []

    def snapshot(t: int) -> None:
        for name, vec in engine.subjects.items():
            row: dict[str, object] = {SUBJECT_COL: name, TIME_COL: t}
            row.update(
                {spec.name.upper(): float(v) for spec, v in zip(kpi_specs, vec)}
            )
            rows.append(row)

    snapshot(0)
    alloc_rows: list[dict[str, object]] = []
    for t in range(1, horizon + 1):
        budget_t = budgets[t - 1]
        if adaptive:
            # Feedback loop: re-optimize the split on the CURRENT field state.
            weights = optimize_allocation(
                kpi_specs,
                cost_specs,
                engine.subjects,
                focus_subject,
                budget_t,
            )
        alloc_rows.append(
            {
                TIME_COL: t,
                **{
                    spec.name.upper(): budget_t * weights[spec.name]
                    for spec in kpi_specs
                },
            }
        )
        for name in sorted(engine.subjects):
            if name == focus_subject:
                current = engine.subjects[name].copy()
                for idx, spec in enumerate(kpi_specs):
                    kpi_budget = budget_t * weights[spec.name]
                    delta_pct = cost_by_name[spec.name].max_delta_pct(kpi_budget)
                    direction = 1.0 if spec.higher_is_better else -1.0
                    current[idx] *= 1.0 + direction * delta_pct
                engine.set_subject(name, current)
            else:
                engine.apply_drift(name, rng)
        snapshot(t)

    history = pl.DataFrame(rows).sort([TIME_COL, SUBJECT_COL])
    allocations = pl.DataFrame(alloc_rows)
    return history, allocations


def time_to_frontier(
    history: pl.DataFrame,
    kpi_specs: Sequence[KpiSpec],
    focus_subject: str,
) -> int | None:
    """First period at which the focus subject sits on the Pareto frontier.

    Returns ``None`` if the frontier is never reached within the history.
    """
    for t in sorted(history[TIME_COL].unique().to_list()):
        engine = PosetEngine(
            kpi_specs=kpi_specs,
            subjects=extract_snapshot(history, kpi_specs, time=int(t)),
        )
        if focus_subject in engine.get_maximal_elements():
            return int(t)
    return None
