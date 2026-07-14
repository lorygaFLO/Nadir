"""Greedy per-period budget allocator (Phase 1 adaptive engine).

Each period, chooses the budget split across KPIs that maximizes the
signed distance metric D of the *next* state (myopic one-step search on
the allocation simplex). This is the feedback loop that stops the focus
subject from over-investing in KPIs it already leads: with min-max
normalization and a positive cap, budget flows to the KPIs where the
focus is still behind.

The full multi-period ``MultiPeriodPlanner`` (joint trajectory
optimization) is a Phase 2 concern; this module deliberately stays
single-period and deterministic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterator, Mapping, Sequence

import numpy as np

from core.cost_functions import CostFunctionSpec
from core.metrics import aggregate_signed_distance, oriented_bounds
from core.poset import KpiSpec


def _simplex_grid(n_dims: int, steps: int) -> Iterator[np.ndarray]:
    """All weight vectors with components k/steps summing to 1."""
    for cuts in combinations(range(steps + n_dims - 1), n_dims - 1):
        parts = []
        prev = -1
        for cut in (*cuts, steps + n_dims - 1):
            parts.append(cut - prev - 1)
            prev = cut
        yield np.array(parts, dtype=float) / steps


def _grid_resolution(n_dims: int) -> int:
    """Coarsen the grid as dimensionality grows to bound the search."""
    if n_dims <= 4:
        return 10  
    if n_dims <= 6:
        return 5
    return 2


def optimize_allocation(
    kpi_specs: Sequence[KpiSpec],
    cost_specs: Sequence[CostFunctionSpec],
    subjects: Mapping[str, np.ndarray],
    focus_subject: str,
    budget: float,
    positive_cap: float | None = 0.0,
) -> dict[str, float]:
    """Budget shares (summing to 1) that maximize next-period D.

    Deterministic grid search on the allocation simplex: for every
    candidate split, the focus subject's next state is computed through
    the inverse cost functions and scored with the signed distance
    metric; the best candidate wins (ties resolve to the first, hence
    reproducibly). Normalization bounds are frozen to the current field
    so that partial progress toward a competitor counts smoothly.
    """
    cost_by_name = {spec.kpi_name: spec for spec in cost_specs}
    current = np.asarray(subjects[focus_subject], dtype=float)
    bounds = oriented_bounds(subjects, kpi_specs)

    def next_state(weights: np.ndarray) -> np.ndarray:
        state = current.copy()
        for idx, spec in enumerate(kpi_specs):
            delta_pct = cost_by_name[spec.name].max_delta_pct(
                budget * float(weights[idx])
            )
            direction = 1.0 if spec.higher_is_better else -1.0
            state[idx] *= 1.0 + direction * delta_pct
        return state

    best_weights: np.ndarray | None = None
    best_score = -np.inf
    trial = dict(subjects)
    for weights in _simplex_grid(len(kpi_specs), _grid_resolution(len(kpi_specs))):
        trial[focus_subject] = next_state(weights)
        score = aggregate_signed_distance(
            trial,
            kpi_specs,
            focus_subject,
            positive_cap=positive_cap,
            normalization=bounds,
        )
        if score > best_score:
            best_score = score
            best_weights = weights

    assert best_weights is not None  # the grid is never empty
    return {
        spec.name: float(best_weights[idx]) for idx, spec in enumerate(kpi_specs)
    }
