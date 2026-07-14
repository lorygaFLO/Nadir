"""Signed Euclidean distance metric (design doc, Section D).

For a focus subject ``s`` and each competitor ``C_j`` the distance is
computed in the min-max normalized, polarity-oriented KPI space, but
**projected onto the subspace of KPIs where the focus still lags**
behind ``C_j``:

    lag_dims(s, C_j) = {k : oriented(s)_k < oriented(C_j)_k}
    d_j(s) = -||s - C_j||_2 restricted to lag_dims(s, C_j)

If ``lag_dims`` is empty (the focus dominates or ties ``C_j`` on every
KPI) ``d_j = 0``: there is nothing left to gain from that competitor.
Otherwise ``d_j`` is always <= 0, reflecting the remaining gap. The
aggregate metric is D(s) = sum_j d_j.

This projection is what keeps the optimizer from chasing 'easy' KPIs
the focus already leads on: only the dimensions where the focus is
behind a given competitor contribute to that competitor's term, so
further improving an already-won KPI cannot manufacture extra score.

Normalization still matters to make KPIs with different scales
commensurable before measuring the projected gap.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from core.poset import KpiSpec


def _oriented_matrix(
    subjects: Mapping[str, np.ndarray], kpi_specs: Sequence[KpiSpec]
) -> tuple[list[str], np.ndarray]:
    """Sorted subject names and matrix re-oriented so greater == better."""
    polarity = np.array(
        [1.0 if spec.higher_is_better else -1.0 for spec in kpi_specs]
    )
    names = sorted(subjects)
    matrix = np.vstack([np.asarray(subjects[n], dtype=float) for n in names])
    return names, matrix * polarity


def _min_max_normalize(matrix: np.ndarray) -> np.ndarray:
    """Column-wise min-max scaling; constant columns collapse to 0."""
    lo = matrix.min(axis=0)
    span = matrix.max(axis=0) - lo
    span = np.where(span > 0, span, 1.0)
    return (matrix - lo) / span


def oriented_bounds(
    subjects: Mapping[str, np.ndarray], kpi_specs: Sequence[KpiSpec]
) -> tuple[np.ndarray, np.ndarray]:
    """Per-KPI (lo, span) of the field in oriented space.

    Useful to freeze the normalization to a reference field state: an
    optimizer probing candidate moves should score them against fixed
    bounds, otherwise re-normalizing per trial flattens the objective
    (the trailing subject stays pinned at 0 regardless of the gap).
    """
    _, matrix = _oriented_matrix(subjects, kpi_specs)
    lo = matrix.min(axis=0)
    span = matrix.max(axis=0) - lo
    return lo, np.where(span > 0, span, 1.0)


def signed_distances(
    subjects: Mapping[str, np.ndarray],
    kpi_specs: Sequence[KpiSpec],
    focus_subject: str,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict[str, float]:
    """Per-competitor projected-gap distance from the focus subject.

    For each competitor, the distance is measured only across the KPIs
    where the focus still lags (see module docstring); it is always
    <= 0, and exactly 0 once the focus dominates or ties that
    competitor on every KPI.

    ``normalization`` optionally provides frozen ``(lo, span)`` bounds in
    oriented space (see ``oriented_bounds``); by default bounds are
    recomputed from ``subjects``.
    """
    names, oriented = _oriented_matrix(subjects, kpi_specs)
    if normalization is None:
        normalized = _min_max_normalize(oriented)
    else:
        lo, span = normalization
        normalized = (oriented - lo) / span
    i = names.index(focus_subject)

    distances: dict[str, float] = {}
    for j, name in enumerate(names):
        if name == focus_subject:
            continue
        lagging = oriented[i] < oriented[j]
        if not lagging.any():
            distances[name] = 0.0
            continue
        gap = normalized[i][lagging] - normalized[j][lagging]
        distances[name] = -float(np.linalg.norm(gap))
    return distances


def aggregate_signed_distance(
    subjects: Mapping[str, np.ndarray],
    kpi_specs: Sequence[KpiSpec],
    focus_subject: str,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Aggregate metric D = sum of per-competitor projected-gap distances.

    Always <= 0; it reaches 0 once the focus dominates or ties every
    competitor on every KPI (the Pareto frontier).
    """
    values = signed_distances(
        subjects, kpi_specs, focus_subject, normalization=normalization
    ).values()
    return float(sum(values))
