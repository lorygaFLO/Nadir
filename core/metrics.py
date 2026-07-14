"""Signed Euclidean distance metric (design doc, Section D).

For a focus subject ``s`` and each competitor ``C_j`` a signed distance
is computed in the min-max normalized, polarity-oriented KPI space:

    d_j(s) = sigma(s, C_j) * ||s - C_j||_2

with sigma = +1 if the focus dominates C_j, -1 if it is dominated or
incomparable, 0 if identical. The aggregate metric is D(s) = sum_j d_j.

Normalization matters twice: it makes KPIs with different scales
commensurable, and it caps the reward for overshooting a KPI the focus
already leads (min-max pins the best subject at 1 regardless of margin).
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
    """Per-competitor signed Euclidean distance from the focus subject.

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
        focus_ge = bool(np.all(oriented[i] >= oriented[j]))
        comp_ge = bool(np.all(oriented[j] >= oriented[i]))
        if focus_ge and comp_ge:
            sigma = 0.0  # identical states
        elif focus_ge:
            sigma = 1.0  # focus dominates
        else:
            sigma = -1.0  # dominated or incomparable
        distances[name] = sigma * float(
            np.linalg.norm(normalized[i] - normalized[j])
        )
    return distances


def aggregate_signed_distance(
    subjects: Mapping[str, np.ndarray],
    kpi_specs: Sequence[KpiSpec],
    focus_subject: str,
    positive_cap: float | None = None,
    normalization: tuple[np.ndarray, np.ndarray] | None = None,
) -> float:
    """Aggregate metric D = sum of signed distances.

    ``positive_cap`` clamps each positive (dominating) contribution:
    ``positive_cap=0.0`` means beating a competitor further gains
    nothing — the natural choice when the goal is reaching the Pareto
    frontier rather than maximizing margin. ``None`` leaves the raw
    design-doc metric untouched.
    """
    values = signed_distances(
        subjects, kpi_specs, focus_subject, normalization=normalization
    ).values()
    if positive_cap is None:
        return float(sum(values))
    return float(sum(min(v, positive_cap) for v in values))
