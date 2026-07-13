"""Scenario simulator: focus subject improves, competitors drift.

Produces a tidy history DataFrame with the same wide schema used by
``core.dataset``: SUBJECT | TIME | <KPI_1> | ... | <KPI_n> (uppercase).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Sequence

import numpy as np
import polars as pl

from core.dataset import SUBJECT_COL, TIME_COL
from core.poset import KpiSpec, PosetEngine


def simulate_history(
    kpi_specs: Sequence[KpiSpec],
    initial_states: Mapping[str, np.ndarray],
    focus_subject: str,
    improvement_rates: Mapping[str, float],
    jitter_std: float | None,
    horizon: int,
    seed: int,
) -> pl.DataFrame:
    """Simulate ``horizon`` periods and return the full history.

    The focus subject improves deterministically by ``improvement_rates``
    (fraction per period, applied in the favourable direction of each KPI).
    Every other subject receives seeded Gaussian jitter with std
    ``jitter_std`` and no passive decay; ``jitter_std=None`` (or ``0``)
    keeps competitors constant.
    """
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
    for t in range(1, horizon + 1):
        for name in sorted(engine.subjects):
            if name == focus_subject:
                current = engine.subjects[name].copy()
                for idx, spec in enumerate(kpi_specs):
                    rate = float(improvement_rates.get(spec.name, 0.0))
                    direction = 1.0 if spec.higher_is_better else -1.0
                    current[idx] *= 1.0 + direction * rate
                engine.set_subject(name, current)
            else:
                engine.apply_drift(name, rng)
        snapshot(t)

    return pl.DataFrame(rows).sort([TIME_COL, SUBJECT_COL])
