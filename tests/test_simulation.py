"""Tests for core.simulation.simulate_history."""

from __future__ import annotations

import numpy as np
import polars as pl

from core.dataset import SUBJECT_COL, TIME_COL
from core.poset import KpiSpec
from core.simulation import simulate_history

KPI_SPECS = [
    KpiSpec(name="market_share_pct", higher_is_better=True),
    KpiSpec(name="unit_production_cost", higher_is_better=False),
]
INITIAL = {
    "Focus": np.array([10.0, 100.0]),
    "Rival": np.array([20.0, 80.0]),
}


def _run(jitter=0.01, horizon=5, seed=1, rates=None):
    return simulate_history(
        kpi_specs=KPI_SPECS,
        initial_states=INITIAL,
        focus_subject="Focus",
        improvement_rates=rates
        or {"market_share_pct": 0.05, "unit_production_cost": 0.05},
        jitter_std=jitter,
        horizon=horizon,
        seed=seed,
    )


def test_schema_and_row_count() -> None:
    df = _run(horizon=5)
    assert df.columns == [
        SUBJECT_COL,
        TIME_COL,
        "MARKET_SHARE_PCT",
        "UNIT_PRODUCTION_COST",
    ]
    assert df.height == 2 * 6  # 2 subjects x (horizon + 1) periods


def test_focus_moves_in_favourable_direction() -> None:
    df = _run(horizon=4).filter(pl.col(SUBJECT_COL) == "Focus").sort(TIME_COL)
    share = df["MARKET_SHARE_PCT"].to_list()
    cost = df["UNIT_PRODUCTION_COST"].to_list()
    assert all(b > a for a, b in zip(share, share[1:]))
    assert all(b < a for a, b in zip(cost, cost[1:]))


def test_competitors_constant_without_jitter() -> None:
    df = _run(jitter=None).filter(pl.col(SUBJECT_COL) == "Rival")
    assert df["MARKET_SHARE_PCT"].n_unique() == 1
    assert df["UNIT_PRODUCTION_COST"].n_unique() == 1


def test_seed_reproducibility() -> None:
    assert _run(seed=7).equals(_run(seed=7))
    assert not _run(seed=7).equals(_run(seed=8))
