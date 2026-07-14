"""Tests for core.simulation: budget-driven history and time-to-frontier."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from core.cost_functions import CostFunctionSpec
from core.dataset import SUBJECT_COL, TIME_COL
from core.poset import KpiSpec
from core.simulation import (
    resolve_budget_schedule,
    simulate_history,
    time_to_frontier,
)

KPI_SPECS = [
    KpiSpec(name="market_share_pct", higher_is_better=True),
    KpiSpec(name="unit_production_cost", higher_is_better=False),
]
COST_SPECS = [
    CostFunctionSpec(
        kpi_name="market_share_pct",
        cost_function_type="quadratic",
        alpha=1_000_000.0,
        allow_worsening=False,
    ),
    CostFunctionSpec(
        kpi_name="unit_production_cost",
        cost_function_type="linear",
        alpha=500_000.0,
        allow_worsening=False,
    ),
]
INITIAL = {
    "Focus": np.array([10.0, 100.0]),
    "Rival": np.array([20.0, 80.0]),
}


def _run(budget=50_000.0, jitter=0.01, horizon=5, seed=1, weights=None, adaptive=False):
    history, _ = simulate_history(
        kpi_specs=KPI_SPECS,
        cost_specs=COST_SPECS,
        initial_states=INITIAL,
        focus_subject="Focus",
        annual_budget=budget,
        horizon=horizon,
        seed=seed,
        jitter_std=jitter,
        allocation_weights=weights,
        adaptive=adaptive,
    )
    return history


class TestSimulateHistory:
    def test_schema_and_row_count(self) -> None:
        df = _run(horizon=5)
        assert df.columns == [
            SUBJECT_COL,
            TIME_COL,
            "MARKET_SHARE_PCT",
            "UNIT_PRODUCTION_COST",
        ]
        assert df.height == 2 * 6  # 2 subjects x (horizon + 1) periods

    def test_focus_moves_in_favourable_direction(self) -> None:
        df = _run(horizon=4).filter(pl.col(SUBJECT_COL) == "Focus").sort(TIME_COL)
        share = df["MARKET_SHARE_PCT"].to_list()
        cost = df["UNIT_PRODUCTION_COST"].to_list()
        assert all(b > a for a, b in zip(share, share[1:]))
        assert all(b < a for a, b in zip(cost, cost[1:]))

    def test_step_size_derived_from_budget_and_cost_functions(self) -> None:
        # Equal split of 50k -> 25k per KPI.
        # quadratic: delta = sqrt(25000 / 1e6);  linear: delta = 25000 / 5e5.
        df = _run(budget=50_000.0, jitter=None, horizon=1)
        focus_t1 = df.filter(
            (pl.col(SUBJECT_COL) == "Focus") & (pl.col(TIME_COL) == 1)
        )
        expected_share = 10.0 * (1.0 + np.sqrt(25_000.0 / 1_000_000.0))
        expected_cost = 100.0 * (1.0 - 25_000.0 / 500_000.0)
        assert focus_t1["MARKET_SHARE_PCT"].item() == pytest.approx(expected_share)
        assert focus_t1["UNIT_PRODUCTION_COST"].item() == pytest.approx(expected_cost)

    def test_zero_budget_keeps_focus_constant(self) -> None:
        df = _run(budget=0.0, jitter=None).filter(pl.col(SUBJECT_COL) == "Focus")
        assert df["MARKET_SHARE_PCT"].n_unique() == 1
        assert df["UNIT_PRODUCTION_COST"].n_unique() == 1

    def test_competitors_constant_without_jitter(self) -> None:
        df = _run(jitter=None).filter(pl.col(SUBJECT_COL) == "Rival")
        assert df["MARKET_SHARE_PCT"].n_unique() == 1
        assert df["UNIT_PRODUCTION_COST"].n_unique() == 1

    def test_seed_reproducibility(self) -> None:
        assert _run(seed=7).equals(_run(seed=7))
        assert not _run(seed=7).equals(_run(seed=8))

    def test_missing_cost_spec_raises(self) -> None:
        with pytest.raises(ValueError, match="No cost function spec"):
            simulate_history(
                kpi_specs=KPI_SPECS,
                cost_specs=COST_SPECS[:1],
                initial_states=INITIAL,
                focus_subject="Focus",
                annual_budget=10_000.0,
                horizon=2,
                seed=1,
            )

    def test_all_zero_weights_raise(self) -> None:
        with pytest.raises(ValueError, match="positive weight"):
            _run(weights={"market_share_pct": 0.0, "unit_production_cost": 0.0})

    def test_allocations_schema(self) -> None:
        _, allocations = simulate_history(
            kpi_specs=KPI_SPECS,
            cost_specs=COST_SPECS,
            initial_states=INITIAL,
            focus_subject="Focus",
            annual_budget=50_000.0,
            horizon=4,
            seed=1,
        )
        assert allocations.columns == [
            TIME_COL,
            "MARKET_SHARE_PCT",
            "UNIT_PRODUCTION_COST",
        ]
        assert allocations.height == 4
        # Static equal split: each KPI gets half the budget every year.
        assert allocations["MARKET_SHARE_PCT"].to_list() == [25_000.0] * 4

    def test_adaptive_reroutes_budget_to_lagging_kpi(self) -> None:
        # Focus already leads on share but badly lags on cost: the adaptive
        # policy must send (almost) all budget to unit_production_cost.
        states = {
            "Focus": np.array([100.0, 100.0]),
            "Rival": np.array([10.0, 50.0]),
        }
        _, allocations = simulate_history(
            kpi_specs=KPI_SPECS,
            cost_specs=COST_SPECS,
            initial_states=states,
            focus_subject="Focus",
            annual_budget=50_000.0,
            horizon=3,
            seed=1,
            jitter_std=None,
            adaptive=True,
        )
        first_year = allocations.filter(pl.col(TIME_COL) == 1)
        assert first_year["UNIT_PRODUCTION_COST"].item() >= 45_000.0

    def test_adaptive_eventually_dominates_rival(self) -> None:
        # Incomparable start: Focus leads share, lags cost. The adaptive
        # policy must close the cost gap until Focus strictly dominates.
        states = {
            "Focus": np.array([100.0, 100.0]),
            "Rival": np.array([10.0, 50.0]),
        }
        history, _ = simulate_history(
            kpi_specs=KPI_SPECS,
            cost_specs=COST_SPECS,
            initial_states=states,
            focus_subject="Focus",
            annual_budget=50_000.0,
            horizon=15,
            seed=1,
            jitter_std=None,
            adaptive=True,
        )
        final = history.filter(pl.col(TIME_COL) == 15)
        focus_cost = final.filter(pl.col(SUBJECT_COL) == "Focus")[
            "UNIT_PRODUCTION_COST"
        ].item()
        rival_cost = final.filter(pl.col(SUBJECT_COL) == "Rival")[
            "UNIT_PRODUCTION_COST"
        ].item()
        assert focus_cost < rival_cost  # cost gap closed -> full dominance


class TestResolveBudgetSchedule:
    def test_scalar_repeated(self) -> None:
        assert resolve_budget_schedule(100.0, 3) == [100.0, 100.0, 100.0]

    def test_short_sequence_extended_with_last_value(self) -> None:
        assert resolve_budget_schedule([100.0, 50.0], 4) == [100.0, 50.0, 50.0, 50.0]

    def test_long_sequence_truncated(self) -> None:
        assert resolve_budget_schedule([1.0, 2.0, 3.0], 2) == [1.0, 2.0]

    def test_empty_sequence_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            resolve_budget_schedule([], 3)


class TestTimeToFrontier:
    HISTORY = pl.DataFrame(
        {
            SUBJECT_COL: ["F", "R", "F", "R"],
            TIME_COL: [0, 0, 1, 1],
            "MARKET_SHARE_PCT": [10.0, 20.0, 25.0, 20.0],
            "UNIT_PRODUCTION_COST": [100.0, 80.0, 70.0, 80.0],
        }
    )

    def test_first_period_on_frontier(self) -> None:
        assert time_to_frontier(self.HISTORY, KPI_SPECS, "F") == 1

    def test_already_on_frontier_at_start(self) -> None:
        assert time_to_frontier(self.HISTORY, KPI_SPECS, "R") == 0

    def test_never_reached_returns_none(self) -> None:
        stuck = self.HISTORY.with_columns(
            pl.when(pl.col(SUBJECT_COL) == "F")
            .then(pl.lit(1.0))
            .otherwise(pl.col("MARKET_SHARE_PCT"))
            .alias("MARKET_SHARE_PCT"),
            pl.when(pl.col(SUBJECT_COL) == "F")
            .then(pl.lit(100.0))
            .otherwise(pl.col("UNIT_PRODUCTION_COST"))
            .alias("UNIT_PRODUCTION_COST"),
        )
        assert time_to_frontier(stuck, KPI_SPECS, "F") is None
