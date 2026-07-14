"""Tests for core.metrics (signed Euclidean distance) and core.optimizer."""

from __future__ import annotations

import numpy as np
import pytest

from core.cost_functions import CostFunctionSpec
from core.metrics import aggregate_signed_distance, signed_distances
from core.optimizer import optimize_allocation
from core.poset import KpiSpec

KPI_SPECS = [
    KpiSpec(name="market_share_pct", higher_is_better=True),
    KpiSpec(name="unit_production_cost", higher_is_better=False),
]
COST_SPECS = [
    CostFunctionSpec(
        kpi_name="market_share_pct",
        cost_function_type="quadratic",
        alpha=1_000_000.0,
    ),
    CostFunctionSpec(
        kpi_name="unit_production_cost",
        cost_function_type="linear",
        alpha=500_000.0,
    ),
]


class TestSignedDistances:
    def test_dominated_focus_is_negative(self) -> None:
        subjects = {
            "F": np.array([10.0, 100.0]),  # worse share, worse cost
            "R": np.array([20.0, 80.0]),
        }
        d = signed_distances(subjects, KPI_SPECS, "F")
        assert d["R"] < 0

    def test_dominating_focus_scores_zero(self) -> None:
        subjects = {
            "F": np.array([20.0, 80.0]),
            "R": np.array([10.0, 100.0]),
        }
        d = signed_distances(subjects, KPI_SPECS, "F")
        assert d["R"] == 0.0

    def test_incomparable_ignores_leading_dims(self) -> None:
        # F leads on share (not a lag dim) but lags on cost: only the
        # cost gap should contribute, matching a focus that lags on
        # cost alone.
        subjects = {
            "F": np.array([20.0, 100.0]),  # better share, worse cost
            "R": np.array([10.0, 80.0]),
        }
        only_cost_lag = {
            "F": np.array([10.0, 100.0]),
            "R": np.array([10.0, 80.0]),
        }
        d = signed_distances(subjects, KPI_SPECS, "F")
        d_cost_only = signed_distances(only_cost_lag, KPI_SPECS, "F")
        assert d["R"] < 0
        assert d["R"] == pytest.approx(d_cost_only["R"])

    def test_identical_states_are_zero(self) -> None:
        subjects = {
            "F": np.array([10.0, 80.0]),
            "R": np.array([10.0, 80.0]),
        }
        assert signed_distances(subjects, KPI_SPECS, "F")["R"] == 0.0

    def test_scale_invariance(self) -> None:
        subjects = {
            "F": np.array([10.0, 100.0]),
            "R": np.array([20.0, 80.0]),
        }
        scaled = {name: vec * np.array([1.0, 1000.0]) for name, vec in subjects.items()}
        d_original = aggregate_signed_distance(subjects, KPI_SPECS, "F")
        d_scaled = aggregate_signed_distance(scaled, KPI_SPECS, "F")
        assert d_original == pytest.approx(d_scaled)

    def test_dominating_every_competitor_scores_zero(self) -> None:
        subjects = {
            "F": np.array([20.0, 80.0]),
            "R": np.array([10.0, 100.0]),
        }
        assert aggregate_signed_distance(subjects, KPI_SPECS, "F") == 0.0


class TestOptimizeAllocation:
    def test_budget_flows_to_lagging_kpi(self) -> None:
        # Focus far ahead on share, far behind on cost: overshooting share
        # buys nothing (min-max pins it at 1), so cost must get the budget.
        subjects = {
            "Focus": np.array([100.0, 100.0]),
            "Rival": np.array([10.0, 50.0]),
        }
        weights = optimize_allocation(
            KPI_SPECS, COST_SPECS, subjects, "Focus", budget=50_000.0
        )
        assert weights["unit_production_cost"] >= 0.9
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_symmetric_lag_splits_budget(self) -> None:
        # Focus dominated on both KPIs by the same margin: expect a
        # non-degenerate split (neither KPI starved).
        subjects = {
            "Focus": np.array([10.0, 100.0]),
            "Rival": np.array([20.0, 80.0]),
        }
        weights = optimize_allocation(
            KPI_SPECS, COST_SPECS, subjects, "Focus", budget=50_000.0
        )
        assert 0.0 < weights["market_share_pct"] < 1.0
        assert 0.0 < weights["unit_production_cost"] < 1.0

    def test_deterministic(self) -> None:
        subjects = {
            "Focus": np.array([10.0, 100.0]),
            "Rival": np.array([20.0, 80.0]),
        }
        w1 = optimize_allocation(KPI_SPECS, COST_SPECS, subjects, "Focus", 50_000.0)
        w2 = optimize_allocation(KPI_SPECS, COST_SPECS, subjects, "Focus", 50_000.0)
        assert w1 == w2
