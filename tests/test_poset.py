"""Regression tests for core.poset.PosetEngine."""

from __future__ import annotations

import numpy as np
import pytest

from core.poset import DominanceRelation, KpiSpec, PosetEngine

# Two KPIs with mixed polarity: revenue (higher better), cost (lower better).
MIXED_SPECS = [
    KpiSpec(name="revenue", higher_is_better=True),
    KpiSpec(name="cost", higher_is_better=False),
]


def make_engine(subjects: dict[str, list[float]]) -> PosetEngine:
    return PosetEngine(
        kpi_specs=MIXED_SPECS,
        subjects={k: np.asarray(v, dtype=float) for k, v in subjects.items()},
    )


class TestDominanceEvaluation:
    def test_strict_dominance_mixed_polarity(self) -> None:
        engine = make_engine({"A": [100, 10], "B": [90, 20]})
        assert engine.evaluate_dominance("A", "B") is DominanceRelation.DOMINATES
        assert engine.evaluate_dominance("B", "A") is DominanceRelation.DOMINATED

    def test_dominance_with_tie_on_one_kpi(self) -> None:
        # Equal revenue, A has lower cost -> A still dominates.
        engine = make_engine({"A": [100, 10], "B": [100, 20]})
        assert engine.evaluate_dominance("A", "B") is DominanceRelation.DOMINATES

    def test_incomparable_pair(self) -> None:
        # A better on revenue, B better on cost.
        engine = make_engine({"A": [100, 20], "B": [90, 10]})
        assert engine.evaluate_dominance("A", "B") is DominanceRelation.INCOMPARABLE

    def test_equal_subjects(self) -> None:
        engine = make_engine({"A": [100, 10], "B": [100, 10]})
        assert engine.evaluate_dominance("A", "B") is DominanceRelation.EQUAL

    def test_polarity_matters(self) -> None:
        # With cost polarity inverted (higher better), B would dominate A.
        specs = [
            KpiSpec(name="revenue", higher_is_better=True),
            KpiSpec(name="cost", higher_is_better=True),
        ]
        engine = PosetEngine(
            kpi_specs=specs,
            subjects={"A": np.array([100.0, 10.0]), "B": np.array([100.0, 20.0])},
        )
        assert engine.evaluate_dominance("B", "A") is DominanceRelation.DOMINATES

    def test_wrong_vector_length_rejected(self) -> None:
        engine = make_engine({"A": [100, 10]})
        with pytest.raises(ValueError):
            engine.set_subject("B", [1.0, 2.0, 3.0])


class TestHasseDiagram:
    def test_transitive_reduction_removes_redundant_edges(self) -> None:
        # Chain A ≻ B ≻ C: full graph has edge A->C, Hasse must not.
        engine = make_engine({"A": [100, 5], "B": [90, 10], "C": [80, 20]})
        full = engine.compute_dominance_graph()
        hasse = engine.compute_hasse_diagram()

        assert full.has_edge("A", "C")
        assert not hasse.has_edge("A", "C")
        assert set(hasse.edges) == {("A", "B"), ("B", "C")}

    def test_hasse_preserves_isolated_nodes(self) -> None:
        # D is incomparable to everyone: must still appear as a node.
        engine = make_engine(
            {"A": [100, 5], "B": [90, 10], "D": [10, 1]}
        )
        hasse = engine.compute_hasse_diagram()
        assert "D" in hasse.nodes


class TestParetoFrontier:
    def test_known_frontier(self) -> None:
        engine = make_engine(
            {
                "A": [100, 10],  # frontier
                "B": [90, 5],    # frontier (incomparable with A)
                "C": [80, 20],   # dominated by A and B
                "D": [95, 15],   # dominated by A only
            }
        )
        assert sorted(engine.get_maximal_elements()) == ["A", "B"]

    def test_single_dominator_frontier(self) -> None:
        engine = make_engine({"A": [100, 1], "B": [90, 10], "C": [50, 50]})
        assert engine.get_maximal_elements() == ["A"]

    def test_most_dominated_element(self) -> None:
        engine = make_engine({"A": [100, 5], "B": [90, 10], "C": [80, 20]})
        assert engine.get_most_dominated_element() == "C"


class TestStochasticDrift:
    NOISY_SPECS = [
        KpiSpec(
            name="revenue",
            higher_is_better=True,
            drift_noise_std=0.02,
            passive_decay_rate=-0.01,
        ),
        KpiSpec(name="cost", higher_is_better=False),  # fully deterministic
    ]

    def test_seed_reproducibility(self) -> None:
        results = []
        for _ in range(2):
            engine = PosetEngine(
                kpi_specs=self.NOISY_SPECS,
                subjects={"A": np.array([100.0, 10.0])},
            )
            rng = np.random.default_rng(42)
            for _t in range(5):
                engine.apply_drift("A", rng)
            results.append(engine.subjects["A"].copy())
        np.testing.assert_array_equal(results[0], results[1])

    def test_deterministic_kpi_untouched(self) -> None:
        engine = PosetEngine(
            kpi_specs=self.NOISY_SPECS,
            subjects={"A": np.array([100.0, 10.0])},
        )
        engine.apply_drift("A", np.random.default_rng(0))
        assert engine.subjects["A"][1] == 10.0  # no noise params -> unchanged

    def test_investment_suppresses_passive_decay(self) -> None:
        specs = [
            KpiSpec(name="revenue", higher_is_better=True, passive_decay_rate=-0.10)
        ]
        engine = PosetEngine(
            kpi_specs=specs, subjects={"A": np.array([100.0])}
        )
        engine.apply_drift(
            "A", np.random.default_rng(0), invested_kpis={"revenue"}
        )
        assert engine.subjects["A"][0] == 100.0

        engine.apply_drift("A", np.random.default_rng(0))
        assert engine.subjects["A"][0] == pytest.approx(90.0)
