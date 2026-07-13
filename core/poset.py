"""Partially ordered set engine.

Evaluates dominance relations between subjects across an arbitrary number of
KPIs (each with its own polarity), builds the Hasse diagram via transitive
reduction, and identifies the Pareto Efficient Frontier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

import networkx as nx
import numpy as np


class DominanceRelation(Enum):
    """Outcome of a pairwise dominance evaluation between two subjects."""

    DOMINATES = "dominates"          # A strictly dominates B (A ≻ B)
    DOMINATED = "dominated"          # B strictly dominates A (B ≻ A)
    INCOMPARABLE = "incomparable"    # A ∥ B — each is better somewhere
    EQUAL = "equal"                  # identical on every KPI


@dataclass(frozen=True)
class KpiSpec:
    """Static definition of a single KPI (polarity + optional noise profile)."""

    name: str
    higher_is_better: bool = True
    drift_noise_std: float | None = None
    passive_decay_rate: float | None = None

    @classmethod
    def from_config(cls, name: str, cfg: Mapping[str, object]) -> "KpiSpec":
        """Build a spec from one ``kpi_definitions`` entry of the YAML config."""
        return cls(
            name=name,
            higher_is_better=bool(cfg.get("higher_is_better", True)),
            drift_noise_std=(
                float(cfg["drift_noise_std"]) if "drift_noise_std" in cfg else None
            ),
            passive_decay_rate=(
                float(cfg["passive_decay_rate"])
                if "passive_decay_rate" in cfg
                else None
            ),
        )


@dataclass
class PosetEngine:
    """Structural state calculator for a set of subjects over ``n`` KPIs.

    Parameters
    ----------
    kpi_specs:
        Ordered sequence of KPI definitions. The order fixes the column
        order expected in every subject's value vector.
    subjects:
        Mapping of subject name -> KPI value vector (length must equal
        ``len(kpi_specs)``).
    """

    kpi_specs: Sequence[KpiSpec]
    subjects: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.subjects = {
            name: self._validated_vector(values)
            for name, values in self.subjects.items()
        }

    # ------------------------------------------------------------------
    # Subject management
    # ------------------------------------------------------------------
    @property
    def n_kpis(self) -> int:
        return len(self.kpi_specs)

    @property
    def kpi_names(self) -> list[str]:
        return [spec.name for spec in self.kpi_specs]

    def set_subject(self, name: str, values: Sequence[float] | np.ndarray) -> None:
        """Add or update a subject's KPI state vector."""
        self.subjects[name] = self._validated_vector(values)

    def _validated_vector(
        self, values: Sequence[float] | np.ndarray
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        if vector.shape != (self.n_kpis,):
            raise ValueError(
                f"Expected a vector of {self.n_kpis} KPI values "
                f"({self.kpi_names}), got shape {vector.shape}."
            )
        return vector

    # ------------------------------------------------------------------
    # Dominance evaluation
    # ------------------------------------------------------------------
    def _oriented(self, values: np.ndarray) -> np.ndarray:
        """Re-orient a raw KPI vector so that 'greater' always means 'better'."""
        polarity = np.array(
            [1.0 if spec.higher_is_better else -1.0 for spec in self.kpi_specs]
        )
        return values * polarity

    def evaluate_dominance(
        self, subject_a: str, subject_b: str
    ) -> DominanceRelation:
        """Evaluate the dominance relation A vs B respecting KPI polarities."""
        a = self._oriented(self.subjects[subject_a])
        b = self._oriented(self.subjects[subject_b])

        a_ge_b = bool(np.all(a >= b))
        b_ge_a = bool(np.all(b >= a))

        if a_ge_b and b_ge_a:
            return DominanceRelation.EQUAL
        if a_ge_b:
            return DominanceRelation.DOMINATES
        if b_ge_a:
            return DominanceRelation.DOMINATED
        return DominanceRelation.INCOMPARABLE

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def compute_dominance_graph(self) -> nx.DiGraph:
        """Full dominance DAG: edge u -> v means u ≻ v (u dominates v)."""
        graph = nx.DiGraph()
        graph.add_nodes_from(self.subjects)
        names = list(self.subjects)
        for i, u in enumerate(names):
            for v in names[i + 1 :]:
                relation = self.evaluate_dominance(u, v)
                if relation is DominanceRelation.DOMINATES:
                    graph.add_edge(u, v)
                elif relation is DominanceRelation.DOMINATED:
                    graph.add_edge(v, u)
        return graph

    def compute_hasse_diagram(self) -> nx.DiGraph:
        """Transitive reduction of the dominance DAG (the Hasse diagram)."""
        full = self.compute_dominance_graph()
        hasse = nx.transitive_reduction(full)
        hasse.add_nodes_from(full.nodes(data=True))
        return hasse

    def get_maximal_elements(self) -> list[str]:
        """Non-dominated subjects: the Pareto Efficient Frontier."""
        graph = self.compute_dominance_graph()
        return [node for node in graph.nodes if graph.in_degree(node) == 0]

    def get_most_dominated_element(self) -> str:
        """Subject with the greatest number of incoming dominance edges.

        Ties are broken deterministically by subject name.
        """
        graph = self.compute_dominance_graph()
        return max(
            sorted(graph.nodes),
            key=lambda node: graph.in_degree(node),
        )

    # ------------------------------------------------------------------
    # Stochastic drift
    # ------------------------------------------------------------------
    def apply_drift(
        self,
        name: str,
        rng: np.random.Generator,
        invested_kpis: frozenset[str] | set[str] = frozenset(),
    ) -> np.ndarray:
        """Advance one subject's state by one period of passive drift + noise.

        For each KPI without active investment, applies the configured
        ``passive_decay_rate`` mean drift. All KPIs with a ``drift_noise_std``
        receive an additive Gaussian perturbation. Deterministic KPIs
        (no noise parameters) are left untouched. Returns the new vector
        and stores it in place.
        """
        current = self.subjects[name].copy()
        for idx, spec in enumerate(self.kpi_specs):
            delta = 0.0
            if spec.passive_decay_rate is not None and spec.name not in invested_kpis:
                delta += spec.passive_decay_rate
            if spec.drift_noise_std is not None:
                delta += rng.normal(0.0, spec.drift_noise_std)
            current[idx] *= 1.0 + delta
        self.subjects[name] = current
        return current
