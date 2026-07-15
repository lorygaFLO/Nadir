"""Cost-function framework for KPI investment effort.

Every KPI is mapped to a customizable cost function of its percentage
variance (``delta_pct``, i.e. Delta%) relative to its current state,
centered at zero. Cost functions are effort proxies: they quantify the
investment required to achieve a given KPI change. They are **not** the
optimization objective; they define the budget constraint / price of a
chosen trajectory. See docs/agents/project-design (Section C) for the
full mathematical framework.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

CostFunction = Callable[[float, float], float]


def quadratic_cost(delta_pct: float, alpha: float) -> float:
    """Diminishing-returns cost: ``C(Delta%) = alpha * (Delta%)^2``.

    Marginal cost scales parabolically as the deviation grows farther
    from zero, in either direction.
    """
    return alpha * delta_pct**2


def linear_cost(delta_pct: float, alpha: float) -> float:
    """Constant-returns cost: ``C(Delta%) = alpha * |Delta%|``.

    Effort maps linearly to the change rate.
    """
    return alpha * abs(delta_pct)


# Registry of available cost-function types, keyed by the
# ``cost_function_type`` string used in config/cost_settings.yaml.
COST_FUNCTIONS: Mapping[str, CostFunction] = {
    "quadratic": quadratic_cost,
    "linear": linear_cost,
}


def quadratic_max_delta_pct(budget: float, alpha: float) -> float:
    """Inverse of ``quadratic_cost``: largest Delta% affordable with ``budget``."""
    return math.sqrt(max(budget, 0.0) / alpha)


def linear_max_delta_pct(budget: float, alpha: float) -> float:
    """Inverse of ``linear_cost``: largest Delta% affordable with ``budget``."""
    return max(budget, 0.0) / alpha


# Inverse registry: budget -> largest achievable Delta%, per type.
INVERSE_COST_FUNCTIONS: Mapping[str, CostFunction] = {
    "quadratic": quadratic_max_delta_pct,
    "linear": linear_max_delta_pct,
}

# Display formulas (LaTeX) for each registered cost-function type.
COST_FUNCTION_FORMULAS: Mapping[str, str] = {
    "quadratic": r"C(\Delta\%) = \alpha \cdot (\Delta\%)^2",
    "linear": r"C(\Delta\%) = \alpha \cdot |\Delta\%|",
}


@dataclass(frozen=True)
class CostFunctionSpec:
    """Per-KPI cost-function configuration."""

    kpi_name: str
    cost_function_type: str
    alpha: float

    @classmethod
    def from_config(cls, name: str, cfg: Mapping[str, object]) -> "CostFunctionSpec":
        """Build a spec from one ``kpi_definitions`` entry of the YAML config."""
        return cls(
            kpi_name=name,
            cost_function_type=str(cfg["cost_function_type"]),
            alpha=float(cfg["alpha"]),
        )

    def max_delta_pct(self, budget: float) -> float:
        """Largest Delta% achievable with ``budget`` (inverse of ``cost``)."""
        try:
            fn = INVERSE_COST_FUNCTIONS[self.cost_function_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown cost_function_type {self.cost_function_type!r} for "
                f"KPI {self.kpi_name!r}. Available: {list(INVERSE_COST_FUNCTIONS)}"
            ) from exc
        return fn(budget, self.alpha)
