from __future__ import annotations

from typing import Any, Callable, Mapping

from dynamics.dynamics import DynamicsSimulator
from planning.casadi_planner import CasadiPlanner
from planning.ompl_casadi_planner import OmplCasadiPlanner
from planning.planner import Planner

PlannerBuilder = Callable[[DynamicsSimulator, Mapping[str, Any]], Planner]

# The experts the config can choose (``expert``): the MPC alone, or the MPC with the goal configuration and the first
# guess of the path planner. The path alone is no expert: a jerk-limited arm cannot follow it step by step.
PLANNER_REGISTRY: dict[str, PlannerBuilder] = {
    "casadi": CasadiPlanner,
    "ompl_casadi": OmplCasadiPlanner,
}


class PlannerFactory:
    """Factory for planner instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(PLANNER_REGISTRY.keys())

    @staticmethod
    def create(
        planner_name: str,
        simulator: DynamicsSimulator,
        config: Mapping[str, Any],
    ) -> Planner:
        try:
            planner_cls = PLANNER_REGISTRY[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available: {sorted(PLANNER_REGISTRY.keys())}"
            ) from exc
        return planner_cls(simulator, config)

