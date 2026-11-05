from __future__ import annotations

from typing import Any, Callable, Mapping

from systems.triple_integrator import TripleIntegratorBase
from planning.ompl_casadi_planner import OmplCasadiPlanner
from planning.planner import Planner
from planning.planner_config import ExpertConfig, ProjectionConfig

PlannerBuilder = Callable[[TripleIntegratorBase, Mapping[str, Any]], Planner]


def build_ompl_casadi_planner(simulator: TripleIntegratorBase, config: Mapping[str, Any]) -> Planner:
    """The expert that follows an OMPL RRT-Connect path timed by time-optimal trajectory generation."""
    return OmplCasadiPlanner(simulator, ExpertConfig.from_mapping(config), ProjectionConfig.from_mapping(config))


# The experts the config can choose (``expert``)
PLANNER_REGISTRY: dict[str, PlannerBuilder] = {
    "ompl_casadi": build_ompl_casadi_planner,
}


class PlannerFactory:
    """Factory for planner instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(PLANNER_REGISTRY.keys())

    @staticmethod
    def create(
        planner_name: str,
        simulator: TripleIntegratorBase,
        config: Mapping[str, Any],
    ) -> Planner:
        try:
            planner_builder = PLANNER_REGISTRY[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available: {sorted(PLANNER_REGISTRY.keys())}"
            ) from exc
        return planner_builder(simulator, config)
