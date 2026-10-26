from __future__ import annotations

from typing import Any, Callable, Mapping

from systems.kinematic_collision_system import KinematicCollisionSystem
from planning.casadi_planner import CasadiPlanner
from planning.ompl_initial_guess import OmplInitialGuess
from planning.ompl_planner import OmplPlanner
from planning.planner import Planner

PlannerBuilder = Callable[[KinematicCollisionSystem, Mapping[str, Any]], Planner]


def build_ompl_casadi_planner(simulator: KinematicCollisionSystem, config: Mapping[str, Any]) -> Planner:
    """The MPC with the goal configuration and the first guess of an OMPL RRT-Connect path (``OmplInitialGuess``)."""
    path_planner = OmplPlanner(simulator, config)
    return CasadiPlanner(
        simulator, config, initial_guess_generator=OmplInitialGuess(path_planner, config["ompl_replan_distance"])
    )


# The experts the config can choose (``expert``): the MPC alone, or the MPC with the goal configuration and the first
# guess of the path planner. The path alone is no expert: a jerk-limited arm cannot follow it step by step.
PLANNER_REGISTRY: dict[str, PlannerBuilder] = {
    "casadi": CasadiPlanner,
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
        simulator: KinematicCollisionSystem,
        config: Mapping[str, Any],
    ) -> Planner:
        try:
            planner_builder = PLANNER_REGISTRY[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available: {sorted(PLANNER_REGISTRY.keys())}"
            ) from exc
        return planner_builder(simulator, config)
