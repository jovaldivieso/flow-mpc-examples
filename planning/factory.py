from __future__ import annotations

from typing import Any, Callable, Mapping

from dynamics.dynamics import DynamicsProtocol
from dynamics.franka_dynamics import FrankaDynamics
from dynamics.triple_integrator import TripleIntegrator
from planning.casadi_planner import CasadiPlanner
from planning.ompl_casadi_planner import OmplCasadiPlanner
from planning.planner import Planner

DynamicsBuilder = Callable[[Mapping[str, Any]], DynamicsProtocol]
PlannerBuilder = Callable[[DynamicsProtocol, Mapping[str, Any]], Planner]

# The systems the config can choose (``system``)
SYSTEM_REGISTRY: dict[str, DynamicsBuilder] = {
    "franka": FrankaDynamics,
    "triple_integrator": TripleIntegrator,
}

# The experts the config can choose (``expert``): the MPC alone, or the MPC with the goal configuration and the first
# guess of the path planner. The path alone is no expert: a jerk-limited arm cannot follow it step by step.
PLANNER_REGISTRY: dict[str, PlannerBuilder] = {
    "casadi": CasadiPlanner,
    "ompl_casadi": OmplCasadiPlanner,
}


class DynamicsFactory:
    """Factory for simulator instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(SYSTEM_REGISTRY.keys())

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> DynamicsProtocol:
        try:
            simulator_cls = SYSTEM_REGISTRY[system_name]
        except KeyError as exc:
            raise ValueError(f"Unknown system '{system_name}'. Available: {sorted(SYSTEM_REGISTRY.keys())}") from exc
        return simulator_cls(config)


class PlannerFactory:
    """Factory for planner instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(PLANNER_REGISTRY.keys())

    @staticmethod
    def create(
        planner_name: str,
        simulator: DynamicsProtocol,
        config: Mapping[str, Any],
    ) -> Planner:
        try:
            planner_cls = PLANNER_REGISTRY[planner_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown planner '{planner_name}'. Available: {sorted(PLANNER_REGISTRY.keys())}"
            ) from exc
        return planner_cls(simulator, config)
