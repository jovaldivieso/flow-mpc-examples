from __future__ import annotations

from typing import Any, Callable, Mapping

from dynamics.dynamics import DynamicsSimulator
from dynamics.franka_dynamics import FrankaDynamics
from dynamics.triple_integrator import TripleIntegrator

DynamicsBuilder = Callable[[Mapping[str, Any]], DynamicsSimulator]

# The systems the config can choose (``system``)
SYSTEM_REGISTRY: dict[str, DynamicsBuilder] = {
    "franka": FrankaDynamics,
    "triple_integrator": TripleIntegrator,
}


class DynamicsFactory:
    """Factory for simulator instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(SYSTEM_REGISTRY.keys())

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> DynamicsSimulator:
        try:
            simulator_cls = SYSTEM_REGISTRY[system_name]
        except KeyError as exc:
            raise ValueError(f"Unknown system '{system_name}'. Available: {sorted(SYSTEM_REGISTRY.keys())}") from exc
        return simulator_cls(config)

