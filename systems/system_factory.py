from __future__ import annotations

from typing import Any, Callable, Mapping

from systems.system import SystemSimulator
from systems.franka_system import FrankaSystem
from systems.point_mass_system import PointMassSystem

SystemBuilder = Callable[[Mapping[str, Any]], SystemSimulator]

# The systems the config can choose (``system``)
SYSTEM_REGISTRY: dict[str, SystemBuilder] = {
    "franka": FrankaSystem,
    "triple_integrator": PointMassSystem,
}


class SystemFactory:
    """Factory for simulator instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(SYSTEM_REGISTRY.keys())

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> SystemSimulator:
        try:
            simulator_cls = SYSTEM_REGISTRY[system_name]
        except KeyError as exc:
            raise ValueError(f"Unknown system '{system_name}'. Available: {sorted(SYSTEM_REGISTRY.keys())}") from exc
        return simulator_cls(config)

