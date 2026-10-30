from __future__ import annotations

from typing import Any, Mapping

from systems.system import SystemSimulator
from systems.franka_system import FrankaSystem
from systems.point_mass_system import PointMassSystem

# The systems the config can choose (``system``)
SYSTEM_REGISTRY: dict[str, type[SystemSimulator]] = {
    "franka": FrankaSystem,
    "triple_integrator": PointMassSystem,
}


class SystemFactory:
    """Factory for simulator instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(SYSTEM_REGISTRY.keys())

    @staticmethod
    def _system_class(system_name: str) -> type[SystemSimulator]:
        try:
            return SYSTEM_REGISTRY[system_name]
        except KeyError as exc:
            raise ValueError(f"Unknown system '{system_name}'. Available: {sorted(SYSTEM_REGISTRY.keys())}") from exc

    @staticmethod
    def create(system_name: str, config: Mapping[str, Any]) -> SystemSimulator:
        return SystemFactory._system_class(system_name)(config)

    @staticmethod
    def config_keys(system_name: str) -> set[str]:
        """The keys a config of the system may set."""
        return SystemFactory._system_class(system_name).config_keys()

    @staticmethod
    def validate_config(system_name: str, raw_config: Mapping[str, Any]) -> dict[str, Any]:
        """The normalized fields of the system in a raw config."""
        return SystemFactory._system_class(system_name).validate_config(raw_config)

