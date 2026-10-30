from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from systems.config_fields import (
    ConfigurationError,
    read_bool,
    read_float,
    read_int,
    read_positive_vector,
)


@dataclass(frozen=True)
class PlannerConfig:
    # The expert planner: a name of planning.planner_factory.PLANNER_REGISTRY (``validate_system_config`` checks it)
    expert: str = "casadi"
    horizon: int = 20
    q_diag: tuple[float, ...] = ()
    r_weight: float = 0.1
    r_diag: tuple[float, ...] = ()
    terminal_cost_multiplier: float = 10.0
    # Read by CasadiTrajectoryProjector (planning/casadi_projector.py); a config
    # that omits them gets these defaults.
    terminal_velocity_weight: float = 1.0
    fallback_terminal_velocity_tol: float = 1e-2
    max_consecutive_fallbacks: int = 5

    def __post_init__(self) -> None:
        if self.r_weight <= 0:
            raise ConfigurationError("'R_weight' must be positive.")
        if self.horizon <= 0:
            raise ConfigurationError("'horizon' must be positive.")
        if self.terminal_cost_multiplier <= 0:
            raise ConfigurationError("'terminal_cost_multiplier' must be positive.")
        if self.terminal_velocity_weight <= 0:
            raise ConfigurationError("'terminal_velocity_weight' must be positive.")
        if self.fallback_terminal_velocity_tol < 0:
            raise ConfigurationError("'fallback_terminal_velocity_tol' must be non-negative.")
        if self.max_consecutive_fallbacks <= 0:
            raise ConfigurationError("'max_consecutive_fallbacks' must be positive.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> PlannerConfig:
        """The planner settings of a (raw) config; keys it omits keep their defaults, wrong types and values are
        rejected. ``Q_diag`` and ``R_diag`` stay empty when omitted: their lengths and defaults depend on the system, so
        ``CasadiPlanner`` checks and completes them."""
        defaults = cls()
        return cls(
            expert=str(config.get("expert", defaults.expert)),
            horizon=read_int(config, "horizon", defaults.horizon),
            q_diag=read_positive_vector(config["Q_diag"], key="Q_diag") if "Q_diag" in config else (),
            r_weight=read_float(config, "R_weight", defaults.r_weight),
            r_diag=read_positive_vector(config["R_diag"], key="R_diag") if "R_diag" in config else (),
            terminal_cost_multiplier=read_float(config, "terminal_cost_multiplier", defaults.terminal_cost_multiplier),
            terminal_velocity_weight=read_float(config, "terminal_velocity_weight", defaults.terminal_velocity_weight),
            fallback_terminal_velocity_tol=read_float(
                config, "fallback_terminal_velocity_tol", defaults.fallback_terminal_velocity_tol
            ),
            max_consecutive_fallbacks=read_int(config, "max_consecutive_fallbacks", defaults.max_consecutive_fallbacks),
        )


@dataclass(frozen=True)
class OmplConfig:
    """Global path planning of the expert ``ompl_casadi``: an OMPL RRT-Connect path to the goal configuration.

    The field names are the YAML keys (and the keys of the validated config).
    """

    # Speed [rad/s] in joint space the OMPL path is traversed at (its timing: one configuration per control period)
    ompl_speed: float = 2.0
    # Distance [rad] from the optimized trajectory at which a new path is planned from where the arm is
    ompl_replan_distance: float = 0.5
    ompl_timeout: float = 5.0  # [s] per RRT-Connect solve
    ompl_seed: int = 0

    def __post_init__(self) -> None:
        positive = ("ompl_speed", "ompl_replan_distance", "ompl_timeout")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"'{name}' must be positive.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> OmplConfig:
        """The settings of a (raw) config; keys it omits keep their defaults, wrong types and values are rejected."""
        read = {bool: read_bool, int: read_int, float: read_float}
        defaults = asdict(cls())
        return cls(**{name: read[type(default)](config, name, default) for name, default in defaults.items()})

    @classmethod
    def keys(cls) -> set[str]:
        return {field.name for field in fields(cls)}
