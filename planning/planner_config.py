from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from systems.config_fields import ConfigurationError, read_float, read_int


@dataclass(frozen=True)
class ExpertConfig:
    """The expert ``ompl_casadi``: an OMPL RRT-Connect path to the goal configuration, timed by time-optimal trajectory
    generation (``planning/totg.py``), whose positions ahead of the arm are the plan.

    The field names are the YAML keys (and the keys of the validated config).
    """

    # The expert's name, a key of planning.planner_factory.PLANNER_REGISTRY (``validate_system_config`` checks it)
    expert: str = "ompl_casadi"
    # Rows of positions the expert plans (at least the policy's ``prediction_horizon``)
    horizon: int = 20
    ompl_timeout: float = 5.0  # [s] per RRT-Connect solve
    # RRT-Connect paths planned per path; the one along which the end effector travels the shortest is kept
    ompl_candidates: int = 1
    ompl_seed: int = 0
    # Distance [rad] from the timed path at which a new path is planned from where the arm is (the learner drove it away)
    replan_distance: float = 0.5
    # Path length [rad] ahead of the arm's progress searched for the point of the path nearest to it
    progress_window: float = 1.5
    # The fractions of the velocity and acceleration limits the path is timed within: the arm tracks it within its jerk limit
    totg_velocity_scale: float = 0.8
    totg_acceleration_scale: float = 0.5
    # How far [rad] the timed path may cut the corner of a waypoint of the OMPL path (the circular blend of the path)
    totg_max_deviation: float = 0.05

    def __post_init__(self) -> None:
        if self.horizon <= 0:
            raise ConfigurationError("'horizon' must be positive.")
        if self.ompl_candidates < 1:
            raise ConfigurationError("'ompl_candidates' must be at least 1.")
        for name in ("ompl_timeout", "replan_distance", "progress_window", "totg_max_deviation"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"'{name}' must be positive.")
        for name in ("totg_velocity_scale", "totg_acceleration_scale"):
            if not 0 < getattr(self, name) <= 1:
                raise ConfigurationError(f"'{name}' must be in (0, 1].")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ExpertConfig:
        """The settings of a (raw) config; keys it omits keep their defaults, wrong types and values are rejected."""
        defaults = asdict(cls())
        values: dict[str, Any] = {}
        for name, default in defaults.items():
            if isinstance(default, str):
                values[name] = str(config.get(name, default))
            elif isinstance(default, int):
                values[name] = read_int(config, name, default)
            else:
                values[name] = read_float(config, name, default)
        return cls(**values)

    @classmethod
    def keys(cls) -> set[str]:
        return {field.name for field in fields(cls)}


@dataclass(frozen=True)
class ProjectionConfig:
    """The cost of ``CasadiTrajectoryProjector`` (the tracking of a chunk of positions, after SafeFlowMPC's safety filter,
    which weights the positions 1, the velocities and accelerations 1e-3 and the jerks 1e-4) and its fallback."""

    velocity_weight: float = 1e-3
    acceleration_weight: float = 1e-3
    jerk_weight: float = 1e-4
    # Weight of the soft rest at the end of the chunk, which keeps the problem feasible from a high speed
    terminal_velocity_weight: float = 10.0
    fallback_terminal_velocity_tol: float = 1e-2
    max_consecutive_fallbacks: int = 5

    def __post_init__(self) -> None:
        for name in ("velocity_weight", "acceleration_weight", "jerk_weight", "terminal_velocity_weight"):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"'{name}' must be positive.")
        if self.fallback_terminal_velocity_tol < 0:
            raise ConfigurationError("'fallback_terminal_velocity_tol' must be non-negative.")
        if self.max_consecutive_fallbacks <= 0:
            raise ConfigurationError("'max_consecutive_fallbacks' must be positive.")

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> ProjectionConfig:
        """The settings of a (raw) config, under the keys ``projection_<field>`` (``terminal_velocity_weight`` and the two
        fallback fields under their own names); keys it omits keep their defaults."""
        defaults = asdict(cls())
        values: dict[str, Any] = {}
        for name, default in defaults.items():
            key = cls.key(name)
            values[name] = read_int(config, key, default) if isinstance(default, int) else read_float(config, key, default)
        return cls(**values)

    @staticmethod
    def key(name: str) -> str:
        return name if name in ("terminal_velocity_weight", "fallback_terminal_velocity_tol", "max_consecutive_fallbacks") else f"projection_{name}"

    @classmethod
    def keys(cls) -> set[str]:
        return {cls.key(field.name) for field in fields(cls)}
