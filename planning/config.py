from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from systems.franka_system import (
    FRANKA_DEFAULT_MAX_ACCELERATION,
    FRANKA_DEFAULT_MAX_JERK,
    FRANKA_NUM_JOINTS,
    POSE_DIM,
)
from systems.system_factory import SystemFactory
from planning.planner_factory import PlannerFactory


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is missing or malformed."""


@dataclass(frozen=True)
class PlannerConfig:
    # The expert planner: a name of planning.planner_factory.PLANNER_REGISTRY
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
        read = {bool: _bool, int: _int, float: _float}
        defaults = asdict(cls())
        return cls(**{name: read[type(default)](config, name, default) for name, default in defaults.items()})

    @classmethod
    def keys(cls) -> set[str]:
        return {field.name for field in fields(cls)}


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML into a dictionary and fail with context when invalid."""
    path = Path(config_path)
    if not path.exists():
        raise ConfigurationError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ConfigurationError(f"Expected YAML mapping in {path}, got {type(data).__name__}.")
    return data


def _not_numeric(key: str, idx: int, value: Any) -> ConfigurationError:
    hint = " (YAML reads 1e-3 as a string, write 1.0e-3)" if isinstance(value, str) else ""
    return ConfigurationError(f"'{key}[{idx}]' must be numeric, got {type(value).__name__} {value!r}{hint}.")


def _float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be numeric, got {type(value).__name__}.")
    return float(value)


def _int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"'{key}' must be int, got {type(value).__name__}.")
    return int(value)


def _optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"'{key}' must be int when provided, got {type(value).__name__}.")
    return int(value)


def _optional_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(config, key)
    if value is None:
        return None
    if value <= 0:
        raise ConfigurationError(f"'{key}' must be positive when provided.")
    return value


def _optional_non_negative_int(config: Mapping[str, Any], key: str) -> int | None:
    value = _optional_int(config, key)
    if value is None:
        return None
    if value < 0:
        raise ConfigurationError(f"'{key}' must be non-negative when provided.")
    return value


def _resolved_action_noise_seed(config: Mapping[str, Any]) -> int:
    action_noise_seed = _optional_non_negative_int(config, "action_noise_seed")
    if action_noise_seed is not None:
        return action_noise_seed

    initial_state_seed = _optional_non_negative_int(config, "initial_state_seed")
    if initial_state_seed is not None:
        return initial_state_seed

    return 0


def _bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be bool, got {type(value).__name__}.")
    return value


def _vector(config: Mapping[str, Any], key: str, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = config.get(key, list(default))
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list of length {size}.")
    if len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")
    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _not_numeric(key, idx, value)
        out.append(float(value))
    return tuple(out)


def _positive_vector(raw: Any, key: str, size: int) -> tuple[float, ...]:
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list of length {size}.")
    if len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")

    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _not_numeric(key, idx, value)
        value_f = float(value)
        if value_f <= 0:
            raise ConfigurationError(f"'{key}[{idx}]' must be positive.")
        out.append(value_f)
    return tuple(out)


def _require_only_known_keys(raw: Mapping[str, Any], allowed_keys: set[str], *, context: str) -> None:
    unknown_keys = sorted(key for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        formatted = ", ".join(f"'{key}'" for key in unknown_keys)
        raise ConfigurationError(f"Unknown keys in {context}: {formatted}.")


COMMON_KEYS = OmplConfig.keys() | {
    "system",
    "dt",
    "horizon",
    "Q_diag",
    "R_weight",
    "R_diag",
    "terminal_cost_multiplier",
    "expert",
    "terminal_velocity_weight",
    "fallback_terminal_velocity_tol",
    "max_consecutive_fallbacks",
    "initial_state_seed",
    "action_noise_seed",
    "done_hold_steps",
    "goal",
    "randomize_goal",
    "max_acceleration",
    "max_jerk",
    "min_start_goal_distance",
    "pos_tol",
    "vel_tol",
}
SYSTEM_KEYS = {
    "franka": set(),
    "triple_integrator": {"dimension", "max_velocity", "workspace_bounds"},
}


def _limits(raw_config: Mapping[str, Any], key: str, default: tuple[float, ...] | float, size: int) -> tuple[float, ...]:
    """A positive limit per joint (or axis), given as one number for all of them or a list."""
    raw = raw_config.get(key, list(default) if isinstance(default, tuple) else default)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [float(raw)] * size
    return _positive_vector(raw, key, size)


def _validate_planner(
    config: Mapping[str, Any],
    nx: int,
    nu: int,
) -> PlannerConfig:
    r_weight = _float(config, "R_weight", 0.1)
    if r_weight <= 0:
        raise ConfigurationError("'R_weight' must be positive.")

    raw_r_diag = config.get("R_diag")
    if raw_r_diag is not None:
        r_diag = _positive_vector(raw_r_diag, key="R_diag", size=nu)
    else:
        r_diag = tuple([r_weight] * nu)

    planner = PlannerConfig(
        expert=str(config.get("expert", "casadi")),
        horizon=_int(config, "horizon", 20),
        q_diag=_vector(config, "Q_diag", nx, tuple([10.0] * nx)),
        r_weight=r_weight,
        r_diag=r_diag,
        terminal_cost_multiplier=_float(config, "terminal_cost_multiplier", 10.0),
        terminal_velocity_weight=_float(config, "terminal_velocity_weight", 1.0),
        fallback_terminal_velocity_tol=_float(config, "fallback_terminal_velocity_tol", 1e-2),
        max_consecutive_fallbacks=_int(config, "max_consecutive_fallbacks", 5),
    )

    if planner.expert not in PlannerFactory.names():
        raise ConfigurationError(f"'expert' must be one of {', '.join(PlannerFactory.names())}, got {planner.expert!r}.")
    if planner.horizon <= 0:
        raise ConfigurationError("'horizon' must be positive.")
    if planner.terminal_cost_multiplier <= 0:
        raise ConfigurationError("'terminal_cost_multiplier' must be positive.")
    if planner.terminal_velocity_weight <= 0:
        raise ConfigurationError("'terminal_velocity_weight' must be positive.")
    if planner.fallback_terminal_velocity_tol < 0:
        raise ConfigurationError("'fallback_terminal_velocity_tol' must be non-negative.")
    if planner.max_consecutive_fallbacks <= 0:
        raise ConfigurationError("'max_consecutive_fallbacks' must be positive.")
    return planner


def load_and_validate_system_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and return a validated config dictionary for the system it names."""
    raw = load_yaml_config(config_path)
    return validate_system_config(raw_config=raw)


def _tolerances(raw_config: Mapping[str, Any], pos_tol: float, vel_tol: float) -> dict[str, float]:
    tolerances = {"pos_tol": _float(raw_config, "pos_tol", pos_tol), "vel_tol": _float(raw_config, "vel_tol", vel_tol)}
    for key, value in tolerances.items():
        if value <= 0:
            raise ConfigurationError(f"'{key}' must be positive.")
    return tolerances


def _franka_system(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of the arm: limits, tolerances, sampling and goal."""
    min_start_goal_distance = _float(raw_config, "min_start_goal_distance", 0.25)
    if min_start_goal_distance < 0:
        raise ConfigurationError("'min_start_goal_distance' must be non-negative.")

    fields: dict[str, Any] = {
        "randomize_goal": _bool(raw_config, "randomize_goal", "goal" not in raw_config),
        "min_start_goal_distance": min_start_goal_distance,
        "max_acceleration": list(_limits(raw_config, "max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION, FRANKA_NUM_JOINTS)),
        "max_jerk": list(_limits(raw_config, "max_jerk", FRANKA_DEFAULT_MAX_JERK, FRANKA_NUM_JOINTS)),
        **_tolerances(raw_config, 0.03, 0.05),
    }
    if "goal" in raw_config:
        goal = _vector(raw_config, "goal", POSE_DIM, ())
        rotation = np.array(goal[3:]).reshape(3, 3)
        if not (np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) and np.linalg.det(rotation) > 0):
            raise ConfigurationError("'goal' must be a position followed by a rotation matrix (row-major, 9 numbers).")
        fields["goal"] = list(goal)
    return fields


def _triple_integrator_system(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of the point mass: dimension, limits, tolerances, sampling and goal."""
    dimension = _int(raw_config, "dimension", 2)
    if dimension < 1:
        raise ConfigurationError("'dimension' must be positive.")
    bounds = _vector(raw_config, "workspace_bounds", 2, (-1.0, 1.0))
    if bounds[0] >= bounds[1]:
        raise ConfigurationError("'workspace_bounds' minimum must be smaller than its maximum.")
    min_start_goal_distance = _float(raw_config, "min_start_goal_distance", 0.5)
    if min_start_goal_distance < 0:
        raise ConfigurationError("'min_start_goal_distance' must be non-negative.")
    fields: dict[str, Any] = {
        "dimension": dimension,
        "workspace_bounds": list(bounds),
        "randomize_goal": _bool(raw_config, "randomize_goal", "goal" not in raw_config),
        "min_start_goal_distance": min_start_goal_distance,
        "max_velocity": list(_limits(raw_config, "max_velocity", 1.0, dimension)),
        "max_acceleration": list(_limits(raw_config, "max_acceleration", 2.0, dimension)),
        "max_jerk": list(_limits(raw_config, "max_jerk", 10.0, dimension)),
        **_tolerances(raw_config, 0.02, 0.02),
    }
    if "goal" in raw_config:
        fields["goal"] = list(_vector(raw_config, "goal", dimension, ()))
    return fields


def validate_system_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate system and planner fields, returning normalized config values."""
    system = raw_config.get("system", "franka")
    if system not in SystemFactory.names():
        raise ConfigurationError(f"'system' must be one of {', '.join(SystemFactory.names())}, got {system!r}.")
    _require_only_known_keys(raw_config, COMMON_KEYS | SYSTEM_KEYS[system], context=f"'{system}' config")

    dt = _float(raw_config, "dt", 0.05)
    if dt <= 0:
        raise ConfigurationError("'dt' must be positive.")
    config_out: dict[str, Any] = {"system": system, "dt": dt, "action_noise_seed": _resolved_action_noise_seed(raw_config)}
    initial_state_seed = _optional_non_negative_int(raw_config, "initial_state_seed")
    if initial_state_seed is not None:
        config_out["initial_state_seed"] = initial_state_seed
    done_hold_steps = _optional_positive_int(raw_config, "done_hold_steps")
    if done_hold_steps is not None:
        config_out["done_hold_steps"] = done_hold_steps

    if system == "franka":
        config_out.update(_franka_system(raw_config))
        nx, nu = 3 * FRANKA_NUM_JOINTS, FRANKA_NUM_JOINTS
    else:
        config_out.update(_triple_integrator_system(raw_config))
        nx, nu = 3 * config_out["dimension"], config_out["dimension"]

    planner_cfg = _validate_planner(raw_config, nx=nx, nu=nu)
    config_out.update(
        {
            **asdict(OmplConfig.from_mapping(raw_config)),
            "expert": planner_cfg.expert,
            "horizon": planner_cfg.horizon,
            "Q_diag": list(planner_cfg.q_diag),
            "R_weight": planner_cfg.r_weight,
            "R_diag": list(planner_cfg.r_diag),
            "terminal_cost_multiplier": planner_cfg.terminal_cost_multiplier,
            "terminal_velocity_weight": planner_cfg.terminal_velocity_weight,
            "fallback_terminal_velocity_tol": planner_cfg.fallback_terminal_velocity_tol,
            "max_consecutive_fallbacks": planner_cfg.max_consecutive_fallbacks,
        }
    )
    return config_out
