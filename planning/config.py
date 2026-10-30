from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from systems.config_fields import ConfigurationError, read_float, require_only_known_keys
from systems.system_factory import SystemFactory
from planning.planner_config import OmplConfig, PlannerConfig
from planning.planner_factory import PlannerFactory


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


# The keys of every config beside the ones of its system (``SystemFactory.config_keys``)
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
}


def load_and_validate_system_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config and return a validated config dictionary for the system it names."""
    raw = load_yaml_config(config_path)
    return validate_system_config(raw_config=raw)


def validate_system_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate system and planner fields, returning normalized config values."""
    system = raw_config.get("system", "franka")
    if system not in SystemFactory.names():
        raise ConfigurationError(f"'system' must be one of {', '.join(SystemFactory.names())}, got {system!r}.")
    require_only_known_keys(raw_config, COMMON_KEYS | SystemFactory.config_keys(system), context=f"'{system}' config")

    dt = read_float(raw_config, "dt", 0.05)
    if dt <= 0:
        raise ConfigurationError("'dt' must be positive.")
    config_out: dict[str, Any] = {"system": system, "dt": dt, "action_noise_seed": _resolved_action_noise_seed(raw_config)}
    initial_state_seed = _optional_non_negative_int(raw_config, "initial_state_seed")
    if initial_state_seed is not None:
        config_out["initial_state_seed"] = initial_state_seed
    done_hold_steps = _optional_positive_int(raw_config, "done_hold_steps")
    if done_hold_steps is not None:
        config_out["done_hold_steps"] = done_hold_steps

    config_out.update(SystemFactory.validate_config(system, raw_config))

    planner_cfg = PlannerConfig.from_mapping(raw_config)
    if planner_cfg.expert not in PlannerFactory.names():
        raise ConfigurationError(f"'expert' must be one of {', '.join(PlannerFactory.names())}, got {planner_cfg.expert!r}.")
    config_out.update(
        {
            **asdict(OmplConfig.from_mapping(raw_config)),
            "expert": planner_cfg.expert,
            "horizon": planner_cfg.horizon,
            "R_weight": planner_cfg.r_weight,
            "terminal_cost_multiplier": planner_cfg.terminal_cost_multiplier,
            "terminal_velocity_weight": planner_cfg.terminal_velocity_weight,
            "fallback_terminal_velocity_tol": planner_cfg.fallback_terminal_velocity_tol,
            "max_consecutive_fallbacks": planner_cfg.max_consecutive_fallbacks,
            **({"Q_diag": list(planner_cfg.q_diag)} if planner_cfg.q_diag else {}),
            **({"R_diag": list(planner_cfg.r_diag)} if planner_cfg.r_diag else {}),
        }
    )
    return config_out
