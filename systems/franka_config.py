"""The config of ``FrankaSystem``: its constants and the validation of its fields."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .config_fields import (
    ConfigurationError,
    read_bool,
    read_float,
    read_limits,
    read_obstacles,
    read_tolerances,
    read_vector,
    require_only_known_keys,
)

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7
POSE_COMPONENTS = ["x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)]
POSE_DIM = len(POSE_COMPONENTS)

# The keys of a Franka config beside the ones every system has
FRANKA_CONFIG_KEYS = {"start_region", "goal_region"}


def _read_region(raw_config: Mapping[str, Any], key: str, extra_keys: set[str] = frozenset()) -> dict[str, Any]:
    """A box of TCP positions [m]: ``x``, ``y`` and ``z``, each as [min, max], plus the optional ``extra_keys``."""
    raw = raw_config[key]
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"'{key}' must be a mapping with 'x', 'y' and 'z', got {type(raw).__name__}.")
    require_only_known_keys(raw, {"x", "y", "z"} | extra_keys, context=f"'{key}'")
    if not {"x", "y", "z"} <= set(raw):
        raise ConfigurationError(f"'{key}' must define 'x', 'y' and 'z'.")
    region: dict[str, Any] = {}
    for axis in "xyz":
        low, high = read_vector(raw, axis, 2, ())
        if low >= high:
            raise ConfigurationError(f"'{key}.{axis}' must be [min, max] with min < max.")
        region[axis] = [low, high]
    return region


def _read_goal_region(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The box of the goals and, optionally, the world direction the hand (tool z-axis) points along (``tool_axis``) within
    ``tool_tilt`` [rad]."""
    region = _read_region(raw_config, "goal_region", {"tool_axis", "tool_tilt"})
    raw = raw_config["goal_region"]
    if "tool_axis" in raw:
        axis = read_vector(raw, "tool_axis", 3, ())
        if not np.linalg.norm(axis) > 0:
            raise ConfigurationError("'goal_region.tool_axis' must not be zero.")
        region["tool_axis"] = list(axis)
        region["tool_tilt"] = read_float(raw, "tool_tilt", 0.0)
        if region["tool_tilt"] < 0:
            raise ConfigurationError("'goal_region.tool_tilt' must be non-negative.")
    elif "tool_tilt" in raw:
        raise ConfigurationError("'goal_region.tool_tilt' needs a 'goal_region.tool_axis'.")
    return region


def validate_franka_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of the arm: limits, tolerances, sampling and goal."""
    min_start_goal_distance = read_float(raw_config, "min_start_goal_distance", 0.25)
    if min_start_goal_distance < 0:
        raise ConfigurationError("'min_start_goal_distance' must be non-negative.")

    fields: dict[str, Any] = {
        "randomize_goal": read_bool(raw_config, "randomize_goal", "goal" not in raw_config),
        "min_start_goal_distance": min_start_goal_distance,
        "max_acceleration": list(read_limits(raw_config, "max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION, FRANKA_NUM_JOINTS)),
        "max_jerk": list(read_limits(raw_config, "max_jerk", FRANKA_DEFAULT_MAX_JERK, FRANKA_NUM_JOINTS)),
        "obstacles": read_obstacles(raw_config, 3),
        **read_tolerances(raw_config, 0.03, 0.05),
    }
    if "start_region" in raw_config:
        fields["start_region"] = _read_region(raw_config, "start_region")
    if "goal_region" in raw_config:
        fields["goal_region"] = _read_goal_region(raw_config)
    if "goal" in raw_config:
        goal = read_vector(raw_config, "goal", POSE_DIM, ())
        rotation = np.array(goal[3:]).reshape(3, 3)
        if not (np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) and np.linalg.det(rotation) > 0):
            raise ConfigurationError("'goal' must be a position followed by a rotation matrix (row-major, 9 numbers).")
        fields["goal"] = list(goal)
    return fields
