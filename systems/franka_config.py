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
FRANKA_CONFIG_KEYS = {"sampling", "sampling_modes"}

# Where a start or a goal is drawn unless a region says otherwise: a box of TCP positions in front of the base and the
# hand (tool z-axis) within HAND_TILT of HAND_AXIS, which spans pointing forward to pointing down
WORKSPACE_BOX = {"x": [0.2, 0.8], "y": [-0.6, 0.6], "z": [0.1, 0.8]}
HAND_AXIS = [1.0, 0.0, -1.0]
HAND_TILT = 0.9
DEFAULT_MODE = "workspace"


def _read_pose_region(raw: Any, context: str) -> dict[str, Any]:
    """A box of TCP positions [m] (``x``, ``y`` and ``z``, each as [min, max]) and the world direction the hand points along
    (``tool_axis``) within ``tool_tilt`` [rad], which default to ``HAND_AXIS`` and ``HAND_TILT``."""
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{context} must be a mapping with 'x', 'y' and 'z', got {type(raw).__name__}.")
    require_only_known_keys(raw, {"x", "y", "z", "tool_axis", "tool_tilt"}, context=context)
    if not {"x", "y", "z"} <= set(raw):
        raise ConfigurationError(f"{context} must define 'x', 'y' and 'z'.")
    region: dict[str, Any] = {}
    for axis in "xyz":
        low, high = read_vector(raw, axis, 2, ())
        if low >= high:
            raise ConfigurationError(f"{context}.{axis} must be [min, max] with min < max.")
        region[axis] = [low, high]
    axis = read_vector(raw, "tool_axis", 3, tuple(HAND_AXIS))
    if not np.linalg.norm(axis) > 0:
        raise ConfigurationError(f"{context}.tool_axis must not be zero.")
    region["tool_axis"] = list(axis)
    region["tool_tilt"] = read_float(raw, "tool_tilt", HAND_TILT)
    if not 0 <= region["tool_tilt"] <= np.pi:
        raise ConfigurationError(f"{context}.tool_tilt must be in [0, pi].")
    return region


def _read_sampling(raw_config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The modes an episode is drawn from, by name: each a ``weight``, the ``start`` and ``goal`` region, and the
    ``start_speed``, the fraction of the joint velocity limits the arm starts with (at random velocities)."""
    raw = raw_config.get("sampling", {DEFAULT_MODE: {}})
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigurationError("'sampling' must be a mapping from a mode name to its settings.")
    modes: dict[str, dict[str, Any]] = {}
    for name, mode in raw.items():
        context = f"'sampling.{name}'"
        if not isinstance(mode, Mapping):
            raise ConfigurationError(f"{context} must be a mapping.")
        require_only_known_keys(mode, {"weight", "start", "goal", "start_speed"}, context=context)
        weight, speed = read_float(mode, "weight", 1.0), read_float(mode, "start_speed", 0.0)
        if weight <= 0:
            raise ConfigurationError(f"{context}.weight must be positive.")
        if not 0 <= speed <= 1:
            raise ConfigurationError(f"{context}.start_speed must be in [0, 1].")
        modes[str(name)] = {
            "weight": weight,
            "start": _read_pose_region(mode.get("start", WORKSPACE_BOX), f"{context}.start"),
            "goal": _read_pose_region(mode.get("goal", WORKSPACE_BOX), f"{context}.goal"),
            "start_speed": speed,
        }
    return modes


def _read_sampling_modes(raw_config: Mapping[str, Any], modes: Mapping[str, Any]) -> list[str]:
    """The names of the modes that are drawn from: every one unless ``sampling_modes`` selects some."""
    selected = raw_config.get("sampling_modes", list(modes))
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ConfigurationError("'sampling_modes' must be a non-empty list of mode names.")
    unknown = [name for name in selected if name not in modes]
    if unknown:
        raise ConfigurationError(f"'sampling_modes' names {unknown}, which 'sampling' does not define ({list(modes)}).")
    return [str(name) for name in selected]


def validate_franka_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of the arm: limits, tolerances, the modes starts and goals are drawn from, and the goal."""
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
    fields["sampling"] = _read_sampling(raw_config)
    fields["sampling_modes"] = _read_sampling_modes(raw_config, fields["sampling"])
    if "goal" in raw_config:
        goal = read_vector(raw_config, "goal", POSE_DIM, ())
        rotation = np.array(goal[3:]).reshape(3, 3)
        if not (np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) and np.linalg.det(rotation) > 0):
            raise ConfigurationError("'goal' must be a position followed by a rotation matrix (row-major, 9 numbers).")
        fields["goal"] = list(goal)
    return fields
