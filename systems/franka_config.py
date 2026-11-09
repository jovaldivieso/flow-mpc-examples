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
)

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7
POSE_COMPONENTS = ["x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)]
POSE_DIM = len(POSE_COMPONENTS)


def validate_franka_config(raw_config: Mapping[str, Any]) -> dict[str, Any]:
    """The fields of the arm: limits, tolerances, obstacles and the goal."""
    min_start_goal_distance = read_float(raw_config, "min_start_goal_distance", 0.25)
    if min_start_goal_distance < 0:
        raise ConfigurationError("'min_start_goal_distance' must be non-negative.")

    fields: dict[str, Any] = {
        "randomize_goal": read_bool(raw_config, "randomize_goal", "goal" not in raw_config),
        "min_start_goal_distance": min_start_goal_distance,
        "max_acceleration": list(read_limits(raw_config, "max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION, FRANKA_NUM_JOINTS)),
        "max_jerk": list(read_limits(raw_config, "max_jerk", FRANKA_DEFAULT_MAX_JERK, FRANKA_NUM_JOINTS)),
        "obstacles": read_obstacles(raw_config, 3),
        **read_tolerances(raw_config, 0.03, 0.30, 0.05),
    }
    if "goal" in raw_config:
        goal = read_vector(raw_config, "goal", POSE_DIM, ())
        rotation = np.array(goal[3:]).reshape(3, 3)
        if not (np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) and np.linalg.det(rotation) > 0):
            raise ConfigurationError("'goal' must be a position followed by a rotation matrix (row-major, 9 numbers).")
        fields["goal"] = list(goal)
    return fields
