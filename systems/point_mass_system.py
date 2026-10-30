"""``PointMassSystem``: a point mass in ``dimension`` dimensions (default 2) with jerk as the action, a toy system to
calibrate the pipeline.

Goal   g = target position
Observation o = [g, x], what the planners read. The point mass has no position limits, so it is translation-invariant and
the policy sees nothing in world coordinates (``policy_observation``): the goal relative to the point, ``g - p``, as
``observation.environment_state`` and the velocity and acceleration as ``observation.state``.
"""

from typing import Any, Mapping

import numpy as np

from .config_fields import (
    SYSTEM_CONFIG_KEYS,
    ConfigurationError,
    read_bool,
    read_float,
    read_int,
    read_limits,
    read_obstacles,
    read_tolerances,
    read_vector,
)
from .triple_integrator import SAMPLING_ATTEMPTS, PlantLimits, TripleIntegratorBase

DEFAULT_DIMENSION = 2
DEFAULT_WORKSPACE_BOUNDS = (-1.0, 1.0)  # [min, max] of every axis, where starts and goals are drawn
DEFAULT_MAX_VELOCITY = 1.0
DEFAULT_MAX_ACCELERATION = 2.0
DEFAULT_MAX_JERK = 10.0

# The keys of a point mass config beside the ones every system has
POINT_MASS_CONFIG_KEYS = {"dimension", "max_velocity", "workspace_bounds"}


class PointMassSystem(TripleIntegratorBase):
    """Point mass in ``dimension`` dimensions (default 2) with jerk as the action: a toy system to calibrate the pipeline."""

    task_description = "reach target position"

    @classmethod
    def config_keys(cls) -> set[str]:
        return SYSTEM_CONFIG_KEYS | POINT_MASS_CONFIG_KEYS

    @classmethod
    def validate_config(cls, raw_config: Mapping[str, Any]) -> dict[str, Any]:
        """The fields of the point mass: dimension, limits, tolerances, sampling, obstacles and goal."""
        dimension = read_int(raw_config, "dimension", DEFAULT_DIMENSION)
        if dimension < 1:
            raise ConfigurationError("'dimension' must be positive.")
        bounds = read_vector(raw_config, "workspace_bounds", 2, DEFAULT_WORKSPACE_BOUNDS)
        if bounds[0] >= bounds[1]:
            raise ConfigurationError("'workspace_bounds' minimum must be smaller than its maximum.")
        min_start_goal_distance = read_float(raw_config, "min_start_goal_distance", 0.5)
        if min_start_goal_distance < 0:
            raise ConfigurationError("'min_start_goal_distance' must be non-negative.")
        fields: dict[str, Any] = {
            "dimension": dimension,
            "workspace_bounds": list(bounds),
            "randomize_goal": read_bool(raw_config, "randomize_goal", "goal" not in raw_config),
            "min_start_goal_distance": min_start_goal_distance,
            "max_velocity": list(read_limits(raw_config, "max_velocity", DEFAULT_MAX_VELOCITY, dimension)),
            "max_acceleration": list(read_limits(raw_config, "max_acceleration", DEFAULT_MAX_ACCELERATION, dimension)),
            "max_jerk": list(read_limits(raw_config, "max_jerk", DEFAULT_MAX_JERK, dimension)),
            "obstacles": read_obstacles(raw_config, dimension),
            **read_tolerances(raw_config, 0.02, 0.02),
        }
        if "goal" in raw_config:
            fields["goal"] = list(read_vector(raw_config, "goal", dimension, ()))
        return fields

    def __init__(self, config: Mapping[str, Any]):
        fields = self.validate_config(config)
        n = fields["dimension"]
        self.workspace_bounds = tuple(fields["workspace_bounds"])
        super().__init__(
            config,
            fields,
            limits=PlantLimits(
                n=n,
                lower_position=np.full(n, -np.inf),
                upper_position=np.full(n, np.inf),
                max_velocity=np.asarray(fields["max_velocity"], dtype=float),
                max_acceleration=np.asarray(fields["max_acceleration"], dtype=float),
                max_jerk=np.asarray(fields["max_jerk"], dtype=float),
                path_lower=np.full(n, self.workspace_bounds[0]),
                path_upper=np.full(n, self.workspace_bounds[1]),
            ),
            default_goal=np.zeros(n),
        )

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """Position error of ``state`` with respect to the goal, and the orientation error (none: 0)."""
        return float(np.linalg.norm(np.asarray(state, dtype=float)[: self.n] - self.goal)), 0.0

    def goal_task_error(self, q: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """The position error to the goal, with the identity as Jacobian."""
        error = np.asarray(goal, dtype=float) - np.asarray(q, dtype=float)[: self.n]
        return error, np.eye(self.n), float(np.linalg.norm(error))

    def goal_configuration(self, goal: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
        """The configuration the planners drive to for ``goal`` (here the goal is a configuration itself)."""
        return np.asarray(goal, dtype=float).copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position the viewers draw for the position ``q``: padded with zeros (or cut) to three axes."""
        return np.pad(np.asarray(q, dtype=float)[:3], (0, max(0, 3 - self.n)))

    def goal_transform(self, goal: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, 3] = self.tcp_position(goal)
        return transform

    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The point mass is one sphere of radius 0 at its position."""
        q_pos = np.asarray(q, dtype=float)[: self.n]
        return q_pos[None, :], np.eye(self.n)[None, :, :], np.zeros(1)

    def collides(self, state: np.ndarray) -> bool:
        """The point mass is its own bounding sphere: it collides when that sphere penetrates an obstacle."""
        return not self.collision.is_collision_free(np.asarray(state, dtype=float)[: self.n])

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """What the policy sees: the goal relative to the point, ``g - p``, and the velocity and acceleration. Nothing
        is in world coordinates, which the unbounded dynamics make irrelevant."""
        obs_array = np.asarray(obs, dtype=np.float32)
        goal, state = obs_array[: self.n], obs_array[self.n :]
        return {"observation.environment_state": goal - state[: self.n], "observation.state": state[self.n :]}

    def state_from_policy(self, state: np.ndarray) -> np.ndarray:
        """The plant state of the velocity and acceleration the policy sees, with the point at the origin: where it is
        does not matter to the dynamics."""
        return np.concatenate([np.zeros(self.n), np.asarray(state, dtype=float)])

    def observation_from_policy(self, environment_state: np.ndarray, state: np.ndarray) -> np.ndarray:
        """An observation ``[goal, x]`` that gives this policy observation, with the point at the origin."""
        return np.concatenate([np.asarray(environment_state, dtype=float), self.state_from_policy(state)])

    @property
    def coordinate_names(self) -> list[str]:
        return [f"p{i}" for i in range(self.n)]

    @property
    def environment_state_names(self) -> list[str]:
        return [f"goal_rel.{name}" for name in self.coordinate_names]

    @property
    def state_names(self) -> list[str]:
        return [f"{name}.{kind}" for kind in ("vel", "acc") for name in self.coordinate_names]

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random collision-free position inside the workspace."""
        for _ in range(SAMPLING_ATTEMPTS):
            q = rng.uniform(*self.workspace_bounds, size=self.n)
            if self.collision.is_collision_free(q):
                return q
        raise RuntimeError(f"Unable to sample a collision-free configuration in {SAMPLING_ATTEMPTS} attempts.")

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        return self._sample_configuration(rng)
