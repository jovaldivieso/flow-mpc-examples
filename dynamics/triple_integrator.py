"""Point mass in ``dimension`` dimensions (default 2) with jerk as the action: a toy system to calibrate the pipeline.

State  x = [p, v, a]  (position, velocity, acceleration), nx = 3 * dimension
Action u = j          (jerk), nu = dimension
Goal   g = target position
Observation o = [g, x]; the LeRobot frame stores g as ``observation.environment_state`` and x as ``observation.state``.

The same plant as the arm's joints (``constant_jerk_step`` and ``saturated_step``): every axis is a triple integrator
with piecewise constant jerk, limited in velocity, acceleration and jerk, with a box on the position. There is no
collision.
"""

from typing import Any, Mapping

import casadi as ca
import numpy as np

from .dynamics import DynamicsSimulator, constant_jerk_step, saturated_step

DEFAULT_DIMENSION = 2
DEFAULT_WORKSPACE_BOUNDS = (-1.0, 1.0)  # [min, max] of every axis, where starts and goals are drawn
DEFAULT_POSITION_LIMIT = 1.5  # the plant stops at +-this
DEFAULT_MAX_VELOCITY = 1.0
DEFAULT_MAX_ACCELERATION = 2.0
DEFAULT_MAX_JERK = 10.0
SAMPLING_ATTEMPTS = 256


class TripleIntegrator(DynamicsSimulator):
    """Triple-integrator point mass; the goal is a position."""

    task_description = "reach target position"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.n = int(config.get("dimension", DEFAULT_DIMENSION))
        self.nx, self.nu = 3 * self.n, self.n
        self.obs_dim = self.n + self.nx

        per_axis = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (self.n,))
        position_limit = per_axis("position_limit", DEFAULT_POSITION_LIMIT)
        max_velocity = per_axis("max_velocity", DEFAULT_MAX_VELOCITY)
        max_acceleration = per_axis("max_acceleration", DEFAULT_MAX_ACCELERATION)
        self.max_action = per_axis("max_jerk", DEFAULT_MAX_JERK).copy()
        self.state_lower_bounds = -np.concatenate([position_limit, max_velocity, max_acceleration])
        self.state_upper_bounds = np.concatenate([position_limit, max_velocity, max_acceleration])

        self.workspace_bounds = tuple(float(value) for value in config.get("workspace_bounds", DEFAULT_WORKSPACE_BOUNDS))
        self.goal = np.asarray(config["goal"], dtype=float) if "goal" in config else np.zeros(self.n)
        self.randomize_goal = config.get("randomize_goal", "goal" not in config)
        self.min_start_goal_distance = float(config.get("min_start_goal_distance", 0.5))
        self.pos_tol = float(config.get("pos_tol", 0.02))
        self.vel_tol = float(config.get("vel_tol", 0.02))

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.nx))  # velocity and acceleration

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        return ca.vertcat(*constant_jerk_step(x[: self.n], x[self.n : 2 * self.n], x[2 * self.n :], u, self.dt))

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        return saturated_step(
            state_array, action_array, self.dt, self.state_lower_bounds, self.state_upper_bounds, self.max_action
        )

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        return self.predict_next_state(state, action, validate=validate)

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """Position error of ``state`` with respect to the goal, and the orientation error (none: 0)."""
        return float(np.linalg.norm(np.asarray(state, dtype=float)[: self.n] - self.goal)), 0.0

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        """At the goal (within ``pos_tol``) and at rest (every velocity within ``vel_tol``)."""
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        return bool(
            self.goal_errors(state_array)[0] < self.pos_tol and np.abs(state_array[self.n : 2 * self.n]).max() < self.vel_tol
        )

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position the viewers draw for the position ``q``: padded with zeros (or cut) to three axes."""
        return np.pad(np.asarray(q, dtype=float)[:3], (0, max(0, 3 - self.n)))

    # --- dataset and sampling --------------------------------------------------

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """LeRobot features: state [p, v, a], the goal position as environment state, the expert's planned jerks
        as action (its next ``prediction_horizon`` jerks, flattened row by row)."""
        axes = [f"p{i}" for i in range(self.n)]
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (self.n,),
                "names": [f"goal.{axis}" for axis in axes],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.nx,),
                "names": [f"{axis}.{kind}" for kind in ("pos", "vel", "acc") for axis in axes],
            },
            "action": {
                "dtype": "float32",
                "shape": (prediction_horizon * self.nu,),
                "names": [f"{axis}.jerk[t+{k}]" for k in range(prediction_horizon) for axis in axes],
            },
        }

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Rest state at a random position at least ``min_start_goal_distance`` from the goal."""
        for _ in range(SAMPLING_ATTEMPTS):
            position = rng.uniform(*self.workspace_bounds, size=self.n)
            if np.linalg.norm(position - self.goal) > self.min_start_goal_distance:
                return np.concatenate([position, np.zeros(2 * self.n)])
        raise RuntimeError(
            f"Unable to sample a start {self.min_start_goal_distance} away from the goal in {SAMPLING_ATTEMPTS} attempts."
        )

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            self.set_goal(rng.uniform(*self.workspace_bounds, size=self.n))
