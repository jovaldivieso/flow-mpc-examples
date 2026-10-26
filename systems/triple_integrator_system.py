"""Point mass in ``dimension`` dimensions (default 2) with jerk as the action: a toy system to calibrate the pipeline,
and the base class for any kinematic triple-integrator system (such as ``FrankaSystem``).

State  x = [p, v, a]  (position, velocity, acceleration), nx = 3 * dimension
Action u = j          (jerk), nu = dimension
Goal   g = target position
Observation o = [g, x], what the planners read. The point mass has no position limits, so it is translation-invariant and
the policy sees nothing in world coordinates (``policy_observation``): the goal relative to the point, ``g - p``, as
``observation.environment_state`` and the velocity and acceleration as ``observation.state``.

Every axis (or joint) is a triple integrator with piecewise constant jerk, limited in velocity, acceleration, and jerk
(and, for a system that sets them, in position).
"""

from typing import Any, Mapping

import casadi as ca
import numpy as np

from .system import SystemSimulator

DEFAULT_DIMENSION = 2
DEFAULT_WORKSPACE_BOUNDS = (-1.0, 1.0)  # [min, max] of every axis, where starts and goals are drawn
DEFAULT_MAX_VELOCITY = 1.0
DEFAULT_MAX_ACCELERATION = 2.0
DEFAULT_MAX_JERK = 10.0
SAMPLING_ATTEMPTS = 256


class TripleIntegratorSystem(SystemSimulator):
    """Triple-integrator system with state ``[q, v, a]`` and jerk action ``u = j``."""

    task_description = "reach target position"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        n = int(config.get("dimension", DEFAULT_DIMENSION))
        per_axis = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (n,)).copy()
        max_velocity = per_axis("max_velocity", DEFAULT_MAX_VELOCITY)
        max_acceleration = per_axis("max_acceleration", DEFAULT_MAX_ACCELERATION)
        max_jerk = per_axis("max_jerk", DEFAULT_MAX_JERK)
        self.workspace_bounds = tuple(float(value) for value in config.get("workspace_bounds", DEFAULT_WORKSPACE_BOUNDS))
        self._init_triple_integrator(
            n=n,
            lower_position=np.full(n, -np.inf),
            upper_position=np.full(n, np.inf),
            path_lower=np.full(n, self.workspace_bounds[0]),
            path_upper=np.full(n, self.workspace_bounds[1]),
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            max_jerk=max_jerk,
            default_goal=np.zeros(n),
            default_min_distance=0.5,
            default_pos_tol=0.02,
            default_vel_tol=0.02,
        )

    def _init_triple_integrator(
        self,
        *,
        n: int,
        lower_position: np.ndarray,
        upper_position: np.ndarray,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        max_jerk: np.ndarray,
        default_goal: np.ndarray,
        default_min_distance: float,
        default_pos_tol: float,
        default_vel_tol: float,
        path_lower: np.ndarray | None = None,
        path_upper: np.ndarray | None = None,
    ) -> None:
        """Initialize dimensions, bounds, goal, and tolerances shared by all triple-integrator systems. The position
        limits may be infinite; ``path_lower`` and ``path_upper`` are the finite box path planners search in (default:
        the position limits)."""
        self.n = n
        self.nx = 3 * n
        self.nu = n
        self.obs_dim = n + self.nx
        self.max_acceleration = np.asarray(max_acceleration, dtype=float).copy()
        self.max_action = np.asarray(max_jerk, dtype=float).copy()
        self.state_lower_bounds = np.concatenate([lower_position, -max_velocity, -self.max_acceleration])
        self.state_upper_bounds = np.concatenate([upper_position, max_velocity, self.max_acceleration])
        self.path_lower = np.asarray(lower_position if path_lower is None else path_lower, dtype=float).copy()
        self.path_upper = np.asarray(upper_position if path_upper is None else path_upper, dtype=float).copy()

        self.goal = np.asarray(self.config["goal"], dtype=float) if "goal" in self.config else default_goal.copy()
        self.randomize_goal = bool(self.config.get("randomize_goal", "goal" not in self.config))
        self.min_start_goal_distance = float(self.config.get("min_start_goal_distance", default_min_distance))
        self.pos_tol = float(self.config.get("pos_tol", default_pos_tol))
        self.vel_tol = float(self.config.get("vel_tol", default_vel_tol))

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.nx))  # velocity and acceleration

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Exact step of a triple integrator under constant jerk, matching ``predict_next_state`` away from limits."""
        q, v, a = x[: self.n], x[self.n : 2 * self.n], x[2 * self.n :]
        dt = self.dt
        return ca.vertcat(
            q + dt * v + dt**2 / 2 * a + dt**3 / 6 * u,
            v + dt * a + dt**2 / 2 * u,
            a + dt * u,
        )

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """The plant step of a triple integrator with state ``[q, v, a]``: the jerk, then the acceleration and the
        velocity are saturated, and each coordinate stops at its position limit."""
        state_array = np.asarray(state, dtype=float)
        jerk = np.asarray(action, dtype=float)
        n, dt = self.n, self.dt
        lower, upper = self.state_lower_bounds, self.state_upper_bounds
        q, v, a = state_array[:n], state_array[n : 2 * n], state_array[2 * n :]
        a_next = np.clip(a + dt * np.clip(jerk, -self.max_action, self.max_action), lower[2 * n :], upper[2 * n :])
        v_next = np.clip(v + dt * (a + a_next) / 2, lower[n : 2 * n], upper[n : 2 * n])
        q_free = q + dt * (v + v_next) / 2 - dt**2 * (a_next - a) / 12
        q_next = np.clip(q_free, lower[:n], upper[:n])
        stopped = q_next != q_free
        return np.concatenate([q_next, np.where(stopped, 0.0, v_next), np.where(stopped, 0.0, a_next)])

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """Position error of ``state`` with respect to the goal, and the orientation error (none: 0)."""
        return float(np.linalg.norm(np.asarray(state, dtype=float)[: self.n] - self.goal)), 0.0

    def is_done(self, state: np.ndarray) -> bool:
        """At the goal (within ``pos_tol``) and at rest (every velocity within ``vel_tol``)."""
        state_array = np.asarray(state, dtype=float)
        return bool(
            self.goal_errors(state_array)[0] < self.pos_tol and np.abs(state_array[self.n : 2 * self.n]).max() < self.vel_tol
        )

    def goal_configuration(self, goal: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
        """The configuration the planners drive to for ``goal``, the nearest one to ``q_reference`` if there are several
        (here the goal is a configuration itself)."""
        return np.asarray(goal, dtype=float).copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position the viewers draw for the position ``q``: padded with zeros (or cut) to three axes."""
        return np.pad(np.asarray(q, dtype=float)[:3], (0, max(0, 3 - self.n)))

    def goal_transform(self, goal: np.ndarray) -> np.ndarray:
        """The 4x4 homogeneous transform the viewers place the goal marker at."""
        transform = np.eye(4)
        transform[:3, 3] = self.tcp_position(goal)
        return transform

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """What the policy sees: the goal relative to the point, ``g - p``, and the velocity and acceleration. Nothing
        is in world coordinates, which the unbounded dynamics make irrelevant."""
        obs_array = np.asarray(obs, dtype=np.float32)
        goal, state = obs_array[: self.n], obs_array[self.n :]
        return {"observation.environment_state": goal - state[: self.n], "observation.state": state[self.n :]}

    def observation_from_policy(self, environment_state: np.ndarray, state: np.ndarray) -> np.ndarray:
        """An observation ``[goal, x]`` that gives this policy observation, with the point at the origin: where it is
        does not matter to the dynamics."""
        return np.concatenate([np.asarray(environment_state, dtype=float), np.zeros(self.n), np.asarray(state, dtype=float)])

    # --- dataset and sampling (template methods) -----------------------------

    def _coordinate_names(self) -> list[str]:
        """Names of the ``n`` coordinates for dataset feature metadata."""
        return [f"p{i}" for i in range(self.n)]

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """LeRobot features: state ``[q, v, a]``, the goal position/configuration as environment state, and the
        expert's next ``prediction_horizon`` planned jerks flattened row by row."""
        names = self._coordinate_names()
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (self.n,),
                "names": [f"goal_rel.{name}" for name in names],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2 * self.n,),
                "names": [f"{name}.{kind}" for kind in ("vel", "acc") for name in names],
            },
            "action": {
                "dtype": "float32",
                "shape": (prediction_horizon * self.nu,),
                "names": [f"{name}.jerk[t+{k}]" for k in range(prediction_horizon) for name in names],
            },
        }

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a random position/configuration inside the workspace or joint bounds."""
        return rng.uniform(*self.workspace_bounds, size=self.n)

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        return self._sample_configuration(rng)

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Rest state at a random configuration at least ``min_start_goal_distance`` (position error) from the goal."""
        for _ in range(SAMPLING_ATTEMPTS):
            q = self._sample_configuration(rng)
            if self.goal_errors(q)[0] > self.min_start_goal_distance:
                return np.concatenate([q, np.zeros(2 * self.n)])
        raise RuntimeError(
            f"Unable to sample a start {self.min_start_goal_distance} away from the goal in {SAMPLING_ATTEMPTS} attempts."
        )

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        if self.randomize_goal:
            self.set_goal(self._sample_goal(rng))
