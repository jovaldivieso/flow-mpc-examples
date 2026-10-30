"""``TripleIntegratorBase``: the abstract core of the triple-integrator systems (``FrankaSystem`` and ``PointMassSystem``).

State  x = [q, v, a]  (positions, velocities, accelerations), nx = 3 * n
Action u = j          (jerk), nu = n

Every axis (or joint) is a triple integrator with piecewise constant jerk, limited in velocity, acceleration, and jerk
(and, for a system that sets them, in position). The base class holds the exact plant step (numeric and CasADi), the
limits, the goal and tolerances, the ``SphereCollisionModel`` of its obstacles and bounding spheres (``collision``), and the
start sampling; a concrete system supplies its goal space, its bounding spheres, and what its policy observes.
"""

from abc import abstractmethod
from typing import Any, Mapping

import casadi as ca
import numpy as np

from .sphere_collision import SphereCollisionModel
from .system import SystemSimulator

SAMPLING_ATTEMPTS = 256


class TripleIntegratorBase(SystemSimulator):
    """Abstract triple-integrator system with state ``[q, v, a]`` and jerk action ``u = j``."""

    def __init__(
        self,
        config: Mapping[str, Any],
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
        """Dimensions, limits, goal, tolerances and obstacles. The position limits may be infinite; ``path_lower`` and
        ``path_upper`` are the finite box path planners search in (default: the position limits)."""
        super().__init__(config)
        self._n = n
        self._nx = 3 * n
        self.max_acceleration = np.asarray(max_acceleration, dtype=float).copy()
        self._max_action = np.asarray(max_jerk, dtype=float).copy()
        self._state_lower_bounds = np.concatenate([lower_position, -max_velocity, -self.max_acceleration])
        self._state_upper_bounds = np.concatenate([upper_position, max_velocity, self.max_acceleration])
        self._path_lower = np.asarray(lower_position if path_lower is None else path_lower, dtype=float).copy()
        self._path_upper = np.asarray(upper_position if path_upper is None else path_upper, dtype=float).copy()

        self._goal = np.asarray(config["goal"], dtype=float) if "goal" in config else np.asarray(default_goal, dtype=float).copy()
        self._randomize_goal = bool(config.get("randomize_goal", "goal" not in config))
        self.min_start_goal_distance = float(config.get("min_start_goal_distance", default_min_distance))
        self.pos_tol = float(config.get("pos_tol", default_pos_tol))
        self.vel_tol = float(config.get("vel_tol", default_vel_tol))
        obstacles = [(np.asarray(obs["position"], dtype=float).copy(), float(obs["radius"])) for obs in config.get("obstacles", []) or []]
        self.collision = SphereCollisionModel(obstacles, self.collision_spheres, n)

    # --- the properties the base class declares ---------------------------------

    @property
    def n(self) -> int:
        return self._n

    @property
    def nx(self) -> int:
        return self._nx

    @property
    def nu(self) -> int:
        return self._n

    @property
    def obs_dim(self) -> int:
        return self.goal_dim + self._nx

    @property
    def max_action(self) -> np.ndarray:
        return self._max_action

    @property
    def state_lower_bounds(self) -> np.ndarray:
        return self._state_lower_bounds

    @property
    def state_upper_bounds(self) -> np.ndarray:
        return self._state_upper_bounds

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value: np.ndarray) -> None:
        self._goal = np.asarray(value, dtype=float).copy()

    @property
    def randomize_goal(self) -> bool:
        return self._randomize_goal

    @property
    def path_lower(self) -> np.ndarray:
        return self._path_lower

    @property
    def path_upper(self) -> np.ndarray:
        return self._path_upper

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.nx))  # velocity and acceleration

    # --- the plant --------------------------------------------------------------

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

    # --- goal and kinematics: what a concrete system supplies -------------------

    @abstractmethod
    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """Position error of ``state`` with respect to the goal, and its orientation error (0 where there is none)."""

    @abstractmethod
    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (S, d), position Jacobians (S, d, n), and radii (S,) at configuration ``q``."""

    @abstractmethod
    def goal_configuration(self, goal: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
        """The configuration the planners drive to for ``goal``, the nearest one to ``q_reference`` if there are several.
        Raises ``RuntimeError`` if none reaches the goal."""

    @abstractmethod
    def goal_task_error(self, q: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """What moves the configuration ``q`` toward ``goal``: the task-space error ``e`` (m,) (the position error, then
        the orientation error where there is one), the Jacobian ``J`` (m, n) with ``dx = J dq``, and the position error
        norm (the first part of ``e``)."""

    @abstractmethod
    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position the viewers draw for the configuration ``q``."""

    @abstractmethod
    def goal_transform(self, goal: np.ndarray) -> np.ndarray:
        """The 4x4 homogeneous transform the viewers place the goal marker at."""

    @property
    @abstractmethod
    def coordinate_names(self) -> list[str]:
        """Names of the ``n`` coordinates (of the dataset feature metadata)."""

    @abstractmethod
    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """A random collision-free configuration to start from or to take a goal from."""

    @abstractmethod
    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """A random reachable goal."""

    def is_done(self, state: np.ndarray) -> bool:
        """At the goal (within ``pos_tol``) and at rest (every velocity within ``vel_tol``)."""
        state_array = np.asarray(state, dtype=float)
        return bool(
            self.goal_errors(state_array)[0] < self.pos_tol and np.abs(state_array[self.n : 2 * self.n]).max() < self.vel_tol
        )

    @property
    def action_names(self) -> list[str]:
        return [f"{name}.jerk" for name in self.coordinate_names]

    # --- sampling ---------------------------------------------------------------

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
