from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


DEFAULT_DONE_HOLD_STEPS = 5
DEFAULT_INITIAL_STATE_SEED = 0


class VectorValidationError(ValueError):
    """Raised when a state, observation, or action vector has the wrong shape."""


def _as_vector(values: Any, name: str, size: int) -> np.ndarray:
    """Convert values into a flat float vector with strict dimensional validation."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise VectorValidationError(f"{name} must be 1D with shape ({size},), got {arr.shape}.")
    if arr.shape[0] != size:
        raise VectorValidationError(f"{name} must have length {size}, got {arr.shape[0]}.")
    return arr


def constant_jerk_step(q: Any, v: Any, a: Any, jerk: Any, dt: float) -> tuple[Any, Any, Any]:
    """The exact step of a triple integrator under constant jerk, on numpy arrays and CasADi expressions alike."""
    return (
        q + dt * v + dt**2 / 2 * a + dt**3 / 6 * jerk,
        v + dt * a + dt**2 / 2 * jerk,
        a + dt * jerk,
    )


def saturated_step(
    state: np.ndarray, jerk: np.ndarray, dt: float, lower: np.ndarray, upper: np.ndarray, max_jerk: np.ndarray
) -> np.ndarray:
    """The plant step of a triple integrator with state [q, v, a], the (nx,) ``lower``/``upper`` bounds of the state and
    the jerk limit: the jerk, then the acceleration and the velocity are saturated, and a joint stops at its position
    limit."""
    n = len(max_jerk)
    q, v, a = state[:n], state[n : 2 * n], state[2 * n :]
    a_next = np.clip(a + dt * np.clip(jerk, -max_jerk, max_jerk), lower[2 * n :], upper[2 * n :])
    v_next = np.clip(v + dt * (a + a_next) / 2, lower[n : 2 * n], upper[n : 2 * n])
    q_free = q + dt * (v + v_next) / 2 - dt**2 * (a_next - a) / 12
    q_next = np.clip(q_free, lower[:n], upper[:n])
    stopped = q_next != q_free
    return np.concatenate([q_next, np.where(stopped, 0.0, v_next), np.where(stopped, 0.0, a_next)])


@runtime_checkable
class DynamicsProtocol(Protocol):
    """Structural contract for the dynamics simulator.

    Any class with these attributes and methods is accepted by planners,
    collectors, and training code without requiring inheritance from
    DynamicsSimulator.

    Observation and dataset invariants
    ----------------------------------
    - ``observe()`` returns a vector of length ``obs_dim`` and may differ from ``nx``.
    - ``get_dataset_features()`` defines the canonical observation/action
        schema consumed by learners.
    - ``format_dataset_frame()`` must return tensors matching that schema
        exactly (keys and dimensions).

    The planner and the projector additionally read the state limits
    (``state_lower_bounds``, ``state_upper_bounds``).
    """

    config: Mapping[str, Any]
    state: np.ndarray | None
    time: int
    dt: float
    nx: int
    nu: int
    n: int
    obs_dim: int
    max_action: float | np.ndarray
    state_lower_bounds: np.ndarray
    state_upper_bounds: np.ndarray
    randomize_goal: bool
    rest_state_indices: tuple[int, ...]

    def validate_state(self, state: np.ndarray) -> np.ndarray: ...

    def validate_action(self, action: np.ndarray) -> np.ndarray: ...

    def validate_observation(self, observation: np.ndarray) -> np.ndarray: ...

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool: ...

    def casadi_dynamics(self, x: Any, u: Any) -> Any: ...

    def rollout_positions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray: ...

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]: ...

    def reset_random(self) -> np.ndarray: ...

    def reset_rollout_termination(self) -> None: ...

    def set_early_rollout_termination(self, enabled: bool) -> None: ...

    def should_terminate_rollout(self, state: np.ndarray) -> bool: ...

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]: ...

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]: ...

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray: ...

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None: ...

    def set_goal(self, goal: np.ndarray) -> None: ...

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray: ...

    @property
    def goal_dim(self) -> int: ...

    @property
    def goal(self) -> np.ndarray: ...

    def reset(self, initial_state: np.ndarray) -> np.ndarray: ...

    def is_collision(self, state: np.ndarray) -> bool: ...

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]: ...


class DynamicsSimulator(ABC):
    """Base class for dynamics simulation."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.state = None
        self.time = 0
        self.dt = config.get("dt", 0.05)
        initial_state_seed = config.get("initial_state_seed", DEFAULT_INITIAL_STATE_SEED)
        if initial_state_seed is None:
            initial_state_seed = DEFAULT_INITIAL_STATE_SEED
        self._sampling_rng = np.random.default_rng(int(initial_state_seed))
        self._rollout_done_counter = 0
        self._early_rollout_termination_enabled = True

        # Agnostic dimensions so planners can read them automatically
        self.nx = None
        self.nu = None
        self.obs_dim = getattr(self, "obs_dim", self.nx)
        self.max_action = None

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        del rng

    def set_goal(self, goal: np.ndarray) -> None:
        goal_array = np.asarray(goal, dtype=float)
        if goal_array.shape != self.goal.shape:
            raise ValueError(
                f"Goal shape mismatch: expected {self.goal.shape}, got {goal_array.shape}."
            )
        self.goal = goal_array.copy()

    @property
    def goal_dim(self) -> int:
        """Dimensionality of the goal vector accepted by ``set_goal`` ."""
        return int(self.goal.shape[0])

    def reset_rollout_termination(self) -> None:
        self._rollout_done_counter = 0

    def set_early_rollout_termination(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        if enabled_flag != self._early_rollout_termination_enabled:
            self._rollout_done_counter = 0
        self._early_rollout_termination_enabled = enabled_flag

    def should_terminate_rollout(self, state: np.ndarray) -> bool:
        if not self._early_rollout_termination_enabled:
            return False

        hold_steps = int(self.config.get("done_hold_steps", DEFAULT_DONE_HOLD_STEPS))
        if hold_steps <= 0:
            raise ValueError("'done_hold_steps' must be a positive integer.")

        if self.is_done(state):
            self._rollout_done_counter += 1
        else:
            self._rollout_done_counter = 0

        return self._rollout_done_counter >= hold_steps

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        """State indices that are zero at rest (velocities, accelerations), which the projector drives to zero at the
        end of its horizon; a system with derivative states must override this."""
        return ()

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        if self.nx is None:
            raise ValueError("Simulator state dimension 'nx' must be set before use.")
        return _as_vector(state, "state", self.nx)

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        if self.nu is None:
            raise ValueError("Simulator action dimension 'nu' must be set before use.")
        return _as_vector(action, "action", self.nu)

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        if self.obs_dim is None:
            if self.nx is None:
                raise ValueError(
                    "Simulator observation dimension cannot be validated before 'obs_dim' or 'nx' is set."
                )
            self.obs_dim = int(self.nx)
        return _as_vector(observation, "observation", int(self.obs_dim))

    @abstractmethod
    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """Return the next state without changing simulator state or time."""
        pass

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """Get next state"""
        pass

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        """The observation [goal, state]."""
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        obs = np.concatenate([self.goal, state_array])
        return self.validate_observation(obs) if validate else obs

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        """The state of the observation (the goal part is dropped)."""
        obs_array = self.validate_observation(obs) if validate else np.asarray(obs, dtype=float)
        state = obs_array[self.goal_dim :]
        return self.validate_state(state) if validate else state

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Must return symbolic CasADi representation of the dynamics"""
        pass

    @abstractmethod
    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """Return the LeRobot features dictionary for this specific robot; the action is a chunk of
        ``prediction_horizon`` actions"""
        pass

    def reset_random(self) -> np.ndarray:
        """Return a random, dynamically valid initial state"""
        self.randomize_goal_for_reset(self._sampling_rng)
        return self.reset(self.random_initial_state(self._sampling_rng))

    def reset_random_state_only(self) -> np.ndarray:
        """Like reset_random(), but keeps the current goal unchanged."""
        return self.reset(self.random_initial_state(self._sampling_rng))

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """What the policy sees of an observation [goal, state]: the two dataset observation features."""
        obs_array = self.validate_observation(obs)
        return {
            "observation.environment_state": np.asarray(obs_array[: self.goal_dim], dtype=np.float32),
            "observation.state": np.asarray(obs_array[self.goal_dim :], dtype=np.float32),
        }

    def format_dataset_frame(self, obs: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        """An observation and the expert's planned actions (prediction_horizon, nu) in the LeRobot format (without
        ``task``, which the caller adds)."""
        actions_array = np.asarray(actions, dtype=float)
        if actions_array.ndim != 2 or actions_array.shape[1] != self.nu:
            raise ValueError(f"actions must have shape (prediction_horizon, {self.nu}), got {actions_array.shape}.")
        return {**self.policy_observation(obs), "action": actions_array.reshape(-1).astype(np.float32)}

    def rollout_positions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The positions (len(actions) + 1, n) of the plant applying ``actions`` from ``state``, row 0 the current one."""
        states = [np.asarray(state, dtype=float)]
        for action in actions:
            states.append(self.predict_next_state(states[-1], action, validate=False))
        return np.array(states)[:, : self.n]

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal"""
        pass

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Reset system state and time step"""
        state = self.validate_state(initial_state)
        self.state = state.copy()
        self.time = 0
        self.reset_rollout_termination()
        return self.state

    def is_collision(self, state: np.ndarray) -> bool:
        return False

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]:
        """Simulate a trajectory"""
        states, observations, actions = [], [], []
        state = self.reset(initial_state)

        for _ in range(num_steps):
            obs = self.observe(state)
            action = policy_fn(obs)  # Call your motion planner here
            state = self.step(state, action)
            states.append(state.copy())
            observations.append(obs)
            actions.append(action)

            if self.should_terminate_rollout(state):
                break

        return {
            "states": np.array(states),
            "observations": np.array(observations),
            "actions": np.array(actions),
        }
