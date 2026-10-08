from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

import numpy as np


DEFAULT_DONE_HOLD_STEPS = 5
DEFAULT_INITIAL_STATE_SEED = 0


@runtime_checkable
class DynamicsProtocol(Protocol):
    """Structural contract for single-robot dynamics used by planners,
    data collection and training code, without requiring inheritance from
    ``DynamicsSimulator``.

    Observation and dataset invariants
    ----------------------------------
    - ``observe()`` returns a vector of length ``obs_dim`` (may differ from ``nx``).
    - ``get_dataset_features()`` defines the canonical observation/action
      schema consumed by learners.
    - ``format_dataset_frame()`` must return a frame matching that schema
      exactly (keys and dimensions).
    """

    config: Mapping[str, Any]
    state: np.ndarray | None
    time: int
    dt: float
    nx: int
    nu: int
    obs_dim: int
    max_action: float | np.ndarray
    is_euclidean: bool
    angular_state_indices: tuple[int, ...]

    def validate_state(self, state: np.ndarray) -> np.ndarray: ...

    def validate_action(self, action: np.ndarray) -> np.ndarray: ...

    def validate_observation(self, observation: np.ndarray) -> np.ndarray: ...

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray: ...

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool: ...

    def casadi_dynamics(self, x: Any, u: Any) -> Any: ...

    def get_dataset_features(self) -> dict[str, Any]: ...

    def reset_random(self) -> np.ndarray: ...

    def reset_rollout_termination(self) -> None: ...

    def set_early_rollout_termination(self, enabled: bool) -> None: ...

    def should_terminate_rollout(self, state: np.ndarray) -> bool: ...

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]: ...

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray: ...

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None: ...

    def set_goal(self, goal: np.ndarray) -> None: ...

    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray: ...

    @property
    def goal_state(self) -> np.ndarray: ...

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
    """Base class for single-robot dynamics simulation.

    Subclasses set ``nx``, ``nu``, ``obs_dim`` and ``max_action`` and implement
    the abstract methods.
    """

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
        """Dimensionality of the goal vector accepted by ``set_goal`` (may differ from ``goal_state``)."""
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
    def is_euclidean(self) -> bool:
        """Whether squared Euclidean residuals are valid across all state coordinates."""
        return True

    @property
    def angular_state_indices(self) -> tuple[int, ...]:
        """State coordinates whose planner residuals should use wrapped angular distance."""
        return ()

    @staticmethod
    def _as_vector(values: np.ndarray, name: str, size: int | None) -> np.ndarray:
        if size is None:
            raise ValueError(f"Simulator dimension for '{name}' must be set before use.")
        arr = np.asarray(values, dtype=float)
        if arr.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {arr.shape}.")
        return arr

    def validate_state(self, state: np.ndarray) -> np.ndarray:
        return self._as_vector(state, "state", self.nx)

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        return self._as_vector(action, "action", self.nu)

    def validate_observation(self, observation: np.ndarray) -> np.ndarray:
        return self._as_vector(observation, "observation", self.obs_dim)

    @abstractmethod
    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """Return the next state without changing simulator state or time."""
        pass

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        """Get next state"""
        pass

    @abstractmethod
    def observe(self, state: np.ndarray, validate: bool = True) -> np.ndarray:
        """Get observation"""
        pass

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Must return symbolic CasADi representation of the dynamics"""
        pass

    @abstractmethod
    def get_dataset_features(self) -> dict[str, Any]:
        """Return the LeRobot features dictionary for this robot"""
        pass

    def reset_random(self) -> np.ndarray:
        """Return a random, dynamically valid initial state"""
        self.randomize_goal_for_reset(self._sampling_rng)
        return self.reset(self.random_initial_state(self._sampling_rng))

    def reset_random_state_only(self) -> np.ndarray:
        """Like reset_random(), but keeps the current goal unchanged."""
        return self.reset(self.random_initial_state(self._sampling_rng))

    @abstractmethod
    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """Package the observation and action into a frame dictionary for the dataset."""
        pass

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal"""
        pass

    @abstractmethod
    def invert_obs(self, obs: np.ndarray, validate: bool = True) -> np.ndarray:
        """Reconstruct absolute state from observation (inverse of observe())"""
        pass

    @property
    @abstractmethod
    def goal_state(self) -> np.ndarray:
        """Return the full nx-dim goal vector in state space"""
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
