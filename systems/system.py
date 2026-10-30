from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


DEFAULT_DONE_HOLD_STEPS = 5


class SystemSimulator(ABC):
    """Abstract base class for systems used by planners, rollouts, and imitation learning."""

    task_description = "reach target"

    @classmethod
    @abstractmethod
    def config_keys(cls) -> set[str]:
        """The keys a config of this system may set."""

    @classmethod
    @abstractmethod
    def validate_config(cls, raw_config: Mapping[str, Any]) -> dict[str, Any]:
        """The normalized fields of the system in a raw config (limits, tolerances, sampling, obstacles, goal); raises
        ``ConfigurationError`` for what is missing or malformed."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.dt = float(config.get("dt", 0.05))

    # --- dimensions, limits and goal: what a concrete system must provide -------

    @property
    @abstractmethod
    def n(self) -> int:
        """Number of positions (joints or axes) of the state."""

    @property
    @abstractmethod
    def nx(self) -> int:
        """Dimension of the state."""

    @property
    @abstractmethod
    def nu(self) -> int:
        """Dimension of the action."""

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Dimension of the observation ``[goal, state]`` the planners read."""

    @property
    @abstractmethod
    def max_action(self) -> np.ndarray:
        """The largest magnitude of every action component, ``(nu,)``."""

    @property
    @abstractmethod
    def state_lower_bounds(self) -> np.ndarray:
        """The lower limit of every state component, ``(nx,)`` (``-inf`` where there is none)."""

    @property
    @abstractmethod
    def state_upper_bounds(self) -> np.ndarray:
        """The upper limit of every state component, ``(nx,)`` (``inf`` where there is none)."""

    @property
    @abstractmethod
    def goal(self) -> np.ndarray:
        """The current goal vector."""

    @goal.setter
    @abstractmethod
    def goal(self, value: np.ndarray) -> None:
        """Replace the goal (``set_goal`` checks its shape first)."""

    @property
    @abstractmethod
    def randomize_goal(self) -> bool:
        """Whether ``randomize_goal_for_reset`` draws a new goal."""

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
        """Dimensionality of the goal vector accepted by ``set_goal``."""
        return int(self.goal.shape[0])

    @property
    def environment_dim(self) -> int:
        """Dimensionality of the observation's part before the state: the goal, and what else the system observes."""
        return self.obs_dim - self.nx

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        """State indices that are zero at rest (velocities, accelerations), which the projector drives to zero at the
        end of its horizon; a system with derivative states must override this."""
        return ()

    def collides(self, state: np.ndarray) -> bool:
        """Ground truth: whether the system in ``state`` touches an obstacle (a system with no obstacles never does).
        What evaluation counts as a collision, as opposed to the bounding spheres the planners keep clear."""
        return False

    @property
    def done_hold_steps(self) -> int:
        """How many consecutive steps ``is_done`` has to hold for a rollout to end (``RolloutTerminationTracker``)."""
        hold_steps = int(self.config.get("done_hold_steps", DEFAULT_DONE_HOLD_STEPS))
        if hold_steps <= 0:
            raise ValueError("'done_hold_steps' must be a positive integer.")
        return hold_steps

    @abstractmethod
    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Return the next state without changing simulator state or time."""

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Advance the plant by one control period."""
        return self.predict_next_state(state, action)

    @abstractmethod
    def is_done(self, state: np.ndarray) -> bool:
        """Return whether ``state`` satisfies the goal and rest tolerances."""

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Return the symbolic CasADi discrete-time step expression."""

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal."""

    def observe(self, state: np.ndarray) -> np.ndarray:
        """The observation ``[goal, state]``; the goal comes first, and a system can put more before the state."""
        return np.concatenate([self.goal, np.asarray(state, dtype=float)])

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        """The state of the observation (the part before it is dropped)."""
        return np.asarray(obs, dtype=float)[self.obs_dim - self.nx :]

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """What the policy sees of an observation: the part before the state and the state, the two dataset features."""
        obs_array = np.asarray(obs, dtype=np.float32)
        return {
            "observation.environment_state": obs_array[: self.environment_dim],
            "observation.state": obs_array[self.environment_dim :],
        }

    @property
    @abstractmethod
    def environment_state_names(self) -> list[str]:
        """Names of the components of ``policy_observation``'s ``observation.environment_state`` (dataset metadata)."""

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """Names of the components of ``policy_observation``'s ``observation.state`` (dataset metadata)."""

    @property
    @abstractmethod
    def action_names(self) -> list[str]:
        """Names of the ``nu`` action components (dataset metadata)."""

    def state_from_policy(self, state: np.ndarray) -> np.ndarray:
        """The plant state of the ``observation.state`` the policy sees (``nx`` entries): the state itself unless the
        policy cannot see all of it."""
        return np.asarray(state, dtype=float)

    def observation_from_policy(self, environment_state: np.ndarray, state: np.ndarray) -> np.ndarray:
        """An observation ``[goal, state]`` that gives the policy observation (the inverse of ``policy_observation``,
        up to what the policy cannot see); here the policy sees the observation itself."""
        return np.concatenate([np.asarray(environment_state, dtype=float), np.asarray(state, dtype=float)])

    def rollout_positions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The positions ``(len(actions) + 1, n)`` of the plant applying ``actions`` from ``state``, row 0 the current one."""
        states = [np.asarray(state, dtype=float)]
        for action in actions:
            states.append(self.predict_next_state(states[-1], action))
        return np.array(states)[:, : self.n]

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Start a rollout: a copy of ``initial_state``."""
        return np.asarray(initial_state, dtype=float).copy()
