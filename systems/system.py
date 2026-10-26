from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import numpy as np


DEFAULT_DONE_HOLD_STEPS = 5
DEFAULT_INITIAL_STATE_SEED = 0


class SystemSimulator(ABC):
    """Abstract base class for systems used by planners, rollouts, and imitation learning."""

    task_description = "reach target"

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.state: np.ndarray | None = None
        self.time = 0
        self.dt = float(config.get("dt", 0.05))
        initial_state_seed = config.get("initial_state_seed", DEFAULT_INITIAL_STATE_SEED)
        if initial_state_seed is None:
            initial_state_seed = DEFAULT_INITIAL_STATE_SEED
        self._sampling_rng = np.random.default_rng(int(initial_state_seed))
        self._rollout_done_counter = 0
        self._early_rollout_termination_enabled = True

        self.n: int = 0
        self.nx: int = 0
        self.nu: int = 0
        self.obs_dim: int = 0
        self.max_action: np.ndarray = np.empty(0)
        self.state_lower_bounds: np.ndarray = np.empty(0)
        self.state_upper_bounds: np.ndarray = np.empty(0)
        self.goal: np.ndarray = np.empty(0)
        self.randomize_goal: bool = False

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
    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """Return the LeRobot features dictionary for this system; the action is a chunk of
        ``prediction_horizon`` actions."""

    @abstractmethod
    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Return a random initial state without changing the goal."""

    def observe(self, state: np.ndarray) -> np.ndarray:
        """The observation ``[goal, state]``; the goal comes first, and a system can put more before the state."""
        return np.concatenate([self.goal, np.asarray(state, dtype=float)])

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        """The state of the observation (the part before it is dropped)."""
        return np.asarray(obs, dtype=float)[self.obs_dim - self.nx :]

    def reset_random(self) -> np.ndarray:
        """Randomize goal (if enabled) and reset to a random valid initial state."""
        self.randomize_goal_for_reset(self._sampling_rng)
        return self.reset(self.random_initial_state(self._sampling_rng))

    def reset_random_state_only(self) -> np.ndarray:
        """Like ``reset_random()``, but keeps the current goal unchanged."""
        return self.reset(self.random_initial_state(self._sampling_rng))

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """What the policy sees of an observation: the part before the state and the state, the two dataset features."""
        obs_array = np.asarray(obs, dtype=np.float32)
        return {
            "observation.environment_state": obs_array[: self.environment_dim],
            "observation.state": obs_array[self.environment_dim :],
        }

    def observation_from_policy(self, environment_state: np.ndarray, state: np.ndarray) -> np.ndarray:
        """An observation ``[goal, state]`` that gives the policy observation (the inverse of ``policy_observation``,
        up to what the policy cannot see); here the policy sees the observation itself."""
        return np.concatenate([np.asarray(environment_state, dtype=float), np.asarray(state, dtype=float)])

    def format_dataset_frame(self, obs: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
        """An observation and the expert's planned actions ``(prediction_horizon, nu)`` in the LeRobot format (without
        ``task``, which the caller adds)."""
        actions_array = np.asarray(actions, dtype=float)
        if actions_array.ndim != 2 or actions_array.shape[1] != self.nu:
            raise ValueError(f"actions must have shape (prediction_horizon, {self.nu}), got {actions_array.shape}.")
        return {**self.policy_observation(obs), "action": actions_array.reshape(-1).astype(np.float32)}

    def rollout_positions(self, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The positions ``(len(actions) + 1, n)`` of the plant applying ``actions`` from ``state``, row 0 the current one."""
        states = [np.asarray(state, dtype=float)]
        for action in actions:
            states.append(self.predict_next_state(states[-1], action))
        return np.array(states)[:, : self.n]

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Reset system state and time step."""
        self.state = np.asarray(initial_state, dtype=float).copy()
        self.time = 0
        self.reset_rollout_termination()
        return self.state

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]:
        """Simulate a trajectory."""
        states, observations, actions = [], [], []
        state = self.reset(initial_state)

        for _ in range(num_steps):
            obs = self.observe(state)
            action = policy_fn(obs)
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
