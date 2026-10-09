from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

import numpy as np


class DynamicsSimulator(ABC):
    """Single-robot dynamics: numeric simulation, CasADi model and dataset schema.

    Subclasses set ``nx``, ``nu``, ``max_action`` and the state bounds, and
    implement the abstract methods.
    """

    nx: int
    nu: int
    max_action: np.ndarray
    state_lower_bounds: np.ndarray
    state_upper_bounds: np.ndarray

    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.dt = config.get("dt", 0.05)

    @abstractmethod
    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Next state under the numeric simulator."""

    @abstractmethod
    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Symbolic CasADi version of ``step``."""

    @abstractmethod
    def observe(self, state: np.ndarray) -> np.ndarray:
        """Observation recorded in the dataset."""

    @property
    @abstractmethod
    def goal_state(self) -> np.ndarray:
        """Full nx-dim reference state used by the planner cost."""

    @abstractmethod
    def is_done(self, state: np.ndarray) -> bool:
        """Whether the task is accomplished at ``state``."""

    @abstractmethod
    def get_dataset_features(self) -> dict[str, Any]:
        """LeRobot features dictionary."""

    @abstractmethod
    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """One dataset frame matching ``get_dataset_features``."""

    def simulate(
        self,
        initial_state: np.ndarray,
        policy_fn: Callable[[np.ndarray], np.ndarray],
        num_steps: int,
    ) -> dict[str, np.ndarray]:
        """Roll out ``policy_fn(state)``; stops after ``done_hold_steps`` consecutive done steps."""
        hold_steps = int(self.config.get("done_hold_steps", 5))
        states, observations, actions = [], [], []
        state = np.asarray(initial_state, dtype=float)
        done_count = 0

        for _ in range(num_steps):
            action = policy_fn(state)
            observations.append(self.observe(state))
            actions.append(action)
            state = self.step(state, action)
            states.append(state.copy())

            done_count = done_count + 1 if self.is_done(state) else 0
            if done_count >= hold_steps:
                break

        return {
            "states": np.array(states),
            "observations": np.array(observations),
            "actions": np.array(actions),
        }
