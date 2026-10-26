import numpy as np

from .system import SystemSimulator


class RolloutTerminationTracker:
    """Ends a rollout once the system has been done (``SystemSimulator.is_done``) for ``done_hold_steps`` steps in a row.

    One tracker per rollout: the count of consecutive done steps is its only state.
    """

    def __init__(self, simulator: SystemSimulator) -> None:
        self._simulator = simulator
        self._hold_steps = simulator.done_hold_steps
        self._done_steps = 0

    def reset(self) -> None:
        self._done_steps = 0

    def update(self, state: np.ndarray) -> bool:
        """Count ``state`` and return whether the rollout is to end."""
        self._done_steps = self._done_steps + 1 if self._simulator.is_done(state) else 0
        return self._done_steps >= self._hold_steps
