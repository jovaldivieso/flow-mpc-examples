from abc import ABC, abstractmethod

import numpy as np


class PlannerSolveError(RuntimeError):
    """Planner could not produce a valid solution."""


class Planner(ABC):
    """A planner maps the current state to the next action."""

    @abstractmethod
    def reset(self) -> None:
        """Clear internal state (cached plans, warm starts) for a new episode."""

    @abstractmethod
    def __call__(self, state: np.ndarray) -> np.ndarray:
        """Next action for ``state``."""
