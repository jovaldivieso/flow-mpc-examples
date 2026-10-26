from abc import ABC, abstractmethod

import numpy as np

from .planner import Plan


class InitialGuessGenerator(ABC):
    """A global strategy that gives an optimizing planner (``CasadiPlanner``) the first guess of its solve.

    It is asked for a plan before every solve. A plan it returns ends at the goal configuration (which the planner then
    drives to) and is where the solve starts from; if that solve fails the plan itself is the solution. Returning none
    leaves the planner to warm start from its own previous solution.
    """

    @abstractmethod
    def reset(self) -> None:
        """A new episode starts."""

    @abstractmethod
    def begin_goal(self) -> None:
        """The goal changed: what was planned for the last one is no use."""

    @abstractmethod
    def initial_plan(self, observation: np.ndarray, q: np.ndarray, last_positions: np.ndarray | None) -> Plan | None:
        """A plan from the configuration ``q`` of ``observation`` to its goal, or none. ``last_positions`` are the
        planner's previous solution (rows of configurations), none right after a reset or a new goal."""

    @property
    def statistics(self) -> dict[str, int]:
        """Counters of what the strategy has done since it was built (it keeps none unless a subclass says so)."""
        return {}
