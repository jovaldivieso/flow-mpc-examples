from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from systems.triple_integrator import TripleIntegratorBase


class PlannerSolveError(RuntimeError):
    """Planner could not find/produce a valid solution."""
    pass


def goal_configuration(simulator: TripleIntegratorBase, goal: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
    """The configuration of ``goal`` nearest to ``q_reference``, or ``PlannerSolveError`` if there is none."""
    try:
        return simulator.goal_configuration(goal, q_reference)
    except RuntimeError as exc:
        raise PlannerSolveError(f"No joint configuration reaches the goal {np.asarray(goal).tolist()}.") from exc


@dataclass(frozen=True)
class Plan:
    """What an expert plans from a state: where it wants the arm and the actions that track it.

    ``positions`` has one row of joint positions per control period: row k is the configuration k + 1 control periods
    from now, which is what the policy imitates; it holds only collision-free configurations. ``actions`` has the
    action of every control period of the tracking of ``positions`` (row 0 is executed).
    """

    positions: np.ndarray
    actions: np.ndarray


class Planner(ABC):
    """
    Abstract base class for all experts: an expert plans from the arm's state to its goal pose.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Clear internal state (e.g., cached trajectories) for a new episode.
        Stateless planners can implement this as a no-op.
        """

    @abstractmethod
    def plan(self, observation: np.ndarray) -> Plan:
        """Plan from the arm's state in ``observation`` to the goal pose in it; raises ``PlannerSolveError`` when it cannot."""

    @property
    def statistics(self) -> dict[str, int]:
        """Counters of what the planner has done since it was built (it keeps none unless a subclass says so)."""
        return {}
