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
    """A plan: where the system is to be at the control steps and the actions that take it there.

    ``positions`` has one row of joint positions per control period: row k is the configuration k control periods
    from now, row 0 the configuration the plan starts from. ``actions`` has one row per control period: row k is the
    action at step k (a path planner has none).
    """

    positions: np.ndarray
    actions: np.ndarray | None = None

    def first_actions(self, rows: int) -> np.ndarray:
        """The first ``rows`` actions; a plan that ends sooner goes on with zero actions. A path (no actions) has none."""
        if self.actions is None:
            raise ValueError("A path plan has no actions.")
        actions = np.asarray(self.actions, dtype=float)
        return np.vstack([actions, np.zeros((max(0, rows - len(actions)), actions.shape[1]))])[:rows]


class Planner(ABC):
    """
    Abstract base class for all motion planners: a planner plans from the arm's state to its goal pose.
    """

    @abstractmethod
    def reset(self) -> None:
        """
        Clear internal state (e.g., cached trajectories) for a new episode.
        Stateless planners can implement this as a no-op.
        """

    @abstractmethod
    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        """
        Plan from the arm's state in ``observation`` to the goal pose in it.

        ``initial_plan`` is a plan to start from, for planners that optimize: it is their initial guess (a plan shorter
        than their horizon is held at its last configuration), and nothing holds their solution to it. A planner that
        has no use for one ignores it.
        """

    @property
    def statistics(self) -> dict[str, int]:
        """Counters of what the planner has done since it was built (it keeps none unless a subclass says so)."""
        return {}
