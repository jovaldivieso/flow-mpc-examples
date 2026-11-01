from __future__ import annotations

import numpy as np

from planning.initial_guess import InitialGuessGenerator
from planning.ompl_planner import OmplPlanner
from planning.planner import Plan, PlannerSolveError
from planning.reference_path import ReferencePath


class OmplInitialGuess(InitialGuessGenerator):
    """The first guess of ``CasadiPlanner`` from a global planner (OMPL RRT-Connect), the expert ``ompl_casadi``.

    ``OmplPlanner`` finds a path to the goal configuration: the end of that path is the goal configuration of the MPC,
    and the path is where its first solve starts from. The MPC heads for the point ``lookahead`` (rad of path length)
    ahead of the arm on that path (``target``), not for the end, so the path steers it around what a straight line
    to the goal would run into. Later solves start from the previous solution as they do without a path, unless the arm
    is further than ``replan_distance`` (rad) from it (the learner drove it away): then a new path is planned from where
    the arm is. If no path can be found the MPC runs as it does alone (counted in ``path_failures``) until the arm has
    moved ``replan_distance`` from where planning failed.
    """

    def __init__(self, path_planner: OmplPlanner, replan_distance: float, lookahead: float) -> None:
        self.path_planner = path_planner
        self.replan_distance = float(replan_distance)
        self.lookahead = float(lookahead)
        self._path: ReferencePath | None = None
        self.path_plans = 0
        self.replans = 0
        self.path_failures = 0
        self._paths_attempted = 0  # RRT-Connect calls for the current goal
        self._failed_at: np.ndarray | None = None

    def reset(self) -> None:
        self._paths_attempted = 0
        self._failed_at = None
        self._path = None

    def begin_goal(self) -> None:
        self.reset()

    @property
    def statistics(self) -> dict[str, int]:
        """Cumulative counters: the paths found (``ompl_paths``), the planning calls that found none
        (``ompl_path_failures``) and the planning calls for a goal that already had a first one (``ompl_replans``: the
        arm had left its solution, or planning failed before and was retried), successful or not."""
        return {
            "ompl_paths": self.path_plans,
            "ompl_replans": self.replans,
            "ompl_path_failures": self.path_failures,
        }

    def target(self, q: np.ndarray) -> np.ndarray | None:
        return None if self._path is None else self._path.ahead(q, self.lookahead)

    def initial_plan(self, observation: np.ndarray, q: np.ndarray, last_positions: np.ndarray | None) -> Plan | None:
        if last_positions is not None:
            distance = float(np.min(np.linalg.norm(last_positions - q, axis=1)))
            if distance <= self.replan_distance:
                return None
        if self._failed_at is not None and float(np.linalg.norm(q - self._failed_at)) <= self.replan_distance:
            return None
        return self._plan_path(observation, q)

    def _plan_path(self, observation: np.ndarray, q: np.ndarray) -> Plan | None:
        self.replans += int(self._paths_attempted > 0)
        self._paths_attempted += 1
        try:
            path = self.path_planner.plan(observation)
        except PlannerSolveError:
            self._failed_at = q.copy()
            self.path_failures += 1
            return None
        self._failed_at = None
        self.path_plans += 1
        self._path = ReferencePath(path.positions)
        return path
