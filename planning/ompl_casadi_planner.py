from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from systems.system import SystemSimulator
from planning.casadi_planner import CasadiPlanner
from planning.ompl_planner import OmplPlanner
from planning.planner import Plan, PlannerSolveError


class OmplCasadiPlanner(CasadiPlanner):
    """The CasADi MPC with the goal configuration and the first guess of a global planner (OMPL RRT-Connect).

    Here the global planner finds a path to the goal configuration (``OmplPlanner``): the end of that
    path is the goal configuration of the MPC, and the path is where the first solve starts from. Later solves
    start from the previous solution as they do without a path, unless the arm is further than
    ``ompl_replan_distance`` (rad) from it (the learner drove it away): then a new path is planned from where the arm
    is. If no path can be found the MPC runs as it does alone (counted in ``path_failures``) until the arm has moved
    ``ompl_replan_distance`` from where planning failed. If a solve that started from a path fails, the path is the
    plan instead (``trajectory_fallbacks``).
    """

    def __init__(self, simulator: SystemSimulator, config: Mapping[str, Any]) -> None:
        super().__init__(simulator, config)
        self.path_planner = OmplPlanner(simulator, config)
        self.replan_distance = float(config["ompl_replan_distance"])
        self.path_plans = 0
        self.replans = 0
        self.path_failures = 0
        self.fallbacks = 0
        self._paths_attempted = 0  # RRT-Connect calls for the current goal
        self._failed_at: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._paths_attempted = 0
        self._failed_at = None

    @property
    def statistics(self) -> dict[str, int]:
        """Cumulative counters: the paths found (``ompl_paths``), the planning calls that found none
        (``ompl_path_failures``) and the planning calls for a goal that already had a first one (``ompl_replans``:
        the arm had left its solution, or planning failed before and was retried), successful or not, the NLP
        solves (``trajectory_optimizations``) and those that failed, for which the path was the plan
        (``trajectory_fallbacks``)."""
        return {
            "ompl_paths": self.path_plans,
            "ompl_replans": self.replans,
            "ompl_path_failures": self.path_failures,
            **super().statistics,
            "trajectory_fallbacks": self.fallbacks,
        }

    def _begin_goal(self, goal: np.ndarray) -> None:
        super()._begin_goal(goal)
        self._paths_attempted = 0
        self._failed_at = None

    def _initial_plan(self, observation: np.ndarray, x0: np.ndarray) -> Plan | None:
        q = x0[: self.n_joints]
        if self.last_X_sol is not None:
            distance = float(np.min(np.linalg.norm(self.last_X_sol[: self.n_joints].T - q, axis=1)))
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
        self.goal_configuration = path.positions[-1].copy()
        return path

    def _solve(self, x0: np.ndarray, initial_plan: Plan | None) -> None:
        try:
            super()._solve(x0, initial_plan)
        except PlannerSolveError:
            if initial_plan is None:
                raise
            self.fallbacks += 1
            self.last_X_sol, self.last_U_sol = self._states_through(x0, initial_plan)
