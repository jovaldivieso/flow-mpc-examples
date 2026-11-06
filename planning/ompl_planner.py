from __future__ import annotations

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from systems.triple_integrator import TripleIntegratorBase
from planning.planner import PlannerSolveError, goal_configuration
from planning.planner_config import ExpertConfig

def initialize_ompl(seed: int, log_level: ou.LogLevel = ou.LogLevel.LOG_ERROR) -> None:
    """Set OMPL's process-wide random seed and log level.

    OMPL keeps both globally, and accepts its seed only before its generator is first used (a later call prints an
    error and keeps the old seed). Call this once, at the entry point of the application (or of a worker process),
    before any planner is built: ``OmplPlanner`` itself does not touch OMPL's global state.
    """
    ou.setLogLevel(log_level)
    ou.RNG.setSeed(int(seed))


# Fraction of the largest joint range between two validity checks along a motion, fine enough that no motion penetrates an
# obstacle between two checks
VALIDITY_RESOLUTION = 0.003
# Largest joint-space step [rad] between the configurations the end effector's path length is summed over
LENGTH_STEP = 0.05


class OmplPlanner:
    """Global joint-space path planner: a path from a configuration to the goal configuration (OMPL RRT-Connect).

    A configuration is valid if it is within the joint limits, in the workspace of the system (``in_workspace``: the end
    effector never leaves it along the path) and clear of the obstacles. The path is shortcut and smoothed with OMPL's path
    simplifier and returned as its waypoints; the straight motions between them are validity checked.

    The planner is stateless regarding OMPL's global environment: its random seed (``ompl_seed``) and log level are set
    by ``initialize_ompl``, which the entry point of the application calls once.
    """

    def __init__(self, simulator: TripleIntegratorBase, config: ExpertConfig) -> None:
        self.sim = simulator
        self.n_joints = simulator.n
        self.timeout = config.ompl_timeout
        self.candidates = config.ompl_candidates

    def path(self, observation: np.ndarray) -> np.ndarray:
        """The waypoints (M, n) of a path from the configuration in ``observation`` (the first row) to its goal
        configuration (the last): of ``ompl_candidates`` RRT-Connect paths (the ones that found none are left out), the one
        along which the end effector (``tcp_position``) travels the shortest. Raises ``PlannerSolveError`` when the start or goal is out of bounds, outside the workspace or in collision, or when none
        of the candidates finds an exact path within ``ompl_timeout`` seconds."""
        q_start = np.asarray(self.sim.invert_obs(observation), dtype=float)[: self.n_joints]
        lower, upper = self.sim.path_lower, self.sim.path_upper
        if np.any(q_start < lower) or np.any(q_start > upper):
            raise PlannerSolveError("OMPL: the start configuration is out of bounds.")
        if not self.sim.in_workspace(q_start):
            raise PlannerSolveError("OMPL: the start configuration is outside the workspace.")
        if self.sim.collision.obstacles and not self.sim.collision.is_collision_free(q_start):
            raise PlannerSolveError("OMPL: the start configuration is in collision.")
        q_goal = goal_configuration(self.sim, observation[: self.sim.goal_dim], q_start)
        if np.any(q_goal < lower) or np.any(q_goal > upper):
            raise PlannerSolveError("OMPL: the goal configuration is out of bounds.")
        if not self.sim.in_workspace(q_goal):
            raise PlannerSolveError("OMPL: the goal configuration is outside the workspace.")
        if self.sim.collision.obstacles and not self.sim.collision.is_collision_free(q_goal):
            raise PlannerSolveError("OMPL: the goal configuration is in collision.")
        paths, failure = [], None
        for _ in range(self.candidates):
            try:
                paths.append(self._connect(q_start, [q_goal], lower, upper))
            except PlannerSolveError as exc:
                failure = exc
        if not paths:
            raise failure
        path = min(paths, key=self._end_effector_length)
        path[0] = q_start
        return path

    def _end_effector_length(self, path: np.ndarray) -> float:
        """The length of the path the end effector travels along the straight joint-space motions between the waypoints, summed
        over configurations at most ``LENGTH_STEP`` apart."""
        configurations = [path[:1]]
        for start, end in zip(path[:-1], path[1:]):
            steps = max(int(np.ceil(np.linalg.norm(end - start) / LENGTH_STEP)), 1)
            configurations.append(start + np.linspace(0.0, 1.0, steps + 1)[1:, None] * (end - start))
        positions = np.array([self.sim.tcp_position(q) for q in np.concatenate(configurations)])
        return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())

    def _state(self, space_information: ob.SpaceInformation, q: np.ndarray) -> ob.State:
        state = space_information.allocState()
        for joint in range(self.n_joints):
            state[joint] = float(q[joint])
        return state

    def _connect(self, q_start: np.ndarray, goals: list[np.ndarray], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """RRT-Connect from the start to any of the goal configurations: the waypoints of the simplified path."""
        space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)
        for joint in range(self.n_joints):
            bounds.setLow(joint, float(lower[joint]))
            bounds.setHigh(joint, float(upper[joint]))
        space.setBounds(bounds)
        space_information = ob.SpaceInformation(space)
        has_obstacles = bool(self.sim.collision.obstacles)

        def is_valid(state: ob.State) -> bool:
            if not space_information.satisfiesBounds(state):
                return False
            q = np.array([state[joint] for joint in range(self.n_joints)], dtype=float)
            return self.sim.in_workspace(q) and (not has_obstacles or self.sim.collision.is_collision_free(q))

        space_information.setStateValidityChecker(is_valid)
        space_information.setStateValidityCheckingResolution(VALIDITY_RESOLUTION)
        space_information.setup()

        problem = ob.ProblemDefinition(space_information)
        problem.addStartState(self._state(space_information, q_start))
        goal_states = ob.GoalStates(space_information)
        for q in goals:
            goal_states.addState(self._state(space_information, q))
        problem.setGoal(goal_states)
        planner = og.RRTConnect(space_information)
        planner.setProblemDefinition(problem)
        planner.setup()
        planner.solve(self.timeout)
        if not problem.hasExactSolution():
            raise PlannerSolveError(f"OMPL: RRT-Connect found no path to the goal pose within {self.timeout} s.")
        path = problem.getSolutionPath()
        og.PathSimplifier(space_information).simplifyMax(path)
        return np.array(
            [[path.getState(index)[joint] for joint in range(self.n_joints)] for index in range(path.getStateCount())]
        )
