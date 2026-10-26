from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from systems.kinematic_collision_system import KinematicCollisionSystem
from planning.planner import Plan, Planner, PlannerSolveError, goal_configuration

def initialize_ompl(seed: int, log_level: ou.LogLevel = ou.LogLevel.LOG_ERROR) -> None:
    """Set OMPL's process-wide random seed and log level.

    OMPL keeps both globally, and accepts its seed only before its generator is first used (a later call prints an
    error and keeps the old seed). Call this once, at the entry point of the application (or of a worker process),
    before any planner is built: ``OmplPlanner`` itself does not touch OMPL's global state.
    """
    ou.setLogLevel(log_level)
    ou.RNG.setSeed(int(seed))


# Fraction of the largest joint range between two validity checks along a motion
VALIDITY_RESOLUTION = 0.01
# Points the simplified path is first resampled to (evenly spaced in joint space), before it is timed
SIMPLIFIED_PATH_POINTS = 200


class OmplPlanner(Planner):
    """Global joint-space planner: a path from a configuration to the goal configuration (OMPL RRT-Connect).

    A configuration is valid if it is within the joint limits. The path is shortcut and smoothed with OMPL's path
    simplifier and traversed at the constant joint-space speed ``ompl_speed`` (rad/s), one configuration per control
    period.

    The planner is stateless regarding OMPL's global environment: its random seed (``ompl_seed``) and log level are set
    by ``initialize_ompl``, which the entry point of the application calls once.
    """

    def __init__(self, simulator: KinematicCollisionSystem, config: Mapping[str, Any]) -> None:
        self.sim = simulator
        self.n_joints = simulator.n
        self.timeout = float(config["ompl_timeout"])
        self.speed = float(config["ompl_speed"])

    def reset(self) -> None:
        """Planning from scratch every time: there is nothing to forget."""

    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        """A path, timed at ``ompl_speed``, from the configuration in ``observation`` (the first row) to its goal
        configuration. ``initial_plan`` is ignored. Raises ``PlannerSolveError`` when the start or goal is out of
        bounds or when no exact path is found within ``ompl_timeout`` seconds."""
        q_start = np.asarray(self.sim.invert_obs(observation), dtype=float)[: self.n_joints]
        lower, upper = self.sim.path_lower, self.sim.path_upper
        if np.any(q_start < lower) or np.any(q_start > upper):
            raise PlannerSolveError("OMPL: the start configuration is out of bounds.")
        if self.sim.obstacles and not self.sim.is_collision_free(q_start):
            raise PlannerSolveError("OMPL: the start configuration is in collision.")
        q_goal = goal_configuration(self.sim, observation[: self.sim.goal_dim], q_start)
        if np.any(q_goal < lower) or np.any(q_goal > upper):
            raise PlannerSolveError("OMPL: the goal configuration is out of bounds.")
        if self.sim.obstacles and not self.sim.is_collision_free(q_goal):
            raise PlannerSolveError("OMPL: the goal configuration is in collision.")
        path = self._connect(q_start, [q_goal], lower, upper)
        path[0] = q_start
        return Plan(positions=self._timed(path))

    def _timed(self, path: np.ndarray) -> np.ndarray:
        """The path at the constant joint-space speed: one configuration per control period (the last step may be shorter)."""
        arc_length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        arcs = np.append(np.arange(0.0, arc_length[-1], self.speed * self.sim.dt), arc_length[-1])
        if len(arcs) == 1:  # the start already is the goal configuration: a plan has a first and a next row
            arcs = np.repeat(arcs, 2)
        return np.stack([np.interp(arcs, arc_length, joint) for joint in path.T], axis=1)

    def _state(self, space_information: ob.SpaceInformation, q: np.ndarray) -> ob.State:
        state = space_information.allocState()
        for joint in range(self.n_joints):
            state[joint] = float(q[joint])
        return state

    def _connect(self, q_start: np.ndarray, goals: list[np.ndarray], lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        """RRT-Connect from the start to any of the goal configurations: the simplified path (SIMPLIFIED_PATH_POINTS)."""
        space = ob.RealVectorStateSpace(self.n_joints)
        bounds = ob.RealVectorBounds(self.n_joints)
        for joint in range(self.n_joints):
            bounds.setLow(joint, float(lower[joint]))
            bounds.setHigh(joint, float(upper[joint]))
        space.setBounds(bounds)
        space_information = ob.SpaceInformation(space)
        has_obstacles = bool(self.sim.obstacles)

        def is_valid(state: ob.State) -> bool:
            if not space_information.satisfiesBounds(state):
                return False
            if not has_obstacles:
                return True
            q = np.array([state[joint] for joint in range(self.n_joints)], dtype=float)
            return self.sim.is_collision_free(q)

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
        path.interpolate(SIMPLIFIED_PATH_POINTS)
        return np.array(
            [[path.getState(index)[joint] for joint in range(self.n_joints)] for index in range(path.getStateCount())]
        )
