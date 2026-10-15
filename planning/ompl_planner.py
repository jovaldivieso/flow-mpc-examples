from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from dynamics.dynamics import DynamicsProtocol
from planning.planner import Plan, Planner, PlannerSolveError

ou.setLogLevel(ou.LogLevel.LOG_ERROR)

# Fraction of the largest joint range between two collision checks along a motion
VALIDITY_RESOLUTION = 0.01
# Points the simplified path is first resampled to (evenly spaced in joint space), before it is timed
SIMPLIFIED_PATH_POINTS = 200
IK_ITERATIONS = 200
IK_ATTEMPTS_PER_SOLUTION = 20
# Inverse-kinematics solutions closer than this (rad, largest joint difference) to one already found are the same
IK_DISTINCT = 0.05


class OmplPlanner(Planner):
    """Global joint-space planner: a collision-free path from a configuration to a TCP goal pose (OMPL RRT-Connect).

    The goal is a pose, not a configuration, so the goal states are the collision-free, within-limits solutions of the
    arm's inverse kinematics for it, found by damped least squares from the start configuration and from random ones;
    RRT-Connect plans to whichever it reaches first. A configuration is valid if the simulator reports no collision
    (``sim.is_collision``) and it is within the joint limits. The path is shortcut and smoothed with OMPL's path
    simplifier and traversed at the constant joint-space speed ``ompl_speed`` (rad/s), one configuration per control
    period.

    OMPL's random generator is process-wide and accepts its seed only before it is first used, so ``ompl_seed`` is
    applied once, when the first planner of the process is built.
    """

    _rng_is_seeded = False

    def __init__(self, simulator: DynamicsProtocol, config: Mapping[str, Any]) -> None:
        self.sim = simulator
        self.n_joints = simulator.n
        self.timeout = float(config["ompl_timeout"])
        self.seed = int(config["ompl_seed"])
        self.speed = float(config["ompl_speed"])
        self.ik_solutions = int(config["ompl_ik_solutions"])
        # A later call would only make OMPL print an error and keep the old seed
        if not OmplPlanner._rng_is_seeded:
            ou.RNG.setSeed(self.seed)
            OmplPlanner._rng_is_seeded = True

    def reset(self) -> None:
        """Planning from scratch every time: there is nothing to forget."""

    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        """A path, timed at ``ompl_speed``, from the arm's configuration in ``observation`` (the first row) to one that
        reaches its goal pose. ``initial_plan`` is ignored. Raises ``PlannerSolveError`` when the start is in collision,
        when no inverse-kinematics solution of the goal is valid, or when no exact path is found within
        ``ompl_timeout`` seconds (an approximate path that ends short of the goal is no path)."""
        q_start = np.asarray(self.sim.invert_obs(observation), dtype=float)[: self.n_joints]
        if self.sim.is_collision(q_start):
            raise PlannerSolveError("OMPL: the start configuration is in collision.")
        lower = self.sim.state_lower_bounds[: self.n_joints]
        upper = self.sim.state_upper_bounds[: self.n_joints]
        goals = self._goal_configurations(observation[: self.sim.goal_dim], q_start, lower, upper)
        if not goals:
            raise PlannerSolveError("OMPL: no collision-free inverse-kinematics solution of the goal pose.")
        path = self._connect(q_start, goals, lower, upper)
        path[0] = q_start
        return Plan(positions=self._timed(path))

    def _timed(self, path: np.ndarray) -> np.ndarray:
        """The path at the constant joint-space speed: one configuration per control period (the last step may be shorter)."""
        arc_length = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        arcs = np.append(np.arange(0.0, arc_length[-1], self.speed * self.sim.dt), arc_length[-1])
        if len(arcs) == 1:  # the start already is the goal configuration: a plan has a first and a next row
            arcs = np.repeat(arcs, 2)
        return np.stack([np.interp(arcs, arc_length, joint) for joint in path.T], axis=1)

    def _goal_configurations(
        self, goal: np.ndarray, q_start: np.ndarray, lower: np.ndarray, upper: np.ndarray
    ) -> list[np.ndarray]:
        goal_pose = self.sim.goal_to_pose(goal)
        rng = np.random.default_rng(self.seed)
        solutions: list[np.ndarray] = []
        for attempt in range(self.ik_solutions * IK_ATTEMPTS_PER_SOLUTION):
            # The first attempt starts from the current configuration, which finds the nearest branch of the solutions
            q_init = q_start if attempt == 0 else rng.uniform(lower, upper)
            try:
                q = self.sim.inverse_kinematics(goal_pose, q_init, iters=IK_ITERATIONS)
            except RuntimeError:
                continue
            is_new = all(np.max(np.abs(q - other)) > IK_DISTINCT for other in solutions)
            if is_new and not self.sim.is_collision(q):
                solutions.append(q)
            if len(solutions) == self.ik_solutions:
                break
        return solutions

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
        space_information.setStateValidityChecker(
            lambda state: not self.sim.is_collision(np.array([state[joint] for joint in range(self.n_joints)]))
        )
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
