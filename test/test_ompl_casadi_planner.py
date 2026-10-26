from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from learning.dagger.utils import sample_initial_state
from planning.casadi_planner import CasadiPlanner
from planning.ompl_casadi_planner import OmplCasadiPlanner
from planning.planner import Planner, PlannerSolveError
from test.planner_fixtures import pinned_config, pose_goal

N = 7
SEED = 3  # a start and goal pair far enough apart for the MPC to plan a long motion


def _step_along(simulator: FrankaSystem, planner: Planner, state: np.ndarray) -> np.ndarray:
    """The state after one control step along the first action of the planner's plan from ``state``."""
    return simulator.step(state, planner.plan(simulator.observe(state)).actions[0])


def _closed_loop(simulator: FrankaSystem, planner: Planner, start: np.ndarray, steps: int) -> bool:
    """Whether the planner in closed loop on the plant reaches the goal."""
    state = simulator.reset(start)
    planner.reset()
    for _ in range(steps):
        state = _step_along(simulator, planner, state)
        if simulator.should_terminate_rollout(state):
            return True
    return False


class OmplCasadiPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.start = sample_initial_state(cls.simulator, SEED)  # also sets the goal
        cls.goal = cls.simulator.goal.copy()

    def setUp(self) -> None:
        self.simulator.set_goal(self.goal)
        self.planner = OmplCasadiPlanner(self.simulator, self.config)

    def _first_observation(self) -> np.ndarray:
        self.planner.reset()
        return self.simulator.observe(self.simulator.reset(self.start))

    def _elsewhere(self) -> np.ndarray:
        """An observation of the arm far from the last solution, with the planner's goal."""
        elsewhere = sample_initial_state(self.simulator, 5)
        self.simulator.set_goal(self.goal)  # the sample drew a new goal; the planner's goal stays the same
        distance = np.min(np.linalg.norm(self.planner.last_X_sol[:N].T - elsewhere[:N], axis=1))
        self.assertGreater(distance, self.planner.replan_distance)
        return self.simulator.observe(elsewhere)

    def test_it_is_a_casadi_planner(self) -> None:
        self.assertIsInstance(self.planner, CasadiPlanner)

    def test_it_reaches_the_goal_pose_and_comes_to_rest(self) -> None:
        reached = _closed_loop(self.simulator, self.planner, self.start, 80)

        self.assertTrue(reached)
        self.assertEqual((self.planner.path_plans, self.planner.path_failures), (1, 0))

    def test_the_goal_configuration_is_the_end_of_the_path_and_the_first_solve_starts_from_the_path(self) -> None:
        observation = self._first_observation()
        path = self.planner.path_planner.plan(observation)

        with mock.patch.object(self.planner.path_planner, "plan", return_value=path):
            guessed = {}
            original = self.planner.opti.set_initial
            self.planner.opti.set_initial = lambda variable, value: (guessed.update({id(variable): np.asarray(value)}), original(variable, value))[1]
            self.planner.plan(observation)

        np.testing.assert_array_equal(self.planner.goal_configuration, path.positions[-1])
        ahead = path.positions[1 : self.planner.N + 1]  # a path shorter than the horizon is held at its end
        np.testing.assert_allclose(guessed[id(self.planner.X)][:N, 1 : 1 + len(ahead)], ahead.T, atol=1e-9)

    def test_later_solves_start_from_the_previous_solution_and_plan_no_new_path(self) -> None:
        state = self.simulator.reset(self.start)
        self.planner.reset()

        for _ in range(5):
            state = _step_along(self.simulator, self.planner, state)

        self.assertEqual((self.planner.path_plans, self.planner.optimizations), (1, 5))

    def test_the_plan_is_the_mpcs_over_its_horizon_and_starts_at_the_state(self) -> None:
        plan = self.planner.plan(self._first_observation())

        self.assertEqual(plan.positions.shape, (self.planner.N + 1, N))
        np.testing.assert_allclose(plan.positions[0], self.start[:N], atol=1e-9)

    def test_a_failing_solve_falls_back_to_the_path_it_started_from(self) -> None:
        observation = self._first_observation()
        path = self.planner.path_planner.plan(observation)

        with mock.patch.object(self.planner.path_planner, "plan", return_value=path):  # the path is the planner's own
            with mock.patch.object(CasadiPlanner, "_solve", side_effect=PlannerSolveError("no solution")):
                plan = self.planner.plan(observation)

        self.assertEqual((self.planner.optimizations, self.planner.statistics["trajectory_fallbacks"]), (0, 1))
        np.testing.assert_allclose(plan.positions[1 : len(path.positions)], path.positions[1 : self.planner.N + 1], atol=1e-9)

    def test_a_failing_solve_without_a_path_to_fall_back_to_is_the_planners_failure(self) -> None:
        observation = self._first_observation()

        with mock.patch.object(self.planner.path_planner, "plan", side_effect=PlannerSolveError("no path")):
            with mock.patch.object(CasadiPlanner, "_solve", side_effect=PlannerSolveError("no solution")):
                with self.assertRaises(PlannerSolveError):
                    self.planner.plan(observation)

    def test_a_failing_path_planner_leaves_the_mpc_alone_and_is_not_retried_every_step(self) -> None:
        state = self.simulator.reset(self.start)
        self.planner.reset()

        with mock.patch.object(self.planner.path_planner, "plan", side_effect=PlannerSolveError("no path")) as plan:
            for _ in range(3):
                state = _step_along(self.simulator, self.planner, state)

        self.assertEqual(plan.call_count, 1)
        self.assertEqual((self.planner.path_plans, self.planner.path_failures), (0, 1))
        self.assertEqual(self.planner.optimizations, 3)
        # the goal configuration then comes from IK
        self.assertLess(max(self.simulator.goal_errors(self.planner.goal_configuration)), 1e-5)

    def test_the_arm_driven_far_from_the_last_solution_gets_a_new_path_from_where_it_is(self) -> None:
        self.planner.plan(self._first_observation())
        self.assertEqual(self.planner.path_plans, 1)

        self.planner.plan(self._elsewhere())

        self.assertEqual(self.planner.path_plans, 2)

    def test_the_statistics_count_paths_replans_failures_solves_and_fallbacks(self) -> None:
        def counts(paths: int, replans: int, failures: int, optimizations: int) -> dict[str, int]:
            return {
                "ompl_paths": paths,
                "ompl_replans": replans,
                "ompl_path_failures": failures,
                "trajectory_optimizations": optimizations,
                "trajectory_fallbacks": 0,
            }

        self.assertEqual(self.planner.statistics, counts(0, 0, 0, 0))

        self.planner.plan(self._first_observation())  # the first path of the episode is no replan
        self.assertEqual(self.planner.statistics, counts(1, 0, 0, 1))

        self.planner.plan(self._elsewhere())  # the arm left the last solution: a replan
        self.assertEqual(self.planner.statistics, counts(2, 1, 0, 2))

        with mock.patch.object(self.planner.path_planner, "plan", side_effect=PlannerSolveError("no path")):
            self.planner.plan(self.simulator.observe(self.start))  # far from the new solution again, planning fails
        self.assertEqual(self.planner.statistics, counts(2, 2, 1, 3))

    def test_the_statistics_survive_a_reset_while_a_new_episode_starts_with_a_first_path(self) -> None:
        self.planner.plan(self._first_observation())

        self.planner.reset()
        self.planner.plan(self.simulator.observe(self.start))

        self.assertEqual(self.planner.statistics["ompl_paths"], 2)
        self.assertEqual(self.planner.statistics["ompl_replans"], 0)

    def test_a_new_goal_gets_a_new_path(self) -> None:
        observation = self._first_observation()
        self.planner.plan(observation)

        self.simulator.set_goal(pose_goal(self.simulator, self.start[:N] + 0.2))
        self.planner.plan(self.simulator.observe(self.start))

        self.assertEqual(self.planner.path_plans, 2)

    def test_reset_forgets_the_solution_and_the_goal_configuration(self) -> None:
        self.planner.plan(self._first_observation())
        self.assertIsNotNone(self.planner.last_X_sol)

        self.planner.reset()

        self.assertIsNone(self.planner.last_X_sol)
        self.assertIsNone(self.planner.goal_configuration)


if __name__ == "__main__":
    unittest.main()
