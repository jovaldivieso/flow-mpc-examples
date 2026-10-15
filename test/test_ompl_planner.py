from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dynamics.franka_dynamics import FrankaDynamics
from learning.dagger.utils import sample_initial_state
from planning.ompl_planner import OmplPlanner
from planning.planner import Plan, Planner, PlannerSolveError
from test.planner_fixtures import pinned_config

N = 7
IN_COLLISION = np.array([2.82, 1.602, -0.228, -0.861, -0.015, 2.65, 1.714])  # 18 cm below the floor


def _observation(q: np.ndarray, goal: np.ndarray) -> np.ndarray:
    """The observation of the arm at rest in configuration ``q`` with the goal pose ``goal``."""
    return np.concatenate([goal, q, np.zeros(2 * N)])


class OmplPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaDynamics(cls.config)
        cls.start = sample_initial_state(cls.simulator, 3)  # also sets the goal of the sample
        cls.observation = _observation(cls.start[:N], cls.simulator.goal)
        cls.planner = OmplPlanner(cls.simulator, cls.config)
        cls.path = cls.planner.plan(cls.observation).positions

    def test_it_is_a_planner_and_plans_a_timed_path(self) -> None:
        self.assertIsInstance(self.planner, Planner)
        plan = self.planner.plan(self.observation)
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.positions.shape[1], N)

    def test_the_path_is_traversed_at_the_configured_speed(self) -> None:
        # One configuration per control period: constant joint-space speed, the last step (to the goal) shorter
        step = self.config["ompl_speed"] * self.simulator.dt
        distances = np.linalg.norm(np.diff(self.path, axis=0), axis=1)

        np.testing.assert_allclose(distances[:-1], step, rtol=1e-6)
        self.assertLessEqual(distances[-1], step + 1e-9)

    def test_a_slower_speed_gives_more_steps_for_the_same_path(self) -> None:
        slow = OmplPlanner(self.simulator, {**self.config, "ompl_speed": 1.0}).plan(self.observation).positions

        self.assertGreater(len(slow), len(self.path))

    def test_the_path_runs_from_the_start_to_a_configuration_that_reaches_the_goal_pose(self) -> None:
        np.testing.assert_array_equal(self.path[0], self.start[:N])
        self.assertLess(max(self.simulator.goal_errors(self.path[-1])), 1e-4)

    def test_every_waypoint_is_inside_the_joint_limits_and_clear_of_the_floor(self) -> None:
        lower = self.simulator.state_lower_bounds[:N]
        upper = self.simulator.state_upper_bounds[:N]
        self.assertTrue(np.all(self.path >= lower - 1e-9) and np.all(self.path <= upper + 1e-9))
        for waypoint in self.path:
            self.assertFalse(self.simulator.is_collision(waypoint))

    def test_the_ompl_seed_is_applied_once_per_process(self) -> None:
        # OMPL accepts its seed only before the generator is first used; a later call prints an error and does nothing
        with mock.patch.object(OmplPlanner, "_rng_is_seeded", False), mock.patch("planning.ompl_planner.ou.RNG") as rng:
            OmplPlanner(self.simulator, {**self.config, "ompl_seed": 7})
            OmplPlanner(self.simulator, {**self.config, "ompl_seed": 9})

        rng.setSeed.assert_called_once_with(7)

    def test_replanning_gives_a_valid_path_again(self) -> None:
        again = self.planner.plan(self.observation).positions

        np.testing.assert_array_equal(again[0], self.start[:N])
        self.assertLess(max(self.simulator.goal_errors(again[-1])), 1e-4)

    def test_a_plan_to_refine_is_ignored(self) -> None:
        refined = self.planner.plan(self.observation, Plan(positions=np.zeros((5, N))))

        np.testing.assert_array_equal(refined.positions[0], self.start[:N])

    def test_reset_is_a_no_op(self) -> None:
        self.planner.reset()

    def test_a_start_in_collision_raises(self) -> None:
        start = IN_COLLISION.copy()
        self.assertTrue(self.simulator.is_collision(start))
        with self.assertRaises(PlannerSolveError):
            self.planner.plan(_observation(start, self.simulator.goal))

    def test_a_goal_pose_out_of_reach_raises(self) -> None:
        goal = self.simulator.goal.copy()
        goal[:3] = [5.0, 5.0, 5.0]
        with self.assertRaises(PlannerSolveError):
            self.planner.plan(_observation(self.start[:N], goal))


if __name__ == "__main__":
    unittest.main()
