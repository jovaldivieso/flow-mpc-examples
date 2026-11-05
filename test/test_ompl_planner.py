from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.config_fields import ConfigurationError
from systems.franka_system import FrankaSystem
from systems.point_mass_system import PointMassSystem
from learning.dagger.utils import sample_initial_state
from ompl import util as ou
from planning.config import validate_system_config
from planning.ompl_planner import OmplPlanner, initialize_ompl
from planning.planner import Planner, PlannerSolveError
from planning.planner_config import ExpertConfig
from test.planner_fixtures import PINNED_EXPERT_SETTINGS, pinned_config

N = 7
TABLE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_table_config.yaml")


def _table_config(**overrides) -> dict:
    with open(TABLE_CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


class OmplPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.start = sample_initial_state(cls.simulator, 3)  # also sets the goal of the sample
        cls.observation = cls.simulator.observe(cls.start)
        cls.planner = OmplPlanner(cls.simulator, ExpertConfig.from_mapping(cls.config))
        cls.path = cls.planner.path(cls.observation)

    def test_it_only_plans_a_path_and_is_no_expert(self) -> None:
        self.assertNotIsInstance(self.planner, Planner)
        self.assertFalse(hasattr(self.planner, "plan"))

    def test_the_path_is_a_matrix_of_waypoints_of_joint_positions(self) -> None:
        self.assertEqual(self.path.ndim, 2)
        self.assertEqual(self.path.shape[1], N)
        self.assertGreaterEqual(len(self.path), 2)
        self.assertTrue(np.isfinite(self.path).all())

    def test_the_path_runs_from_the_start_to_the_goal_configuration(self) -> None:
        np.testing.assert_array_equal(self.path[0], self.start[:N])
        self.assertLess(max(self.simulator.goal_errors(self.path[-1])), 1e-4)

    def test_every_waypoint_is_inside_the_joint_limits(self) -> None:
        self.assertTrue(np.all(self.path >= self.simulator.path_lower - 1e-9) and np.all(self.path <= self.simulator.path_upper + 1e-9))

    def test_the_path_is_the_start_of_the_observation_whatever_the_velocity_in_it(self) -> None:
        moving = self.start.copy()
        moving[N : 2 * N] = 0.2

        path = self.planner.path(self.simulator.observe(moving))

        np.testing.assert_array_equal(path[0], self.start[:N])

    def test_planning_again_gives_a_valid_path_again(self) -> None:
        again = self.planner.path(self.observation)

        np.testing.assert_array_equal(again[0], self.start[:N])
        self.assertLess(max(self.simulator.goal_errors(again[-1])), 1e-4)

    def test_initialize_ompl_sets_the_seed_and_the_log_level_and_building_a_planner_does_not(self) -> None:
        with mock.patch("planning.ompl_planner.ou.RNG") as rng, mock.patch("planning.ompl_planner.ou.setLogLevel") as log:
            OmplPlanner(self.simulator, ExpertConfig.from_mapping({**self.config, "ompl_seed": 7}))
            rng.setSeed.assert_not_called()
            log.assert_not_called()

            initialize_ompl(7)

        rng.setSeed.assert_called_once_with(7)
        log.assert_called_once_with(ou.LogLevel.LOG_ERROR)

    def test_a_goal_out_of_reach_raises(self) -> None:
        observation = self.observation.copy()
        observation[:3] = [5.0, 0.0, 0.0]
        with self.assertRaises(PlannerSolveError):
            self.planner.path(observation)

    def test_a_start_out_of_bounds_raises(self) -> None:
        start = self.start.copy()
        start[0] = self.simulator.path_upper[0] + 0.1
        with self.assertRaisesRegex(PlannerSolveError, "start configuration is out of bounds"):
            self.planner.path(self.simulator.observe(start))


class OmplPlannerAmongObstaclesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _table_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.start = sample_initial_state(cls.simulator, 2)
        cls.observation = cls.simulator.observe(cls.start)
        cls.planner = OmplPlanner(cls.simulator, ExpertConfig.from_mapping(cls.config))
        cls.path = cls.planner.path(cls.observation)

    def test_every_waypoint_and_the_motions_between_them_are_collision_free(self) -> None:
        collision = self.simulator.collision
        self.assertTrue(collision.obstacles)
        for a, b in zip(self.path[:-1], self.path[1:]):
            motion = a + np.linspace(0.0, 1.0, 50)[:, None] * (b - a)
            self.assertTrue(collision.trajectory_is_collision_free(motion))

    def test_the_path_ends_at_the_goal(self) -> None:
        self.assertLess(max(self.simulator.goal_errors(self.path[-1])), 1e-4)

    def test_a_start_in_collision_raises(self) -> None:
        simulator = self.simulator
        in_collision = self.start.copy()
        in_collision[:N] = self.path[len(self.path) // 2]
        centers, _, _ = simulator.collision_spheres(in_collision[:N])
        blocked = FrankaSystem(_table_config(obstacles=[{"position": centers[-1].tolist(), "radius": 0.05}]))
        blocked.set_goal(simulator.goal)

        with self.assertRaisesRegex(PlannerSolveError, "start configuration is in collision"):
            OmplPlanner(blocked, ExpertConfig.from_mapping(self.config)).path(blocked.observe(in_collision))

    def test_a_goal_in_collision_raises(self) -> None:
        goal_centers, _, _ = self.simulator.collision_spheres(self.path[-1])
        blocked = FrankaSystem(_table_config(obstacles=[{"position": goal_centers[-1].tolist(), "radius": 0.05}]))
        blocked.set_goal(self.simulator.goal)

        with self.assertRaises(PlannerSolveError):
            OmplPlanner(blocked, ExpertConfig.from_mapping(self.config)).path(blocked.observe(self.start))

    def test_no_path_within_the_timeout_raises(self) -> None:
        planner = OmplPlanner(self.simulator, ExpertConfig.from_mapping({**self.config, "ompl_timeout": 1e-9}))

        with self.assertRaisesRegex(PlannerSolveError, "no path"):
            planner.path(self.observation)


if __name__ == "__main__":
    unittest.main()


class OmplCandidatesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.observation = cls.simulator.observe(sample_initial_state(cls.simulator, 3))
        cls.start = cls.simulator.invert_obs(cls.observation)[:N]

    def _planner(self, candidates: int) -> OmplPlanner:
        return OmplPlanner(self.simulator, ExpertConfig.from_mapping({**self.config, "ompl_candidates": candidates}))

    def _turn(self, joint: int, angle: float) -> np.ndarray:
        """A path that turns one joint of the start configuration by ``angle``."""
        goal = self.start.copy()
        goal[joint] += angle
        return np.array([self.start, goal])

    def test_the_candidate_the_end_effector_travels_the_shortest_along_is_the_path(self) -> None:
        base, wrist, long_wrist = self._turn(0, 1.0), self._turn(6, 1.0), self._turn(6, 2.0)

        with mock.patch.object(OmplPlanner, "_connect", side_effect=[base.copy(), wrist.copy(), long_wrist.copy()]) as connect:
            path = self._planner(3).path(self.observation)

        self.assertEqual(connect.call_count, 3)
        np.testing.assert_allclose(path, wrist)

    def test_the_length_is_the_one_of_the_end_effector_and_not_of_the_joints(self) -> None:
        planner = self._planner(1)

        self.assertLess(planner._end_effector_length(self._turn(6, 1.0)), planner._end_effector_length(self._turn(0, 1.0)))

    def test_the_end_effector_length_of_a_straight_path_of_the_point_mass_is_its_euclidean_length(self) -> None:
        simulator = PointMassSystem(validate_system_config(raw_config={"system": "triple_integrator", "dimension": 2}))
        planner = OmplPlanner(simulator, ExpertConfig())

        length = planner._end_effector_length(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]))

        self.assertAlmostEqual(length, 7.0, places=9)

    def test_a_candidate_that_finds_no_path_is_left_out(self) -> None:
        candidate = self._turn(0, 1.0)

        with mock.patch.object(OmplPlanner, "_connect", side_effect=[PlannerSolveError("none"), candidate.copy(), PlannerSolveError("none")]):
            path = self._planner(3).path(self.observation)

        np.testing.assert_allclose(path, candidate)

    def test_no_candidate_with_a_path_raises(self) -> None:
        with mock.patch.object(OmplPlanner, "_connect", side_effect=PlannerSolveError("none")):
            with self.assertRaises(PlannerSolveError):
                self._planner(3).path(self.observation)

    def test_a_single_candidate_is_planned_once(self) -> None:
        with mock.patch.object(OmplPlanner, "_connect", side_effect=[self._turn(0, 1.0)]) as connect:
            self._planner(1).path(self.observation)

        self.assertEqual(connect.call_count, 1)

    def test_fewer_than_one_candidate_is_rejected(self) -> None:
        with self.assertRaises(ConfigurationError):
            ExpertConfig.from_mapping({**self.config, "ompl_candidates": 0})
