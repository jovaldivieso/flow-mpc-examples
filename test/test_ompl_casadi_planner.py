from __future__ import annotations

import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.utils import sample_initial_state
from planning.config import validate_system_config
from planning.ompl_casadi_planner import GOAL_BLEND_OUTER, OmplCasadiPlanner, path_weight
from planning.planner import Plan, Planner, PlannerSolveError
from planning.planner_config import ExpertConfig, ProjectionConfig
from planning.planner_factory import PlannerFactory
from systems.franka_system import FrankaSystem
from test.planner_fixtures import PINNED_EXPERT_SETTINGS, pinned_config, pose_goal, rest_state

N = 7
HORIZON = PINNED_EXPERT_SETTINGS["horizon"]
TABLE_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_table_config.yaml")
OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
]
STATISTICS = ("ompl_paths", "ompl_replans", "ompl_path_failures", "blocked_replans", "trimmed_labels", "goal_labels")


def _far_goal_configuration() -> np.ndarray:
    """A configuration far from the rest state (a path of more than a dozen rows) with the end effector in the workspace."""
    return np.array([0.8, -0.44, -0.78, -1.75, -1.32, 1.31, -0.26])


def _table_config(**overrides) -> dict:
    with open(TABLE_CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


def _expert(simulator: FrankaSystem, config: dict) -> OmplCasadiPlanner:
    return OmplCasadiPlanner(simulator, ExpertConfig.from_mapping(config), ProjectionConfig.from_mapping(config))


def _rollout(expert: OmplCasadiPlanner, simulator: FrankaSystem, state: np.ndarray, steps: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Close the loop with the first action of every plan until the goal is reached: the final state and every label."""
    labels = []
    state = simulator.reset(state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a fallback of the projection warns
        for _ in range(steps):
            plan = expert.plan(simulator.observe(state))
            labels.append(plan.positions)
            state = simulator.step(state, plan.actions[0])
            assert simulator.collision.is_collision_free(state[:N]), "the arm collided"
            if simulator.is_done(state):
                break
    return state, labels


class ExpertLabelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.simulator.set_goal(pose_goal(cls.simulator, _far_goal_configuration()))
        cls.state = rest_state(cls.simulator)
        cls.expert = _expert(cls.simulator, cls.config)
        cls.plan = cls.expert.plan(cls.simulator.observe(cls.state))
        cls.max_velocity = PINNED_EXPERT_SETTINGS["totg_velocity_scale"] * cls.simulator.state_upper_bounds[N : 2 * N]

    def test_the_expert_is_a_planner_the_factory_builds_under_its_name(self) -> None:
        self.assertIsInstance(self.expert, Planner)
        self.assertIsInstance(PlannerFactory.create("ompl_casadi", self.simulator, self.config), OmplCasadiPlanner)

    def test_the_plan_has_the_positions_of_the_horizon_and_the_actions_that_track_them(self) -> None:
        self.assertIsInstance(self.plan, Plan)
        self.assertEqual(self.plan.positions.shape, (HORIZON, N))
        self.assertEqual(self.plan.actions.shape, (HORIZON, self.simulator.nu))
        self.assertTrue(np.isfinite(self.plan.positions).all() and np.isfinite(self.plan.actions).all())
        self.assertEqual(self.plan.actions[0].shape, (self.simulator.nu,))

    def test_the_horizon_is_the_number_of_rows_of_the_label(self) -> None:
        config = pinned_config(horizon=7)
        expert = _expert(self.simulator, config)

        plan = expert.plan(self.simulator.observe(self.state))

        self.assertEqual((plan.positions.shape, plan.actions.shape), ((7, N), (7, self.simulator.nu)))

    def test_the_label_starts_near_the_arm_and_heads_to_the_goal(self) -> None:
        q = self.state[:N]
        reach = np.linalg.norm(self.max_velocity) * self.simulator.dt  # what the arm covers in a control period at most

        self.assertLessEqual(np.linalg.norm(self.plan.positions[0] - q), reach)
        start_error = self.simulator.goal_errors(self.state)[0]
        errors = [self.simulator.goal_errors(np.concatenate([row, np.zeros(2 * N)]))[0] for row in self.plan.positions]
        self.assertLess(errors[-1], start_error)
        self.assertLess(errors[-1], errors[0])

    def test_the_label_respects_the_velocity_limits(self) -> None:
        rows = np.vstack([self.state[:N], self.plan.positions])

        velocities = np.abs(np.diff(rows, axis=0)) / self.simulator.dt

        self.assertTrue(np.all(velocities <= self.max_velocity * 1.05))

    def test_the_label_speeds_up_smoothly_from_rest(self) -> None:
        rows = np.vstack([self.state[:N], self.plan.positions])
        steps = np.linalg.norm(np.diff(rows, axis=0), axis=1)

        self.assertLess(steps[0], steps[2])

    def test_the_actions_track_the_label_from_the_state(self) -> None:
        rolled = self.simulator.rollout_positions(self.state, self.plan.actions)[1:]

        np.testing.assert_allclose(rolled[:5], self.plan.positions[:5], atol=0.1)  # within what the jerk limit lets the arm follow

    def test_the_first_action_is_within_the_limits_of_the_plant(self) -> None:
        self.assertTrue(np.all(np.abs(self.plan.actions[0]) <= self.simulator.max_action + 1e-6))

    def test_the_label_is_the_same_from_the_same_state_with_a_new_expert(self) -> None:
        again = _expert(self.simulator, self.config).plan(self.simulator.observe(self.state))

        np.testing.assert_allclose(again.positions, self.plan.positions, atol=1e-9)

    def test_the_label_follows_the_path_ahead_of_the_arm_and_never_goes_back(self) -> None:
        expert = _expert(self.simulator, self.config)
        first = expert.plan(self.simulator.observe(self.state))
        ahead = np.concatenate([first.positions[4], np.zeros(2 * N)])
        behind = np.concatenate([first.positions[0], np.zeros(2 * N)])

        at_ahead = expert.plan(self.simulator.observe(ahead))
        at_behind = expert.plan(self.simulator.observe(behind))

        self.assertEqual(expert.statistics["ompl_paths"], 1)
        np.testing.assert_allclose(at_behind.positions, at_ahead.positions, atol=1e-9)

    def test_an_arm_that_is_already_at_the_goal_is_labelled_with_holding_still(self) -> None:
        q_goal = self.simulator.goal_configuration(self.simulator.goal, self.state[:N])
        state = np.concatenate([q_goal, np.zeros(2 * N)])
        expert = _expert(self.simulator, self.config)

        plan = expert.plan(self.simulator.observe(state))

        np.testing.assert_allclose(plan.positions, np.tile(q_goal, (HORIZON, 1)), atol=1e-6)
        np.testing.assert_allclose(plan.actions, 0.0, atol=1e-2)


class PathWeightTests(unittest.TestCase):
    def test_the_path_has_no_weight_in_the_goal_set_and_all_of_it_from_the_blend_outer(self) -> None:
        self.assertEqual(path_weight(0.0), 0.0)
        self.assertEqual(path_weight(1.0), 0.0)
        self.assertEqual(path_weight(GOAL_BLEND_OUTER), 1.0)
        self.assertEqual(path_weight(10.0 * GOAL_BLEND_OUTER), 1.0)
        self.assertAlmostEqual(path_weight(0.5 * (1.0 + GOAL_BLEND_OUTER)), 0.5)

    def test_it_rises_monotonically_with_no_slope_at_either_end(self) -> None:
        ratios = np.linspace(1.0, GOAL_BLEND_OUTER, 101)
        weights = np.array([path_weight(ratio) for ratio in ratios])

        self.assertTrue(np.all(np.diff(weights) > 0))
        step = 1e-4
        self.assertLess((path_weight(1.0 + step) - path_weight(1.0)) / step, 1e-2)
        self.assertLess((path_weight(GOAL_BLEND_OUTER) - path_weight(GOAL_BLEND_OUTER - step)) / step, 1e-2)


class ExpertGoalSetTests(unittest.TestCase):
    """The goal is a pose: the label is the path blended into the nearest configuration of the goal pose, the more the
    closer the pose is, wherever in the nullspace of the pose the arm is."""

    NULLSPACE_MOVE = 0.2  # rad of the joints that leaves the pose within the tolerances of the pinned settings

    def setUp(self) -> None:
        self.config = pinned_config()
        self.simulator = FrankaSystem(self.config)
        q = rest_state(self.simulator)[:N]
        self.simulator.set_goal(pose_goal(self.simulator, q))
        jacobian = self.simulator.goal_task_error(q, self.simulator.goal)[1]
        self.in_goal_set = np.concatenate([q + self.NULLSPACE_MOVE * np.linalg.svd(jacobian)[2][-1], np.zeros(2 * N)])
        self.expert = _expert(self.simulator, self.config)

    def _pose_error_of(self, ratio: float) -> np.ndarray:
        """A state of the goal set's configuration with the pose error moved to ``ratio`` goal sets, by moving the joints in the
        pose error alone."""
        q = self.in_goal_set[:N]
        jacobian = self.simulator.goal_task_error(q, self.simulator.goal)[1]
        direction = np.linalg.pinv(jacobian)[:, 0]
        scale = 0.01
        for _ in range(12):
            state = np.concatenate([q - scale * direction, np.zeros(2 * N)])
            scale *= ratio / self.simulator.goal_ratio(state)
        state = np.concatenate([q - scale * direction, np.zeros(2 * N)])
        self.assertAlmostEqual(self.simulator.goal_ratio(state), ratio, delta=0.05)
        return state

    def test_the_state_is_in_the_goal_set_in_another_configuration_than_the_one_the_goal_is_solved_in(self) -> None:
        q_goal = self.simulator.goal_configuration(self.simulator.goal, self.in_goal_set[:N])

        self.assertTrue(self.simulator.at_goal_pose(self.in_goal_set))
        self.assertGreater(np.linalg.norm(q_goal - self.in_goal_set[:N]), 0.5 * self.NULLSPACE_MOVE)

    def test_a_state_in_the_goal_set_is_labelled_with_the_nearest_configuration_of_the_goal_pose_and_plans_no_path(self) -> None:
        state = self._pose_error_of(0.8)
        q = state[:N]
        error, jacobian, _ = self.simulator.goal_task_error(q, self.simulator.goal)

        plan = self.expert.plan(self.simulator.observe(state))

        np.testing.assert_allclose(plan.positions, np.tile(q + self.simulator.goal_step(error, jacobian), (HORIZON, 1)), atol=1e-9)
        self.assertEqual((self.expert.statistics["goal_labels"], self.expert.statistics["ompl_paths"]), (1, 0))

    def test_the_label_in_the_goal_set_moves_the_arm_in_the_pose_error_alone_and_closes_it(self) -> None:
        state = self._pose_error_of(0.8)
        q = state[:N]
        jacobian = self.simulator.goal_task_error(q, self.simulator.goal)[1]

        move = self.expert.plan(self.simulator.observe(state)).positions[0] - q

        self.assertLess(abs(np.linalg.svd(jacobian)[2][-1] @ move), 1e-9)
        self.assertLess(self.simulator.goal_ratio(np.concatenate([q + move, np.zeros(2 * N)])), 0.1 * self.simulator.goal_ratio(state))

    def test_a_moving_arm_in_the_goal_set_is_labelled_alike(self) -> None:
        state = self._pose_error_of(0.8)
        moving = state.copy()
        moving[N : 2 * N] = 0.1

        at_rest = self.expert.plan(self.simulator.observe(state)).positions
        in_motion = self.expert.plan(self.simulator.observe(moving)).positions

        np.testing.assert_allclose(in_motion, at_rest, atol=1e-9)

    def test_a_state_beyond_the_blend_is_labelled_with_the_path_alone(self) -> None:
        state = self._pose_error_of(GOAL_BLEND_OUTER + 1.0)

        plan = self.expert.plan(self.simulator.observe(state))

        self.assertEqual((self.expert.statistics["goal_labels"], self.expert.statistics["ompl_paths"]), (0, 1))
        np.testing.assert_allclose(plan.positions, self.expert._progress.reference(state[:N], HORIZON, self.simulator.dt), atol=1e-9)

    def test_a_state_in_the_blend_is_labelled_with_the_weighted_mean_of_the_path_and_the_goal_pose_configuration(self) -> None:
        state = self._pose_error_of(0.5 * (1.0 + GOAL_BLEND_OUTER))
        q = state[:N]

        plan = self.expert.plan(self.simulator.observe(state))

        path = self.expert._progress.reference(q, HORIZON, self.simulator.dt)
        weight = path_weight(self.simulator.goal_ratio(state))
        self.assertGreater(weight, 0.0)
        self.assertLess(weight, 1.0)
        np.testing.assert_allclose(plan.positions, weight * path + (1.0 - weight) * self.expert._goal_pose_configuration(q), atol=1e-9)
        self.assertEqual((self.expert.statistics["goal_labels"], self.expert.statistics["ompl_paths"]), (1, 1))

    def test_the_label_is_continuous_across_the_edge_of_the_goal_set(self) -> None:
        inside, outside = self._pose_error_of(0.99), self._pose_error_of(1.01)

        label_inside = self.expert.plan(self.simulator.observe(inside)).positions
        label_outside = self.expert.plan(self.simulator.observe(outside)).positions

        self.assertLess(np.abs(label_outside - label_inside).max(), 0.05)


class ExpertBlockedPathTests(unittest.TestCase):
    """A path whose first row is inside an obstacle holds the arm still: it is planned again from where the arm is."""

    def setUp(self) -> None:
        self.config = pinned_config()
        self.simulator = FrankaSystem(self.config)
        q = rest_state(self.simulator)[:N]
        self.simulator.set_goal(pose_goal(self.simulator, q))
        self.state = np.concatenate([q + 0.5, np.zeros(2 * N)])
        self.expert = _expert(self.simulator, self.config)
        self.expert.plan(self.simulator.observe(self.state))

    def _plan_with_the_first_row_blocked(self) -> Plan:
        real = self.simulator.collision.free_prefix
        outcomes = [0]
        with mock.patch.object(self.simulator.collision, "free_prefix", lambda rows: outcomes.pop() if outcomes else real(rows)):
            return self.expert.plan(self.simulator.observe(self.state))

    def test_a_blocked_first_row_plans_a_new_path(self) -> None:
        self._plan_with_the_first_row_blocked()

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["blocked_replans"]), (2, 1))

    def test_the_old_path_is_kept_if_there_is_no_new_one(self) -> None:
        old = self.expert._progress
        with mock.patch.object(self.expert._ompl, "path", side_effect=PlannerSolveError("no path")):
            self._plan_with_the_first_row_blocked()

        self.assertIs(self.expert._progress, old)
        self.assertEqual((self.expert.statistics["ompl_path_failures"], self.expert.statistics["blocked_replans"]), (1, 0))

    def test_a_free_first_row_keeps_the_path(self) -> None:
        self.expert.plan(self.simulator.observe(self.state))

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["blocked_replans"]), (1, 0))


class ExpertAmongObstaclesTests(unittest.TestCase):
    def test_every_label_row_is_collision_free_on_the_table_scene(self) -> None:
        config = _table_config()
        simulator = FrankaSystem(config)
        for seed in (0, 1):
            expert = _expert(simulator, config)
            state = sample_initial_state(simulator, seed)
            with self.subTest(seed=seed), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for _ in range(6):
                    plan = expert.plan(simulator.observe(state))
                    self.assertTrue(simulator.collision.trajectory_is_collision_free(plan.positions))
                    state = simulator.step(state, plan.actions[0])

    def test_a_rollout_from_rest_reaches_the_goal_among_obstacles_without_collision(self) -> None:
        config = pinned_config(obstacles=OBSTACLES)
        simulator = FrankaSystem(config)
        simulator.set_goal(pose_goal(simulator, _far_goal_configuration()))
        expert = _expert(simulator, config)

        state, labels = _rollout(expert, simulator, rest_state(simulator), steps=150)

        self.assertTrue(simulator.is_done(state), simulator.goal_errors(state))
        self.assertEqual(expert.statistics["ompl_replans"], 0)
        for label in labels:
            self.assertTrue(simulator.collision.trajectory_is_collision_free(label))

    def test_a_rollout_across_the_table_reaches_the_goal_without_collision(self) -> None:
        config = _table_config()
        simulator = FrankaSystem(config)
        state = sample_initial_state(simulator, 0)
        expert = _expert(simulator, config)

        state, _ = _rollout(expert, simulator, state, steps=100)

        self.assertTrue(simulator.is_done(state), simulator.goal_errors(state))


class ExpertReplanningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = pinned_config()
        self.simulator = FrankaSystem(self.config)
        self.simulator.set_goal(pose_goal(self.simulator, _far_goal_configuration()))
        self.state = rest_state(self.simulator)
        self.expert = _expert(self.simulator, self.config)

    def test_a_rollout_that_follows_the_path_plans_it_once(self) -> None:
        state, _ = _rollout(self.expert, self.simulator, self.state, steps=150)

        self.assertTrue(self.simulator.is_done(state))
        self.assertEqual(self.expert.statistics["ompl_paths"], 1)
        self.assertEqual(self.expert.statistics["ompl_replans"], 0)

    def test_an_arm_near_the_path_does_not_replan(self) -> None:
        self.expert.plan(self.simulator.observe(self.state))
        nudged = self.state.copy()
        nudged[0] += 0.5 * self.config["replan_distance"]

        self.expert.plan(self.simulator.observe(nudged))

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["ompl_replans"]), (1, 0))

    def test_a_new_path_is_planned_when_the_arm_is_pushed_far_from_the_path(self) -> None:
        self.expert.plan(self.simulator.observe(self.state))
        pushed = self.state.copy()
        pushed[2] += 2.0 * self.config["replan_distance"]
        pushed[4] -= 0.6 * self.config["replan_distance"]

        plan = self.expert.plan(self.simulator.observe(pushed))

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["ompl_replans"]), (2, 1))
        reach = np.linalg.norm(PINNED_EXPERT_SETTINGS["totg_velocity_scale"] * self.simulator.state_upper_bounds[N : 2 * N]) * self.simulator.dt
        self.assertLessEqual(np.linalg.norm(plan.positions[0] - pushed[:N]), reach)

    def test_reset_forgets_the_path_and_the_next_plan_is_not_a_replan(self) -> None:
        self.expert.plan(self.simulator.observe(self.state))

        self.expert.reset()
        self.expert.plan(self.simulator.observe(self.state))

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["ompl_replans"]), (2, 0))


class ExpertTrimmingTests(unittest.TestCase):
    def setUp(self) -> None:
        # an obstacle far from the arm: the scene has obstacles, so the tracker has its half-spaces
        self.config = pinned_config(obstacles=[{"position": [0.0, -0.9, 0.9], "radius": 0.02}])
        self.simulator = FrankaSystem(self.config)
        self.simulator.set_goal(pose_goal(self.simulator, _far_goal_configuration()))
        self.state = rest_state(self.simulator)
        self.expert = _expert(self.simulator, self.config)

    def _obstacle_on_row(self, label: np.ndarray, row: int) -> int:
        """Put a small obstacle on the bounding sphere that moves most between ``row`` and the ends of ``label``."""
        centers = np.array([self.simulator.collision_spheres(q)[0] for q in label])
        motion = np.minimum(np.linalg.norm(centers[row] - centers[0], axis=1), np.linalg.norm(centers[row] - centers[-1], axis=1))
        sphere = int(np.argmax(motion))
        self.simulator.collision.obstacles[:] = [(centers[row][sphere], 0.02)]
        return self.simulator.collision.free_prefix(label)

    def test_a_free_reference_is_not_trimmed(self) -> None:
        reference = self.expert.plan(self.simulator.observe(self.state)).positions

        trimmed = self.expert._trimmed(reference, self.state[:N])

        np.testing.assert_array_equal(trimmed, reference)
        self.assertEqual(self.expert.statistics["trimmed_labels"], 0)

    def test_the_rows_from_the_first_penetration_hold_the_last_free_row(self) -> None:
        reference = self.expert.plan(self.simulator.observe(self.state)).positions
        free = self._obstacle_on_row(reference, 10)
        self.assertTrue(0 < free < len(reference))

        trimmed = self.expert._trimmed(reference, self.state[:N])

        np.testing.assert_array_equal(trimmed[:free], reference[:free])
        np.testing.assert_array_equal(trimmed[free:], np.tile(reference[free - 1], (len(reference) - free, 1)))
        self.assertTrue(self.simulator.collision.trajectory_is_collision_free(trimmed))
        self.assertEqual(self.expert.statistics["trimmed_labels"], 1)

    def test_a_reference_that_penetrates_from_its_first_row_holds_the_current_configuration(self) -> None:
        reference = self.expert.plan(self.simulator.observe(self.state)).positions
        centers = self.simulator.collision_spheres(reference[0])[0]
        self.simulator.collision.obstacles[:] = [(centers[-1], 0.02)]
        self.assertEqual(self.simulator.collision.free_prefix(reference), 0)

        trimmed = self.expert._trimmed(reference, self.state[:N])

        np.testing.assert_array_equal(trimmed, np.tile(self.state[:N], (len(reference), 1)))

    def test_a_plan_is_trimmed_where_an_obstacle_lies_on_the_path(self) -> None:
        reference = self.expert.plan(self.simulator.observe(self.state)).positions
        free = self._obstacle_on_row(reference, 10)
        self.assertTrue(0 < free < len(reference))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plan = self.expert.plan(self.simulator.observe(self.state))

        np.testing.assert_allclose(plan.positions[:free], reference[:free], atol=1e-9)
        np.testing.assert_allclose(plan.positions[free:], np.tile(reference[free - 1], (len(reference) - free, 1)), atol=1e-9)
        self.assertTrue(self.simulator.collision.trajectory_is_collision_free(plan.positions))
        self.assertEqual(self.expert.statistics["trimmed_labels"], 1)


class ExpertFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = pinned_config()
        self.simulator = FrankaSystem(self.config)
        self.simulator.set_goal(pose_goal(self.simulator, _far_goal_configuration()))
        self.state = rest_state(self.simulator)
        self.expert = _expert(self.simulator, self.config)

    def test_a_goal_that_no_configuration_reaches_raises_and_is_counted(self) -> None:
        goal = self.simulator.goal.copy()
        goal[:3] = [5.0, 0.0, 0.0]
        self.simulator.set_goal(goal)

        with self.assertRaises(PlannerSolveError):
            self.expert.plan(self.simulator.observe(self.state))
        with self.assertRaises(PlannerSolveError):  # the path is planned again at the next call
            self.expert.plan(self.simulator.observe(self.state))

        self.assertEqual((self.expert.statistics["ompl_paths"], self.expert.statistics["ompl_path_failures"]), (0, 2))

    def test_a_start_in_collision_raises(self) -> None:
        centers = self.simulator.collision_spheres(self.state[:N])[0]
        self.simulator.collision.obstacles[:] = [(centers[-1], 0.02)]

        with self.assertRaises(PlannerSolveError):
            self.expert.plan(self.simulator.observe(self.state))

        self.assertEqual(self.expert.statistics["ompl_path_failures"], 1)

    def test_a_timeout_without_a_path_raises(self) -> None:
        config = pinned_config(ompl_timeout=1e-9, obstacles=OBSTACLES)
        simulator = FrankaSystem(config)
        simulator.set_goal(pose_goal(simulator, _far_goal_configuration()))
        expert = _expert(simulator, config)

        with self.assertRaises(PlannerSolveError):
            expert.plan(simulator.observe(self.state))

        self.assertEqual(expert.statistics["ompl_path_failures"], 1)

    def test_a_plan_the_arm_cannot_track_safely_raises(self) -> None:
        def cannot_track(state, reference):
            raise PlannerSolveError("no safe plan")

        self.expert._tracker.project = cannot_track

        with self.assertRaises(PlannerSolveError):
            self.expert.plan(self.simulator.observe(self.state))


class ExpertStatisticsTests(unittest.TestCase):
    def test_the_counters_start_at_zero(self) -> None:
        config = pinned_config()

        statistics = _expert(FrankaSystem(config), config).statistics

        self.assertEqual(statistics, {name: 0 for name in STATISTICS})

    def test_the_counters_are_cumulative_over_episodes(self) -> None:
        config = pinned_config()
        simulator = FrankaSystem(config)
        simulator.set_goal(pose_goal(simulator, _far_goal_configuration()))
        expert = _expert(simulator, config)
        observation = simulator.observe(rest_state(simulator))

        for episode in range(1, 4):
            expert.reset()
            expert.plan(observation)
            expert.plan(observation)  # the same episode: the path is kept

            self.assertEqual(expert.statistics["ompl_paths"], episode)

        self.assertEqual(expert.statistics["ompl_replans"], 0)


if __name__ == "__main__":
    unittest.main()
