import copy
import os
import sys
import unittest
from unittest import mock

import casadi as ca
import numpy as np
import pinocchio as pin
import yaml

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import torch

from planning.config import ConfigurationError, load_and_validate_system_config, validate_system_config
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from systems.franka_system import FLOOR_CLEARANCE, Q_READY, SAMPLING_RADIUS, WORKSPACE_BOUNDS, FrankaSystem
from systems.triple_integrator_system import TripleIntegratorSystem
from learning.models.policy_factory import PolicyFactory
from planning.planner_factory import PlannerFactory
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")
CHUNK = 16  # actions per state the policy predicts

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_casadi_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "pos_tol": 0.002,
    "vel_tol": 0.02,
}


def _build_simulator(**overrides) -> FrankaSystem:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return FrankaSystem(validate_system_config(raw_config=raw_config))


def _rest_state(simulator: FrankaSystem) -> np.ndarray:
    """Ready pose at rest."""
    return np.concatenate([Q_READY, np.zeros(2 * simulator.n)])


class SimulatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = _build_simulator()

    def test_satisfies_the_dynamics_hierarchy(self) -> None:
        self.assertIsInstance(self.simulator, SystemSimulator)
        self.assertIsInstance(self.simulator, TripleIntegratorSystem)

    def test_dimensions_and_rest_state_indices(self) -> None:
        simulator = self.simulator
        # observation = goal pose (12) + state (21); the policy sees the goal in the TCP frame (12) + q_rel (7) = 19
        self.assertEqual((simulator.nx, simulator.nu, simulator.obs_dim, simulator.goal_dim), (21, 7, 33, 12))
        self.assertEqual(simulator.environment_dim, 19)
        self.assertEqual(simulator.max_action.shape, (simulator.nu,))
        self.assertEqual(simulator.state_lower_bounds.shape, (simulator.nx,))
        # state = [q, v, a]: the projector's terminal come-to-rest cost reads the velocities and accelerations
        self.assertEqual(simulator.rest_state_indices, tuple(range(7, 21)))
        self.assertEqual(simulator.model.nv, 7)

    def test_predict_next_state_is_side_effect_free(self) -> None:
        simulator = self.simulator
        state = simulator.reset(simulator.random_initial_state(np.random.default_rng(11)))
        action = np.zeros(int(simulator.nu), dtype=float)
        time_before = simulator.time
        stored_state_before = simulator.state.copy()

        predicted = simulator.predict_next_state(state, action)

        self.assertEqual(simulator.time, time_before)
        np.testing.assert_array_equal(simulator.state, stored_state_before)
        np.testing.assert_allclose(predicted, simulator.step(state, action))

    def test_dataset_schema_matches_formatted_frames(self) -> None:
        simulator = self.simulator
        rng = np.random.default_rng(7)
        simulator.randomize_goal_for_reset(rng)
        state = simulator.reset(simulator.random_initial_state(rng))
        observation = simulator.observe(state)
        plan = np.linspace(-1.0, 1.0, CHUNK * simulator.n).reshape(CHUNK, simulator.n)

        features = simulator.get_dataset_features(CHUNK)
        frame = simulator.format_dataset_frame(observation, plan)

        self.assertEqual(set(frame), set(features))
        self.assertEqual(set(features), {"observation.environment_state", "observation.state", "action"})
        for name, feature in features.items():
            self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
            self.assertEqual(str(frame[name].dtype), feature["dtype"], name)
            self.assertEqual(len(feature["names"]), feature["shape"][0], name)

        q = state[: simulator.n]
        relative = simulator.tcp_placement(q).actInv(simulator.vector_pose(simulator.goal))
        q_rel = simulator.goal_configuration(simulator.goal, q) - q
        np.testing.assert_allclose(
            frame["observation.environment_state"],
            np.concatenate([simulator.pose_vector(relative), q_rel]),
            atol=1e-6,
        )
        np.testing.assert_allclose(frame["observation.state"], state, atol=1e-6)
        # The dataset action is the expert's plan, one joint configuration per row, flattened row by row
        self.assertEqual(features["action"]["shape"], (CHUNK * simulator.n,))
        np.testing.assert_allclose(frame["action"], plan.reshape(-1), atol=1e-6)
        self.assertEqual(features["action"]["names"][simulator.n].endswith("[t+1]"), True)

        # The policy sees exactly the two observation keys of the dataset, as float32
        policy_observation = simulator.policy_observation(observation)
        self.assertEqual(set(policy_observation), {"observation.environment_state", "observation.state"})
        for name, value in policy_observation.items():
            self.assertEqual(value.dtype, np.float32, name)
            np.testing.assert_array_equal(value, frame[name])

    def test_observe_stacks_goal_and_state_and_invert_obs_recovers_the_state(self) -> None:
        simulator = self.simulator
        state = _rest_state(simulator)
        observation = simulator.observe(state)

        self.assertEqual(observation.shape, (simulator.obs_dim,))
        np.testing.assert_array_equal(observation[: simulator.goal_dim], simulator.goal)
        np.testing.assert_array_equal(observation[simulator.goal_dim :], state)
        np.testing.assert_array_equal(simulator.invert_obs(observation), state)

    def test_the_policy_sees_the_goal_in_the_tcp_frame(self) -> None:
        simulator = _build_simulator()
        rng = np.random.default_rng(4)
        simulator.randomize_goal_for_reset(rng)
        state = simulator.reset(simulator.random_initial_state(rng))
        placement = simulator.tcp_placement(state[: simulator.n])
        goal = simulator.vector_pose(simulator.goal)

        seen = simulator.policy_observation(simulator.observe(state))["observation.environment_state"].astype(float)

        np.testing.assert_allclose(seen[:3], placement.rotation.T @ (goal.translation - placement.translation), atol=1e-6)
        np.testing.assert_allclose(seen[3:12].reshape(3, 3), placement.rotation.T @ goal.rotation, atol=1e-6)
        np.testing.assert_allclose(
            seen[12:], simulator.goal_configuration(simulator.goal, state[: simulator.n]) - state[: simulator.n], atol=1e-6
        )
        # the goal comes back from what the policy saw and the state
        rebuilt = simulator.observation_from_policy(seen, state[:])
        np.testing.assert_allclose(rebuilt[: simulator.goal_dim], simulator.goal, atol=1e-5)
        np.testing.assert_array_equal(rebuilt[simulator.goal_dim :], state)

    def test_a_reached_goal_is_the_identity_in_the_tcp_frame(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))

        seen = simulator.policy_observation(simulator.observe(state))["observation.environment_state"]

        np.testing.assert_allclose(
            seen, np.concatenate([np.zeros(3), np.eye(3).ravel(), np.zeros(simulator.n)]), atol=1e-6
        )

    def test_set_goal_rejects_a_wrong_shape_and_copies_its_input(self) -> None:
        simulator = _build_simulator()
        with self.assertRaises(ValueError):
            simulator.set_goal(np.zeros(7))
        goal = simulator.goal.copy()
        goal[0] += 0.1
        simulator.set_goal(goal)
        goal[0] = -5.0
        self.assertNotEqual(simulator.goal[0], -5.0)

    def test_randomize_goal_contract(self) -> None:
        self.assertIsInstance(self.simulator.randomize_goal, bool)
        self.assertTrue(callable(getattr(self.simulator, "randomize_goal_for_reset", None)))
        self.assertFalse(_build_simulator(goal=list(pose_goal(self.simulator, Q_READY))).randomize_goal)


class DynamicsModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = _build_simulator()

    def test_casadi_dynamics_matches_the_plant_inside_all_limits(self) -> None:
        simulator = self.simulator
        x_sym, u_sym = ca.SX.sym("x", simulator.nx), ca.SX.sym("u", simulator.nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [simulator.casadi_dynamics(x_sym, u_sym)])
        state = _rest_state(simulator)
        state[simulator.n : 2 * simulator.n] = 0.05 * np.linspace(-1.0, 1.0, simulator.n)
        state[2 * simulator.n :] = 0.5 * np.linspace(1.0, -1.0, simulator.n)
        action = np.random.default_rng(1).uniform(-2.0, 2.0, simulator.nu)

        np.testing.assert_allclose(
            np.array(step_fn(state, action)).ravel(), simulator.predict_next_state(state, action), atol=1e-9
        )

    def test_zero_jerk_keeps_the_arm_at_rest(self) -> None:
        simulator = self.simulator
        state = _rest_state(simulator)
        np.testing.assert_allclose(simulator.predict_next_state(state, np.zeros(simulator.nu)), state, atol=1e-9)

    def test_plant_clips_the_commanded_jerk_to_max_action(self) -> None:
        simulator = self.simulator
        state = _rest_state(simulator)
        np.testing.assert_allclose(
            simulator.predict_next_state(state, np.full(simulator.nu, 1e3)),
            simulator.predict_next_state(state, simulator.max_action),
        )

    def test_plant_is_a_triple_integrator_with_constant_jerk_inside_all_limits(self) -> None:
        # Masses and torques are not modelled: a joint follows its commanded jerk exactly
        simulator = self.simulator
        n, dt = simulator.n, simulator.dt
        state = _rest_state(simulator)
        state[n : 2 * n] = 0.2 * np.linspace(-1.0, 1.0, n)
        state[2 * n :] = 1.0 * np.linspace(1.0, -1.0, n)
        jerk = np.linspace(-3.0, 3.0, simulator.nu)
        q, v, a = state[:n], state[n : 2 * n], state[2 * n :]
        expected = np.concatenate(
            [q + v * dt + a * dt**2 / 2 + jerk * dt**3 / 6, v + a * dt + jerk * dt**2 / 2, a + jerk * dt]
        )
        np.testing.assert_allclose(simulator.predict_next_state(state, jerk), expected, atol=1e-12)

    def test_plant_stops_at_position_limits_and_saturates_velocity(self) -> None:
        simulator = self.simulator
        state = _rest_state(simulator)
        state[0] = simulator.state_upper_bounds[0] - 1e-4
        state[simulator.n] = simulator.state_upper_bounds[simulator.n]  # joint 1 at its speed limit, moving up
        next_state = simulator.predict_next_state(state, np.zeros(simulator.nu))
        self.assertEqual(next_state[0], simulator.state_upper_bounds[0])
        self.assertEqual(next_state[simulator.n], 0.0)

        state = _rest_state(simulator)
        state[simulator.n + 4] = simulator.state_upper_bounds[simulator.n + 4] - 1e-3
        next_state = simulator.predict_next_state(state, np.full(simulator.nu, simulator.max_action.max()))
        self.assertTrue(np.all(next_state[simulator.n :] <= simulator.state_upper_bounds[simulator.n :] + 1e-12))

    def test_goal_errors_and_is_done_need_the_goal_and_rest(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))

        position_error, rotation_error = simulator.goal_errors(state)
        self.assertAlmostEqual(position_error, 0.0, places=9)
        self.assertAlmostEqual(rotation_error, 0.0, places=9)
        self.assertTrue(simulator.is_done(state))

        # only the position counts: a turned wrist at the goal position is done, and is reported as turned
        turned = state.copy()
        turned[6] += 0.5
        simulator.set_goal(np.concatenate([simulator.tcp_position(state[: simulator.n]), simulator.goal[3:]]))
        self.assertGreater(simulator.goal_errors(turned)[1], 0.1)
        self.assertLess(simulator.goal_errors(turned)[0], simulator.pos_tol)
        self.assertTrue(simulator.is_done(turned))
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))

        moving = state.copy()
        moving[simulator.n] = 10 * simulator.vel_tol
        self.assertFalse(simulator.is_done(moving))
        displaced = state.copy()
        displaced[0] += 0.05
        self.assertFalse(simulator.is_done(displaced))
        self.assertGreater(simulator.goal_errors(displaced)[0], simulator.pos_tol)


class RolloutTerminationTests(unittest.TestCase):
    def test_rollout_termination_api_exists(self) -> None:
        simulator = _build_simulator()
        for name in ("reset_rollout_termination", "set_early_rollout_termination", "should_terminate_rollout"):
            self.assertTrue(callable(getattr(simulator, name, None)), name)

    def test_rollout_termination_state_resets_and_honors_hold_steps(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        self.assertTrue(simulator.is_done(state), "a rest state at the goal must satisfy is_done()")

        hold_steps = int(simulator.config.get("done_hold_steps", 5))
        self.assertGreaterEqual(hold_steps, 1)

        simulator.reset(state)
        for step_idx in range(hold_steps - 1):
            self.assertFalse(
                simulator.should_terminate_rollout(state),
                f"termination triggered too early on step {step_idx + 1}",
            )

        self.assertTrue(
            simulator.should_terminate_rollout(state),
            f"termination did not trigger on hold step {hold_steps}",
        )

        simulator.reset(state)
        self.assertFalse(simulator.should_terminate_rollout(state), "termination counter was not reset by reset()")

    def test_disabling_early_termination_never_terminates(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        simulator.set_early_rollout_termination(False)
        for _ in range(20):
            self.assertFalse(simulator.should_terminate_rollout(state))

    def test_simulate_stops_after_the_hold_steps_and_returns_matching_arrays(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        observations = []

        def policy(observation: np.ndarray) -> np.ndarray:
            observations.append(observation)
            return np.zeros(simulator.nu)

        trajectory = simulator.simulate(state, policy, num_steps=50)
        hold_steps = int(simulator.config.get("done_hold_steps", 5))
        self.assertEqual(len(trajectory["actions"]), hold_steps)
        self.assertEqual(trajectory["states"].shape, (hold_steps, simulator.nx))
        # The policy is handed the observation, not the state
        self.assertEqual(observations[0].shape, (simulator.obs_dim,))


class SamplingTests(unittest.TestCase):
    def test_seeded_reset_random_is_deterministic_and_inside_joint_limits(self) -> None:
        simulator_a = _build_simulator(initial_state_seed=13)
        simulator_b = _build_simulator(initial_state_seed=13)

        state_a = simulator_a.reset_random()
        state_b = simulator_b.reset_random()

        np.testing.assert_allclose(state_a, state_b, atol=1e-12)
        np.testing.assert_allclose(simulator_a.goal, simulator_b.goal, atol=1e-12)
        lower, upper = simulator_a.model.lowerPositionLimit, simulator_a.model.upperPositionLimit
        self.assertTrue(np.all(state_a[: simulator_a.n] >= lower) and np.all(state_a[: simulator_a.n] <= upper))

    def test_random_goals_are_reachable_poses_and_their_configuration_reproduces_them(self) -> None:
        simulator = _build_simulator()
        rng = np.random.default_rng(5)
        for _ in range(10):
            simulator.randomize_goal_for_reset(rng)
            rotation = simulator.goal[3:].reshape(3, 3)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)

            q_start_a = simulator.random_initial_state(rng)[: simulator.n]
            q_start_b = simulator.random_initial_state(rng)[: simulator.n]
            q_goal = simulator.goal_configuration(simulator.goal, q_start_a)
            np.testing.assert_allclose(q_goal, simulator.goal_configuration(simulator.goal, q_start_b), atol=1e-12)

            np.testing.assert_allclose(pose_goal(simulator, q_goal), simulator.goal, atol=1e-5)
            self.assertTrue(np.all(q_goal >= simulator.model.lowerPositionLimit))
            self.assertTrue(np.all(q_goal <= simulator.model.upperPositionLimit))

    def test_the_casadi_forward_kinematics_matches_pinocchio(self) -> None:
        simulator = _build_simulator()
        rng = np.random.default_rng(2)
        for _ in range(20):
            q = simulator._sample_configuration(rng)
            position, rotation = simulator.kinematics.fk(q)

            placement = simulator.tcp_placement(q)
            np.testing.assert_allclose(np.array(position).ravel(), placement.translation, atol=1e-12)
            np.testing.assert_allclose(np.array(rotation), placement.rotation, atol=1e-12)

    def test_the_kinematics_are_built_once_and_shared_by_every_simulator_and_copy(self) -> None:
        self.assertIs(_build_simulator().kinematics, _build_simulator().kinematics)
        simulator = _build_simulator()
        self.assertIs(copy.deepcopy(simulator).kinematics, simulator.kinematics)

    def test_a_goal_pose_out_of_reach_has_no_configuration(self) -> None:
        simulator = _build_simulator()
        goal = simulator.goal.copy()
        goal[:3] = [5.0, 0.0, 0.0]
        with self.assertRaises(RuntimeError):
            simulator.goal_configuration(goal, Q_READY)

    def test_random_initial_state_does_not_change_the_goal(self) -> None:
        simulator = _build_simulator()
        goal_before = simulator.goal.copy()
        simulator.random_initial_state(np.random.default_rng(3))
        np.testing.assert_array_equal(simulator.goal, goal_before)

    def test_fixed_goal_is_kept_by_randomize_goal_for_reset(self) -> None:
        simulator = _build_simulator(goal=list(pose_goal(_build_simulator(), Q_READY)))
        goal_before = simulator.goal.copy()
        simulator.randomize_goal_for_reset(np.random.default_rng(3))
        np.testing.assert_array_equal(simulator.goal, goal_before)

    def test_sampled_start_is_a_safe_rest_state_away_from_the_goal(self) -> None:
        simulator = _build_simulator()
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = WORKSPACE_BOUNDS
        for seed in range(25):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            state = simulator.random_initial_state(rng)
            q = state[: simulator.n]
            tcp = simulator.tcp_position(q)

            np.testing.assert_array_equal(state[simulator.n :], 0.0)
            self.assertTrue(np.all(state >= simulator.state_lower_bounds - 1e-12))
            self.assertTrue(np.all(state <= simulator.state_upper_bounds + 1e-12))
            self.assertTrue(np.all(np.abs(q - Q_READY) <= SAMPLING_RADIUS + 1e-12))
            self.assertGreater(simulator.goal_errors(state)[0], simulator.min_start_goal_distance)
            for point in (simulator.goal[:3], tcp):
                self.assertTrue(x_min <= point[0] <= x_max and y_min <= point[1] <= y_max and z_min <= point[2] <= z_max)
            self.assertTrue(all(joint.translation[2] >= FLOOR_CLEARANCE for joint in simulator.data.oMi[1:]))

    def test_infeasible_min_start_goal_distance_raises_instead_of_looping_forever(self) -> None:
        simulator = _build_simulator(min_start_goal_distance=100.0)
        with self.assertRaises(RuntimeError):
            simulator.random_initial_state(np.random.default_rng(0))


POINT_MASS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "triple_integrator_casadi_config.yaml")
# The tests pin the dimension they assume, so scaling the shipped config up does not break them
POINT_MASS_SETTINGS = {"dimension": 2, "Q_diag": [20.0, 20.0, 2.0, 2.0, 0.2, 0.2], "R_diag": [1.0e-2, 1.0e-2]}


def _point_mass_config(**overrides) -> dict:
    with open(POINT_MASS_CONFIG_PATH) as f:
        return validate_system_config({**yaml.safe_load(f), **POINT_MASS_SETTINGS, **overrides})


class TripleIntegratorSystemTests(unittest.TestCase):
    """The toy system: the same plant as the arm's joints, the same pipeline hooks, a position as the goal."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _point_mass_config()
        cls.simulator = SystemFactory.create(cls.config["system"], cls.config)

    def test_the_factory_builds_it_from_the_config_and_it_satisfies_the_base_class(self) -> None:
        self.assertIsInstance(self.simulator, TripleIntegratorSystem)
        self.assertIsInstance(self.simulator, SystemSimulator)
        self.assertEqual(
            (self.simulator.nx, self.simulator.nu, self.simulator.obs_dim, self.simulator.goal_dim), (6, 2, 8, 2)
        )
        with self.assertRaises(ValueError):
            SystemFactory.create("pendulum", self.config)

    def test_the_plant_is_the_triple_integrator_the_planner_models(self) -> None:
        simulator, n = self.simulator, 2
        x_sym, u_sym = ca.SX.sym("x", simulator.nx), ca.SX.sym("u", simulator.nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [simulator.casadi_dynamics(x_sym, u_sym)])
        state = np.array([0.1, -0.2, 0.3, -0.1, 0.5, -0.4])
        jerk = np.array([2.0, -3.0])

        np.testing.assert_allclose(np.array(step_fn(state, jerk)).ravel(), simulator.predict_next_state(state, jerk), atol=1e-12)
        # the positions of a rollout are those of the plant stepping through the actions
        actions = np.array([[2.0, -3.0], [0.0, 1.0], [-1.0, 0.5]])
        positions = simulator.rollout_positions(state, actions)
        x = state
        for k, action in enumerate(actions):
            x = simulator.predict_next_state(x, action)
            np.testing.assert_allclose(positions[k + 1], x[:n])
        np.testing.assert_allclose(positions[0], state[:n])

    def test_the_viewers_draw_the_position_padded_to_three_axes(self) -> None:
        np.testing.assert_array_equal(self.simulator.tcp_position(np.array([0.5, -0.25])), [0.5, -0.25, 0.0])

    def test_the_plant_saturates_the_jerk_and_has_no_position_limit(self) -> None:
        simulator = self.simulator
        rest = np.zeros(simulator.nx)
        np.testing.assert_allclose(
            simulator.predict_next_state(rest, np.full(2, 1e3)), simulator.predict_next_state(rest, simulator.max_action)
        )
        self.assertTrue(np.all(np.isinf(simulator.state_upper_bounds[:2])))
        far = np.array([100.0, -100.0, 0.5, -0.5, 0.0, 0.0])
        moved = simulator.predict_next_state(far, np.zeros(2))
        self.assertTrue(np.all(moved[:2] != far[:2]))  # it keeps moving at its velocity, nowhere to stop

    def test_the_policy_sees_the_goal_relative_to_the_point_and_nothing_in_world_coordinates(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        state = np.array([0.1, 0.2, 0.3, -0.1, 0.05, 0.0])
        shift = np.array([5.0, -3.0])

        simulator.set_goal(np.array([0.3, -0.4]))
        seen = simulator.policy_observation(simulator.observe(state))
        simulator.set_goal(np.array([0.3, -0.4]) + shift)
        moved = simulator.policy_observation(simulator.observe(np.concatenate([state[:2] + shift, state[2:]])))

        np.testing.assert_allclose(seen["observation.environment_state"], [0.2, -0.6], atol=1e-6)
        np.testing.assert_allclose(seen["observation.state"], state[2:], atol=1e-6)
        for name in seen:  # moving the point and the goal together changes nothing
            np.testing.assert_allclose(moved[name], seen[name], atol=1e-6)
        # an observation for the planners can be rebuilt from it, with the point at the origin
        rebuilt = simulator.observation_from_policy(seen["observation.environment_state"], seen["observation.state"])
        np.testing.assert_allclose(rebuilt, [0.2, -0.6, 0.0, 0.0, *state[2:]], atol=1e-6)

    def test_observation_dataset_frame_and_policy_input_follow_the_arm_convention(self) -> None:
        simulator = self.simulator
        simulator.set_goal(np.array([0.3, -0.4]))
        state = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        observation = simulator.observe(state)
        plan = np.arange(CHUNK * 2, dtype=float).reshape(CHUNK, 2)

        np.testing.assert_array_equal(observation, np.concatenate([simulator.goal, state]))
        np.testing.assert_array_equal(simulator.invert_obs(observation), state)
        frame = simulator.format_dataset_frame(observation, plan)
        features = simulator.get_dataset_features(CHUNK)
        self.assertEqual(set(frame), set(features))
        for name, feature in features.items():
            self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
        np.testing.assert_allclose(frame["action"], plan.reshape(-1))

    def test_starts_are_at_rest_and_away_from_the_goal_and_goals_are_in_the_box(self) -> None:
        simulator = _TripleIntegratorFactory.build(initial_state_seed=3)
        for _ in range(20):
            state = simulator.reset_random()
            np.testing.assert_array_equal(state[2:], 0.0)
            self.assertGreater(np.linalg.norm(state[:2] - simulator.goal), simulator.min_start_goal_distance)
            self.assertTrue(np.all(np.abs(simulator.goal) <= 1.0))

    def test_the_expert_reaches_random_goals(self) -> None:
        simulator = _TripleIntegratorFactory.build(initial_state_seed=5)
        planner = PlannerFactory.create(self.config["expert"], simulator, self.config)
        for _ in range(5):
            state = simulator.reset_random()
            planner.reset()
            for _ in range(80):
                plan = planner.plan(simulator.observe(state))
                state = simulator.step(state, plan.actions[0])
                if simulator.is_done(state):
                    break
            self.assertTrue(simulator.is_done(state), simulator.goal_errors(state))

    def test_the_ompl_casadi_expert_reaches_random_goals_too(self) -> None:
        config = _point_mass_config(expert="ompl_casadi")
        simulator = TripleIntegratorSystem(config)
        planner = PlannerFactory.create("ompl_casadi", simulator, config)
        for _ in range(3):
            state = simulator.reset_random()
            planner.reset()
            for _ in range(80):
                state = simulator.step(state, planner.plan(simulator.observe(state)).actions[0])
                if simulator.is_done(state):
                    break
            self.assertTrue(simulator.is_done(state), simulator.goal_errors(state))
        self.assertEqual(planner.statistics["ompl_paths"], 3)

    def test_both_policies_run_on_it(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        simulator.reset_random()
        observation = simulator.policy_observation(simulator.observe(np.concatenate([simulator.goal, np.zeros(4)])))
        batch = {name: torch.as_tensor(value)[None] for name, value in observation.items()}
        for policy_type, extra in (("flow", {}), ("safeflow", {"simulator": simulator, "planner_config": self.config})):
            with self.subTest(policy_type=policy_type):
                policy = PolicyFactory.create(
                    policy_type, action_dim=2, state_dim=sum(value.shape[0] for value in observation.values()),
                    unet_dim=8, unet_dim_mults=(1, 2),
                    prediction_horizon=CHUNK, num_inference_steps=2, action_scale=[1.0, 1.0], **extra,
                ).eval()
                self.assertEqual(tuple(policy.select_action(batch).shape), (1, CHUNK, 2))

    def test_the_config_rejects_what_a_point_mass_has_no_use_for(self) -> None:
        raw = {"system": "triple_integrator", "Q_diag": [1.0] * 6, "R_diag": [1.0] * 2}
        self.assertEqual(validate_system_config(raw)["dimension"], 2)
        for invalid in ({"floor_clearance": 0.1}, {"system": "pendulum"}, {"Q_diag": [1.0] * 14}):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                validate_system_config({**raw, **invalid})


class _TripleIntegratorFactory:
    @staticmethod
    def build(**overrides) -> TripleIntegratorSystem:
        return TripleIntegratorSystem(_point_mass_config(**overrides))


if __name__ == "__main__":
    unittest.main()
