import copy
import os
import sys
import unittest
from unittest import mock

import casadi as ca
import numpy as np
import pinocchio as pin
import yaml
from xacro import substitution_args

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import torch

from planning.config import load_and_validate_system_config, validate_system_config
from systems.config_fields import ConfigurationError
from learning.data_utils import build_dataset_features, format_dataset_frame
from systems.position_history import PositionHistory
from systems.rollout_termination import RolloutTerminationTracker
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from systems import franka_sampling
from systems.franka_config import HAND_AXIS, HAND_TILT, WORKSPACE_BOX
from systems.franka_sampling import IK_STARTS, EpisodeSampler, PoseRegion, SamplingMode
from systems.franka_system import FLOOR_CLEARANCE, Q_READY, FrankaSystem
from systems.collision_spheres import RADIUS_MARGIN, fit_collision_spheres, surface_points
from systems.franka_kinematics import COLLISION_SPHERE_MAX_RADIUS
from systems import franka_model
from systems.franka_model import build_robot
from systems.point_mass_system import PointMassSystem
from systems.sphere_collision import SphereCollisionModel
from systems.triple_integrator import TripleIntegratorBase
from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.planner_factory import PlannerFactory
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_config.yaml")
CHUNK = 16  # configurations per state the policy predicts
HISTORY_LENGTH = 3  # configurations of the history the policy observes

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
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


def _observed(simulator: SystemSimulator, state: np.ndarray) -> dict[str, np.ndarray]:
    """The policy observation of ``state`` at the start of a rollout (the history is the configuration repeated)."""
    history = PositionHistory(simulator.n, HISTORY_LENGTH)
    history.reset(state)
    return simulator.policy_observation(simulator.observe(state), history.flat)


def _stepped_rollout(simulator: TripleIntegratorBase, state: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """The rollout one ``predict_next_state`` at a time, the definition ``rollout_states`` has to agree with."""
    states = [np.asarray(state, dtype=float)]
    for action in actions:
        states.append(simulator.predict_next_state(states[-1], action))
    return np.array(states)


def _assert_rollouts_follow_the_plant(test: unittest.TestCase, simulator: TripleIntegratorBase, state: np.ndarray) -> None:
    rng = np.random.default_rng(4)
    gentle = 0.05 * simulator.max_action * rng.uniform(-1, 1, (8, simulator.nu))
    beyond_the_jerk_limit = 3.0 * simulator.max_action * rng.uniform(-1, 1, (8, simulator.nu))
    saturating = np.tile(simulator.max_action, (12, 1))  # drives the acceleration and the velocity into their limits
    for name, actions in (("gentle", gentle), ("beyond the jerk limit", beyond_the_jerk_limit), ("saturating", saturating)):
        with test.subTest(actions=name):
            np.testing.assert_allclose(simulator.rollout_states(state, actions), _stepped_rollout(simulator, state, actions), atol=1e-10)
    with mock.patch.object(simulator, "predict_next_state", side_effect=AssertionError("stepped")):
        simulator.rollout_states(state, gentle)  # nothing saturates, so it is one matrix product
    test.assertEqual(simulator.rollout_states(state, np.zeros((0, simulator.nu))).shape, (1, simulator.nx))
    for steps in (3, 8):  # the maps of other lengths are separate
        actions = gentle[:steps]
        np.testing.assert_allclose(simulator.rollout_states(state, actions), _stepped_rollout(simulator, state, actions), atol=1e-10)


class SimulatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = _build_simulator()

    def test_satisfies_the_dynamics_hierarchy(self) -> None:
        self.assertIsInstance(self.simulator, SystemSimulator)
        self.assertIsInstance(self.simulator, TripleIntegratorBase)
        self.assertNotIsInstance(self.simulator, PointMassSystem)  # the arm is no point mass: they share only the base

    def test_the_abstract_bases_cannot_be_instantiated(self) -> None:
        for abstract in (SystemSimulator, TripleIntegratorBase):
            with self.subTest(abstract=abstract.__name__), self.assertRaises(TypeError):
                abstract({})

    def test_the_goal_is_an_abstract_property_implemented_by_the_systems_and_set_through_set_goal(self) -> None:
        self.assertIn("goal", SystemSimulator.__abstractmethods__)
        for name in ("max_action", "state_lower_bounds", "state_upper_bounds", "n", "nx", "nu", "obs_dim"):
            self.assertIn(name, SystemSimulator.__abstractmethods__)
        simulator = _build_simulator()
        goal = pose_goal(simulator, Q_READY)
        simulator.set_goal(goal)
        np.testing.assert_array_equal(simulator.goal, goal)
        with self.assertRaises(ValueError):
            simulator.set_goal(goal[:3])

    def test_a_reset_drops_the_cached_joint_configuration_of_the_goal(self) -> None:
        simulator = _build_simulator()
        q_goal = Q_READY + 0.3
        goal = pose_goal(simulator, q_goal)
        far_reference = Q_READY - 0.4
        with mock.patch.object(simulator.kinematics, "solve_ik", wraps=simulator.kinematics.solve_ik) as solve_ik:
            first = simulator.goal_configuration(goal, far_reference)
            simulator.goal_configuration(goal, far_reference)
            self.assertEqual(solve_ik.call_count, 1)  # the second call came from the cache

            simulator.reset(np.concatenate([Q_READY, np.zeros(2 * simulator.n)]))
            self.assertIsNone(simulator._cached_goal)
            again = simulator.goal_configuration(goal, far_reference)

            self.assertEqual(solve_ik.call_count, 2)  # a new rollout solves again
        np.testing.assert_allclose(again, first, atol=1e-6)

    def test_xacro_is_only_redirected_while_the_urdf_is_built(self) -> None:
        original = substitution_args._eval_find
        franka_model._urdf.cache_clear()

        franka_model._urdf()

        self.assertIs(substitution_args._eval_find, original)

    def test_dimensions_and_rest_state_indices(self) -> None:
        simulator = self.simulator
        # observation = goal pose (12) + state (21); the policy sees the goal in the TCP frame (12), no goal configuration,
        # and the centers of the 23 bounding spheres (69)
        self.assertEqual((simulator.nx, simulator.nu, simulator.obs_dim, simulator.goal_dim), (21, 7, 33, 12))
        self.assertEqual(simulator.environment_dim, 12 + 3 * 23)
        self.assertEqual(simulator.max_action.shape, (simulator.nu,))
        self.assertEqual(simulator.state_lower_bounds.shape, (simulator.nx,))
        # state = [q, v, a]: the projector's terminal come-to-rest cost reads the velocities and accelerations
        self.assertEqual(simulator.rest_state_indices, tuple(range(7, 21)))
        self.assertEqual(simulator.model.nv, 7)

    def test_step_matches_predict_next_state(self) -> None:
        simulator = self.simulator
        state = simulator.reset(simulator.random_initial_state(np.random.default_rng(11)))
        action = np.zeros(int(simulator.nu), dtype=float)
        np.testing.assert_allclose(simulator.predict_next_state(state, action), simulator.step(state, action))

    def test_dataset_schema_matches_formatted_frames(self) -> None:
        simulator = self.simulator
        rng = np.random.default_rng(7)
        simulator.randomize_goal_for_reset(rng)
        state = simulator.reset(simulator.random_initial_state(rng))
        observation = simulator.observe(state)
        plan = np.linspace(-1.0, 1.0, CHUNK * simulator.n).reshape(CHUNK, simulator.n)
        history = PositionHistory(simulator.n, HISTORY_LENGTH)
        history.reset(state)
        history.push(state + 0.1)

        features = build_dataset_features(simulator, CHUNK, HISTORY_LENGTH)
        frame = format_dataset_frame(simulator, observation, history.flat, plan)

        self.assertEqual(set(frame), set(features))
        self.assertEqual(set(features), {"observation.environment_state", "observation.state", "observation.history", "action"})
        for name, feature in features.items():
            self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
            self.assertEqual(str(frame[name].dtype), feature["dtype"], name)
            self.assertEqual(len(feature["names"]), feature["shape"][0], name)

        relative = simulator.tcp_placement(state[: simulator.n]).actInv(simulator.vector_pose(simulator.goal))
        sphere_centers = simulator.collision_spheres(state[: simulator.n])[0]
        np.testing.assert_allclose(
            frame["observation.environment_state"],
            np.concatenate([simulator.pose_vector(relative), sphere_centers.ravel()]),
            atol=1e-6,
        )
        np.testing.assert_allclose(frame["observation.state"], state, atol=1e-6)
        # The history is the last positions, the oldest first
        self.assertEqual(features["observation.history"]["shape"], (HISTORY_LENGTH * simulator.n,))
        np.testing.assert_allclose(
            frame["observation.history"],
            np.concatenate([state[: simulator.n]] * (HISTORY_LENGTH - 1) + [state[: simulator.n] + 0.1]),
            atol=1e-6,
        )
        # The dataset action is the expert's plan, one joint configuration per row, flattened row by row
        self.assertEqual(features["action"]["shape"], (CHUNK * simulator.n,))
        np.testing.assert_allclose(frame["action"], plan.reshape(-1), atol=1e-6)
        self.assertTrue(features["action"]["names"][0].endswith("[t+1]"))
        self.assertTrue(features["action"]["names"][simulator.n].endswith("[t+2]"))
        self.assertTrue(features["observation.history"]["names"][0].endswith(f"[t-{HISTORY_LENGTH - 1}]"))
        self.assertTrue(features["observation.history"]["names"][-1].endswith("[t-0]"))

        # The policy sees exactly the three observation keys of the dataset, as float32
        policy_observation = simulator.policy_observation(observation, history.flat)
        self.assertEqual(
            set(policy_observation), {"observation.environment_state", "observation.state", "observation.history"}
        )
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

        seen = _observed(simulator, state)["observation.environment_state"].astype(float)

        np.testing.assert_allclose(seen[:3], placement.rotation.T @ (goal.translation - placement.translation), atol=1e-6)
        np.testing.assert_allclose(seen[3:12].reshape(3, 3), placement.rotation.T @ goal.rotation, atol=1e-6)
        self.assertEqual(seen.shape, (simulator.environment_dim,))  # no goal joint configuration: IK is the experts'
        # the goal comes back from what the policy saw and the state
        rebuilt = simulator.observation_from_policy(seen, state[:])
        np.testing.assert_allclose(rebuilt[: simulator.goal_dim], simulator.goal, atol=1e-5)
        np.testing.assert_array_equal(rebuilt[simulator.goal_dim :], state)

    def test_the_policy_sees_where_its_body_is_whatever_the_goal(self) -> None:
        simulator = _build_simulator()
        rng = np.random.default_rng(6)
        state = simulator.reset(simulator.random_initial_state(rng))
        simulator.randomize_goal_for_reset(rng)
        first = _observed(simulator, state)["observation.environment_state"]
        simulator.randomize_goal_for_reset(rng)
        second = _observed(simulator, state)["observation.environment_state"]

        expected = simulator.collision_spheres(state[: simulator.n])[0].ravel()  # base frame: forward kinematics of q only
        np.testing.assert_allclose(first[12:], expected, atol=1e-6)
        np.testing.assert_array_equal(first[12:], second[12:])
        self.assertEqual(len(simulator.environment_state_names), simulator.environment_dim)

    def test_a_reached_goal_is_the_identity_in_the_tcp_frame(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))

        seen = _observed(simulator, state)["observation.environment_state"]

        np.testing.assert_allclose(seen[:12], np.concatenate([np.zeros(3), np.eye(3).ravel()]), atol=1e-6)

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


class CollisionModelTests(unittest.TestCase):
    """The bounding spheres the planners keep clear of obstacles must cover the real collision meshes."""

    OBSTACLES = [{"position": [0.45, -0.12, 0.40], "radius": 0.08}, {"position": [0.45, 0.15, 0.48], "radius": 0.08}]

    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = _build_simulator(obstacles=cls.OBSTACLES)

    def test_the_fitted_spheres_cover_an_independent_sample_of_every_mesh_surface(self) -> None:
        collision_model = build_robot().collision_model
        spheres = fit_collision_spheres(collision_model, COLLISION_SPHERE_MAX_RADIUS)

        self.assertEqual({joint for joint, _, _ in spheres}, set(range(1, 8)))
        self.assertTrue(all(0.0 < radius <= COLLISION_SPHERE_MAX_RADIUS + RADIUS_MARGIN for _, _, radius in spheres))
        rng = np.random.default_rng(123)  # not the fit's own sample
        for geometry in collision_model.geometryObjects:
            if geometry.parentJoint == 0:
                continue
            points = surface_points(geometry.geometry, geometry.placement, rng)
            inside = np.any(
                [np.linalg.norm(points - center, axis=1) <= radius for joint, center, radius in spheres if joint == geometry.parentJoint],
                axis=0,
            )
            self.assertTrue(inside.all(), f"{geometry.name}: {(~inside).sum()} surface points outside the spheres")

    def test_the_kinematics_export_the_spheres_with_their_jacobians(self) -> None:
        centers, jacobians, radii = self.simulator.collision_spheres(Q_READY)

        self.assertEqual((centers.shape, jacobians.shape), ((len(radii), 3), (len(radii), 3, 7)))
        self.assertGreater(len(radii), 7)

    def test_a_rollout_follows_the_plant_where_it_saturates_too(self) -> None:
        state = _rest_state(self.simulator)
        state[7:14] = 0.2  # moving already
        _assert_rollouts_follow_the_plant(self, self.simulator, state)

    def test_the_spheres_along_a_trajectory_are_those_of_each_configuration(self) -> None:
        q_trajectory = Q_READY + 0.3 * np.random.default_rng(3).standard_normal((5, 7))

        centers, jacobians, radii = self.simulator.collision_spheres_along(q_trajectory)

        self.assertEqual((centers.shape, jacobians.shape), ((5, len(radii), 3), (5, len(radii), 3, 7)))
        for row, q in enumerate(q_trajectory):
            single_centers, single_jacobians, single_radii = self.simulator.collision_spheres(q)
            np.testing.assert_allclose(centers[row], single_centers, atol=1e-12)
            np.testing.assert_allclose(jacobians[row], single_jacobians, atol=1e-12)
            np.testing.assert_array_equal(radii, single_radii)

    def test_the_spheres_miss_no_collision_of_the_real_meshes(self) -> None:
        simulator = self.simulator
        rng = np.random.default_rng(0)
        lower, upper = simulator.model.lowerPositionLimit, simulator.model.upperPositionLimit
        collisions = 0
        for _ in range(3000):
            q = rng.uniform(lower, upper)
            if simulator.collides(q):
                collisions += 1
                self.assertFalse(simulator.collision.is_collision_free(q), f"the spheres miss a real collision at q={q.tolist()}")
        self.assertGreater(collisions, 50)  # the sample does reach the obstacles

    def test_the_hand_and_fingers_count_not_only_the_axis_of_the_arm(self) -> None:
        # an obstacle beside the hand, 9 cm off the axis of the TCP frame: outside spheres on the axis alone, inside the hand
        simulator = _build_simulator()
        placement = simulator.tcp_placement(Q_READY)
        beside_the_hand = placement.act(np.array([0.0, 0.09, -0.08]))
        simulator = _build_simulator(obstacles=[{"position": beside_the_hand.tolist(), "radius": 0.02}])

        self.assertTrue(simulator.collides(Q_READY))
        self.assertFalse(simulator.collision.is_collision_free(Q_READY))

    def test_a_system_without_obstacles_never_collides(self) -> None:
        self.assertFalse(_build_simulator().collides(Q_READY))


class CollisionInterfaceTests(unittest.TestCase):
    """What the planners use of a system is declared by ``TripleIntegratorBase``: its ``collision`` model, the path box and
    the goal configuration."""

    def test_both_systems_have_a_collision_model_without_obstacles_and_a_path_box(self) -> None:
        for simulator in (_build_simulator(), _TripleIntegratorFactory.build()):
            with self.subTest(system=type(simulator).__name__):
                self.assertIsInstance(simulator, TripleIntegratorBase)
                self.assertIsInstance(simulator.collision, SphereCollisionModel)
                self.assertEqual(simulator.collision.obstacles, [])
                self.assertEqual(simulator.path_lower.shape, (simulator.n,))
                self.assertTrue(np.all(simulator.path_lower < simulator.path_upper))

    def test_the_collision_model_uses_the_bounding_spheres_of_its_system(self) -> None:
        simulator = _build_simulator()
        centers, _, radii = simulator.collision_spheres(Q_READY)
        simulator.collision.obstacles.append((centers[0].copy(), 0.01))

        self.assertFalse(simulator.collision.is_collision_free(Q_READY))

    def test_the_interface_cannot_be_instantiated_without_its_geometry(self) -> None:
        with self.assertRaises(TypeError):
            TripleIntegratorBase({})

    def test_the_policy_observes_the_full_plant_state_and_the_history_of_both_systems(self) -> None:
        arm = _build_simulator()
        state = _rest_state(arm)
        seen = _observed(arm, state)
        np.testing.assert_array_equal(seen["observation.state"], state.astype(np.float32))
        np.testing.assert_array_equal(seen["observation.history"], np.tile(Q_READY, HISTORY_LENGTH).astype(np.float32))
        point = _TripleIntegratorFactory.build()
        point.set_goal(np.array([0.5, 0.1]))
        point_state = np.array([0.4, -0.2, 0.1, 0.2, 0.0, 0.3])
        seen = _observed(point, point_state)
        np.testing.assert_allclose(seen["observation.state"], point_state, atol=1e-6)
        np.testing.assert_allclose(seen["observation.environment_state"], [0.1, 0.3], atol=1e-6)
        np.testing.assert_allclose(seen["observation.history"], np.tile(point_state[:2], HISTORY_LENGTH), atol=1e-6)


class RolloutTerminationTests(unittest.TestCase):
    def test_rollout_termination_state_resets_and_honors_hold_steps(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        self.assertTrue(simulator.is_done(state), "a rest state at the goal must satisfy is_done()")

        hold_steps = simulator.done_hold_steps
        self.assertEqual(hold_steps, int(simulator.config.get("done_hold_steps", 5)))
        self.assertGreaterEqual(hold_steps, 1)

        termination = RolloutTerminationTracker(simulator)
        for step_idx in range(hold_steps - 1):
            self.assertFalse(termination.update(state), f"termination triggered too early on step {step_idx + 1}")

        self.assertTrue(termination.update(state), f"termination did not trigger on hold step {hold_steps}")

        termination.reset()
        self.assertFalse(termination.update(state), "termination counter was not reset by reset()")

    def test_a_rollout_that_leaves_the_goal_starts_counting_again(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        moving = state.copy()
        moving[simulator.n] = 1.0
        termination = RolloutTerminationTracker(simulator)

        for _ in range(simulator.done_hold_steps - 1):
            termination.update(state)
        termination.update(moving)

        self.assertFalse(termination.update(state))

    def test_the_trackers_of_two_rollouts_are_independent(self) -> None:
        simulator = _build_simulator()
        state = _rest_state(simulator)
        simulator.set_goal(pose_goal(simulator, state[: simulator.n]))
        first, second = RolloutTerminationTracker(simulator), RolloutTerminationTracker(simulator)

        for _ in range(simulator.done_hold_steps):
            first.update(state)

        self.assertFalse(second.update(state))


class SamplingTests(unittest.TestCase):
    def test_seeded_sampling_is_deterministic_and_inside_joint_limits(self) -> None:
        from learning.dagger.utils import sample_initial_state

        simulator_a = _build_simulator()
        simulator_b = _build_simulator()

        state_a = sample_initial_state(simulator_a, 13)
        state_b = sample_initial_state(simulator_b, 13)

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
        for seed in range(6):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            state = simulator.random_initial_state(rng)
            q = state[: simulator.n]
            tcp = simulator.tcp_position(q)

            np.testing.assert_array_equal(state[simulator.n :], 0.0)
            self.assertTrue(np.all(state >= simulator.state_lower_bounds - 1e-12))
            self.assertTrue(np.all(state <= simulator.state_upper_bounds + 1e-12))
            self.assertGreater(simulator.goal_errors(state)[0], simulator.min_start_goal_distance)
            for point in (simulator.goal[:3], tcp):
                self.assertTrue(_in_region(point, WORKSPACE_BOX))
            self.assertTrue(all(joint.translation[2] >= FLOOR_CLEARANCE for joint in simulator.data.oMi[1:]))

    def test_infeasible_min_start_goal_distance_raises_instead_of_looping_forever(self) -> None:
        simulator = _build_simulator(min_start_goal_distance=100.0)
        with self.assertRaises(RuntimeError):
            simulator.random_initial_state(np.random.default_rng(0))


TABLE = [{"position": [x, y, 0.50], "radius": 0.10} for x in (0.40, 0.54) for y in (-0.28, -0.14, 0.0, 0.14, 0.28)]
START_REGION = {"x": [0.38, 0.56], "y": [-0.25, 0.25], "z": [0.70, 0.90]}
GOAL_REGION = {"x": [0.38, 0.56], "y": [-0.25, 0.25], "z": [0.15, 0.34], "tool_axis": [1.0, 0.0, 0.0], "tool_tilt": 0.4}


def _in_region(point: np.ndarray, region: dict) -> bool:
    return all(region[axis][0] - 1e-3 <= value <= region[axis][1] + 1e-3 for axis, value in zip("xyz", point))


class PoseRegionTests(unittest.TestCase):
    def test_a_pose_has_its_position_in_the_box_and_the_hand_within_the_tilt_of_the_axis(self) -> None:
        region = PoseRegion.from_fields({**GOAL_REGION})
        rng = np.random.default_rng(0)
        for _ in range(200):
            position, rotation = region.pose(rng)

            self.assertTrue(_in_region(position, GOAL_REGION))
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            self.assertGreaterEqual(rotation[:, 2] @ np.array([1.0, 0.0, 0.0]), np.cos(GOAL_REGION["tool_tilt"]) - 1e-9)

    def test_the_hand_cone_of_the_workspace_spans_pointing_forward_to_pointing_down(self) -> None:
        axis = np.array(HAND_AXIS) / np.linalg.norm(HAND_AXIS)
        for direction in ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0]):
            self.assertLessEqual(np.arccos(axis @ np.array(direction)), HAND_TILT)


class EpisodeSamplerTests(unittest.TestCase):
    def _sampler(self, inverse_kinematics, is_valid=lambda q: True) -> EpisodeSampler:
        region = PoseRegion.from_fields({**GOAL_REGION})
        mode = SamplingMode(1.0, region, region, 0.0)
        return EpisodeSampler({"a": mode}, ["a"], inverse_kinematics, np.full(7, -1.0), np.full(7, 1.0), is_valid)

    def test_the_first_valid_configuration_of_a_pose_is_taken_from_a_random_start(self) -> None:
        starts = []

        def reach(position, rotation, q_init):
            starts.append(q_init)
            return q_init

        q = self._sampler(reach).configuration(PoseRegion.from_fields({**GOAL_REGION}), np.random.default_rng(0))

        self.assertEqual(len(starts), 1)
        np.testing.assert_array_equal(q, starts[0])
        self.assertTrue(np.all(np.abs(q) <= 1.0))

    def test_failed_and_invalid_solutions_are_skipped_within_the_starts_of_a_pose(self) -> None:
        calls = []

        def reach(position, rotation, q_init):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("no convergence")
            return q_init

        invalid_first = iter([False, True])
        sampler = self._sampler(reach, is_valid=lambda q: next(invalid_first))

        sampler.configuration(PoseRegion.from_fields({**GOAL_REGION}), np.random.default_rng(0))

        self.assertEqual(len(calls), 3)  # a failure, an invalid solution, then a valid one

    def test_a_pose_no_start_reaches_is_dropped_after_the_starts_and_the_region_raises_after_the_poses(self) -> None:
        calls = []

        def never(position, rotation, q_init):
            calls.append(1)
            raise RuntimeError("no convergence")

        with mock.patch.object(franka_sampling, "POSE_ATTEMPTS", 2), self.assertRaises(RuntimeError):
            self._sampler(never).configuration(PoseRegion.from_fields({**GOAL_REGION}), np.random.default_rng(0))

        self.assertEqual(len(calls), 2 * IK_STARTS)

    def test_the_starts_are_random_between_the_limits_and_reproducible_by_the_seed(self) -> None:
        def draw(seed):
            starts = []

            def reach(position, rotation, q_init):
                starts.append(q_init)
                if len(starts) < 4:
                    raise RuntimeError("no convergence")
                return q_init

            q = self._sampler(reach).configuration(PoseRegion.from_fields({**GOAL_REGION}), np.random.default_rng(seed))
            return q, starts

        q_a, starts_a = draw(5)
        q_b, _ = draw(5)

        np.testing.assert_array_equal(q_a, q_b)
        self.assertEqual(len({tuple(start) for start in starts_a}), 4)
        self.assertTrue(all(np.all(np.abs(start) <= 1.0) for start in starts_a))


class SamplingModeTests(unittest.TestCase):
    BOX = {"x": [0.40, 0.60], "y": [-0.4, -0.2], "z": [0.45, 0.65]}
    OPPOSITE = {**BOX, "y": [0.2, 0.4]}

    def _two_modes(self, **overrides) -> FrankaSystem:
        modes = {
            "left_to_right": {"weight": 1, "start": self.BOX, "goal": self.OPPOSITE},
            "right_to_left": {"weight": 1, "start": self.OPPOSITE, "goal": self.BOX},
        }
        return _build_simulator(sampling=modes, **overrides)

    def test_an_episode_starts_and_ends_in_the_regions_of_the_same_mode(self) -> None:
        simulator = self._two_modes()
        sides = set()
        for seed in range(8):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            start = simulator.tcp_position(simulator.random_initial_state(rng)[: simulator.n])

            self.assertLess(start[1] * simulator.goal[1], 0.0)
            sides.add(bool(start[1] < 0))
        self.assertEqual(sides, {True, False})

    def test_a_start_without_a_drawn_goal_draws_its_own_mode(self) -> None:
        simulator = self._two_modes()
        sides = {bool(simulator.tcp_position(simulator.random_initial_state(np.random.default_rng(seed))[: simulator.n])[1] < 0) for seed in range(8)}
        self.assertEqual(sides, {True, False})

    def test_only_the_selected_modes_are_drawn(self) -> None:
        simulator = self._two_modes(sampling_modes=["right_to_left"])
        for seed in range(4):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            self.assertTrue(_in_region(simulator.goal[:3], self.BOX))

    def test_the_modes_are_drawn_by_their_weights(self) -> None:
        modes = {"rare": {"weight": 1}, "common": {"weight": 3}}
        simulator = _build_simulator(sampling=modes)
        rng = np.random.default_rng(0)
        common = sum(simulator.sampler.draw_mode(rng) is simulator.sampler.modes[1] for _ in range(400))
        self.assertTrue(260 <= common <= 340, common)

    def test_the_arm_starts_in_motion_at_a_fraction_of_the_velocity_limits(self) -> None:
        simulator = _build_simulator(sampling={"moving": {"start_speed": 0.5}})
        limit = simulator.state_upper_bounds[simulator.n : 2 * simulator.n]
        for seed in range(3):
            state = simulator.random_initial_state(np.random.default_rng(seed))
            velocity = state[simulator.n : 2 * simulator.n]

            self.assertTrue(np.any(velocity != 0.0))
            self.assertTrue(np.all(np.abs(velocity) <= 0.5 * limit))
            np.testing.assert_array_equal(state[2 * simulator.n :], 0.0)
            self.assertTrue(np.all(state <= simulator.state_upper_bounds) and np.all(state >= simulator.state_lower_bounds))
            self.assertTrue(simulator._can_brake(state[: simulator.n], velocity))


    def test_a_start_in_motion_must_be_able_to_brake_inside_the_limits_and_clear_of_the_obstacles(self) -> None:
        simulator = _build_simulator()
        at_rest, fast = np.zeros(simulator.n), np.full(simulator.n, 1.0)
        near_the_limit = simulator.model.upperPositionLimit - 0.01

        self.assertTrue(simulator._can_brake(Q_READY, at_rest))
        self.assertFalse(simulator._can_brake(near_the_limit, fast))  # it stops beyond the joint limit
        self.assertTrue(simulator._can_brake(near_the_limit, -fast))  # away from it

        simulator.collision.obstacles[:] = [(simulator.tcp_position(Q_READY) + np.array([0.0, 0.0, 0.08]), 0.05)]
        self.assertTrue(simulator._can_brake(Q_READY, at_rest) == simulator.collision.is_collision_free(Q_READY))

    def test_a_start_in_motion_that_can_never_brake_raises(self) -> None:
        simulator = _build_simulator(sampling={"moving": {"start_speed": 1.0}})
        with mock.patch.object(simulator, "_can_brake", return_value=False), mock.patch("systems.franka_system.START_ATTEMPTS", 3):
            with self.assertRaisesRegex(RuntimeError, "can brake"):
                simulator.random_initial_state(np.random.default_rng(0))


class RegionSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = _build_simulator(obstacles=TABLE, sampling={"crossing": {"start": START_REGION, "goal": GOAL_REGION}})

    def test_without_sampling_the_workspace_and_the_default_hand_cone_are_sampled(self) -> None:
        [mode] = _build_simulator().sampler.modes

        np.testing.assert_allclose(mode.start.low, [WORKSPACE_BOX[axis][0] for axis in "xyz"])
        np.testing.assert_allclose(mode.goal.high, [WORKSPACE_BOX[axis][1] for axis in "xyz"])
        self.assertEqual(mode.start.tilt, HAND_TILT)
        self.assertEqual(mode.start_speed, 0.0)

    def test_starts_are_collision_free_with_the_tcp_in_the_start_region(self) -> None:
        simulator = self.simulator
        for seed in range(4):
            rng = np.random.default_rng(seed)
            simulator.randomize_goal_for_reset(rng)
            q = simulator.random_initial_state(rng)[: simulator.n]

            self.assertTrue(_in_region(simulator.tcp_position(q), START_REGION))
            self.assertTrue(simulator.collision.is_collision_free(q))

    def test_goals_are_in_the_goal_region_with_the_hand_along_the_tool_axis_and_reached_without_a_collision(self) -> None:
        simulator = self.simulator
        for seed in range(4):
            simulator.randomize_goal_for_reset(np.random.default_rng(seed))
            position, rotation = simulator.goal[:3], simulator.goal[3:].reshape(3, 3)

            self.assertTrue(_in_region(position, GOAL_REGION))
            self.assertGreaterEqual(rotation[:, 2] @ np.array([1.0, 0.0, 0.0]), np.cos(GOAL_REGION["tool_tilt"]) - 1e-9)
            q_goal = simulator.goal_configuration(simulator.goal)
            self.assertTrue(simulator.collision.is_collision_free(q_goal))
            np.testing.assert_allclose(simulator.tcp_position(q_goal), position, atol=1e-3)

    def test_an_empty_region_raises_instead_of_looping_forever(self) -> None:
        inside_the_table = {"x": [0.38, 0.56], "y": [-0.25, 0.25], "z": [0.45, 0.55]}
        simulator = _build_simulator(obstacles=TABLE, sampling={"inside": {"start": inside_the_table}})
        with mock.patch.object(franka_sampling, "POSE_ATTEMPTS", 3), self.assertRaises(RuntimeError):
            simulator.random_initial_state(np.random.default_rng(0))


POINT_MASS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "triple_integrator_config.yaml")
# The tests pin the dimension they assume, so scaling the shipped config up does not break them
POINT_MASS_SETTINGS = {"dimension": 2}


def _point_mass_config(**overrides) -> dict:
    with open(POINT_MASS_CONFIG_PATH) as f:
        return validate_system_config({**yaml.safe_load(f), **POINT_MASS_SETTINGS, **overrides})


class PointMassSystemTests(unittest.TestCase):
    """The toy system: the same plant as the arm's joints, the same pipeline hooks, a position as the goal."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _point_mass_config()
        cls.simulator = SystemFactory.create(cls.config["system"], cls.config)

    def test_the_factory_builds_it_from_the_config_and_it_satisfies_the_base_class(self) -> None:
        self.assertIsInstance(self.simulator, PointMassSystem)
        self.assertIsInstance(self.simulator, TripleIntegratorBase)
        self.assertEqual(
            (self.simulator.nx, self.simulator.nu, self.simulator.obs_dim, self.simulator.goal_dim), (6, 2, 8, 2)
        )
        with self.assertRaises(ValueError):
            SystemFactory.create("pendulum", self.config)

    def test_a_rollout_follows_the_plant_where_it_saturates_too(self) -> None:
        _assert_rollouts_follow_the_plant(self, self.simulator, np.array([0.1, -0.2, 0.3, -0.1, 0.5, -0.4]))

    def test_the_spheres_along_a_trajectory_are_those_of_each_configuration(self) -> None:
        q_trajectory = np.random.default_rng(3).standard_normal((4, 2))

        centers, jacobians, radii = self.simulator.collision_spheres_along(q_trajectory)

        for row, q in enumerate(q_trajectory):
            single_centers, single_jacobians, single_radii = self.simulator.collision_spheres(q)
            np.testing.assert_array_equal(centers[row], single_centers)
            np.testing.assert_array_equal(jacobians[row], single_jacobians)
            np.testing.assert_array_equal(radii, single_radii)

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
        # the states of a rollout carry the velocities and accelerations of those positions
        states = simulator.rollout_states(state, actions)
        np.testing.assert_allclose(states[:, :n], positions)
        np.testing.assert_allclose(states[-1], x)
        self.assertEqual(states.shape, (len(actions) + 1, simulator.nx))

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

    def test_the_policy_sees_the_goal_relative_to_the_point_and_its_position_only_in_the_state_and_history(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        state = np.array([0.1, 0.2, 0.3, -0.1, 0.05, 0.0])
        shift = np.array([5.0, -3.0])

        simulator.set_goal(np.array([0.3, -0.4]))
        seen = _observed(simulator, state)
        simulator.set_goal(np.array([0.3, -0.4]) + shift)
        moved = _observed(simulator, np.concatenate([state[:2] + shift, state[2:]]))

        np.testing.assert_allclose(seen["observation.environment_state"], [0.2, -0.6], atol=1e-6)
        np.testing.assert_allclose(seen["observation.state"], state, atol=1e-6)
        np.testing.assert_allclose(seen["observation.history"], np.tile(state[:2], HISTORY_LENGTH), atol=1e-6)
        # moving the point and the goal together changes the goal the policy sees by nothing
        np.testing.assert_allclose(moved["observation.environment_state"], seen["observation.environment_state"], atol=1e-6)
        # the observation of the planners can be rebuilt from it
        rebuilt = simulator.observation_from_policy(moved["observation.environment_state"], moved["observation.state"])
        np.testing.assert_allclose(rebuilt, simulator.observe(np.concatenate([state[:2] + shift, state[2:]])), atol=1e-5)

    def test_observation_dataset_frame_and_policy_input_follow_the_arm_convention(self) -> None:
        simulator = self.simulator
        simulator.set_goal(np.array([0.3, -0.4]))
        state = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        observation = simulator.observe(state)
        plan = np.arange(CHUNK * 2, dtype=float).reshape(CHUNK, 2)
        history = np.arange(HISTORY_LENGTH * 2, dtype=float)

        np.testing.assert_array_equal(observation, np.concatenate([simulator.goal, state]))
        np.testing.assert_array_equal(simulator.invert_obs(observation), state)
        frame = format_dataset_frame(simulator, observation, history, plan)
        features = build_dataset_features(simulator, CHUNK, HISTORY_LENGTH)
        self.assertEqual(set(frame), set(features))
        for name, feature in features.items():
            self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
        np.testing.assert_allclose(frame["action"], plan.reshape(-1))
        np.testing.assert_allclose(frame["observation.history"], history)

    def test_starts_are_at_rest_and_away_from_the_goal_and_goals_are_in_the_box(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        rng = np.random.default_rng(3)
        for _ in range(20):
            simulator.randomize_goal_for_reset(rng)
            state = simulator.random_initial_state(rng)
            np.testing.assert_array_equal(state[2:], 0.0)
            self.assertGreater(np.linalg.norm(state[:2] - simulator.goal), simulator.min_start_goal_distance)
            self.assertTrue(np.all(np.abs(simulator.goal) <= 1.0))

    def test_the_expert_reaches_random_goals(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        planner = PlannerFactory.create(self.config["expert"], simulator, self.config)
        rng = np.random.default_rng(5)
        for _ in range(5):
            simulator.randomize_goal_for_reset(rng)
            state = simulator.reset(simulator.random_initial_state(rng))
            planner.reset()
            for _ in range(150):
                plan = planner.plan(simulator.observe(state))
                self.assertEqual(plan.positions.shape, (self.config["horizon"], simulator.n))
                state = simulator.step(state, plan.actions[0])
                if simulator.is_done(state):
                    break
            self.assertTrue(simulator.is_done(state), simulator.goal_errors(state))
        self.assertEqual(planner.statistics["ompl_paths"], 5)

    def test_both_policies_run_on_it(self) -> None:
        simulator = _TripleIntegratorFactory.build()
        simulator.randomize_goal_for_reset(np.random.default_rng(0))
        observation = _observed(simulator, np.concatenate([simulator.goal, np.zeros(4)]))
        batch = {name: torch.as_tensor(value)[None] for name, value in observation.items()}
        cond_dim = observation["observation.environment_state"].shape[0] + observation["observation.history"].shape[0]
        for policy_type in PolicyFactory.names():
            with self.subTest(policy_type=policy_type):
                spec = PolicySpec(
                    policy_type=policy_type, action_dim=2, cond_dim=cond_dim, history_length=HISTORY_LENGTH,
                    execution_horizon=2, prediction_horizon=CHUNK, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=2, action_scale=(1.0, 1.0),
                )
                policy = PolicyFactory.create(spec, simulator, self.config).eval()
                self.assertEqual(tuple(policy.select_action(batch).shape), (1, CHUNK, 2))
                self.assertEqual(policy.execution_horizon, 2)

    def test_the_config_rejects_what_a_point_mass_has_no_use_for(self) -> None:
        raw = {"system": "triple_integrator"}
        self.assertEqual(validate_system_config(raw)["dimension"], 2)
        for invalid in ({"floor_clearance": 0.1}, {"system": "pendulum"}, {"Q_diag": [1.0] * 6}):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                validate_system_config({**raw, **invalid})


class _TripleIntegratorFactory:
    @staticmethod
    def build(**overrides) -> PointMassSystem:
        return PointMassSystem(_point_mass_config(**overrides))


if __name__ == "__main__":
    unittest.main()
