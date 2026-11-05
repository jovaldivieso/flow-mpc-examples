from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import pinocchio as pin
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from systems.franka_system import Q_READY, FrankaSystem
from systems.point_mass_system import PointMassSystem
from learning.models.pose_guidance import PoseGuidance
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_config.yaml")
HORIZON = 10
Q_GOAL = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])


def _arm(goal: np.ndarray) -> FrankaSystem:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    return FrankaSystem(validate_system_config(raw_config={**raw_config, "goal": list(goal)}))


def _arm_goal() -> np.ndarray:
    return pose_goal(_arm(np.concatenate([[0.4, 0.0, 0.4], np.eye(3).ravel()])), Q_GOAL)


HISTORY = 3


def _rest_at(simulator, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The policy observation (goal relative to the TCP, state) of the arm at rest in ``q``."""
    state = np.concatenate([q, np.zeros(2 * simulator.n)])
    observation = simulator.policy_observation(simulator.observe(state), np.tile(q, HISTORY))
    return observation["observation.environment_state"].astype(float), observation["observation.state"].astype(float)


def _final_velocity_and_acceleration(chunk: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """The backward differences of the last three positions of a chunk."""
    return (chunk[-1] - chunk[-2]) / dt, (chunk[-1] - 2 * chunk[-2] + chunk[-3]) / dt**2


def _drift(q: np.ndarray, rate: float) -> np.ndarray:
    """The chunk of positions (HORIZON, n) that leaves ``q`` at a constant velocity ``rate`` in every joint."""
    return q + rate * np.arange(1, HORIZON + 1)[:, None] * np.ones(7)


def _hold(q: np.ndarray) -> np.ndarray:
    """The chunk of positions (HORIZON, n) that stays in ``q``."""
    return np.tile(q, (HORIZON, 1))


class PoseGuidanceOnTheArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.goal = _arm_goal()
        cls.simulator = _arm(cls.goal)

    def test_the_jacobian_of_the_task_error_is_its_derivative_whatever_the_orientation_error(self) -> None:
        rng = np.random.default_rng(1)
        for angle in (0.05, 0.5, 1.5, 2.8):
            q = Q_GOAL + 0.1 * rng.standard_normal(7)
            placement = self.simulator.tcp_placement(q)
            axis = rng.standard_normal(3)
            rotated = pin.exp3(angle * axis / np.linalg.norm(axis)) @ placement.rotation
            goal = np.concatenate([placement.translation + 0.1, rotated.ravel()])

            error, jacobian, distance = self.simulator.goal_task_error(q, goal)

            relative = placement.actInv(pin.SE3(rotated, goal[:3]))
            np.testing.assert_allclose(error, pin.log6(relative).vector, atol=1e-12)
            self.assertAlmostEqual(distance, float(np.linalg.norm(goal[:3] - placement.translation)))

            step = 1e-6
            derivative = np.zeros((6, 7))
            for joint in range(7):
                delta = np.zeros(7)
                delta[joint] = step
                derivative[:, joint] = (
                    self.simulator.goal_task_error(q + delta, goal)[0] - self.simulator.goal_task_error(q - delta, goal)[0]
                ) / (2 * step)
            np.testing.assert_allclose(derivative, -jacobian, atol=1e-6, err_msg=f"orientation error of {angle} rad")

    def test_close_to_the_goal_the_positions_move_toward_it_the_more_the_later_they_are(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        q = Q_GOAL + 0.03 * np.random.default_rng(0).standard_normal(7)
        environment_state, state = _rest_at(self.simulator, q)
        guidance._start_distance = 0.5  # the episode started far away: the goal is close
        proposal = _hold(q)

        guided = guidance.guide(environment_state, state, proposal)

        self.assertEqual(guided.shape, (HORIZON, 7))
        self.assertLess(self.simulator.goal_errors(guided[-1])[0], 0.5 * self.simulator.goal_errors(q)[0])
        moves = np.linalg.norm(guided - proposal, axis=1)
        self.assertLess(moves[0], 0.5 * moves[-1])

    def test_a_proposal_at_the_goal_pose_is_left_as_proposed(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        environment_state, state = _rest_at(self.simulator, Q_GOAL + 0.03)
        guidance._start_distance = 0.5

        guided = guidance.guide(environment_state, state, _hold(Q_GOAL))

        self.assertLess(np.abs(guided - _hold(Q_GOAL)).max(), 1e-3)

    def test_close_to_the_goal_the_chunk_comes_to_rest(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        q = Q_GOAL + 0.03
        environment_state, state = _rest_at(self.simulator, q)
        guidance._start_distance = 5.0  # the goal is within the onset fraction of the start distance
        proposal = _drift(q, 0.01)

        guided = guidance.guide(environment_state, state, proposal)

        velocity, acceleration = _final_velocity_and_acceleration(guided, self.simulator.dt)
        proposed_velocity, _ = _final_velocity_and_acceleration(proposal, self.simulator.dt)
        self.assertGreater(np.abs(proposed_velocity).max(), 0.1)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-8)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-8)

    def test_far_from_the_goal_the_final_velocity_is_the_proposed_one(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        environment_state, state = _rest_at(self.simulator, Q_READY)
        proposal = _drift(Q_READY, 0.01)

        guided = guidance.guide(environment_state, state, proposal)

        np.testing.assert_allclose(
            _final_velocity_and_acceleration(guided, self.simulator.dt)[0],
            _final_velocity_and_acceleration(proposal, self.simulator.dt)[0],
            rtol=1e-2,
        )

    def test_the_rest_weight_is_the_closeness_of_the_last_guided_position(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        q = Q_GOAL + 0.3
        environment_state, state = _rest_at(self.simulator, q)
        guidance._start_distance = 1.0 * self.simulator.goal_task_error(q, self.simulator.observation_from_policy(environment_state, state)[: self.simulator.goal_dim])[2]
        proposal = _drift(q, 0.0)

        with mock.patch.object(guidance, "_rest", wraps=guidance._rest) as rest:
            guided = guidance.guide(environment_state, state, proposal)

        guided_before_rest, weight = rest.call_args.args
        goal = self.simulator.observation_from_policy(environment_state, state)[: self.simulator.goal_dim]
        distance = self.simulator.goal_task_error(guided_before_rest[-1], goal)[2]
        self.assertAlmostEqual(weight, guidance._closeness(distance))
        self.assertGreater(weight, 0.0)
        self.assertLess(weight, 1.0)
        np.testing.assert_allclose(guided, guidance._rest(guided_before_rest, weight))

    def test_far_from_the_goal_the_positions_are_left_as_proposed(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        environment_state, state = _rest_at(self.simulator, Q_READY)
        proposal = _hold(Q_READY) + 0.1 * np.random.default_rng(1).standard_normal((HORIZON, 7))

        guided = guidance.guide(environment_state, state, proposal)

        self.assertLess(np.abs(guided - proposal).max(), 1e-3)

    def test_the_start_distance_is_the_one_of_the_first_call_of_an_episode(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        proposal = _hold(Q_READY)
        far = _rest_at(self.simulator, Q_READY)
        near = _rest_at(self.simulator, Q_GOAL + 0.05)

        guidance.guide(*far, proposal)
        first = guidance._start_distance
        guidance.guide(*near, proposal)
        self.assertEqual(guidance._start_distance, first)

        guidance.reset()
        guidance.guide(*near, proposal)
        self.assertLess(guidance._start_distance, first)

    def test_it_reads_no_goal_of_the_simulator(self) -> None:
        # The goal comes from the observation (the goal in the TCP frame): two simulators holding other goals guide alike
        observation = _rest_at(self.simulator, Q_GOAL + 0.03)
        other = _arm(pose_goal(self.simulator, np.array([0.2, -0.7, 0.1, -1.8, -0.2, 1.3, 0.8])))
        proposal = _hold(Q_GOAL + 0.03)
        chunks = []
        for simulator in (self.simulator, other):
            guidance = PoseGuidance(simulator, HORIZON)
            guidance._start_distance = 0.5
            chunks.append(guidance.guide(*observation, proposal))

        np.testing.assert_allclose(chunks[0], chunks[1])


class PoseGuidanceOnThePointMassTests(unittest.TestCase):
    def test_the_positions_move_toward_the_goal(self) -> None:
        simulator = PointMassSystem({"goal": [0.03, 0.0]})
        guidance = PoseGuidance(simulator, HORIZON)
        guidance._start_distance = 0.3
        state = np.zeros(6)
        observation = simulator.policy_observation(simulator.observe(state), np.zeros(HISTORY * 2))

        guided = guidance.guide(observation["observation.environment_state"], observation["observation.state"], np.zeros((HORIZON, 2)))

        self.assertEqual(guided.shape, (HORIZON, 2))
        self.assertLess(np.linalg.norm(guided[-1] - [0.03, 0.0]), 0.015)
        self.assertTrue(np.all(guided[:, 0] >= 0.0))
        np.testing.assert_allclose(guided[:, 1], 0.0, atol=1e-12)

    def test_the_chunk_ends_at_rest_when_the_goal_is_reached(self) -> None:
        simulator = PointMassSystem({"goal": [0.0, 0.0]})
        guidance = PoseGuidance(simulator, HORIZON)
        guidance._start_distance = 1.0
        state = np.zeros(6)
        observation = simulator.policy_observation(simulator.observe(state), np.zeros(HISTORY * 2))
        proposal = 0.001 * np.arange(1, HORIZON + 1)[:, None] * np.ones(2)

        guided = guidance.guide(observation["observation.environment_state"], observation["observation.state"], proposal)

        velocity, acceleration = _final_velocity_and_acceleration(guided, simulator.dt)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-8)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
