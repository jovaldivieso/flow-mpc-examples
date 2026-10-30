from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from systems.franka_system import Q_READY, FrankaSystem
from systems.point_mass_system import PointMassSystem
from learning.models.policy_factory import PolicyFactory
from learning.models.pose_guidance import PoseGuidance
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")
HORIZON = 10
Q_GOAL = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])


def _arm(goal: np.ndarray) -> FrankaSystem:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    return FrankaSystem(validate_system_config(raw_config={**raw_config, "goal": list(goal)}))


def _arm_goal() -> np.ndarray:
    return pose_goal(_arm(np.concatenate([[0.4, 0.0, 0.4], np.eye(3).ravel()])), Q_GOAL)


def _rest_at(simulator: FrankaSystem, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The policy observation (goal in the TCP frame, state) of the arm at rest in ``q``."""
    observation = simulator.policy_observation(simulator.observe(np.concatenate([q, np.zeros(14)])))
    return observation["observation.environment_state"].astype(float), observation["observation.state"].astype(float)


def _final_state(simulator, state: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    for jerk in chunk:
        state = simulator.predict_next_state(state, jerk)
    return state


class PoseGuidanceOnTheArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.goal = _arm_goal()
        cls.simulator = _arm(cls.goal)

    def test_close_to_the_goal_the_chunk_moves_the_tcp_toward_it_and_keeps_its_rest(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        q = Q_GOAL + 0.03 * np.random.default_rng(0).standard_normal(7)
        environment_state, state = _rest_at(self.simulator, q)
        guidance._start_distance = 0.5  # the episode started far away: the goal is close

        guided = guidance.guide(environment_state, state, np.zeros((HORIZON, 7)))

        self.simulator.goal = self.goal
        before, after = _final_state(self.simulator, state, np.zeros((HORIZON, 7))), _final_state(self.simulator, state, guided)
        self.assertLess(self.simulator.goal_errors(after)[0], 0.5 * self.simulator.goal_errors(before)[0])
        np.testing.assert_allclose(after[7:], 0.0, atol=1e-3)  # the final velocity and acceleration are kept
        self.assertTrue(np.all(np.abs(guided) <= self.simulator.max_action + 1e-9))

    def test_far_from_the_goal_the_chunk_is_left_as_proposed(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        environment_state, state = _rest_at(self.simulator, Q_READY)
        proposal = 0.1 * np.random.default_rng(1).standard_normal((HORIZON, 7))

        guided = guidance.guide(environment_state, state, proposal)

        self.assertLess(np.abs(guided - proposal).max(), 0.05)

    def test_the_start_distance_is_the_one_of_the_first_call_of_an_episode(self) -> None:
        guidance = PoseGuidance(self.simulator, HORIZON)
        proposal = np.zeros((HORIZON, 7))
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
        proposal = np.zeros((HORIZON, 7))
        chunks = []
        for simulator in (self.simulator, other):
            guidance = PoseGuidance(simulator, HORIZON)
            guidance._start_distance = 0.5
            chunks.append(guidance.guide(*observation, proposal))

        np.testing.assert_allclose(chunks[0], chunks[1])


class PoseGuidanceOnThePointMassTests(unittest.TestCase):
    def test_the_chunk_moves_the_point_toward_the_goal(self) -> None:
        simulator = PointMassSystem({"goal": [0.03, 0.0]})
        guidance = PoseGuidance(simulator, HORIZON)
        guidance._start_distance = 0.3
        observation = simulator.policy_observation(simulator.observe(np.zeros(6)))

        guided = guidance.guide(observation["observation.environment_state"], observation["observation.state"], np.zeros((HORIZON, 2)))

        final = _final_state(simulator, np.zeros(6), guided)
        self.assertLess(np.linalg.norm(final[:2] - [0.03, 0.0]), 0.015)
        np.testing.assert_allclose(final[2:], 0.0, atol=1e-3)


class GuidedFlowPolicyTests(unittest.TestCase):
    def test_every_euler_step_of_a_flow_policy_is_guided(self) -> None:
        simulator = PointMassSystem({"goal": [0.5, 0.0]})
        policy = PolicyFactory.create(
            "flow", action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2), prediction_horizon=HORIZON,
            num_inference_steps=3, action_scale=[10.0, 10.0], simulator=simulator,
        ).eval()
        observation = simulator.policy_observation(simulator.observe(np.zeros(6)))
        batch = {name: torch.as_tensor(value)[None] for name, value in observation.items()}

        with mock.patch.object(policy.guidance, "guide", wraps=policy.guidance.guide) as guide:
            chunk = policy.select_action(batch)

        self.assertEqual(guide.call_count, 3)
        self.assertEqual(tuple(chunk.shape), (1, HORIZON, 2))

    def test_the_iterates_of_a_flow_policy_stay_within_the_jerk_limits(self) -> None:
        simulator = PointMassSystem({"goal": [0.5, 0.0]})
        policy = PolicyFactory.create(
            "flow", action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2), prediction_horizon=HORIZON,
            num_inference_steps=3, action_scale=[100.0, 100.0], simulator=simulator,  # the noise prior reaches far past the limit of 10
        ).eval()
        observation = simulator.policy_observation(simulator.observe(np.zeros(6)))
        batch = {name: torch.as_tensor(value)[None] for name, value in observation.items()}

        chunk = policy.select_action(batch)[0].numpy()

        self.assertTrue(np.all(np.abs(chunk) <= simulator.max_action + 1e-6))

    def test_a_reset_starts_the_guidance_of_a_new_episode(self) -> None:
        simulator = PointMassSystem({})
        policy = PolicyFactory.create(
            "flow", action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2), prediction_horizon=HORIZON,
            num_inference_steps=1, action_scale=[10.0, 10.0], simulator=simulator,
        )
        policy.guidance._start_distance = 1.0

        policy.reset()

        self.assertIsNone(policy.guidance._start_distance)

    def test_a_guided_flow_policy_serves_one_system(self) -> None:
        simulator = PointMassSystem({})
        policy = PolicyFactory.create(
            "flow", action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2), prediction_horizon=HORIZON,
            num_inference_steps=1, action_scale=[10.0, 10.0], simulator=simulator,
        ).eval()
        batch = {"observation.environment_state": torch.zeros(2, 2), "observation.state": torch.zeros(2, 4)}

        with self.assertRaises(ValueError):
            policy.select_action(batch)


if __name__ == "__main__":
    unittest.main()
