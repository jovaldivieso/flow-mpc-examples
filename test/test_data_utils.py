from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.data_utils import (
    build_dataset_features,
    collate_batch_for_policy,
    create_collate_fn,
    format_dataset_frame,
    format_sample_for_policy,
)
from systems.point_mass_system import PointMassSystem

HORIZON = 4
JOINTS = 3
HISTORY = 5
SEGMENTS = 3


class _Pairs:
    """The safe paths of a dataset (N, SEGMENTS + 1, HORIZON, JOINTS), as ``SafePairs`` holds them."""

    def __init__(self, paths: np.ndarray) -> None:
        self.paths = paths.astype(np.float32)


def _sample(offset: float, index: int = 0) -> dict[str, object]:
    """A synthetic dataset row whose plan row k, joint j holds offset + 10k + j."""
    plan = np.array(
        [[offset + 10.0 * k + j for j in range(JOINTS)] for k in range(HORIZON)], dtype=np.float32
    )
    return {
        "observation.environment_state": np.array([100.0 + offset], dtype=np.float32),
        "observation.state": np.array([200.0 + offset, 201.0 + offset], dtype=np.float32),
        "observation.history": np.arange(HISTORY * JOINTS, dtype=np.float32) + 300.0 + offset,
        "action": plan.reshape(-1),
        "index": np.array([index]),
    }


class FormatSampleTests(unittest.TestCase):
    def test_the_flat_action_is_reshaped_row_by_row_into_the_plan(self) -> None:
        observation, plan = format_sample_for_policy(_sample(1.0), HORIZON)

        self.assertEqual(tuple(plan.shape), (HORIZON, JOINTS))
        self.assertEqual(plan[2, 1].item(), 1.0 + 10.0 * 2 + 1)
        self.assertEqual(plan.dtype, torch.float32)

    def test_observations_carry_exactly_the_environment_state_state_and_history_keys(self) -> None:
        observation, _ = format_sample_for_policy(_sample(1.0), HORIZON)

        self.assertEqual(set(observation), {"observation.environment_state", "observation.state", "observation.history"})
        self.assertEqual(observation["observation.environment_state"].tolist(), [101.0])
        self.assertEqual(observation["observation.state"].tolist(), [201.0, 202.0])
        self.assertEqual(tuple(observation["observation.history"].shape), (HISTORY * JOINTS,))
        self.assertEqual(observation["observation.history"][0].item(), 301.0)

    def test_a_plan_that_does_not_split_into_rows_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_sample_for_policy(_sample(0.0), HORIZON + 1)

    def test_a_missing_field_is_rejected(self) -> None:
        for name in ("observation.state", "observation.history", "action"):
            with self.subTest(name=name):
                sample = _sample(0.0)
                del sample[name]

                with self.assertRaises(ValueError):
                    format_sample_for_policy(sample, HORIZON)

    def test_non_positive_prediction_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_sample_for_policy(_sample(0.0), 0)


class CollationTests(unittest.TestCase):
    def test_a_batch_stacks_observations_and_plans_in_order(self) -> None:
        observations, plans, paths = collate_batch_for_policy([_sample(1.0), _sample(2.0)], HORIZON)

        self.assertIsNone(paths)
        self.assertEqual(tuple(plans.shape), (2, HORIZON, JOINTS))
        self.assertEqual(plans[1, 0, 0].item(), 2.0)
        self.assertEqual(tuple(observations["observation.state"].shape), (2, 2))
        self.assertEqual(tuple(observations["observation.history"].shape), (2, HISTORY * JOINTS))
        self.assertEqual(observations["observation.environment_state"][:, 0].tolist(), [101.0, 102.0])

    def test_create_collate_fn_binds_the_prediction_horizon(self) -> None:
        collate = create_collate_fn(HORIZON)

        _, plans, paths = collate([_sample(0.0)])

        self.assertEqual(tuple(plans.shape), (1, HORIZON, JOINTS))
        self.assertIsNone(paths)

    def test_an_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_batch_for_policy([], HORIZON)


class CollationWithPairsTests(unittest.TestCase):
    """With pairs every row takes the safe path of its sample index; the plans are the raw expert chunks."""

    def setUp(self) -> None:
        self.pairs = _Pairs(np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1) + np.linspace(0.0, 0.5, SEGMENTS + 1).reshape(1, -1, 1, 1) * np.ones((3, 1, HORIZON, JOINTS)))
        self.batch = [_sample(10.0, index=2), _sample(20.0, index=0)]

    def test_the_paths_are_those_of_the_sample_indices_and_the_plans_are_raw(self) -> None:
        _, plans, paths = collate_batch_for_policy(self.batch, HORIZON, self.pairs)

        self.assertEqual(tuple(paths.shape), (2, SEGMENTS + 1, HORIZON, JOINTS))
        np.testing.assert_array_equal(paths.numpy(), self.pairs.paths[[2, 0]])
        self.assertEqual(plans[0, 0, 0].item(), 10.0)
        self.assertEqual(plans[1, 0, 0].item(), 20.0)

    def test_the_observations_are_the_same_with_and_without_pairs(self) -> None:
        with_pairs, _, _ = collate_batch_for_policy(self.batch, HORIZON, self.pairs)
        without, _, _ = collate_batch_for_policy(self.batch, HORIZON)

        for name in without:
            torch.testing.assert_close(with_pairs[name], without[name])

    def test_create_collate_fn_binds_the_pairs(self) -> None:
        collate = create_collate_fn(HORIZON, self.pairs)

        _, plans, paths = collate(self.batch)

        np.testing.assert_array_equal(paths.numpy(), self.pairs.paths[[2, 0]])
        self.assertEqual(plans[0, 0, 0].item(), 10.0)

    def test_a_sample_without_an_index_is_rejected_when_pairs_are_used(self) -> None:
        sample = _sample(1.0)
        del sample["index"]

        with self.assertRaises(ValueError):
            collate_batch_for_policy([sample], HORIZON, self.pairs)


class DatasetFormatTests(unittest.TestCase):
    """The dataset of a point mass in 2-D: environment 2, state 6, history of HISTORY positions, plan of HORIZON positions."""

    def setUp(self) -> None:
        self.simulator = PointMassSystem({"goal": [0.5, 0.0]})

    def test_the_features_describe_the_three_observations_and_the_flat_plan(self) -> None:
        features = build_dataset_features(self.simulator, HORIZON, HISTORY)

        self.assertEqual(set(features), {"observation.environment_state", "observation.state", "observation.history", "action"})
        self.assertEqual(features["observation.environment_state"]["shape"], (2,))
        self.assertEqual(features["observation.state"]["shape"], (6,))
        self.assertEqual(features["observation.history"]["shape"], (HISTORY * 2,))
        self.assertEqual(features["action"]["shape"], (HORIZON * 2,))
        for feature in features.values():
            self.assertEqual(len(feature["names"]), feature["shape"][0])

    def test_a_frame_carries_the_observation_and_the_plan_flattened_row_by_row(self) -> None:
        state = np.array([0.1, 0.2, 0.0, 0.0, 0.0, 0.0])
        history = np.arange(HISTORY * 2, dtype=float)
        positions = np.arange(HORIZON * 2, dtype=float).reshape(HORIZON, 2)

        frame = format_dataset_frame(self.simulator, self.simulator.observe(state), history, positions)

        self.assertEqual(set(frame), {"observation.environment_state", "observation.state", "observation.history", "action"})
        np.testing.assert_allclose(frame["observation.environment_state"], [0.4, -0.2], atol=1e-6)
        np.testing.assert_array_equal(frame["observation.history"], history)
        self.assertEqual(frame["action"].dtype, np.float32)
        np.testing.assert_array_equal(frame["action"], positions.reshape(-1))
        _, plan = format_sample_for_policy(frame, HORIZON)
        np.testing.assert_array_equal(plan.numpy(), positions)

    def test_a_plan_that_is_not_one_row_of_positions_per_step_is_rejected(self) -> None:
        state = np.zeros(6)
        with self.assertRaises(ValueError):
            format_dataset_frame(self.simulator, self.simulator.observe(state), np.zeros(HISTORY * 2), np.zeros(HORIZON * 2))
        with self.assertRaises(ValueError):
            format_dataset_frame(self.simulator, self.simulator.observe(state), np.zeros(HISTORY * 2), np.zeros((HORIZON, 3)))


if __name__ == "__main__":
    unittest.main()
