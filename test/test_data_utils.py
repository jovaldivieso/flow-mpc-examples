from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.data_utils import (
    collate_batch_for_policy,
    create_collate_fn,
    format_sample_for_policy,
)

HORIZON = 4
JOINTS = 3


def _sample(offset: float) -> dict[str, object]:
    """A synthetic dataset row whose plan row k, joint j holds offset + 10k + j."""
    plan = np.array(
        [[offset + 10.0 * k + j for j in range(JOINTS)] for k in range(HORIZON)], dtype=np.float32
    )
    return {
        "observation.environment_state": np.array([100.0 + offset], dtype=np.float32),
        "observation.state": np.array([200.0 + offset, 201.0 + offset], dtype=np.float32),
        "action": plan.reshape(-1),
    }


class FormatSampleTests(unittest.TestCase):
    def test_the_flat_action_is_reshaped_row_by_row_into_the_plan(self) -> None:
        observation, plan = format_sample_for_policy(_sample(1.0), HORIZON)

        self.assertEqual(tuple(plan.shape), (HORIZON, JOINTS))
        self.assertEqual(plan[2, 1].item(), 1.0 + 10.0 * 2 + 1)
        self.assertEqual(plan.dtype, torch.float32)

    def test_observations_carry_exactly_the_environment_state_and_state_keys(self) -> None:
        observation, _ = format_sample_for_policy(_sample(1.0), HORIZON)

        self.assertEqual(set(observation), {"observation.environment_state", "observation.state"})
        self.assertEqual(observation["observation.environment_state"].tolist(), [101.0])
        self.assertEqual(observation["observation.state"].tolist(), [201.0, 202.0])

    def test_a_plan_that_does_not_split_into_rows_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_sample_for_policy(_sample(0.0), HORIZON + 1)

    def test_a_missing_field_is_rejected(self) -> None:
        sample = _sample(0.0)
        del sample["observation.state"]

        with self.assertRaises(ValueError):
            format_sample_for_policy(sample, HORIZON)

    def test_non_positive_prediction_horizon_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            format_sample_for_policy(_sample(0.0), 0)


class CollationTests(unittest.TestCase):
    def test_a_batch_stacks_observations_and_plans_in_order(self) -> None:
        observations, plans = collate_batch_for_policy([_sample(1.0), _sample(2.0)], HORIZON)

        self.assertEqual(tuple(plans.shape), (2, HORIZON, JOINTS))
        self.assertEqual(plans[1, 0, 0].item(), 2.0)
        self.assertEqual(tuple(observations["observation.state"].shape), (2, 2))
        self.assertEqual(observations["observation.environment_state"][:, 0].tolist(), [101.0, 102.0])

    def test_create_collate_fn_binds_the_prediction_horizon(self) -> None:
        collate = create_collate_fn(HORIZON)

        _, plans = collate([_sample(0.0)])

        self.assertEqual(tuple(plans.shape), (1, HORIZON, JOINTS))

    def test_an_empty_batch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            collate_batch_for_policy([], HORIZON)


if __name__ == "__main__":
    unittest.main()
