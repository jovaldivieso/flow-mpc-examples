from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.rest_projection import RestProjection

HORIZON = 10
DT = 0.05


def _final_velocity_and_acceleration(chunk: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """The backward differences of the last three positions of each joint of a chunk (H, n)."""
    return (chunk[-1] - chunk[-2]) / dt, (chunk[-1] - 2 * chunk[-2] + chunk[-3]) / dt**2


class RestProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rest = RestProjection(HORIZON, DT)
        self.chunk = np.random.default_rng(0).standard_normal((HORIZON, 4))

    def test_the_full_weight_brings_the_final_velocity_and_acceleration_to_rest(self) -> None:
        velocity, acceleration = _final_velocity_and_acceleration(self.rest(self.chunk, 1.0))

        np.testing.assert_allclose(velocity, 0.0, atol=1e-9)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-9)

    def test_a_partial_weight_scales_the_final_velocity_and_acceleration_of_every_joint(self) -> None:
        before = _final_velocity_and_acceleration(self.chunk)

        for weight in (0.25, 0.5, 0.9):
            with self.subTest(weight=weight):
                after = _final_velocity_and_acceleration(self.rest(self.chunk, weight))

                np.testing.assert_allclose(after[0], (1 - weight) * before[0], atol=1e-9)
                np.testing.assert_allclose(after[1], (1 - weight) * before[1], atol=1e-9)

    def test_no_weight_leaves_the_chunk_as_it_is(self) -> None:
        np.testing.assert_array_equal(self.rest(self.chunk, 0.0), self.chunk)

    def test_the_change_is_spread_along_the_chunk_and_largest_at_its_end(self) -> None:
        change = np.linalg.norm(self.rest(self.chunk, 1.0) - self.chunk, axis=1)

        self.assertEqual(int(np.argmax(change)), HORIZON - 1)
        self.assertGreater(change[HORIZON // 2], 1e-3 * change[-1])  # not confined to the last rows
        self.assertLess(change[0], 0.1 * change[-1])

    def test_the_change_is_smooth(self) -> None:
        # In velocity and acceleration the change is far smaller than the same change landing on the last row alone
        change = self.rest(self.chunk, 1.0) - self.chunk
        spike = np.zeros_like(change)
        spike[-1] = np.abs(change[-1])

        self.assertLess(np.abs(np.diff(change, 2, axis=0)).max(), 0.5 * np.abs(np.diff(spike, 2, axis=0)).max())

    def test_the_change_is_linear_in_the_chunk_and_the_weight(self) -> None:
        other = np.random.default_rng(1).standard_normal(self.chunk.shape)
        change = lambda chunk, weight: self.rest(chunk, weight) - chunk

        np.testing.assert_allclose(change(self.chunk + 2 * other, 0.5), change(self.chunk, 0.5) + 2 * change(other, 0.5), atol=1e-10)
        np.testing.assert_allclose(change(self.chunk, 0.5), 0.5 * change(self.chunk, 1.0), atol=1e-10)

    def test_the_joints_are_projected_independently(self) -> None:
        together = self.rest(self.chunk, 0.7)

        for joint in range(self.chunk.shape[1]):
            np.testing.assert_allclose(together[:, joint], self.rest(self.chunk[:, joint : joint + 1], 0.7)[:, 0], atol=1e-12)

    def test_a_chunk_that_already_rests_is_left_as_it_is(self) -> None:
        resting = np.tile(self.chunk[:1], (HORIZON, 1))

        np.testing.assert_allclose(self.rest(resting, 1.0), resting, atol=1e-10)

    def test_the_result_does_not_depend_on_the_scale_of_the_time_step(self) -> None:
        # The change that scales the final velocity and acceleration is the same for any dt: both are rescaled alike
        for dt in (0.01, 1.0):
            with self.subTest(dt=dt):
                np.testing.assert_allclose(RestProjection(HORIZON, dt)(self.chunk, 0.6), self.rest(self.chunk, 0.6), atol=1e-8)

    def test_the_final_velocity_and_acceleration_are_scaled_with_any_time_step(self) -> None:
        dt = 0.2
        before = _final_velocity_and_acceleration(self.chunk, dt)
        after = _final_velocity_and_acceleration(RestProjection(HORIZON, dt)(self.chunk, 1.0), dt)

        self.assertGreater(np.abs(before[0]).max(), 1e-3)
        np.testing.assert_allclose(after[0], 0.0, atol=1e-9)
        np.testing.assert_allclose(after[1], 0.0, atol=1e-9)

    def test_the_shortest_chunk_has_three_positions(self) -> None:
        for horizon in (0, 1, 2):
            with self.subTest(horizon=horizon):
                with self.assertRaises(ValueError):
                    RestProjection(horizon, DT)

        chunk = np.random.default_rng(2).standard_normal((3, 2))
        velocity, acceleration = _final_velocity_and_acceleration(RestProjection(3, DT)(chunk, 1.0))

        np.testing.assert_allclose(velocity, 0.0, atol=1e-9)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
