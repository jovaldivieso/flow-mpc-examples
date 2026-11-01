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


def _terminal_state(chunk: np.ndarray, velocity: np.ndarray, acceleration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The velocity and acceleration (n,) after the jerks (H, n) are applied one step each from the given ones."""
    for jerk in chunk:
        velocity, acceleration = velocity + DT * acceleration + DT**2 / 2 * jerk, acceleration + DT * jerk
    return velocity, acceleration


def _terminal_map() -> np.ndarray:
    """The (2, H) map from the jerks of one joint, from rest, to its final velocity and acceleration."""
    columns = []
    for k in range(HORIZON):
        jerk = np.zeros((HORIZON, 1))
        jerk[k] = 1.0
        velocity, acceleration = _terminal_state(jerk, np.zeros(1), np.zeros(1))
        columns.append([velocity[0], acceleration[0]])
    return np.array(columns).T


class RestProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rest = RestProjection(_terminal_map())
        rng = np.random.default_rng(0)
        self.chunk = rng.standard_normal((HORIZON, 3))
        self.velocity, self.acceleration = rng.standard_normal(3), rng.standard_normal(3)

    def _project(self, weight: float) -> np.ndarray:
        final = _terminal_state(self.chunk, self.velocity, self.acceleration)
        return self.rest(self.chunk, final[0], final[1], weight)

    def test_the_full_weight_brings_the_final_velocity_and_acceleration_to_zero(self) -> None:
        velocity, acceleration = _terminal_state(self._project(1.0), self.velocity, self.acceleration)

        np.testing.assert_allclose(velocity, 0.0, atol=1e-12)
        np.testing.assert_allclose(acceleration, 0.0, atol=1e-12)

    def test_a_partial_weight_scales_the_final_velocity_and_acceleration(self) -> None:
        before = _terminal_state(self.chunk, self.velocity, self.acceleration)
        after = _terminal_state(self._project(0.4), self.velocity, self.acceleration)

        np.testing.assert_allclose(after[0], 0.6 * before[0], atol=1e-12)
        np.testing.assert_allclose(after[1], 0.6 * before[1], atol=1e-12)

    def test_no_weight_leaves_the_chunk_as_it_is(self) -> None:
        np.testing.assert_array_equal(self._project(0.0), self.chunk)

    def test_the_change_of_the_jerks_is_the_smallest_one_that_does_it(self) -> None:
        change = self._project(1.0) - self.chunk
        terminal_map = _terminal_map()
        _, _, right_singular = np.linalg.svd(terminal_map)
        null_space = right_singular[2:].T  # the jerks that change neither the final velocity nor the final acceleration

        np.testing.assert_allclose(null_space.T @ change, 0.0, atol=1e-12)

    def test_the_joints_are_projected_independently(self) -> None:
        together = self._project(1.0)
        final = _terminal_state(self.chunk, self.velocity, self.acceleration)

        alone = self.rest(self.chunk[:, :1], final[0][:1], final[1][:1], 1.0)

        np.testing.assert_allclose(together[:, :1], alone, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
