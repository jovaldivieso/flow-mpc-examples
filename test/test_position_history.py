from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.position_history import PositionHistory


class PositionHistoryTests(unittest.TestCase):
    def test_a_reset_fills_the_history_with_the_configuration(self) -> None:
        history = PositionHistory(n=3, length=4)

        history.reset(np.array([1.0, 2.0, 3.0]))

        np.testing.assert_array_equal(history.flat, np.tile([1.0, 2.0, 3.0], 4))

    def test_a_reset_takes_the_positions_of_a_whole_state(self) -> None:
        history = PositionHistory(n=2, length=3)

        history.reset(np.array([1.0, 2.0, 9.0, 9.0, 9.0, 9.0]))  # positions, velocities and accelerations

        np.testing.assert_array_equal(history.flat, [1.0, 2.0] * 3)

    def test_a_push_drops_the_oldest_and_appends_the_newest_last(self) -> None:
        history = PositionHistory(n=2, length=3)
        history.reset(np.array([0.0, 0.0]))

        history.push(np.array([1.0, 10.0]))
        history.push(np.array([2.0, 20.0]))

        np.testing.assert_array_equal(history.flat, [0.0, 0.0, 1.0, 10.0, 2.0, 20.0])

    def test_after_more_pushes_than_the_length_only_the_last_ones_remain(self) -> None:
        history = PositionHistory(n=1, length=3)
        history.reset(np.array([0.0]))

        for value in range(1, 8):
            history.push(np.array([float(value)]))

        np.testing.assert_array_equal(history.flat, [5.0, 6.0, 7.0])

    def test_a_history_of_length_one_is_the_last_configuration(self) -> None:
        history = PositionHistory(n=2, length=1)
        history.reset(np.array([1.0, 2.0]))
        history.push(np.array([3.0, 4.0]))

        np.testing.assert_array_equal(history.flat, [3.0, 4.0])

    def test_the_flat_history_has_length_times_n_entries(self) -> None:
        history = PositionHistory(n=7, length=10)

        self.assertEqual(history.flat.shape, (70,))
        self.assertEqual((history.n, history.length), (7, 10))

    def test_the_flat_history_is_a_copy(self) -> None:
        history = PositionHistory(n=2, length=2)
        history.reset(np.array([1.0, 2.0]))

        history.flat[:] = 99.0

        np.testing.assert_array_equal(history.flat, [1.0, 2.0, 1.0, 2.0])

    def test_the_history_does_not_alias_the_array_it_is_given(self) -> None:
        history = PositionHistory(n=2, length=2)
        q = np.array([1.0, 2.0])
        history.reset(q)
        q[:] = 5.0
        history.push(q)
        q[:] = 6.0

        np.testing.assert_array_equal(history.flat, [1.0, 2.0, 5.0, 5.0])

    def test_a_reset_starts_a_new_rollout(self) -> None:
        history = PositionHistory(n=1, length=3)
        history.reset(np.array([0.0]))
        history.push(np.array([1.0]))

        history.reset(np.array([7.0]))

        np.testing.assert_array_equal(history.flat, [7.0, 7.0, 7.0])

    def test_a_length_below_one_is_rejected(self) -> None:
        for length in (0, -1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                PositionHistory(n=2, length=length)


if __name__ == "__main__":
    unittest.main()
