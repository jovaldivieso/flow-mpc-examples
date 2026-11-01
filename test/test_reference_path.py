from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.reference_path import ReferencePath

# a U-shaped path that comes back close to its start: (0, 0) -> (1, 0) -> (1, 1) -> (0, 1) -> (0, 0.1)
U_PATH = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]])


class ReferencePathTests(unittest.TestCase):
    def test_the_arc_length_is_the_length_of_the_polyline(self) -> None:
        self.assertAlmostEqual(ReferencePath(U_PATH).length, 3.9)

    def test_the_point_ahead_is_a_lookahead_of_arc_length_beyond_the_arm(self) -> None:
        path = ReferencePath(U_PATH)

        np.testing.assert_allclose(path.ahead(np.array([0.2, 0.1]), 1.0), [1.0, 0.2])
        self.assertAlmostEqual(path.progress, 0.2)

    def test_the_point_ahead_stops_at_the_end_of_the_path(self) -> None:
        path = ReferencePath(U_PATH)

        np.testing.assert_allclose(path.ahead(np.array([0.0, 0.9]), 5.0), U_PATH[-1])

    def test_the_progress_never_goes_back(self) -> None:
        path = ReferencePath(U_PATH)
        path.ahead(np.array([1.0, 0.5]), 1.0)

        path.ahead(np.array([0.3, 0.0]), 1.0)  # the arm is back near the start

        self.assertGreaterEqual(path.progress, 1.5)

    def test_a_path_that_comes_back_close_to_itself_is_not_skipped(self) -> None:
        path = ReferencePath(U_PATH)
        path.ahead(np.array([0.2, 0.1]), 1.0)

        # the end of the path is nearer to this point than the start, but a lookahead of 1 away from the progress
        path.ahead(np.array([0.05, 0.05]), 1.0)

        self.assertLess(path.progress, 1.3)

    def test_a_path_of_one_point_or_no_length_is_its_own_target(self) -> None:
        np.testing.assert_allclose(ReferencePath(np.zeros((1, 3))).ahead(np.ones(3), 1.0), 0.0)
        np.testing.assert_allclose(ReferencePath(np.zeros((2, 3))).ahead(np.zeros(3), 1.0), 0.0)

    def test_a_path_needs_a_row_of_positions(self) -> None:
        with self.assertRaises(ValueError):
            ReferencePath(np.zeros(3))


if __name__ == "__main__":
    unittest.main()
