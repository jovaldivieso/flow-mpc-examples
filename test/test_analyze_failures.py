from __future__ import annotations

import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.analyze_failures import (
    APPROACHING,
    AT_REST,
    CREEP,
    HUGGING,
    NEAR_GOAL,
    RESTING,
    SLIDING,
    STALL_WINDOW,
    WANDERING,
    classify_timeout,
    stall_features,
)

FAR = 3.0 * NEAR_GOAL


class ClassifyTimeoutTests(unittest.TestCase):
    def test_near_the_goal_is_creep_whatever_the_speed_and_clearance(self) -> None:
        for speed in (0.0, 10 * AT_REST):
            for clearance in (0.0, 10 * HUGGING):
                self.assertEqual(classify_timeout(0.5 * NEAR_GOAL, speed, clearance), CREEP)

    def test_far_from_the_goal_and_at_rest_is_resting_whatever_the_clearance(self) -> None:
        for clearance in (0.0, 10 * HUGGING):
            self.assertEqual(classify_timeout(FAR, 0.5 * AT_REST, clearance), RESTING)

    def test_far_and_moving_along_an_obstacle_is_sliding(self) -> None:
        self.assertEqual(classify_timeout(FAR, 2 * AT_REST, 0.5 * HUGGING), SLIDING)

    def test_far_and_moving_in_free_space_is_wandering(self) -> None:
        self.assertEqual(classify_timeout(FAR, 2 * AT_REST, 5 * HUGGING), WANDERING)


class StallFeaturesTests(unittest.TestCase):
    STEPS = 2 * STALL_WINDOW

    def _traces(self, goal_distance: list[float], clearance: float, speed: float, tcp_y: list[float]) -> dict:
        n = len(goal_distance)
        return stall_features(goal_distance, [clearance] * n, [speed] * n, tcp_y)

    def test_the_window_progress_is_the_fall_of_the_goal_distance_over_the_last_steps(self) -> None:
        distance = [FAR] * (self.STEPS - STALL_WINDOW) + [FAR - APPROACHING * k / STALL_WINDOW for k in range(STALL_WINDOW + 1)]

        features = self._traces(distance, 5 * HUGGING, 2 * AT_REST, [0.1] * len(distance))

        self.assertAlmostEqual(features["window_progress"], APPROACHING)
        self.assertEqual(features["mode"], WANDERING)

    def test_a_goal_distance_that_grows_has_negative_progress(self) -> None:
        distance = [FAR + 0.001 * k for k in range(self.STEPS)]

        features = self._traces(distance, 0.5 * HUGGING, 2 * AT_REST, [0.1] * self.STEPS)

        self.assertLess(features["window_progress"], 0.0)
        self.assertEqual(features["mode"], SLIDING)

    def test_only_the_last_window_decides_the_mode(self) -> None:
        clearance = [10 * HUGGING] * (self.STEPS - STALL_WINDOW - 1) + [0.5 * HUGGING] * (STALL_WINDOW + 1)

        features = stall_features([FAR] * self.STEPS, clearance, [2 * AT_REST] * self.STEPS, [0.1] * self.STEPS)

        self.assertEqual(features["mode"], SLIDING)

    def test_an_episode_crossed_the_plane_when_its_tcp_was_on_both_sides(self) -> None:
        distance = [FAR] * self.STEPS

        self.assertTrue(self._traces(distance, 5 * HUGGING, 2 * AT_REST, [-0.2] + [0.2] * (self.STEPS - 1))["crossed_y0"])
        self.assertFalse(self._traces(distance, 5 * HUGGING, 2 * AT_REST, [0.2] * self.STEPS)["crossed_y0"])


if __name__ == "__main__":
    unittest.main()
