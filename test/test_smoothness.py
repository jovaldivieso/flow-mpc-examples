from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.smoothness import SmoothSpread, smoothness_metric

HORIZON = 10


def _total_variation(sequence: np.ndarray) -> float:
    return float(np.abs(np.diff(sequence, axis=0)).sum())


class SmoothnessMetricTests(unittest.TestCase):
    def test_the_metric_is_symmetric_and_positive_definite(self) -> None:
        metric = smoothness_metric(HORIZON)

        np.testing.assert_allclose(metric, metric.T)
        self.assertGreater(np.linalg.eigvalsh(metric).min(), 0.0)

    def test_a_spiky_change_is_larger_than_a_smooth_one_of_the_same_size(self) -> None:
        metric = smoothness_metric(HORIZON)
        spike = np.zeros(HORIZON)
        spike[5] = 1.0
        smooth = np.sin(np.linspace(0.0, np.pi, HORIZON))
        smooth /= np.linalg.norm(smooth)

        self.assertGreater(spike @ metric @ spike, smooth @ metric @ smooth)


class SmoothSpreadTests(unittest.TestCase):
    def test_a_spike_is_spread_along_the_sequence_and_never_grows(self) -> None:
        spike = np.zeros((HORIZON, 2))
        spike[5, 0] = 1.0

        spread = SmoothSpread(HORIZON)(spike)

        self.assertLessEqual(np.abs(spread).max(), 1.0)
        self.assertLess(_total_variation(spread[:, 0]), _total_variation(spike[:, 0]))
        self.assertGreater(spread[3, 0], 0.0)  # the neighbors move with it
        np.testing.assert_allclose(spread[:, 1], 0.0, atol=1e-15)

    def test_the_columns_are_spread_independently(self) -> None:
        rng = np.random.default_rng(0)
        correction = rng.standard_normal((HORIZON, 3))

        spread = SmoothSpread(HORIZON)(correction)

        np.testing.assert_allclose(spread[:, 1], SmoothSpread(HORIZON)(correction[:, 1:2])[:, 0])

    def test_without_strength_the_correction_is_unchanged(self) -> None:
        correction = np.random.default_rng(1).standard_normal((HORIZON, 2))

        np.testing.assert_allclose(SmoothSpread(HORIZON, strength=0.0)(correction), correction)


if __name__ == "__main__":
    unittest.main()
