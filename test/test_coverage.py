from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.coverage import DatasetCoverage


def _dataset(episodes: int = 20, steps: int = 10, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Episodes that walk around a start of their own, all of them near the origin of a 4-dimensional observation."""
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(episodes, 4))
    observations = starts[:, None, :] + 0.05 * np.cumsum(rng.normal(size=(episodes, steps, 4)), axis=1)
    return observations.reshape(-1, 4), np.repeat(np.arange(episodes), steps)


class DatasetCoverageTests(unittest.TestCase):
    def test_a_query_on_the_data_is_near_and_one_far_from_it_is_far(self) -> None:
        observations, episodes = _dataset()
        coverage = DatasetCoverage(observations, episodes)

        near = coverage.measure(observations[:10] + 0.01)
        far = coverage.measure(observations[:10] + 20.0)

        self.assertLess(near.median, 0.5)
        self.assertEqual(near.far_fraction, 0.0)
        self.assertGreater(far.median, 10 * near.median)
        self.assertEqual(far.far_fraction, 1.0)
        self.assertEqual(near.num_queries, 10)

    def test_the_components_are_measured_in_units_of_their_spread(self) -> None:
        observations, episodes = _dataset()
        stretched = observations * np.array([1.0, 1.0, 1.0, 1000.0])
        queries = observations[:10] + np.array([0.0, 0.0, 0.0, 0.5])

        self.assertEqual(DatasetCoverage(stretched, episodes).measure(queries * np.array([1.0, 1.0, 1.0, 1000.0])).far_fraction, 0.0)

    def test_the_reference_leaves_out_the_episode_of_a_query(self) -> None:
        observations, episodes = _dataset(episodes=5, steps=4)
        coverage = DatasetCoverage(observations, episodes)

        # the first observation of an episode has itself and its successors at distance < the nearest other episode
        self.assertTrue(np.all(coverage._reference > 0.0))

    def test_the_observations_of_a_lerobot_dataset_are_read_from_its_parquet_files(self) -> None:
        observations, episodes = _dataset(episodes=6, steps=5)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data" / "chunk-000").mkdir(parents=True)
            for index, rows in enumerate((slice(0, 15), slice(15, 30))):
                pd.DataFrame(
                    {
                        "episode_index": episodes[rows],
                        "observation.environment_state": list(observations[rows, :3].astype(np.float32)),
                        "observation.state": list(observations[rows, 3:].astype(np.float32)),
                    }
                ).to_parquet(root / "data" / "chunk-000" / f"file-{index:03d}.parquet")

            from_disk = DatasetCoverage.from_dataset(root)

        self.assertEqual(from_disk._points.shape, (30, 4))
        self.assertLess(from_disk.measure(observations[:5]).median, 1e-3)


if __name__ == "__main__":
    unittest.main()
