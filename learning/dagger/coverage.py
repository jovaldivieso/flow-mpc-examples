"""How well the data covers the episodes a policy is asked to solve.

The policy sees the observation ``[goal in the TCP frame, bounding sphere centers, state]``, so a start whose observation
has no near neighbour among the observations of the dataset is one the policy has no example of. ``DatasetCoverage``
measures the distance of the observations of the starts of an evaluation to their nearest neighbour in the dataset, in
units of the standard deviation of every observation component, against the same distance of the first observations of
the dataset's own episodes to the other episodes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REFERENCE_EPISODES = 200  # episodes of the dataset whose first observation the queries are compared with
MIN_STANDARD_DEVIATION = 1e-3
CHUNK = 64  # queries per distance matrix


@dataclass(frozen=True)
class CoverageMetrics:
    """The nearest-neighbour distances of the queries: their median and 90th percentile, and the share of them farther
    than the 90th percentile of the reference distances of the dataset's own episodes."""

    median: float
    p90: float
    far_fraction: float
    num_queries: int

    def to_dict(self) -> dict[str, float | int]:
        return {key: round(value, 3) if isinstance(value, float) else value for key, value in asdict(self).items()}


class DatasetCoverage:
    """The observations of a dataset on disk and the reference distances of its own episodes."""

    def __init__(self, observations: np.ndarray, episodes: np.ndarray, seed: int = 0) -> None:
        self._mean = observations.mean(axis=0)
        self._std = np.maximum(observations.std(axis=0), MIN_STANDARD_DEVIATION)
        self._points = torch.from_numpy(((observations - self._mean) / self._std).astype(np.float32))
        rng = np.random.default_rng(seed)
        episode_ids = np.unique(episodes)
        chosen = rng.choice(episode_ids, size=min(REFERENCE_EPISODES, len(episode_ids)), replace=False)
        first = np.array([np.flatnonzero(episodes == episode)[0] for episode in chosen])
        self._reference = self._nearest(self._points[first], other_episodes=(episodes, episodes[first]))
        self._reference_p90 = float(np.percentile(self._reference, 90))

    @classmethod
    def from_dataset(cls, root: Path) -> DatasetCoverage:
        """The coverage of the LeRobot dataset at ``root`` (the episodes of every parquet file)."""
        columns = ["episode_index", "observation.environment_state", "observation.state"]
        frames = pd.concat([pd.read_parquet(path, columns=columns) for path in sorted(root.glob("data/**/*.parquet"))])
        observations = np.concatenate(
            [np.stack(frames["observation.environment_state"].to_numpy()), np.stack(frames["observation.state"].to_numpy())],
            axis=1,
        )
        return cls(observations, frames["episode_index"].to_numpy())

    def measure(self, queries: np.ndarray) -> CoverageMetrics:
        """The coverage of the observations ``queries`` (one per row)."""
        distances = self._nearest(torch.from_numpy(((queries - self._mean) / self._std).astype(np.float32)))
        return CoverageMetrics(
            float(np.median(distances)),
            float(np.percentile(distances, 90)),
            float(np.mean(distances > self._reference_p90)),
            len(distances),
        )

    def _nearest(self, queries: torch.Tensor, other_episodes: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
        """The distance of every query to its nearest observation of the dataset, with ``other_episodes`` = (the episode of
        every observation, the episode of every query) only to those of another episode."""
        nearest = []
        for start in range(0, len(queries), CHUNK):
            distances = torch.cdist(queries[start : start + CHUNK], self._points)
            if other_episodes is not None:
                episodes, query_episodes = other_episodes
                same = torch.from_numpy(episodes[None, :] == query_episodes[start : start + CHUNK, None])
                distances[same] = float("inf")
            nearest.append(distances.min(dim=1).values)
        return torch.cat(nearest).numpy()
