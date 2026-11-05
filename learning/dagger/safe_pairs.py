from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from learning.models.safe_source import PATH_SEGMENTS, SafeSource
from planning.planner_config import ProjectionConfig
from systems.system_factory import SystemFactory

PAIRS_FILE = "safe_pairs.npz"

# The projected paths of consecutive frames: (states, labeled chunks, index of the first frame) -> (frames, segments + 1, H, n)
PathsOfFrames = Callable[[np.ndarray, np.ndarray, int], np.ndarray]


class SafePairs:
    """The safe path from the safe source to the safe target of every frame of a dataset (``SafeSource``), kept in the
    dataset's folder.

    The frames of a dataset only ever get appended, so the paths of the frames that are there are final and ``update``
    computes the ones of the frames added since. A path is NaN where none was found (the frame is then left out of the training).
    """

    def __init__(self, root: Path, horizon: int, action_dim: int) -> None:
        self._path = Path(root) / PAIRS_FILE
        shape = (0, PATH_SEGMENTS + 1, horizon, action_dim)
        self.paths = np.zeros(shape, dtype=np.float32)
        if self._path.is_file():
            with np.load(self._path) as stored:
                self.paths = stored["paths"]
            if self.paths.shape[1:] != shape[1:]:
                raise ValueError(f"{self._path} holds paths of {self.paths.shape[1:]}, not {shape[1:]}.")

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def sources(self) -> np.ndarray:
        return self.paths[:, 0]

    @property
    def targets(self) -> np.ndarray:
        return self.paths[:, -1]

    def update(self, total: int, load: Callable[[int], tuple[np.ndarray, np.ndarray]], paths_of: PathsOfFrames) -> int:
        """Compute the paths of the frames beyond the ones kept, of a dataset of ``total`` frames; ``load(start)`` returns
        the states (frames, state_dim) and the labeled chunks (frames, horizon, action_dim) of the frames from ``start``
        on, and ``paths_of`` their paths. The number of frames computed."""
        known = len(self)
        if total < known:
            raise ValueError(f"The dataset has {total} frames, fewer than the {known} that have paths.")
        if total == known:
            return 0
        paths = paths_of(*load(known), known)
        self.paths = np.concatenate([self.paths, paths])
        np.savez(self._path, paths=self.paths)
        return total - known


_worker_source: SafeSource | None = None  # the one of this process of a pool


def _init_pair_worker(config: Mapping[str, Any], horizon: int, seed: int) -> None:
    global _worker_source
    _worker_source = SafeSource(SystemFactory.create(config["system"], config), ProjectionConfig.from_mapping(config), horizon, seed)


def _paths_of_chunk(task: tuple[np.ndarray, np.ndarray, int]) -> np.ndarray:
    assert _worker_source is not None
    states, labels, first_index = task
    return _worker_source.paths(states, labels, first_index)


class PooledPairs:
    """The paths of consecutive frames computed by a pool of worker processes, chunk by chunk, to be used as the
    ``paths_of`` of ``SafePairs.update`` inside a ``with`` block. The path of a frame depends on its index alone, so they are
    the ones a single process computes."""

    CHUNK = 200  # frames per task

    def __init__(self, config: Mapping[str, Any], horizon: int, seed: int, workers: int) -> None:
        self._initargs = (dict(config), int(horizon), int(seed))
        self._workers = int(workers)
        self._pool: Any = None

    def __enter__(self) -> PooledPairs:
        self._pool = multiprocessing.get_context("spawn").Pool(self._workers, initializer=_init_pair_worker, initargs=self._initargs)
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._pool.terminate()
        self._pool.join()
        self._pool = None

    def __call__(self, states: np.ndarray, labels: np.ndarray, first_index: int) -> np.ndarray:
        tasks = [(states[start : start + self.CHUNK], labels[start : start + self.CHUNK], first_index + start) for start in range(0, len(states), self.CHUNK)]
        chunks = []
        for done, chunk in enumerate(self._pool.imap(_paths_of_chunk, tasks), start=1):
            chunks.append(chunk)
            print(f"  safe sources: {min(done * self.CHUNK, len(states))}/{len(states)} frames")
        return np.concatenate(chunks)
