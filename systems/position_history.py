import numpy as np


class PositionHistory:
    """The last ``length`` configurations of a rollout, the current one last: what a policy observes of where the arm has
    been, as SafeFlowMPC's flow is conditioned on its last ten. A rollout owns one: ``reset`` at its first state fills it with
    that configuration, ``push`` adds the configuration of every state after it."""

    def __init__(self, n: int, length: int) -> None:
        if length < 1:
            raise ValueError("'length' must be at least 1.")
        self.n, self.length = int(n), int(length)
        self._positions = np.zeros((self.length, self.n))

    def reset(self, q: np.ndarray) -> None:
        self._positions[:] = np.asarray(q, dtype=float)[: self.n]

    def push(self, q: np.ndarray) -> None:
        self._positions = np.vstack([self._positions[1:], np.asarray(q, dtype=float)[None, : self.n]])

    @property
    def flat(self) -> np.ndarray:
        """The positions (length * n,), the oldest configuration first."""
        return self._positions.ravel().copy()
