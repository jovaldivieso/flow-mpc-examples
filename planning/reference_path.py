import numpy as np


class ReferencePath:
    """A joint-space path (M, n) that an optimizing planner follows instead of heading straight for its end.

    The arm's progress along the path is the arc length of the point of the path nearest to it, and never goes back.
    ``ahead`` is the point ``lookahead`` (rad, arc length) further along, or the end of the path. The nearest point is only
    searched within one lookahead of the progress, so a path that comes back close to itself is not skipped over.
    """

    def __init__(self, waypoints: np.ndarray) -> None:
        self.waypoints = np.asarray(waypoints, dtype=float)
        if self.waypoints.ndim != 2 or len(self.waypoints) < 1:
            raise ValueError(f"A path needs at least one row of joint positions, got {self.waypoints.shape}.")
        steps = np.linalg.norm(np.diff(self.waypoints, axis=0), axis=1)
        self.arc_length = np.concatenate([[0.0], np.cumsum(steps)])
        self.progress = 0.0

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    def ahead(self, q: np.ndarray, lookahead: float) -> np.ndarray:
        """The point ``lookahead`` along the path from the arm at ``q``; moves the progress to where ``q`` is."""
        self.progress = max(self.progress, self._arc_of_nearest_point(np.asarray(q, dtype=float), float(lookahead)))
        return self._point_at(min(self.progress + float(lookahead), self.length))

    def _arc_of_nearest_point(self, q: np.ndarray, window: float) -> float:
        """The arc length of the point of the path nearest to ``q`` among those from the progress to ``window`` beyond it."""
        if len(self.waypoints) == 1:
            return 0.0
        start, segment = self.waypoints[:-1], np.diff(self.waypoints, axis=0)
        length = np.linalg.norm(segment, axis=1)
        along = np.einsum("ij,ij->i", q - start, segment) / np.maximum(length, 1e-9) ** 2
        behind = np.clip((self.progress - self.arc_length[:-1]) / np.maximum(length, 1e-9), 0.0, 1.0)
        fraction = np.clip(along, behind, 1.0)
        distance = np.linalg.norm(start + fraction[:, None] * segment - q, axis=1)
        distance[(self.arc_length[1:] < self.progress) | (self.arc_length[:-1] > self.progress + window)] = np.inf
        nearest = int(np.argmin(distance))
        return float(self.arc_length[nearest] + fraction[nearest] * length[nearest])

    def _point_at(self, arc: float) -> np.ndarray:
        return np.array([np.interp(arc, self.arc_length, joint) for joint in self.waypoints.T])
