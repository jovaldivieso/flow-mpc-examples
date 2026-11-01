"""``FreeSpaceSets``: the convex free-space set of every bounding sphere around a segment of its motion (BoundPlanner,
Oelerich et al., 2025, Eq. 1 and 13; used by SafeFlowMPC, 2026).

For a sphere moving from ``p0`` to ``p1`` and a spherical obstacle inflated by the sphere's radius, the half-space

    n^T x >= n^T c + r,    n = (p* - c) / ||p* - c||

touches the obstacle at its point nearest to the segment (``p*`` is the point of the segment nearest to the center
``c``), and the whole segment lies in it. The set of a sphere is made of such half-spaces, nearest obstacle first,
skipping every obstacle that a half-space of the set already excludes entirely. It holds at most ``max_size``
half-spaces, however many obstacles there are.
"""

import numpy as np

MAX_SET_SIZE = 4  # half-spaces per sphere (a stage of the MPC grows with it)
EXCLUDED_TOLERANCE = 1e-4  # [m] how far an obstacle may still reach into a half-space and count as excluded by it


class FreeSpaceSets:
    """Builds the sets of the bounding spheres around spherical obstacles ``(center, radius)``; ``margin`` [m] inflates
    the obstacles on top of the sphere radii."""

    def __init__(self, obstacles: list[tuple[np.ndarray, float]], margin: float, max_size: int = MAX_SET_SIZE) -> None:
        self.obstacles = obstacles
        self.margin = float(margin)
        self.max_size = int(max_size)

    @property
    def size(self) -> int:
        """The half-spaces of every set (``K``): ``max_size`` unless there are fewer obstacles."""
        return min(self.max_size, len(self.obstacles))

    def build(self, start: np.ndarray, end: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The half-spaces ``normals[s, i] . x >= offsets[s, i]`` of every sphere ``s`` moving from the center
        ``start[s]`` to ``end[s]`` (both (S, 3)), with normals (S, K, 3) and offsets (S, K), ``K = size``.

        A set with fewer than ``K`` half-spaces is padded with ``0 . x >= -1``, which every point satisfies. If a segment
        enters an inflated obstacle, the half-space is the one of the obstacle's point nearest to ``start`` instead
        (the segment is no longer in it, the start is on its free side if it is clear of the obstacle).
        """
        num_spheres = len(radii)
        normals = np.zeros((num_spheres, self.size, 3))
        offsets = np.full((num_spheres, self.size), -1.0)
        if self.size == 0:
            return normals, offsets

        centers = np.array([center for center, _ in self.obstacles])
        inflated = np.array([radius for _, radius in self.obstacles])[None, :] + radii[:, None] + self.margin  # (S, O)
        nearest = self._nearest_points(start, end, centers)  # (S, O, 3)
        away = nearest - centers[None, :, :]
        gap = np.linalg.norm(away, axis=-1) - inflated
        inside = gap < 0.0
        away = np.where(inside[..., None], start[:, None, :] - centers[None, :, :], away)
        length = np.linalg.norm(away, axis=-1, keepdims=True)
        direction = np.where(length > 1e-9, away / np.maximum(length, 1e-9), np.eye(3)[0])  # (S, O, 3)
        plane = np.einsum("soc,oc->so", direction, centers) + inflated  # offset of the half-space of every pair

        excluded = np.zeros(gap.shape, dtype=bool)
        spheres = np.arange(num_spheres)
        for i in range(self.size):
            nearest_open = np.argmin(np.where(excluded, np.inf, gap), axis=1)
            has_open = ~excluded[spheres, nearest_open]
            normal, offset = direction[spheres, nearest_open], plane[spheres, nearest_open]
            normals[has_open, i], offsets[has_open, i] = normal[has_open], offset[has_open]
            # An obstacle whose inflated sphere lies entirely on the excluded side of the half-space adds nothing to the set
            reach = np.einsum("sc,oc->so", normal, centers) + inflated
            excluded |= has_open[:, None] & (reach - offset[:, None] <= EXCLUDED_TOLERANCE)
            excluded[spheres, nearest_open] |= has_open
        return normals, offsets

    @staticmethod
    def _nearest_points(start: np.ndarray, end: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """The point of the segment ``start[s]`` -> ``end[s]`` nearest to every center, (S, O, 3)."""
        direction = end - start
        squared_length = np.einsum("sc,sc->s", direction, direction)
        along = np.einsum("soc,sc->so", centers[None, :, :] - start[:, None, :], direction)
        fraction = np.where(squared_length[:, None] > 1e-12, along / np.maximum(squared_length, 1e-12)[:, None], 0.0)
        return start[:, None, :] + np.clip(fraction, 0.0, 1.0)[..., None] * direction[:, None, :]
