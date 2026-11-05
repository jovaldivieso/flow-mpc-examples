"""``SphereCollisionModel``: the spherical obstacles of a system and the checks and linearizations built on its bounding spheres.

A system holds one (``TripleIntegratorBase.collision``) and hands it the function of its own kinematics that gives its
bounding spheres, so the planners ask the model for what they enforce and never need to know how the spheres move.
"""

from collections.abc import Callable

import numpy as np

from .free_space_sets import FreeSpaceSets

COLLISION_BUFFER = 0.005  # [m] safety margin of the free-space sets around the obstacles

# q -> bounding sphere centers (S, d), position Jacobians (S, d, n), and radii (S,)
SphereFunction = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]
# (H, n) configurations -> the same at every row: centers (H, S, d), Jacobians (H, S, d, n), and radii (S,)
TrajectorySphereFunction = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


class SphereCollisionModel:
    """Static spherical obstacles ``(center, radius)`` and the bounding spheres of a system with ``n`` configuration
    coordinates: ``spheres`` at one configuration and ``spheres_along`` at the rows of a trajectory in one call."""

    def __init__(
        self, obstacles: list[tuple[np.ndarray, float]], spheres: SphereFunction, spheres_along: TrajectorySphereFunction, n: int
    ) -> None:
        self.obstacles = obstacles
        self.spheres = spheres
        self.spheres_along = spheres_along
        self.n = int(n)
        self.free_space = FreeSpaceSets(obstacles, COLLISION_BUFFER)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if no bounding sphere at configuration ``q`` penetrates any obstacle."""
        if not self.obstacles:
            return True
        centers, _, radii = self.spheres(np.asarray(q, dtype=float)[: self.n])
        return not self._penetrates(centers[None], radii)

    def trajectory_is_collision_free(self, q_trajectory: np.ndarray) -> bool:
        """True if no bounding sphere penetrates any obstacle at any row of ``q_trajectory`` (H, n)."""
        q_traj = np.asarray(q_trajectory, dtype=float)
        if not self.obstacles or len(q_traj) == 0:
            return True
        centers, _, radii = self.spheres_along(q_traj[:, : self.n])
        return not self._penetrates(centers, radii)

    def free_prefix(self, q_trajectory: np.ndarray) -> int:
        """The number of leading rows of ``q_trajectory`` (H, n) before the first one at which a bounding sphere
        penetrates an obstacle (all of them if none does)."""
        q_traj = np.asarray(q_trajectory, dtype=float)
        if not self.obstacles or len(q_traj) == 0:
            return len(q_traj)
        centers, _, radii = self.spheres_along(q_traj[:, : self.n])
        hit = self._penetrating_rows(centers, radii)
        return int(np.argmax(hit)) if hit.any() else len(q_traj)

    def _penetrating_rows(self, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """For each of the H rows of ``centers`` (H, S, d), whether a sphere of ``radii`` (S,) penetrates an obstacle."""
        positions = np.array([position for position, _ in self.obstacles], dtype=float)
        obstacle_radii = np.array([radius for _, radius in self.obstacles], dtype=float)
        distances = np.linalg.norm(centers[:, :, None, :] - positions, axis=-1)  # (H, S, obstacles)
        return np.any(distances < radii[None, :, None] + obstacle_radii, axis=(1, 2))

    def _penetrates(self, centers: np.ndarray, radii: np.ndarray) -> bool:
        """Whether a sphere of ``centers`` (H, S, d) and ``radii`` (S,) penetrates an obstacle, at any of the H rows."""
        return bool(self._penetrating_rows(centers, radii).any())

    @property
    def num_spheres(self) -> int:
        """The bounding spheres that matter: all of them, or none without obstacles."""
        return len(self.spheres(np.zeros(self.n))[2]) if self.obstacles else 0

    @property
    def num_halfspaces(self) -> int:
        """The rows ``halfspaces`` returns per node: the free-space half-spaces of every bounding sphere."""
        return self.num_spheres * self.free_space.size

    def sets_along(self, q_current: np.ndarray, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The free-space sets (see ``FreeSpaceSets``) of the bounding spheres along ``q_trajectory`` (H, n): the set of
        node ``k`` is the one around the motion of every sphere from its place at the previous node (``q_current`` for
        the first) to its place at node ``k``, so a trajectory that bends around an obstacle is inside its sets.

        Returns the normals ``(H, S, K, 3)`` and offsets ``(H, S, K)`` of the half-spaces ``normal . center >= offset``.
        """
        q_traj = np.asarray(q_trajectory, dtype=float)
        horizon = len(q_traj)
        if not self.obstacles:
            return np.zeros((horizon, 0, 0, 3)), np.zeros((horizon, 0, 0))
        centers, _, radii = self.spheres_along(np.vstack([np.asarray(q_current, dtype=float)[None, : self.n], q_traj]))
        previous, centers = centers[:-1], centers[1:]  # the set of a node is around the motion from the previous one
        normals, offsets = self.free_space.build(previous.reshape(-1, 3), centers.reshape(-1, 3), np.tile(radii, horizon))
        return normals.reshape(horizon, len(radii), -1, 3), offsets.reshape(horizon, len(radii), -1)

    def linearize(self, sets: tuple[np.ndarray, np.ndarray], q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The half-spaces ``A @ q >= b`` that keep the bounding spheres in ``sets`` (``sets_along``), linearized in the
        configuration at the nodes of ``q_trajectory`` (H, n): the sets stay as they are, only the kinematics are
        linearized again.

        Returns ``A`` of shape ``(H, P, n)`` and ``b`` of shape ``(H, P)``, where ``P = num_halfspaces`` and the row
        index is ``sphere * set_size + i``.
        """
        normals, offsets = sets
        q_traj = np.asarray(q_trajectory, dtype=float)
        horizon = len(q_traj)
        if normals.shape[1] == 0:
            return np.zeros((horizon, 0, self.n)), np.zeros((horizon, 0))
        centers, jacobians, _ = self.spheres_along(q_traj)
        # (H, S, K, n): the gradient of normal . center(q) in q, and its value at every node
        a_vec = np.einsum("hski,hsin->hskn", normals, jacobians)
        value = np.einsum("hski,hsi->hsk", normals, centers)
        b_vec = offsets - value + np.einsum("hskn,hn->hsk", a_vec, q_traj)
        return a_vec.reshape(horizon, -1, self.n), b_vec.reshape(horizon, -1)

    def halfspaces(self, q_current: np.ndarray, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The sets along ``q_trajectory`` linearized along it: ``linearize(sets_along(...), q_trajectory)``."""
        return self.linearize(self.sets_along(q_current, q_trajectory), q_trajectory)
