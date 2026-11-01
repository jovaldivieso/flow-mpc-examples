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


class SphereCollisionModel:
    """Static spherical obstacles ``(center, radius)`` and the bounding spheres (``spheres``) of a system with ``n``
    configuration coordinates."""

    def __init__(self, obstacles: list[tuple[np.ndarray, float]], spheres: SphereFunction, n: int) -> None:
        self.obstacles = obstacles
        self.spheres = spheres
        self.n = int(n)
        self.free_space = FreeSpaceSets(obstacles, COLLISION_BUFFER)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if no bounding sphere at configuration ``q`` penetrates any obstacle."""
        if not self.obstacles:
            return True
        centers, _, radii = self.spheres(np.asarray(q, dtype=float)[: self.n])
        for obs_pos, obs_radius in self.obstacles:
            if np.any(np.linalg.norm(centers - obs_pos[None, :], axis=1) < radii + obs_radius):
                return False
        return True

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
        start, _, radii = self.spheres(np.asarray(q_current, dtype=float)[: self.n])
        centers = np.stack([self.spheres(q)[0] for q in q_traj])
        previous = np.concatenate([start[None], centers[:-1]])
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
        spheres = [self.spheres(q) for q in q_traj]
        centers, jacobians = np.stack([sphere[0] for sphere in spheres]), np.stack([sphere[1] for sphere in spheres])
        # (H, S, K, n): the gradient of normal . center(q) in q, and its value at every node
        a_vec = np.einsum("hski,hsin->hskn", normals, jacobians)
        value = np.einsum("hski,hsi->hsk", normals, centers)
        b_vec = offsets - value + np.einsum("hskn,hn->hsk", a_vec, q_traj)
        return a_vec.reshape(horizon, -1, self.n), b_vec.reshape(horizon, -1)

    def halfspaces(self, q_current: np.ndarray, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The sets along ``q_trajectory`` linearized along it: ``linearize(sets_along(...), q_trajectory)``."""
        return self.linearize(self.sets_along(q_current, q_trajectory), q_trajectory)
