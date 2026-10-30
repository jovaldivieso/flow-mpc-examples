"""``SphereCollisionModel``: the spherical obstacles of a system and the checks and linearizations built on its bounding spheres.

A system holds one (``TripleIntegratorBase.collision``) and hands it the function of its own kinematics that gives its
bounding spheres, so the planners ask the model for what they enforce and never need to know how the spheres move.
"""

from collections.abc import Callable

import numpy as np

COLLISION_BUFFER = 0.005  # [m] linearization safety margin for half-space obstacle constraints

# q -> bounding sphere centers (S, d), position Jacobians (S, d, n), and radii (S,)
SphereFunction = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]


class SphereCollisionModel:
    """Static spherical obstacles ``(center, radius)`` and the bounding spheres (``spheres``) of a system with ``n``
    configuration coordinates."""

    def __init__(self, obstacles: list[tuple[np.ndarray, float]], spheres: SphereFunction, n: int) -> None:
        self.obstacles = obstacles
        self.spheres = spheres
        self.n = int(n)

    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if no bounding sphere at configuration ``q`` penetrates any obstacle."""
        if not self.obstacles:
            return True
        centers, _, radii = self.spheres(np.asarray(q, dtype=float)[: self.n])
        for obs_pos, obs_radius in self.obstacles:
            if np.any(np.linalg.norm(centers - obs_pos[None, :], axis=1) < radii + obs_radius):
                return False
        return True

    def halfspaces(self, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearized half-spaces ``A @ q >= b`` keeping all robot spheres outside all obstacles along ``q_trajectory`` (H, n).

        Returns ``A`` of shape ``(H, P, n)`` and ``b`` of shape ``(H, P)``, where ``P = num_spheres * num_obstacles``.
        """
        q_traj = np.asarray(q_trajectory, dtype=float)
        horizon = len(q_traj)
        if not self.obstacles:
            return np.zeros((horizon, 0, self.n)), np.zeros((horizon, 0))
        spheres = [self.spheres(q) for q in q_traj]
        centers, jacobians = np.stack([sphere[0] for sphere in spheres]), np.stack([sphere[1] for sphere in spheres])
        radii = spheres[0][2]
        obstacle_positions = np.array([position for position, _ in self.obstacles])
        obstacle_radii = np.array([radius for _, radius in self.obstacles])
        # (H, S, O, 3): the vector from every obstacle to every sphere at every node
        diff = centers[:, :, None, :] - obstacle_positions[None, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        normal = np.where(dist[..., None] > 1e-9, diff / np.maximum(dist, 1e-9)[..., None], np.eye(diff.shape[-1])[0])
        clearance = dist - (radii[None, :, None] + obstacle_radii[None, None, :] + COLLISION_BUFFER)
        a_vec = np.einsum("hsoc,hscn->hson", normal, jacobians)
        b_vec = np.einsum("hson,hn->hso", a_vec, q_traj) - clearance
        # the pair index is sphere * num_obstacles + obstacle
        return a_vec.reshape(horizon, -1, self.n), b_vec.reshape(horizon, -1)
