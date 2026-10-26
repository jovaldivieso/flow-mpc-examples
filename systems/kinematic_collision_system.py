"""``KinematicCollisionSystem``: a system whose configuration the planners move among spherical obstacles.

What the planners (``CasadiPlanner``, ``CasadiTrajectoryProjector``, ``OmplPlanner``) need of a system beyond
``SystemSimulator``: the obstacles, the bounding spheres of the system at a configuration, the box a path planner
searches in, and the configuration that reaches a goal.
"""

from abc import abstractmethod

import numpy as np

from .system import SystemSimulator

COLLISION_BUFFER = 0.005  # [m] linearization safety margin for half-space obstacle constraints


class KinematicCollisionSystem(SystemSimulator):
    """A system with ``n`` positions (the configuration ``q``), spherical obstacles and a bounding-sphere model."""

    @property
    @abstractmethod
    def obstacles(self) -> list[tuple[np.ndarray, float]]:
        """The static spherical obstacles as ``(center, radius)``."""

    @property
    @abstractmethod
    def path_lower(self) -> np.ndarray:
        """The lower corner ``(n,)`` of the finite box a path planner searches in."""

    @property
    @abstractmethod
    def path_upper(self) -> np.ndarray:
        """The upper corner ``(n,)`` of the finite box a path planner searches in."""

    @abstractmethod
    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (S, d), position Jacobians (S, d, n), and radii (S,) at configuration ``q``."""

    @abstractmethod
    def goal_configuration(self, goal: np.ndarray, q_reference: np.ndarray) -> np.ndarray:
        """The configuration the planners drive to for ``goal``, the nearest one to ``q_reference`` if there are several.
        Raises ``RuntimeError`` if none reaches the goal."""

    def is_collision_free(self, q: np.ndarray) -> bool:
        """True if no bounding sphere at configuration ``q`` penetrates any obstacle."""
        if not self.obstacles:
            return True
        centers, _, radii = self.collision_spheres(np.asarray(q, dtype=float)[: self.n])
        for obs_pos, obs_radius in self.obstacles:
            if np.any(np.linalg.norm(centers - obs_pos[None, :], axis=1) < radii + obs_radius):
                return False
        return True

    def collision_halfspaces(self, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Linearized half-spaces ``A @ q >= b`` keeping all robot spheres outside all obstacles along ``q_trajectory`` (H, n).

        Returns ``A`` of shape ``(H, P, n)`` and ``b`` of shape ``(H, P)``, where ``P = num_spheres * num_obstacles``.
        """
        q_traj = np.asarray(q_trajectory, dtype=float)
        horizon = len(q_traj)
        if not self.obstacles:
            return np.zeros((horizon, 0, self.n)), np.zeros((horizon, 0))
        _, _, radii = self.collision_spheres(q_traj[0])
        num_pairs = len(radii) * len(self.obstacles)
        A = np.zeros((horizon, num_pairs, self.n), dtype=float)
        b = np.zeros((horizon, num_pairs), dtype=float)
        for k, q in enumerate(q_traj):
            centers, jacobians, radii = self.collision_spheres(q)
            pair = 0
            for s in range(len(radii)):
                p_s, J_s, r_s = centers[s], jacobians[s], radii[s]
                for obs_pos, obs_radius in self.obstacles:
                    diff = p_s - obs_pos
                    dist = float(np.linalg.norm(diff))
                    normal = diff / dist if dist > 1e-9 else np.eye(len(obs_pos))[0]
                    clearance = dist - (r_s + obs_radius + COLLISION_BUFFER)
                    a_vec = normal @ J_s
                    A[k, pair] = a_vec
                    b[k, pair] = float(a_vec @ q) - clearance
                    pair += 1
        return A, b
