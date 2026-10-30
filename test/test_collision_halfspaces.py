from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from systems.sphere_collision import COLLISION_BUFFER
from test.planner_fixtures import pinned_config

OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
    {"position": [0.40, -0.30, 0.30], "radius": 0.07},
]
Q_MID = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8])


def _loop_halfspaces(simulator: FrankaSystem, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One half-space per node, sphere and obstacle, straight from their definition."""
    horizon = len(q_trajectory)
    num_pairs = len(simulator.collision_spheres(q_trajectory[0])[2]) * len(simulator.collision.obstacles)
    A = np.zeros((horizon, num_pairs, simulator.n))
    b = np.zeros((horizon, num_pairs))
    for k, q in enumerate(q_trajectory):
        centers, jacobians, radii = simulator.collision_spheres(q)
        pair = 0
        for sphere in range(len(radii)):
            for position, radius in simulator.collision.obstacles:
                diff = centers[sphere] - position
                dist = float(np.linalg.norm(diff))
                normal = diff / dist if dist > 1e-9 else np.eye(3)[0]
                a_vec = normal @ jacobians[sphere]
                A[k, pair] = a_vec
                b[k, pair] = a_vec @ q - (dist - (radii[sphere] + radius + COLLISION_BUFFER))
                pair += 1
    return A, b


class CollisionHalfspacesTests(unittest.TestCase):
    def test_the_vectorized_half_spaces_are_those_of_the_definition(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES))
        q_trajectory = Q_MID + 0.2 * np.random.default_rng(0).standard_normal((10, 7))

        A, b = simulator.collision.halfspaces(q_trajectory)
        A_loop, b_loop = _loop_halfspaces(simulator, q_trajectory)

        self.assertEqual(A.shape, (10, 23 * 3, 7))
        np.testing.assert_allclose(A, A_loop, atol=1e-12)
        np.testing.assert_allclose(b, b_loop, atol=1e-12)

    def test_a_sphere_centered_on_an_obstacle_gets_a_fixed_normal_instead_of_nan(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        center = simulator.collision_spheres(Q_MID)[0][0]
        simulator.collision.obstacles[0] = (center.copy(), 0.05)

        A, b = simulator.collision.halfspaces(Q_MID[None, :])

        self.assertTrue(np.isfinite(A).all() and np.isfinite(b).all())
        np.testing.assert_allclose(A, _loop_halfspaces(simulator, Q_MID[None, :])[0], atol=1e-12)

    def test_without_obstacles_there_are_no_rows(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=[]))

        A, b = simulator.collision.halfspaces(np.tile(Q_MID, (4, 1)))

        self.assertEqual((A.shape, b.shape), ((4, 0, 7), (4, 0)))


if __name__ == "__main__":
    unittest.main()
