from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from systems.free_space_sets import FreeSpaceSets
from test.planner_fixtures import pinned_config

OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
    {"position": [0.40, -0.30, 0.30], "radius": 0.07},
]
Q_MID = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.8])


def _spheres(*centers: tuple, radius: float = 0.2) -> list[tuple[np.ndarray, float]]:
    return [(np.array(center, dtype=float), radius) for center in centers]


class FreeSpaceSetsTests(unittest.TestCase):
    def test_the_half_space_touches_the_inflated_obstacle_and_contains_the_segment(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0)), margin=0.0)
        start, end = np.array([[0.0, 0.1, 0.0]]), np.array([[0.5, 0.0, 0.0]])

        normals, offsets = sets.build(start, end, np.array([0.1]))

        self.assertEqual((normals.shape, offsets.shape), ((1, 1, 3), (1, 1)))
        np.testing.assert_allclose(np.linalg.norm(normals[0, 0]), 1.0)
        for fraction in (0.0, 0.5, 1.0):
            self.assertGreaterEqual(normals[0, 0] @ (start[0] + fraction * (end[0] - start[0])), offsets[0, 0] - 1e-12)
        # the plane touches the obstacle inflated by the radius of the robot sphere (0.2 + 0.1)
        np.testing.assert_allclose(offsets[0, 0] - normals[0, 0] @ np.array([1.0, 0.0, 0.0]), 0.3)

    def test_the_margin_inflates_the_obstacles(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0)), margin=0.05)

        normals, offsets = sets.build(np.zeros((1, 3)), np.array([[0.2, 0.0, 0.0]]), np.array([0.1]))

        np.testing.assert_allclose(offsets[0, 0] - normals[0, 0] @ np.array([1.0, 0.0, 0.0]), 0.35)

    def test_an_obstacle_excluded_by_an_earlier_half_space_is_skipped_and_the_set_is_padded(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (3.0, 0.0, 0.0)), margin=0.0)

        normals, offsets = sets.build(np.zeros((1, 3)), np.array([[0.5, 0.0, 0.0]]), np.array([0.1]))

        np.testing.assert_allclose(normals[0, 0], [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(offsets[0, 0], -0.7)
        np.testing.assert_allclose(normals[0, 1:], 0.0)
        np.testing.assert_allclose(offsets[0, 1:], -1.0)

    def test_an_obstacle_beside_the_path_gets_its_own_half_space(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), margin=0.0)

        normals, _ = sets.build(np.zeros((1, 3)), np.array([[0.4, 0.0, 0.0]]), np.array([0.1]))

        self.assertEqual(int((np.linalg.norm(normals[0], axis=1) > 0.5).sum()), 2)

    def test_a_segment_through_an_obstacle_falls_back_to_the_half_space_of_its_start(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0)), margin=0.0)
        start, end = np.zeros((1, 3)), np.array([[2.0, 0.0, 0.0]])

        normals, offsets = sets.build(start, end, np.array([0.1]))

        np.testing.assert_allclose(normals[0, 0], [-1.0, 0.0, 0.0])
        self.assertGreater(normals[0, 0] @ start[0], offsets[0, 0])

    def test_a_motionless_sphere_and_a_sphere_on_an_obstacle_give_finite_half_spaces(self) -> None:
        sets = FreeSpaceSets(_spheres((1.0, 0.0, 0.0), (0.0, 0.0, 0.0)), margin=0.0)

        normals, offsets = sets.build(np.zeros((2, 3)), np.zeros((2, 3)), np.array([0.1, 0.1]))

        self.assertTrue(np.isfinite(normals).all() and np.isfinite(offsets).all())

    def test_the_set_size_is_capped_whatever_the_number_of_obstacles(self) -> None:
        obstacles = _spheres(*[(1.0, 0.3 * i, 0.0) for i in range(-5, 6)], radius=0.1)

        capped = FreeSpaceSets(obstacles, margin=0.0, max_size=4)

        self.assertEqual(capped.build(np.zeros((1, 3)), np.array([[0.2, 0.0, 0.0]]), np.array([0.05]))[0].shape, (1, 4, 3))
        self.assertEqual(FreeSpaceSets(obstacles[:2], margin=0.0, max_size=4).size, 2)

    def test_without_obstacles_there_are_no_half_spaces(self) -> None:
        normals, offsets = FreeSpaceSets([], margin=0.0).build(np.zeros((3, 3)), np.ones((3, 3)), np.full(3, 0.1))

        self.assertEqual((normals.shape, offsets.shape), ((3, 0, 3), (3, 0)))


class CollisionHalfspacesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES))
        self.model = self.simulator.collision

    def test_the_rows_are_the_free_space_half_spaces_of_the_spheres_linearized_at_every_node(self) -> None:
        q_trajectory = Q_MID + 0.1 * np.random.default_rng(0).standard_normal((6, 7))

        A, b = self.model.halfspaces(Q_MID, q_trajectory)

        num_spheres = len(self.simulator.collision_spheres(Q_MID)[2])
        self.assertEqual(self.model.num_halfspaces, num_spheres * 3)
        self.assertEqual((A.shape, b.shape), ((6, num_spheres * 3, 7), (6, num_spheres * 3)))
        for k in (0, 3, 5):
            previous = self.simulator.collision_spheres(Q_MID if k == 0 else q_trajectory[k - 1])[0]
            centers, jacobians, radii = self.simulator.collision_spheres(q_trajectory[k])
            normals, offsets = self.model.free_space.build(previous, centers, radii)  # the set around the step into node k
            for sphere in range(num_spheres):
                for i in range(3):
                    row = sphere * 3 + i
                    np.testing.assert_allclose(A[k, row], normals[sphere, i] @ jacobians[sphere], atol=1e-12)
                    expected = offsets[sphere, i] - normals[sphere, i] @ centers[sphere] + A[k, row] @ q_trajectory[k]
                    np.testing.assert_allclose(b[k, row], expected, atol=1e-12)

    def test_linearizing_the_same_sets_elsewhere_keeps_the_half_spaces_in_cartesian_space(self) -> None:
        q_trajectory = np.tile(Q_MID, (3, 1))
        sets = self.model.sets_along(Q_MID, q_trajectory)
        shifted = q_trajectory + 0.05

        A, b = self.model.linearize(sets, shifted)

        centers, jacobians, _ = self.simulator.collision_spheres(shifted[0])
        normals, offsets = sets[0][0], sets[1][0]
        # the linearized rows at the new reference reproduce normal . center(q) - offset at that reference
        residual = A[0] @ shifted[0] - b[0]
        expected = np.einsum("ski,si->sk", normals, centers).ravel() - offsets.ravel()
        np.testing.assert_allclose(residual, expected, atol=1e-9)

    def test_the_linearized_rows_hold_at_the_reference_when_the_sphere_is_clear_of_the_obstacles(self) -> None:
        A, b = self.model.halfspaces(Q_MID, np.tile(Q_MID, (3, 1)))

        self.assertTrue(self.model.is_collision_free(Q_MID))
        self.assertGreaterEqual(float((A[0] @ Q_MID - b[0]).min()), -1e-9)

    def test_a_configuration_satisfying_the_rows_exactly_is_clear_of_the_obstacles(self) -> None:
        # the half-spaces exclude every obstacle, so a small step that keeps all rows satisfied cannot touch one
        q_trajectory = np.tile(Q_MID, (2, 1))
        A, b = self.model.halfspaces(Q_MID, q_trajectory)
        step = 0.01 * np.random.default_rng(1).standard_normal((20, 7))

        for delta in step:
            if np.all(A[0] @ (Q_MID + delta) >= b[0]):
                self.assertTrue(self.model.is_collision_free(Q_MID + delta))

    def test_without_obstacles_there_are_no_rows(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=[]))

        A, b = simulator.collision.halfspaces(Q_MID, np.tile(Q_MID, (4, 1)))

        self.assertEqual((A.shape, b.shape, simulator.collision.num_halfspaces), ((4, 0, 7), (4, 0), 0))


if __name__ == "__main__":
    unittest.main()
