from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

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
        self.simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES))
        self.model = self.simulator.collision

    def test_the_sets_and_their_linearization_come_from_one_evaluation_of_the_spheres(self) -> None:
        q_trajectory = Q_MID + 0.1 * np.random.default_rng(1).standard_normal((6, 7))
        expected_sets = self.model.sets_along(Q_MID, q_trajectory)
        expected_A, expected_b = self.model.linearize(expected_sets, q_trajectory)

        with mock.patch.object(self.model, "spheres_along", wraps=self.model.spheres_along) as spheres_along:
            sets, A, b = self.model.linearized_sets(Q_MID, q_trajectory)

        self.assertEqual(spheres_along.call_count, 1)
        np.testing.assert_allclose(sets[0], expected_sets[0])
        np.testing.assert_allclose(sets[1], expected_sets[1])
        np.testing.assert_allclose(A, expected_A, atol=1e-12)
        np.testing.assert_allclose(b, expected_b, atol=1e-12)

    def test_without_obstacles_the_linearized_sets_are_empty(self) -> None:
        free = FrankaSystem(pinned_config())

        (normals, offsets), A, b = free.collision.linearized_sets(Q_MID, np.tile(Q_MID, (3, 1)))

        self.assertEqual((normals.shape, offsets.shape, A.shape, b.shape), ((3, 0, 0, 3), (3, 0, 0), (3, 0, 7), (3, 0)))

    def test_penetration_checks_do_not_evaluate_the_jacobians(self) -> None:
        q_trajectory = Q_MID + 0.1 * np.random.default_rng(2).standard_normal((4, 7))

        with mock.patch.object(self.model, "spheres_along", side_effect=AssertionError("Jacobians evaluated")):
            self.model.trajectory_is_collision_free(q_trajectory)
            self.model.free_prefix(q_trajectory)

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

    def test_a_trajectory_is_collision_free_exactly_when_every_node_is(self) -> None:
        def penetrates(q: np.ndarray) -> bool:  # one obstacle at a time, as the definition reads
            centers, _, radii = self.simulator.collision_spheres(q)
            return any(np.any(np.linalg.norm(centers - position, axis=1) < radii + radius) for position, radius in self.model.obstacles)

        rng = np.random.default_rng(2)
        for _ in range(10):
            q_trajectory = Q_MID + 0.6 * rng.standard_normal((8, 7))
            expected = not any(penetrates(q) for q in q_trajectory)
            self.assertEqual(self.model.trajectory_is_collision_free(q_trajectory), expected)
            for q in q_trajectory:
                self.assertEqual(self.model.is_collision_free(q), not penetrates(q))
        self.assertTrue(self.model.trajectory_is_collision_free(np.zeros((0, 7))))

    def test_the_free_prefix_ends_at_the_first_row_that_penetrates_an_obstacle(self) -> None:
        # a random trajectory around the middle pose, which reaches an obstacle
        rng = np.random.default_rng(3)
        q_trajectory = Q_MID + 0.6 * rng.standard_normal((40, 7))
        rows_free = [self.model.is_collision_free(q) for q in q_trajectory]
        self.assertIn(False, rows_free)  # the draw reaches an obstacle

        free = self.model.free_prefix(q_trajectory)

        self.assertEqual(free, rows_free.index(False))
        self.assertTrue(self.model.trajectory_is_collision_free(q_trajectory[:free]))
        self.assertFalse(self.model.trajectory_is_collision_free(q_trajectory[: free + 1]))

    def test_the_free_prefix_is_every_row_of_a_free_trajectory_and_none_of_an_empty_one(self) -> None:
        free_trajectory = np.tile(Q_MID, (5, 1))
        self.assertTrue(self.model.is_collision_free(Q_MID))

        self.assertEqual(self.model.free_prefix(free_trajectory), 5)
        self.assertEqual(self.model.free_prefix(np.zeros((0, 7))), 0)

    def test_the_free_prefix_is_zero_when_the_first_row_penetrates(self) -> None:
        center = self.simulator.collision_spheres(Q_MID)[0][-1]
        simulator = FrankaSystem(pinned_config(obstacles=[{"position": center.tolist(), "radius": 0.05}]))

        self.assertEqual(simulator.collision.free_prefix(np.tile(Q_MID, (4, 1))), 0)

    def test_without_obstacles_every_row_is_free(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=[]))

        self.assertEqual(simulator.collision.free_prefix(np.tile(Q_MID, (4, 1))), 4)

    def test_without_obstacles_there_are_no_rows(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=[]))

        A, b = simulator.collision.halfspaces(Q_MID, np.tile(Q_MID, (4, 1)))

        self.assertEqual((A.shape, b.shape, simulator.collision.num_halfspaces), ((4, 0, 7), (4, 0), 0))


if __name__ == "__main__":
    unittest.main()
