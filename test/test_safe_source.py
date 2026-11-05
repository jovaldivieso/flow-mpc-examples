from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.dagger.safe_pairs import PAIRS_FILE, PooledPairs, SafePairs
from learning.models.safe_source import PATH_SEGMENTS, SafeSource
from planning.casadi_projector import Projection
from planning.planner import PlannerSolveError
from planning.planner_config import ProjectionConfig
from systems.franka_system import FrankaSystem
from test.planner_fixtures import pinned_config, rest_state

HORIZON = 10
FRAMES = 5
REST_TOLERANCE = 0.1  # of the soft rest of the filter: the final velocity and acceleration of a source


def _states_and_labels(simulator: FrankaSystem, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """Arm states at rest, each with the chunk of positions of small random jerks as its label."""
    rng = np.random.default_rng(0)
    states = np.array([rest_state(simulator) + np.concatenate([0.01 * k * np.ones(simulator.n), np.zeros(2 * simulator.n)]) for k in range(frames)])
    labels = np.array([simulator.rollout_positions(state, 0.1 * rng.uniform(-1, 1, (HORIZON, simulator.nu)) * simulator.max_action)[1:] for state in states])
    return states, labels


def _jerks_of(simulator: FrankaSystem, state: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """The jerks (H, nu) whose rollout from ``state`` has the positions ``positions`` (the dynamics are linear and alike per joint)."""
    horizon = len(positions)
    base = simulator.rollout_positions(state, np.zeros((horizon, simulator.nu)))[1:]
    response = np.zeros((horizon, horizon))
    for step in range(horizon):
        jerks = np.zeros((horizon, simulator.nu))
        jerks[step, 0] = 1.0
        response[:, step] = (simulator.rollout_positions(state, jerks)[1:] - base)[:, 0]
    return np.linalg.solve(response, positions - base)


def _assert_within_limits_and_at_rest(test: unittest.TestCase, sim: FrankaSystem, state: np.ndarray, positions: np.ndarray, rest_tolerance: float) -> None:
    """The jerks and states of the plan ``positions`` from ``state`` are within the limits and its final velocity and acceleration are small."""
    jerks = _jerks_of(sim, state, positions)
    states = sim.rollout_states(state, jerks)[1:]
    test.assertTrue(np.all(np.abs(jerks) <= sim.max_action + 1e-4))
    test.assertTrue(np.all(states <= sim.state_upper_bounds + 1e-6) and np.all(states >= sim.state_lower_bounds - 1e-6))
    test.assertLess(np.abs(states[-1, sim.n :]).max(), rest_tolerance)


class SafeSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.projection = ProjectionConfig.from_mapping(cls.config)
        cls.states, cls.labels = _states_and_labels(cls.simulator)

    def setUp(self) -> None:
        self.source = SafeSource(self.simulator, self.projection, HORIZON, seed=0)

    def _path(self, row: int = 0, index: int = 0) -> np.ndarray:
        path = self.source.path(self.states[row], self.labels[row], index)
        self.assertIsNotNone(path)
        return path

    def _assert_a_safe_plan(self, state: np.ndarray, positions: np.ndarray, rest_tolerance: float) -> None:
        _assert_within_limits_and_at_rest(self, self.simulator, state, positions, rest_tolerance)

    def test_a_path_is_a_finite_sequence_of_chunks_from_the_source_to_the_target(self) -> None:
        path = self._path()

        self.assertEqual(path.shape, (PATH_SEGMENTS + 1, HORIZON, self.simulator.n))
        self.assertTrue(np.isfinite(path).all())

    def test_the_path_starts_at_a_source_within_the_limits_that_ends_at_rest(self) -> None:
        for index in range(3):
            with self.subTest(index=index):
                self._assert_a_safe_plan(self.states[0], self._path(0, index)[0], REST_TOLERANCE)

    def test_the_path_ends_at_the_label_made_safe(self) -> None:
        target = self._path()[-1]

        self._assert_a_safe_plan(self.states[0], target, 0.02)
        self.assertLess(np.abs(target - self.labels[0]).max(), 0.2)  # the filter tracks the label, moving it only a little

    def test_a_label_that_stays_in_place_is_its_own_target(self) -> None:
        label = np.tile(self.states[0][: self.simulator.n], (HORIZON, 1))

        target = self.source.path(self.states[0], label, 0)[-1]

        np.testing.assert_allclose(target, label, atol=1e-3)

    def test_the_source_is_not_the_target(self) -> None:
        path = self._path()

        self.assertGreater(np.abs(path[0] - path[-1]).max(), 1e-3)

    def test_the_endpoints_are_the_filter_plans_of_a_random_chunk_and_of_the_label(self) -> None:
        real = self.source._make_safe
        plans = []

        def make_safe(state, reference):
            plans.append((reference, real(state, reference)))
            return plans[-1][1]

        with mock.patch.object(self.source, "_make_safe", make_safe):
            path = self._path()

        self.assertEqual(len(plans), 2)
        np.testing.assert_array_equal(plans[1][0], self.labels[0])
        np.testing.assert_allclose(path[0], plans[0][1], atol=1e-12)
        np.testing.assert_allclose(path[-1], plans[1][1], atol=1e-12)

    def test_a_line_clear_of_obstacles_is_its_own_projection(self) -> None:
        with mock.patch.object(self.source, "_make_safe", wraps=self.source._make_safe) as make_safe:
            path = self._path()

        self.assertEqual(make_safe.call_count, 2)  # the source and the target only
        for segment in range(PATH_SEGMENTS + 1):
            line = path[0] + segment / PATH_SEGMENTS * (path[-1] - path[0])
            np.testing.assert_allclose(path[segment], line, atol=1e-12)

    def test_the_same_index_gives_the_same_path_whatever_was_computed_before(self) -> None:
        first = self._path(0, 3)
        self._path(1, 7)
        again = self._path(0, 3)
        fresh = SafeSource(self.simulator, self.projection, HORIZON, seed=0).path(self.states[0], self.labels[0], 3)

        np.testing.assert_allclose(first, again, atol=1e-9)
        np.testing.assert_allclose(first, fresh, atol=1e-9)

    def test_different_indices_give_different_sources_for_the_same_frame(self) -> None:
        sources = [self._path(0, index)[0] for index in (0, 1)]

        self.assertGreater(np.abs(sources[0] - sources[1]).max(), 1e-3)

    def test_a_different_seed_gives_a_different_source(self) -> None:
        other = SafeSource(self.simulator, self.projection, HORIZON, seed=1).path(self.states[0], self.labels[0], 0)

        self.assertGreater(np.abs(other[0] - self._path(0, 0)[0]).max(), 1e-3)

    def test_the_filter_is_reset_before_each_of_its_uses(self) -> None:
        with mock.patch.object(self.source._filter, "reset", wraps=self.source._filter.reset) as reset:
            self._path()

        self.assertEqual(reset.call_count, 2)

    def test_a_failing_projection_of_the_source_or_of_the_target_gives_no_path(self) -> None:
        real_project = self.source._filter.project
        for failing_call in (0, 1):
            with self.subTest(failing_call=failing_call):
                calls = []

                def project(state, reference):
                    calls.append(None)
                    if len(calls) == failing_call + 1:
                        raise PlannerSolveError("forced failure")
                    return real_project(state, reference)

                with mock.patch.object(self.source._filter, "project", project):
                    self.assertIsNone(self.source.path(self.states[0], self.labels[0], 0))

    def test_a_fallback_to_an_earlier_plan_counts_as_a_failed_projection(self) -> None:
        def project(state, reference):
            warnings.warn("falling back to the last known-safe trajectory")
            return Projection(np.zeros((HORIZON, self.simulator.n)), np.zeros((HORIZON, self.simulator.nu)))

        with mock.patch.object(self.source._filter, "project", project):
            self.assertIsNone(self.source.path(self.states[0], self.labels[0], 0))

    def test_paths_stacks_float32_paths_of_consecutive_indices(self) -> None:
        paths = self.source.paths(self.states[:3], self.labels[:3], 4)

        self.assertEqual(paths.shape, (3, PATH_SEGMENTS + 1, HORIZON, self.simulator.n))
        self.assertEqual(paths.dtype, np.float32)
        for row in range(3):
            np.testing.assert_allclose(paths[row], self.source.path(self.states[row], self.labels[row], 4 + row), atol=1e-5)

    def test_paths_leaves_nan_where_a_frame_has_no_path(self) -> None:
        real_path = self.source.path

        def path(state, label, index):
            return None if index == 1 else real_path(state, label, index)

        with mock.patch.object(self.source, "path", path):
            paths = self.source.paths(self.states[:3], self.labels[:3], 0)

        self.assertTrue(np.isnan(paths[1]).all())
        self.assertTrue(np.isfinite(paths[[0, 2]]).all())


class SafeSourceObstacleTests(unittest.TestCase):
    """With an obstacle in the workspace, every point of the path is clear of it."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config(obstacles=[{"position": [0.55, 0.30, 0.45], "radius": 0.10}])
        cls.simulator = FrankaSystem(cls.config)
        cls.state = rest_state(cls.simulator)
        cls.reference = np.tile(cls.state[:7], (HORIZON, 1))
        cls.source = SafeSource(cls.simulator, ProjectionConfig.from_mapping(cls.config), HORIZON, 0)
        cls.paths = [cls.source.path(cls.state, cls.reference, index) for index in range(4)]

    def test_the_start_is_clear_of_the_obstacle(self) -> None:
        self.assertTrue(self.simulator.collision.is_collision_free(self.state[:7]))

    def test_paths_are_found(self) -> None:
        self.assertTrue(any(path is not None for path in self.paths))

    def test_every_point_of_every_path_is_clear_of_the_obstacle(self) -> None:
        for index, path in enumerate(self.paths):
            if path is None:
                continue
            for segment, point in enumerate(path):
                with self.subTest(index=index, segment=segment):
                    self.assertTrue(self.simulator.collision.trajectory_is_collision_free(point))

    def test_the_source_and_the_target_are_within_the_limits_and_end_at_rest(self) -> None:
        for index, path in enumerate(self.paths):
            if path is None:
                continue
            for point in (path[0], path[-1]):
                with self.subTest(index=index):
                    _assert_within_limits_and_at_rest(self, self.simulator, self.state, point, REST_TOLERANCE)


class SafeSourceCutLineTests(unittest.TestCase):
    """A small obstacle on the middle chunk of the line between two known chunks cuts some of its interior points."""

    RADIUS = 0.02

    @classmethod
    def setUpClass(cls) -> None:
        free = FrankaSystem(pinned_config())
        cls.state = rest_state(free)
        cls.source_chunk = cls._chunk(free, joint=1, amount=-1.0)
        cls.target_chunk = cls._chunk(free, joint=2, amount=1.0)
        middle = cls.source_chunk + 0.5 * (cls.target_chunk - cls.source_chunk)
        cls.obstacle = cls._obstacle_on_the_middle(free, middle)
        cls.config = pinned_config(obstacles=[{"position": cls.obstacle.tolist(), "radius": cls.RADIUS}])
        cls.simulator = FrankaSystem(cls.config)
        cls.projection = ProjectionConfig.from_mapping(cls.config)

    @classmethod
    def _chunk(cls, simulator: FrankaSystem, joint: int, amount: float) -> np.ndarray:
        """The rest state with ``joint`` moved by ``amount`` over the horizon along a smoothstep that ends at rest."""
        tau = np.arange(1, HORIZON + 1) / HORIZON
        chunk = np.tile(cls.state[: simulator.n], (HORIZON, 1))
        chunk[:, joint] += amount * (3 * tau**2 - 2 * tau**3)
        return chunk

    @classmethod
    def _obstacle_on_the_middle(cls, simulator: FrankaSystem, middle: np.ndarray) -> np.ndarray:
        """The centre of the bounding sphere of the last row of ``middle`` that stays farthest from the start, the source and the target."""
        centers, _, radii = simulator.collision_spheres(middle[-1])
        others = [simulator.collision_spheres(cls.state[: simulator.n])[0][None]]
        for chunk in (cls.source_chunk, cls.target_chunk):
            others.append(simulator.collision_spheres_along(chunk)[0])
        others = np.concatenate(others)  # (rows, S, 3)
        clearance = [
            (np.linalg.norm(others - center, axis=-1) - radii[None] - cls.RADIUS).min() for center in centers
        ]
        return centers[int(np.argmax(clearance))]

    def setUp(self) -> None:
        self.source = SafeSource(self.simulator, self.projection, HORIZON, seed=0)
        self.label = self.target_chunk
        self.lines = [self.source_chunk + k / PATH_SEGMENTS * (self.target_chunk - self.source_chunk) for k in range(1, PATH_SEGMENTS)]
        self.cut = [not self.simulator.collision.trajectory_is_collision_free(line) for line in self.lines]

    def test_the_start_the_source_and_the_target_are_clear_of_the_obstacle(self) -> None:
        collision = self.simulator.collision

        self.assertTrue(collision.is_collision_free(self.state[:7]))
        self.assertTrue(collision.trajectory_is_collision_free(self.source_chunk))
        self.assertTrue(collision.trajectory_is_collision_free(self.target_chunk))

    def test_some_interior_points_of_the_line_are_cut_and_some_are_not(self) -> None:
        self.assertTrue(self.cut[PATH_SEGMENTS // 2 - 1])
        self.assertTrue(any(self.cut) and not all(self.cut))

    def _make_safe_of(self, replacement: np.ndarray | None):
        """A stand-in of ``_make_safe``: the known source, then the known target, then ``replacement`` for each projected point."""
        calls = []

        def make_safe(state, reference):
            calls.append((state, reference))
            if len(calls) == 1:
                return self.source_chunk
            if len(calls) == 2:
                return self.target_chunk
            return replacement

        return make_safe, calls

    def test_each_cut_point_is_projected_once_and_the_others_are_kept(self) -> None:
        replacement = self.source_chunk + 0.1
        make_safe, calls = self._make_safe_of(replacement)

        with mock.patch.object(self.source, "_make_safe", side_effect=make_safe):
            path = self.source.path(self.state, self.label, 0)

        self.assertIsNotNone(path)
        self.assertEqual(len(calls), 2 + sum(self.cut))
        projected = iter(calls[2:])
        np.testing.assert_array_equal(path[0], self.source_chunk)
        np.testing.assert_array_equal(path[-1], self.target_chunk)
        for segment, (line, cut) in enumerate(zip(self.lines, self.cut), start=1):
            with self.subTest(segment=segment, cut=cut):
                if cut:
                    state, reference = next(projected)
                    np.testing.assert_array_equal(state, self.state)
                    np.testing.assert_array_equal(reference, line)
                    np.testing.assert_array_equal(path[segment], replacement)
                else:
                    np.testing.assert_allclose(path[segment], line, atol=1e-12)

    def test_a_cut_point_whose_projection_fails_gives_no_path(self) -> None:
        make_safe, _ = self._make_safe_of(None)

        with mock.patch.object(self.source, "_make_safe", side_effect=make_safe):
            self.assertIsNone(self.source.path(self.state, self.label, 0))

    def _path_with_failing_filter(self, failure) -> np.ndarray | None:
        """The path when the filter returns the known source and target and then ``failure`` on any cut point."""
        plans = iter([self.source_chunk, self.target_chunk])

        def project(state, reference):
            try:
                positions = next(plans)
            except StopIteration:
                return failure()
            return Projection(positions, np.zeros((HORIZON, self.simulator.nu)))

        with mock.patch.object(self.source._filter, "project", project):
            return self.source.path(self.state, self.label, 0)

    def test_a_solve_error_on_a_cut_point_gives_no_path(self) -> None:
        def failure():
            raise PlannerSolveError("forced failure")

        self.assertIsNone(self._path_with_failing_filter(failure))

    def test_a_fallback_warning_on_a_cut_point_gives_no_path(self) -> None:
        def failure():
            warnings.warn("falling back to the last known-safe trajectory")
            return Projection(self.source_chunk, np.zeros((HORIZON, self.simulator.nu)))

        self.assertIsNone(self._path_with_failing_filter(failure))


class SafePairsTests(unittest.TestCase):
    N = 3

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calls: list[tuple[int, int]] = []

    def _paths_of(self, states: np.ndarray, labels: np.ndarray, first_index: int) -> np.ndarray:
        """Paths whose value is 10 times the index of their frame plus the segment."""
        self.calls.append((first_index, len(states)))
        values = 10.0 * (first_index + np.arange(len(states), dtype=np.float32))[:, None] + np.arange(PATH_SEGMENTS + 1, dtype=np.float32)
        return np.broadcast_to(values[:, :, None, None], (len(states), PATH_SEGMENTS + 1, HORIZON, self.N)).copy()

    @staticmethod
    def _load_of(total: int):
        """The loader of a dataset of ``total`` frames: the states and labels of the frames from ``start`` on."""
        return lambda start: (np.zeros((total - start, 4)), np.zeros((total - start, HORIZON, 3)))

    def test_a_new_folder_has_no_paths(self) -> None:
        pairs = SafePairs(self.root, HORIZON, self.N)

        self.assertEqual(len(pairs), 0)
        self.assertEqual(pairs.paths.shape, (0, PATH_SEGMENTS + 1, HORIZON, self.N))
        self.assertEqual(pairs.sources.shape, (0, HORIZON, self.N))

    def test_update_computes_the_frames_and_keeps_them_in_the_dataset_folder(self) -> None:
        pairs = SafePairs(self.root, HORIZON, self.N)

        computed = pairs.update(4, self._load_of(4), self._paths_of)

        self.assertEqual(computed, 4)
        self.assertEqual(len(pairs), 4)
        self.assertTrue((self.root / PAIRS_FILE).is_file())
        reloaded = SafePairs(self.root, HORIZON, self.N)
        np.testing.assert_array_equal(reloaded.paths, pairs.paths)
        np.testing.assert_array_equal(reloaded.paths[:, 0, 0, 0], [0, 10, 20, 30])

    def test_the_sources_and_the_targets_are_the_first_and_the_last_point_of_the_paths(self) -> None:
        pairs = SafePairs(self.root, HORIZON, self.N)
        pairs.update(3, self._load_of(3), self._paths_of)

        np.testing.assert_array_equal(pairs.sources, pairs.paths[:, 0])
        np.testing.assert_array_equal(pairs.targets, pairs.paths[:, -1])
        np.testing.assert_array_equal(pairs.sources[:, 0, 0], [0, 10, 20])
        np.testing.assert_array_equal(pairs.targets[:, 0, 0], [PATH_SEGMENTS, 10 + PATH_SEGMENTS, 20 + PATH_SEGMENTS])

    def test_an_update_computes_only_the_frames_beyond_the_ones_kept(self) -> None:
        SafePairs(self.root, HORIZON, self.N).update(4, self._load_of(4), self._paths_of)
        self.calls.clear()
        pairs = SafePairs(self.root, HORIZON, self.N)

        computed = pairs.update(6, self._load_of(6), self._paths_of)

        self.assertEqual(computed, 2)
        self.assertEqual(self.calls, [(4, 2)])  # the frames from index 4 on, numbered from 4
        np.testing.assert_array_equal(pairs.paths[:, 0, 0, 0], [0, 10, 20, 30, 40, 50])
        np.testing.assert_array_equal(SafePairs(self.root, HORIZON, self.N).paths[:, 0, 0, 0], [0, 10, 20, 30, 40, 50])

    def test_an_update_of_a_dataset_that_has_not_grown_computes_nothing(self) -> None:
        pairs = SafePairs(self.root, HORIZON, self.N)
        pairs.update(4, self._load_of(4), self._paths_of)
        self.calls.clear()

        self.assertEqual(pairs.update(4, self._load_of(4), self._paths_of), 0)
        self.assertEqual(self.calls, [])

    def test_a_dataset_with_fewer_frames_than_the_paths_kept_is_rejected(self) -> None:
        pairs = SafePairs(self.root, HORIZON, self.N)
        pairs.update(4, self._load_of(4), self._paths_of)

        with self.assertRaises(ValueError):
            pairs.update(3, self._load_of(3), self._paths_of)

    def test_stored_paths_of_another_shape_are_rejected(self) -> None:
        SafePairs(self.root, HORIZON, self.N).update(2, self._load_of(2), self._paths_of)

        with self.assertRaises(ValueError):
            SafePairs(self.root, HORIZON + 1, self.N)
        with self.assertRaises(ValueError):
            SafePairs(self.root, HORIZON, self.N + 1)


class PooledPairsTests(unittest.TestCase):
    def test_the_paths_of_the_pool_equal_the_serial_ones_chunk_by_chunk(self) -> None:
        config = pinned_config()
        simulator = FrankaSystem(config)
        states, labels = _states_and_labels(simulator, frames=3)
        serial = SafeSource(simulator, ProjectionConfig.from_mapping(config), HORIZON, seed=3).paths(states, labels, 6)

        with mock.patch.object(PooledPairs, "CHUNK", 2), PooledPairs(config, HORIZON, 3, workers=2) as pooled:
            with contextlib.redirect_stdout(io.StringIO()):
                paths = pooled(states, labels, 6)

        self.assertEqual(paths.shape, (3, PATH_SEGMENTS + 1, HORIZON, simulator.n))
        self.assertTrue(np.isfinite(serial).any())
        np.testing.assert_allclose(paths, serial, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
