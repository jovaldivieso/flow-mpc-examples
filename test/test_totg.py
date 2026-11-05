from __future__ import annotations

import math
import os
import sys
import unittest
from unittest import mock

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.totg import BlendedPath, TimedPath, TrajectoryError

VELOCITY_LIMITS = 0.8 * np.array([2.1, 2.1, 2.1, 2.1, 2.6, 2.6, 2.6])
ACCELERATION_LIMITS = np.full(7, 2.5)
CORNER_PATH = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])


def _distance_to_polyline(point: np.ndarray, waypoints: np.ndarray) -> float:
    best = math.inf
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        direction = end - start
        fraction = np.clip((point - start) @ direction / (direction @ direction), 0.0, 1.0)
        best = min(best, float(np.linalg.norm(point - (start + fraction * direction))))
    return best


def _samples(path: BlendedPath, count: int = 2000) -> np.ndarray:
    return np.array([path.config(s) for s in np.linspace(0.0, path.length, count)])


def _finite_differences(timed: TimedPath, step: float = 0.01) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The positions, velocities and accelerations of ``timed`` on a grid of ``step`` seconds."""
    positions = timed.position(np.arange(0.0, timed.duration + step, step))
    velocities = np.diff(positions, axis=0) / step
    return positions, velocities, np.diff(velocities, axis=0) / step


class BlendedPathTests(unittest.TestCase):
    def test_the_path_starts_and_ends_at_the_waypoints(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        np.testing.assert_allclose(path.config(0.0), CORNER_PATH[0], atol=1e-9)
        np.testing.assert_allclose(path.config(path.length), CORNER_PATH[-1], atol=1e-9)

    def test_the_blend_stays_within_the_maximum_deviation_of_every_corner(self) -> None:
        for max_deviation in (0.02, 0.1, 0.3):
            with self.subTest(max_deviation=max_deviation):
                path = BlendedPath(CORNER_PATH, max_deviation)
                samples = _samples(path, 20000)  # a dense sampling: its spacing bounds the error of the nearest sample

                for corner in CORNER_PATH[1:-1]:
                    nearest = np.linalg.norm(samples - corner, axis=1).min()
                    self.assertLessEqual(nearest, max_deviation + 1e-4)
                    self.assertGreater(nearest, 0.0)  # the corner is cut

    def test_the_blend_cuts_a_corner_by_exactly_the_maximum_deviation_when_the_lines_are_long_enough(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        distance = np.linalg.norm(_samples(path, 20000) - CORNER_PATH[1], axis=1).min()

        self.assertAlmostEqual(distance, 0.1, places=4)

    def test_the_blended_path_stays_within_the_maximum_deviation_of_the_polyline(self) -> None:
        rng = np.random.default_rng(1)
        waypoints = rng.uniform(-1.0, 1.0, (6, 7))
        max_deviation = 0.05

        samples = _samples(BlendedPath(waypoints, max_deviation), 800)

        self.assertLessEqual(max(_distance_to_polyline(point, waypoints) for point in samples), max_deviation + 1e-9)

    def test_the_blend_is_shorter_than_the_polyline_and_longer_than_the_straight_line(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        polyline = float(np.linalg.norm(np.diff(CORNER_PATH, axis=0), axis=1).sum())

        self.assertLess(path.length, polyline)
        self.assertGreater(path.length, np.linalg.norm(CORNER_PATH[-1] - CORNER_PATH[0]))

    def test_the_tangent_is_a_unit_vector(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        for s in np.linspace(0.0, path.length, 200):
            self.assertAlmostEqual(float(np.linalg.norm(path.tangent(s))), 1.0, places=9)

    def test_the_tangent_is_continuous_along_the_path_and_across_the_joins_of_lines_and_blends(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)
        step = 1e-4

        arc_lengths = np.arange(0.0, path.length - step, step)
        jumps = [np.linalg.norm(path.tangent(s + step) - path.tangent(s)) for s in arc_lengths]

        # the tangent turns at the rate of the curvature, which a blend of radius r bounds by step / r
        radius = float(1.0 / max(np.linalg.norm(path.curvature(s)) for s in np.linspace(0.0, path.length, 2000)))
        self.assertLessEqual(max(jumps), 1.5 * step / radius)

    def test_the_tangent_is_the_derivative_of_the_configuration(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)
        step = 1e-6

        for s in (0.3, 0.95, 1.1, 1.6, 2.0, path.length - 0.2):
            derivative = (path.config(s + step) - path.config(s - step)) / (2 * step)
            np.testing.assert_allclose(derivative, path.tangent(s), atol=1e-5)

    def test_the_curvature_is_zero_on_lines_and_one_over_the_radius_in_blends(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        np.testing.assert_array_equal(path.curvature(0.2), 0.0)
        magnitudes = [np.linalg.norm(path.curvature(s)) for s in np.linspace(0.0, path.length, 4000)]
        blend = [m for m in magnitudes if m > 0.0]
        self.assertTrue(blend)
        np.testing.assert_allclose(blend, blend[0], rtol=1e-6)  # a circular arc of one radius
        # curvature is orthogonal to the tangent
        for s in np.linspace(0.0, path.length, 50):
            self.assertAlmostEqual(float(path.tangent(s) @ path.curvature(s)), 0.0, places=9)

    def test_collinear_waypoints_need_no_blend(self) -> None:
        waypoints = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

        path = BlendedPath(waypoints, 0.1)

        self.assertAlmostEqual(path.length, 4.0 * math.sqrt(2.0), places=9)
        for s in np.linspace(0.0, path.length, 20):
            np.testing.assert_array_equal(path.curvature(s), 0.0)
            self.assertLessEqual(_distance_to_polyline(path.config(s), waypoints), 1e-9)

    def test_a_two_waypoint_path_is_the_straight_line(self) -> None:
        path = BlendedPath(np.array([[0.0, 0.0], [3.0, 4.0]]), 0.1)

        self.assertAlmostEqual(path.length, 5.0, places=9)
        np.testing.assert_allclose(path.config(2.5), [1.5, 2.0], atol=1e-9)
        np.testing.assert_allclose(path.tangent(1.0), [0.6, 0.8], atol=1e-9)

    def test_a_blend_is_limited_by_the_half_of_a_short_segment(self) -> None:
        waypoints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 0.1], [2.0, 0.1]])

        path = BlendedPath(waypoints, 1.0)  # far more than the short middle segment allows

        samples = _samples(path)
        self.assertLessEqual(max(_distance_to_polyline(point, waypoints) for point in samples), 0.1 + 1e-9)
        np.testing.assert_allclose(path.config(path.length), waypoints[-1], atol=1e-9)

    def test_duplicate_waypoints_are_dropped(self) -> None:
        with_duplicates = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

        path = BlendedPath(with_duplicates, 0.1)

        self.assertAlmostEqual(path.length, BlendedPath(with_duplicates[[0, 2, 4]], 0.1).length, places=12)

    def test_a_start_equal_to_the_goal_is_no_path(self) -> None:
        with self.assertRaises(ValueError):
            BlendedPath(np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]), 0.1)

    def test_invalid_input_raises_value_error(self) -> None:
        for waypoints, max_deviation in (
            (np.zeros((1, 3)), 0.1),  # a single waypoint
            (np.zeros(3), 0.1),  # not a matrix
            (np.array([[0.0, 0.0], [1.0, 0.0]]), 0.0),  # non-positive deviation
            (np.array([[0.0, 0.0], [1.0, 0.0]]), -0.1),
            (np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]), 0.1),  # turns back on itself
        ):
            with self.subTest(waypoints=waypoints.tolist(), max_deviation=max_deviation), self.assertRaises(ValueError):
                BlendedPath(waypoints, max_deviation)


class TimedPathTests(unittest.TestCase):
    def test_a_trajectory_along_a_straight_line_is_the_triangle_profile_when_the_line_is_short(self) -> None:
        # one joint, v_max = 1, a_max = 2: it never reaches v_max within d < v_max^2 / a_max = 0.5, so T = 2 sqrt(d / a_max)
        distance = 0.25
        timed = TimedPath(BlendedPath(np.array([[0.0], [distance]]), 0.1), np.array([1.0]), np.array([2.0]))

        self.assertAlmostEqual(timed.duration, 2.0 * math.sqrt(distance / 2.0), places=3)
        np.testing.assert_allclose(timed.position(np.array([timed.duration / 2]))[0, 0], distance / 2, atol=1e-3)

    def test_a_trajectory_along_a_straight_line_is_the_trapezoid_profile_when_the_line_is_long(self) -> None:
        # d >= v_max^2 / a_max: accelerate for v_max / a_max, cruise, brake, so T = d / v_max + v_max / a_max
        distance = 2.0
        timed = TimedPath(BlendedPath(np.array([[0.0], [distance]]), 0.1), np.array([1.0]), np.array([2.0]))

        self.assertAlmostEqual(timed.duration, distance / 1.0 + 1.0 / 2.0, places=3)
        np.testing.assert_allclose(timed.position(np.array([0.5]))[0, 0], 0.25, atol=1e-3)  # accelerated: a t^2 / 2
        np.testing.assert_allclose(timed.position(np.array([1.5]))[0, 0], 1.25, atol=1e-3)  # cruising at v_max

    def test_the_joint_with_the_tightest_limit_decides_the_duration_of_a_straight_line(self) -> None:
        waypoints = np.array([[0.0, 0.0], [1.0, 1.0]])
        slow_second = TimedPath(BlendedPath(waypoints, 0.1), np.array([10.0, 0.5]), np.array([100.0, 100.0]))
        slow_first = TimedPath(BlendedPath(waypoints, 0.1), np.array([0.5, 10.0]), np.array([100.0, 100.0]))

        self.assertAlmostEqual(slow_second.duration, slow_first.duration, places=6)
        self.assertGreater(slow_second.duration, 1.0 / 0.5)  # at least the time of the slow joint at its limit

    def test_the_trajectory_starts_and_ends_at_rest_at_the_end_points_of_the_path(self) -> None:
        timed = TimedPath(BlendedPath(CORNER_PATH, 0.1), VELOCITY_LIMITS[:3], ACCELERATION_LIMITS[:3])
        step = 1e-3

        np.testing.assert_allclose(timed.position(np.array([0.0]))[0], CORNER_PATH[0], atol=1e-9)
        np.testing.assert_allclose(timed.position(np.array([timed.duration]))[0], CORNER_PATH[-1], atol=1e-9)
        start_speed = np.linalg.norm(timed.position(np.array([step]))[0] - CORNER_PATH[0]) / step
        end_speed = np.linalg.norm(timed.position(np.array([timed.duration - step]))[0] - CORNER_PATH[-1]) / step
        self.assertLess(start_speed, 2.0 * ACCELERATION_LIMITS[0] * step)
        self.assertLess(end_speed, 2.0 * ACCELERATION_LIMITS[0] * step)

    def test_a_time_beyond_the_duration_is_the_end_of_the_path_and_a_time_before_zero_is_the_start(self) -> None:
        timed = TimedPath(BlendedPath(CORNER_PATH, 0.1), VELOCITY_LIMITS[:3], ACCELERATION_LIMITS[:3])

        np.testing.assert_allclose(timed.position(np.array([-1.0, timed.duration + 5.0])), [CORNER_PATH[0], CORNER_PATH[-1]], atol=1e-9)
        np.testing.assert_allclose(timed.arc(np.array([-1.0, timed.duration + 5.0])), [0.0, timed.path.length], atol=1e-9)

    def test_the_arc_length_increases_monotonically_from_zero_to_the_length_of_the_path(self) -> None:
        timed = TimedPath(BlendedPath(CORNER_PATH, 0.1), VELOCITY_LIMITS[:3], ACCELERATION_LIMITS[:3])

        arcs = timed.arc(np.linspace(0.0, timed.duration, 500))

        self.assertGreaterEqual(float(np.diff(arcs).min()), 0.0)
        self.assertAlmostEqual(float(arcs[0]), 0.0, places=12)
        self.assertAlmostEqual(float(arcs[-1]), timed.path.length, places=9)

    def test_the_positions_follow_the_path(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)
        timed = TimedPath(path, VELOCITY_LIMITS[:3], ACCELERATION_LIMITS[:3])
        times = np.linspace(0.0, timed.duration, 50)

        positions = timed.position(times)

        for arc, position in zip(timed.arc(times), positions):
            np.testing.assert_allclose(position, path.config(arc), atol=1e-12)

    def test_the_trajectory_respects_the_velocity_and_acceleration_limits_of_every_joint(self) -> None:
        rng = np.random.default_rng(0)
        for index in range(4):
            waypoints = rng.uniform(-1.0, 1.0, (5, 7))
            with self.subTest(index=index):
                timed = TimedPath(BlendedPath(waypoints, 0.05), VELOCITY_LIMITS, ACCELERATION_LIMITS)

                _, velocities, accelerations = _finite_differences(timed)

                self.assertLessEqual(float((np.abs(velocities) / VELOCITY_LIMITS).max()), 1.03)
                self.assertLessEqual(float((np.abs(accelerations) / ACCELERATION_LIMITS).max()), 1.05)

    def test_the_trajectory_is_time_optimal_one_joint_is_always_at_a_limit(self) -> None:
        # time-optimal along a path: at every instant some joint is at its velocity or acceleration limit (up to the blends'
        # sampling), so the trajectory is not slower than needed
        waypoints = np.random.default_rng(2).uniform(-1.0, 1.0, (4, 7))
        timed = TimedPath(BlendedPath(waypoints, 0.05), VELOCITY_LIMITS, ACCELERATION_LIMITS)

        _, velocities, accelerations = _finite_differences(timed)

        usage = np.maximum((np.abs(velocities[1:]) / VELOCITY_LIMITS).max(axis=1), (np.abs(accelerations) / ACCELERATION_LIMITS).max(axis=1))
        self.assertGreater(float(np.mean(usage > 0.95)), 0.9)

    def test_the_duration_is_at_least_that_of_every_joint_alone_at_its_velocity_limit(self) -> None:
        waypoints = np.random.default_rng(3).uniform(-1.0, 1.0, (5, 7))
        timed = TimedPath(BlendedPath(waypoints, 0.05), VELOCITY_LIMITS, ACCELERATION_LIMITS)

        travelled = np.abs(np.diff(timed.position(np.linspace(0.0, timed.duration, 4000)), axis=0)).sum(axis=0)

        self.assertTrue(np.all(timed.duration >= travelled / VELOCITY_LIMITS - 1e-6))

    def test_looser_limits_give_a_shorter_trajectory(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)

        tight = TimedPath(path, VELOCITY_LIMITS[:3], ACCELERATION_LIMITS[:3])
        loose = TimedPath(path, 2.0 * VELOCITY_LIMITS[:3], 2.0 * ACCELERATION_LIMITS[:3])

        self.assertLess(loose.duration, tight.duration)

    def test_collinear_waypoints_take_as_long_as_the_straight_line(self) -> None:
        straight = TimedPath(BlendedPath(np.array([[0.0, 0.0], [2.0, 2.0]]), 0.1), np.ones(2), np.ones(2))
        through = TimedPath(BlendedPath(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 0.1), np.ones(2), np.ones(2))

        self.assertAlmostEqual(through.duration, straight.duration, places=6)

    def test_random_seven_joint_paths_all_have_a_trajectory(self) -> None:
        rng = np.random.default_rng(42)
        for index in range(30):
            waypoints = rng.uniform(-2.0, 2.0, (rng.integers(2, 7), 7))
            with self.subTest(index=index):
                timed = TimedPath(BlendedPath(waypoints, 0.05), VELOCITY_LIMITS, ACCELERATION_LIMITS)

                self.assertTrue(math.isfinite(timed.duration) and timed.duration > 0.0)
                np.testing.assert_allclose(timed.position(np.array([timed.duration]))[0], waypoints[-1], atol=1e-6)

    def test_limits_that_are_not_positive_raise_value_error(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.1)
        for velocity, acceleration in ((np.zeros(3), np.ones(3)), (np.ones(3), np.array([1.0, -1.0, 1.0]))):
            with self.assertRaises(ValueError):
                TimedPath(path, velocity, acceleration)

    def test_a_trajectory_error_is_a_runtime_error(self) -> None:
        self.assertTrue(issubclass(TrajectoryError, RuntimeError))


if __name__ == "__main__":
    unittest.main()


# Paths whose forward trajectory ended with a point repeated, which left a zero-length segment to meet when integrating backward
REPEATED_POINT_PATHS = [
    # random path, seed 125
    np.array(
        [[ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           0.000000000000e+00,  0.000000000000e+00],
         [-2.540010017846e-05, -4.393528957545e-04, -1.281569691649e-03, -1.597014818795e-03,  1.120959978761e-03,
           1.145391515870e-03,  1.484377931404e-03],
         [-7.368096854507e-04, -1.963977383259e-03, -8.399411458893e-04, -6.622555700856e-04,  2.067115496785e-06,
           2.298498651847e-03,  5.988798364718e-04],
         [-1.855935508314e-02, -1.112250355959e-02, -2.076011682930e-02,  5.781731344665e-03,  1.049529361372e-02,
          -7.536652583504e-03, -1.593718158124e-02],
         [-3.032435866735e-01, -4.323648531542e-02, -6.932187090614e-02, -2.765354534239e-01,  1.852742628527e-01,
          -2.678032059407e-02,  1.973576108052e-01]]
    ),
    # random path, seed 846
    np.array(
        [[ 0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,  0.000000000000e+00,
           0.000000000000e+00,  0.000000000000e+00],
         [ 1.520835175662e-04, -1.139604119985e-04, -1.164563782564e-03,  1.554745146311e-03, -3.329290213828e-04,
          -1.118775009951e-03,  7.677629406524e-04],
         [-2.751500425608e-03, -2.059361257316e-03,  9.948497803195e-03,  3.183009236104e-03,  4.864551697077e-03,
           3.556865014312e-03, -1.981393294883e-03],
         [-2.991857429491e-01,  2.777937774253e-02, -9.019882216029e-02,  7.047119509382e-02,  2.944517133093e-01,
           9.436886493404e-02, -2.782508349784e-02],
         [-3.046906173900e-01,  3.976525045232e-02, -8.267988376670e-02,  7.509548913715e-02,  2.755097339709e-01,
           1.036400519022e-01, -2.791799958534e-02]]
    ),
]


class DegenerateTrajectoryTests(unittest.TestCase):
    def test_a_path_that_repeats_a_point_of_the_forward_trajectory_is_timed(self) -> None:
        for waypoints in REPEATED_POINT_PATHS:
            with self.subTest(waypoints=waypoints[1].tolist()):
                timed = TimedPath(BlendedPath(waypoints, 0.05), VELOCITY_LIMITS, ACCELERATION_LIMITS)

                self.assertGreater(timed.duration, 0.0)
                np.testing.assert_allclose(timed.position(np.array([timed.duration]))[0], waypoints[-1], atol=1e-9)

    def test_an_arithmetic_error_of_the_search_is_a_failed_search(self) -> None:
        path = BlendedPath(CORNER_PATH, 0.05)

        with mock.patch.object(TimedPath, "_search_phase_plane", side_effect=ZeroDivisionError("float division by zero")):
            with self.assertRaises(TrajectoryError):
                TimedPath(path, np.ones(3), np.ones(3))
