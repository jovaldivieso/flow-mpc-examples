"""Time-optimal trajectory generation along a joint-space path (Kunz and Stilman, RSS 2012).

``BlendedPath`` turns the waypoints of a path planner into a differentiable path by circular blends around the
waypoints; ``TimedPath`` is the fastest trajectory that follows it exactly within the joint velocity and acceleration
limits, found in the phase plane (s, ds/dt) of the arc length s: forward with the largest path acceleration, backward
with the smallest from the switching points of the limit curves (Sec. VI and VII), with the source and sink check of
Sec. VIII-A.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

EPS = 1e-6  # [rad] of path length: the tolerance of comparisons and of the numerical derivatives
FLAT = 1e-6  # [rad] turning angle below which a waypoint needs no blend
LINE_MIN = 1e-9  # [rad] length below which the line between two blends is none
INTEGRATION_STEP = 5e-3  # [s] of the phase-plane integration
VELOCITY_SEARCH_STEP = 1e-3  # [rad] of the search for switching points on the velocity limit curve
VELOCITY_SEARCH_ACCURACY = 1e-6
MAX_STEPS = 2_000_000  # bound on the integration steps of one trajectory


class TrajectoryError(RuntimeError):
    """No trajectory follows the path within the limits (a numerical failure of the phase-plane search)."""


class _Line:
    def __init__(self, start: np.ndarray, end: np.ndarray) -> None:
        self.length = float(np.linalg.norm(end - start))
        self._start, self._direction = start, (end - start) / self.length
        self._zero = np.zeros_like(start)

    def config(self, s: float) -> np.ndarray:
        return self._start + s * self._direction

    def tangent(self, s: float) -> np.ndarray:
        return self._direction

    def curvature(self, s: float) -> np.ndarray:
        return self._zero

    def zero_tangent_points(self) -> list[float]:
        return []


class _Arc:
    """The circular blend of Eq. 4 to 9: the circle of radius ``radius`` around ``center`` in the plane of ``x`` (from
    the center to where the arc starts) and ``y`` (the direction of the line before it), turning by ``angle``."""

    def __init__(self, center: np.ndarray, x: np.ndarray, y: np.ndarray, radius: float, angle: float) -> None:
        self._center, self._x, self._y, self._radius, self.angle = center, x, y, radius, angle
        self.length = radius * angle

    def config(self, s: float) -> np.ndarray:
        theta = s / self._radius
        return self._center + self._radius * (self._x * math.cos(theta) + self._y * math.sin(theta))

    def tangent(self, s: float) -> np.ndarray:
        theta = s / self._radius
        return -self._x * math.sin(theta) + self._y * math.cos(theta)

    def curvature(self, s: float) -> np.ndarray:
        theta = s / self._radius
        return -(self._x * math.cos(theta) + self._y * math.sin(theta)) / self._radius

    def zero_tangent_points(self) -> list[float]:
        """The arc lengths in the arc where a joint stands still (Eq. 39: f'_i = 0), where the acceleration limit curve
        can have a switching point. f'_i = A cos(theta - theta0) with theta0 = atan2(-x_i, y_i), zero at theta0 +- pi/2."""
        points = []
        for x_i, y_i in zip(self._x, self._y):
            if x_i == 0.0 and y_i == 0.0:
                continue
            theta0 = math.atan2(-x_i, y_i)
            for theta in (theta0 + math.pi / 2, theta0 - math.pi / 2, theta0 + 3 * math.pi / 2):
                if 0.0 < theta < self.angle:
                    points.append(theta * self._radius)
        return points


class BlendedPath:
    """The waypoints joined by straight lines, with circular blends of at most ``max_deviation`` [rad] from the
    waypoint at every turn (Sec. IV): the path is then differentiable, its curvature piecewise continuous."""

    def __init__(self, waypoints: np.ndarray, max_deviation: float) -> None:
        points = np.asarray(waypoints, dtype=float)
        if points.ndim != 2 or len(points) < 2:
            raise ValueError(f"A path needs at least two waypoints of joint positions, got {points.shape}.")
        if max_deviation <= 0.0:
            raise ValueError("'max_deviation' must be positive.")
        keep = np.concatenate([[True], np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9])
        points = points[keep]
        if len(points) < 2:
            raise ValueError("A path needs two different waypoints.")
        self._segments = self._build(points, float(max_deviation))
        self._starts = np.concatenate([[0.0], np.cumsum([segment.length for segment in self._segments])])
        self.length = float(self._starts[-1])
        self._start_list = self._starts[:-1].tolist()
        self._switching = self._switching_points()

    @staticmethod
    def _build(points: np.ndarray, max_deviation: float) -> list[_Line | _Arc]:
        segments: list[_Line | _Arc] = []
        line_start = points[0]
        for index in range(1, len(points) - 1):
            before, after = points[index] - points[index - 1], points[index + 1] - points[index]
            y_before, y_after = before / np.linalg.norm(before), after / np.linalg.norm(after)
            angle = math.acos(float(np.clip(y_before @ y_after, -1.0, 1.0)))
            if angle < FLAT:
                continue
            if angle > math.pi - 1e-3:
                raise ValueError("The path turns back on itself: it has to stop at that waypoint.")
            half = angle / 2
            reach = min(
                np.linalg.norm(before) / 2, np.linalg.norm(after) / 2, max_deviation * math.sin(half) / (1.0 - math.cos(half))
            )
            radius = reach / math.tan(half)
            bisector = (y_after - y_before) / np.linalg.norm(y_after - y_before)
            center = points[index] + bisector * radius / math.cos(half)
            x = points[index] - reach * y_before - center
            if np.linalg.norm(points[index] - reach * y_before - line_start) > LINE_MIN:
                segments.append(_Line(line_start, points[index] - reach * y_before))
            segments.append(_Arc(center, x / np.linalg.norm(x), y_before, radius, angle))
            line_start = points[index] + reach * y_after
        if np.linalg.norm(points[-1] - line_start) > LINE_MIN:
            segments.append(_Line(line_start, points[-1]))
        return segments

    def _switching_points(self) -> list[tuple[float, bool]]:
        """The arc lengths where the limit curve can switch between acceleration and braking: where the curvature jumps
        (``True``: a discontinuity) and where a joint stands still inside an arc (``False``)."""
        points = [(float(start), True) for start in self._starts[1:-1]]
        for start, segment in zip(self._starts, self._segments):
            points += [(float(start) + s, False) for s in segment.zero_tangent_points()]
        return sorted(points)

    def next_switching_point(self, s: float) -> tuple[float, bool]:
        """The first switching point beyond ``s`` and whether it is a discontinuity; the end of the path if none is."""
        index = bisect.bisect_right([point for point, _ in self._switching], s)
        return self._switching[index] if index < len(self._switching) else (self.length, True)

    def _locate(self, s: float) -> tuple[_Line | _Arc, float]:
        index = min(max(bisect.bisect_right(self._start_list, s) - 1, 0), len(self._segments) - 1)
        return self._segments[index], s - self._start_list[index]

    def config(self, s: float) -> np.ndarray:
        segment, local = self._locate(s)
        return segment.config(local)

    def tangent(self, s: float) -> np.ndarray:
        segment, local = self._locate(s)
        return segment.tangent(local)

    def curvature(self, s: float) -> np.ndarray:
        segment, local = self._locate(s)
        return segment.curvature(local)


class TimedPath:
    """The time-optimal trajectory along ``path`` within the joint velocity and acceleration limits, from rest to rest.

    ``position(times)`` are the configurations at the given times (the end of the path after ``duration``) and ``arc(times)``
    the arc lengths there. Raises ``TrajectoryError`` when the phase-plane search fails.
    """

    def __init__(
        self,
        path: BlendedPath,
        max_velocity: np.ndarray,
        max_acceleration: np.ndarray,
        step: float = INTEGRATION_STEP,
    ) -> None:
        self.path = path
        self._max_velocity = np.asarray(max_velocity, dtype=float)
        self._max_acceleration = np.asarray(max_acceleration, dtype=float)
        if np.any(self._max_velocity <= 0.0) or np.any(self._max_acceleration <= 0.0):
            raise ValueError("The velocity and acceleration limits must be positive.")
        self._step = float(step)
        self._valid = True
        self._steps = 0
        self._trajectory: list[tuple[float, float]] = [(0.0, 0.0)]
        self._search()
        self._arcs = np.array([s for s, _ in self._trajectory])
        self._velocities = np.array([v for _, v in self._trajectory])
        self._times = self._timing()
        self.duration = float(self._times[-1])

    # --- the search of the time-optimal trajectory in the phase plane ----------------------------------------------

    def _search(self) -> None:
        try:
            self._search_phase_plane()
        except ArithmeticError as exc:  # a degenerate step of the integration: the search failed
            raise TrajectoryError(f"The phase-plane search met an arithmetic error: {exc}") from exc

    def _search_phase_plane(self) -> None:
        after = self._acceleration(0.0, 0.0, True)
        while self._valid and not self._integrate_forward(after) and self._valid:
            end, point, before, after = self._next_switching_point(self._trajectory[-1][0])
            if end:
                break
            self._integrate_backward(point[0], point[1], before)
        if self._valid:
            self._integrate_backward(self.path.length, 0.0, self._acceleration(self.path.length, 0.0, False))
        if not self._valid:
            raise TrajectoryError("The phase-plane search found no trajectory along the path.")

    def _acceleration(self, s: float, velocity: float, largest: bool) -> float:
        """The largest (or smallest) path acceleration at (s, velocity) that keeps every joint within its limit (Eq. 22, 23)."""
        tangent, curvature = self.path.tangent(s), self.path.curvature(s)
        moving = tangent != 0.0
        if not moving.any():
            return math.inf if largest else -math.inf
        factor = 1.0 if largest else -1.0
        bound = self._max_acceleration[moving] / np.abs(tangent[moving]) - factor * curvature[moving] * velocity**2 / tangent[moving]
        return factor * float(bound.min())

    def _slope(self, s: float, velocity: float, largest: bool) -> float:
        return self._acceleration(s, velocity, largest) / velocity

    def _acceleration_velocity(self, s: float) -> float:
        """The path velocity above which no path acceleration is left (Eq. 31)."""
        tangent, curvature = self.path.tangent(s), self.path.curvature(s)
        limit = math.inf
        moving = np.flatnonzero(tangent != 0.0)
        if len(moving) > 1:
            ratio = curvature[moving] / tangent[moving]
            room = self._max_acceleration[moving] / np.abs(tangent[moving])
            difference = np.abs(ratio[:, None] - ratio[None, :])
            upper = np.triu(np.ones((len(moving), len(moving)), dtype=bool), 1) & (difference != 0.0)
            if upper.any():
                limit = float(np.sqrt(((room[:, None] + room[None, :])[upper] / difference[upper]).min()))
        standing = (tangent == 0.0) & (curvature != 0.0)
        if standing.any():
            limit = min(limit, float(np.sqrt((self._max_acceleration[standing] / np.abs(curvature[standing])).min())))
        return limit

    def _acceleration_velocity_slope(self, s: float) -> float:
        return (self._acceleration_velocity(s + EPS) - self._acceleration_velocity(s - EPS)) / (2.0 * EPS)

    def _velocity_velocity(self, s: float) -> float:
        """The path velocity above which a joint exceeds its velocity limit (Eq. 36)."""
        tangent = np.abs(self.path.tangent(s))
        moving = tangent > 0.0
        return float((self._max_velocity[moving] / tangent[moving]).min()) if moving.any() else math.inf

    def _velocity_velocity_slope(self, s: float) -> float:
        """The slope of that limit curve (Eq. 37)."""
        tangent = self.path.tangent(s)
        moving = tangent != 0.0
        ratios = np.full(len(tangent), math.inf)
        ratios[moving] = self._max_velocity[moving] / np.abs(tangent[moving])
        active = int(np.argmin(ratios))
        return -float(self._max_velocity[active] * self.path.curvature(s)[active] / (tangent[active] * abs(tangent[active])))

    def _count(self) -> None:
        self._steps += 1
        if self._steps > MAX_STEPS:
            self._valid = False
            raise TrajectoryError("The phase-plane integration did not end.")

    def _integrate_forward(self, acceleration: float) -> bool:
        """Follow the largest acceleration from the end of the trajectory. True when the end of the path is reached (or the
        search failed), False when a limit curve is met and a switching point has to be found."""
        trajectory, path, step = self._trajectory, self.path, self._step
        s, velocity = trajectory[-1]
        discontinuity = self._next_discontinuity(s)
        while True:
            self._count()
            old_s, old_velocity = s, velocity
            velocity += step * acceleration
            s += step * 0.5 * (old_velocity + velocity)
            if discontinuity is not None and s > discontinuity:
                if s - discontinuity < EPS:
                    continue
                velocity = old_velocity + (discontinuity - old_s) * (velocity - old_velocity) / (s - old_s)
                s = discontinuity
            if s > path.length:
                trajectory.append((s, velocity))
                return True
            if velocity < 0.0:
                self._valid = False
                return True
            velocity_limit = self._velocity_velocity(s)
            if velocity > velocity_limit and self._slope(old_s, self._velocity_velocity(old_s), False) <= self._velocity_velocity_slope(old_s):
                velocity = velocity_limit
            trajectory.append((s, velocity))
            if discontinuity is not None and s >= discontinuity:
                discontinuity = self._next_discontinuity(s)
            acceleration = self._acceleration(s, velocity, True)
            if velocity == 0.0 and acceleration < 0.0:
                self._valid = False
                return True
            if velocity > self._acceleration_velocity(s) + EPS:
                if self._hit_limit_curve(discontinuity):
                    return False

    def _next_discontinuity(self, s: float) -> float | None:
        position, discontinuous = s, False
        while not discontinuous:
            position, discontinuous = self.path.next_switching_point(position)
            if position >= self.path.length:
                return None
        return position if position > s else None

    def _hit_limit_curve(self, discontinuity: float | None) -> bool:
        """The last step overshot the acceleration limit curve: locate the crossing by bisection (Sec. VIII-A) and tell whether
        the trajectory has to switch there, that is, whether the crossing is not a trajectory source."""
        trajectory = self._trajectory
        overshoot = trajectory.pop()
        before, before_velocity = trajectory[-1]
        after, after_velocity = overshoot
        while after - before > EPS:
            middle, middle_velocity = 0.5 * (before + after), 0.5 * (before_velocity + after_velocity)
            if middle_velocity > self._velocity_velocity(middle) and self._slope(before, self._velocity_velocity(before), False) <= self._velocity_velocity_slope(before):
                middle_velocity = self._velocity_velocity(middle)
            if middle_velocity > self._acceleration_velocity(middle):
                after, after_velocity = middle, middle_velocity
            else:
                before, before_velocity = middle, middle_velocity
        if (before, before_velocity) != trajectory[-1]:
            trajectory.append((before, before_velocity))
        if self._acceleration_velocity(after) < self._velocity_velocity(after):
            if discontinuity is not None and after > discontinuity:
                return True
            return self._slope(before, before_velocity, True) > self._acceleration_velocity_slope(before)
        return self._slope(before, before_velocity, False) > self._velocity_velocity_slope(before)

    def _integrate_backward(self, s: float, velocity: float, acceleration: float) -> None:
        """Follow the smallest acceleration backward from (s, velocity) until the trajectory so far is met, and replace its
        end by that."""
        start, step = self._trajectory, self._step
        second = len(start) - 1
        first = second - 1
        backward: list[tuple[float, float]] = []  # from the point nearest to the switching point back
        slope = None
        while first > 0 or s >= 0.0:
            self._count()
            if start[first][0] <= s:
                backward.append((s, velocity))
                velocity -= step * acceleration
                s -= step * 0.5 * (velocity + backward[-1][1])
                acceleration = self._acceleration(s, velocity, False)
                slope = (backward[-1][1] - velocity) / (backward[-1][0] - s)
                if velocity < 0.0:
                    self._valid = False
                    return
            else:
                first -= 1
                second -= 1
            if slope is None:
                continue
            if start[second][0] == start[first][0]:
                continue
            start_slope = (start[second][1] - start[first][1]) / (start[second][0] - start[first][0])
            if slope == start_slope:
                continue
            meeting = (start[first][1] - velocity + slope * s - start_slope * start[first][0]) / (slope - start_slope)
            if max(start[first][0], s) - EPS <= meeting <= EPS + min(start[second][0], backward[-1][0]):
                del start[second:]
                start.append((meeting, start[first][1] + start_slope * (meeting - start[first][0])))
                start.extend(reversed(backward))
                return
        self._valid = False

    def _next_switching_point(self, s: float) -> tuple[bool, tuple[float, float], float, float]:
        """The next switching point after ``s`` on the lower of the two limit curves: (end of the path reached, the point,
        the acceleration before it, the acceleration after it)."""
        acceleration_point = (s, 0.0)
        while True:
            acceleration_end, acceleration_point, acceleration_before, acceleration_after = self._next_acceleration_switching_point(acceleration_point[0])
            if acceleration_end or acceleration_point[1] <= self._velocity_velocity(acceleration_point[0]):
                break
        velocity_point = (s, 0.0)
        while True:
            velocity_end, velocity_point, velocity_before, velocity_after = self._next_velocity_switching_point(velocity_point[0])
            if (
                velocity_end
                or velocity_point[0] > acceleration_point[0]
                or not (
                    velocity_point[1] > self._acceleration_velocity(velocity_point[0] - EPS)
                    or velocity_point[1] > self._acceleration_velocity(velocity_point[0] + EPS)
                )
            ):
                break
        if acceleration_end and velocity_end:
            return True, (0.0, 0.0), 0.0, 0.0
        if not acceleration_end and (velocity_end or acceleration_point[0] <= velocity_point[0]):
            return False, acceleration_point, acceleration_before, acceleration_after
        return False, velocity_point, velocity_before, velocity_after

    def _next_acceleration_switching_point(self, s: float) -> tuple[bool, tuple[float, float], float, float]:
        position = s
        while True:
            position, discontinuous = self.path.next_switching_point(position)
            if position > self.path.length - EPS:
                return True, (position, 0.0), 0.0, 0.0
            if discontinuous:
                velocity_before = self._acceleration_velocity(position - EPS)
                velocity_after = self._acceleration_velocity(position + EPS)
                velocity = min(velocity_before, velocity_after)
                before = self._acceleration(position - EPS, velocity, False)
                after = self._acceleration(position + EPS, velocity, True)
                if (
                    velocity_before > velocity_after
                    or self._slope(position - EPS, velocity, False) > self._acceleration_velocity_slope(position - 2.0 * EPS)
                ) and (
                    velocity_before < velocity_after
                    or self._slope(position + EPS, velocity, True) < self._acceleration_velocity_slope(position + 2.0 * EPS)
                ):
                    return False, (position, velocity), before, after
            else:
                velocity = self._acceleration_velocity(position)
                if self._acceleration_velocity_slope(position - EPS) < 0.0 < self._acceleration_velocity_slope(position + EPS):
                    return False, (position, velocity), 0.0, 0.0

    def _next_velocity_switching_point(self, s: float) -> tuple[bool, tuple[float, float], float, float]:
        """The next switching point on the velocity limit curve (Sec. VII-B), found by stepping along it for where the
        smallest phase slope crosses the slope of the curve, then bisection."""
        length = self.path.length

        def drops(position: float) -> bool:
            return self._slope(position, self._velocity_velocity(position), False) > self._velocity_velocity_slope(position)

        position, started = s - VELOCITY_SEARCH_STEP, False
        while True:
            position += VELOCITY_SEARCH_STEP
            if self._slope(position, self._velocity_velocity(position), False) >= self._velocity_velocity_slope(position):
                started = True
            if not ((not started or drops(position)) and position < length):
                break
        if position >= length:
            return True, (length, 0.0), 0.0, 0.0
        before_position, after_position = position - VELOCITY_SEARCH_STEP, position
        while after_position - before_position > VELOCITY_SEARCH_ACCURACY:
            position = 0.5 * (before_position + after_position)
            if drops(position):
                before_position = position
            else:
                after_position = position
        before = self._acceleration(before_position, self._velocity_velocity(before_position), False)
        after = self._acceleration(after_position, self._velocity_velocity(after_position), True)
        return False, (after_position, self._velocity_velocity(after_position)), before, after

    # --- the trajectory in time --------------------------------------------------------------------------------------

    def _timing(self) -> np.ndarray:
        """The time of every point of the phase-plane trajectory, the path acceleration being constant between them."""
        arcs, velocities = self._arcs, self._velocities
        keep = np.concatenate([[True], np.diff(arcs) > 0.0])
        self._arcs, self._velocities = arcs, velocities = arcs[keep], velocities[keep]
        intervals = 2.0 * np.diff(arcs) / np.maximum(velocities[1:] + velocities[:-1], 1e-12)
        return np.concatenate([[0.0], np.cumsum(intervals)])

    def _phase(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The arc length, path velocity and path acceleration at ``times`` (a time past the end is the end of the path at rest)."""
        times = np.clip(np.atleast_1d(np.asarray(times, dtype=float)), 0.0, self.duration)
        index = np.clip(np.searchsorted(self._times, times, side="right") - 1, 0, len(self._times) - 2)
        elapsed = times - self._times[index]
        interval = self._times[index + 1] - self._times[index]
        acceleration = (self._velocities[index + 1] - self._velocities[index]) / interval
        arcs = np.minimum(self._arcs[index] + self._velocities[index] * elapsed + 0.5 * acceleration * elapsed**2, self.path.length)
        velocities = self._velocities[index] + acceleration * elapsed
        ended = times >= self.duration
        return arcs, np.where(ended, 0.0, velocities), np.where(ended, 0.0, acceleration)

    def arc(self, times: np.ndarray) -> np.ndarray:
        """The arc lengths at ``times`` (a time past the end gives the end of the path)."""
        return self._phase(times)[0]

    def position(self, times: np.ndarray) -> np.ndarray:
        """The configurations ``(len(times), n)`` at ``times``."""
        return np.array([self.path.config(s) for s in self.arc(times)])
