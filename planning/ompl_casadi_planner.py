from __future__ import annotations

import numpy as np

from planning.casadi_projector import CasadiTrajectoryProjector
from planning.ompl_planner import OmplPlanner
from planning.planner import Plan, Planner, PlannerSolveError
from planning.planner_config import ExpertConfig, ProjectionConfig
from planning.totg import BlendedPath, TimedPath, TrajectoryError
from systems.triple_integrator import TripleIntegratorBase

SAMPLES_PER_STEP = 4  # samples of the timed path per control period, for the arm's progress along it
GOAL_BLEND_OUTER = 2.0  # the pose error, in units of the goal set, from which the label is the path alone


def path_weight(ratio: float) -> float:
    """The weight of the path in the label of a state whose pose error is ``ratio`` goal sets (``goal_ratio``): a smoothstep
    from 0 in the goal set (ratio 1) to 1 at ``GOAL_BLEND_OUTER``, with no slope at either end."""
    s = float(np.clip((ratio - 1.0) / (GOAL_BLEND_OUTER - 1.0), 0.0, 1.0))
    return s * s * (3.0 - 2.0 * s)


class _Progress:
    """A timed path sampled finely, and where the arm is along it. The progress is the sample nearest to the arm within
    ``window`` [rad] of path length beyond the current one, and never goes back, so a path that comes close to itself is not
    skipped over."""

    def __init__(self, waypoints: np.ndarray, simulator: TripleIntegratorBase, config: ExpertConfig) -> None:
        self._timed: TimedPath | None = None
        self._window = config.progress_window
        self._index = 0
        if np.ptp(waypoints, axis=0).max() < 1e-9:  # the arm is at the goal already
            self._times, self._arcs, self._q = np.zeros(1), np.zeros(1), waypoints[:1].copy()
            return
        try:
            path = BlendedPath(waypoints, config.totg_max_deviation)
            self._timed = TimedPath(
                path,
                config.totg_velocity_scale * simulator.state_upper_bounds[simulator.n : 2 * simulator.n],
                config.totg_acceleration_scale * simulator.max_acceleration,
            )
        except (TrajectoryError, ValueError) as exc:
            raise PlannerSolveError(f"TOTG: no timed trajectory along the path: {exc}") from exc
        self._times = np.append(np.arange(0.0, self._timed.duration, simulator.dt / SAMPLES_PER_STEP), self._timed.duration)
        self._arcs = self._timed.arc(self._times)
        self._q = self._timed.position(self._times)

    def nearest(self, q: np.ndarray) -> tuple[int, float]:
        """The sample nearest to ``q`` within the window of the progress, and its distance."""
        last = int(np.searchsorted(self._arcs, self._arcs[self._index] + self._window, side="right"))
        distances = np.linalg.norm(self._q[self._index : last] - q, axis=1)
        nearest = int(np.argmin(distances))
        return self._index + nearest, float(distances[nearest])

    def distance(self, q: np.ndarray) -> float:
        return self.nearest(q)[1]

    def reference(self, q: np.ndarray, rows: int, dt: float) -> np.ndarray:
        """The positions (rows, n) of the next ``rows`` control periods along the path, timed from the arm's progress (the
        end of the path held)."""
        self._index = max(self._index, self.nearest(q)[0])
        if self._timed is None:
            return np.repeat(self._q[:1], rows, axis=0)
        return self._timed.position(self._times[self._index] + dt * np.arange(1, rows + 1))


class OmplCasadiPlanner(Planner):
    """The expert of the position-chunk policies: an OMPL RRT-Connect path (``OmplPlanner``) timed by time-optimal
    trajectory generation (``planning/totg.py``) within the velocity and acceleration limits, whose positions ahead of the
    arm are the plan, tracked directly: there is no rest at a lookahead point to head for, the arm follows the timed path.

    A state is labelled with the positions of the next ``horizon`` control periods along the timed path from the point of it
    nearest to the arm (``_Progress``), and the plan executes them with the jerks of ``CasadiTrajectoryProjector``, which
    keep the arm within its limits and clear of the obstacles. The path is planned at the first state of an episode and
    again where the arm is when it is further than ``replan_distance`` from it (the learner drove it away).

    The goal is a pose, not a configuration: the joint configuration the path ends in is one of many that reach it, and the
    policy, who sees the goal pose only, could not know which. The label is the path blended into the nearest
    configuration of the goal pose, q + dq with dq the Gauss-Newton step of the pose error (``goal_step``), which moves
    the arm in the pose error alone, never in the nullspace of the pose, and depends on the state alone:

        positions_k = w q_path,k + (1 - w) (q + dq),

    with the weight w of the path a smoothstep of the pose error in units of the goal set (``path_weight``): 0 in the
    goal set (``at_goal_pose``), 1 from ``GOAL_BLEND_OUTER`` of it. In the goal set no path is planned.

    A label never penetrates an obstacle: its rows from the first one at which a bounding sphere does are replaced by the
    last row before it (the current configuration if there is none), so the state is labelled up to the last
    collision-free configuration and what lies inside an obstacle is never a label. A path whose first row is such a
    row is planned again from the arm's configuration, which keeps the old path if there is no new one. There is no
    relaxation of the obstacle slack: a state whose tracking the projector cannot keep clear of the obstacles raises
    ``PlannerSolveError``, as the planning of a path that does not exist does.
    """

    def __init__(self, simulator: TripleIntegratorBase, config: ExpertConfig, projection: ProjectionConfig) -> None:
        self.sim = simulator
        self.config = config
        self._ompl = OmplPlanner(simulator, config)
        self._tracker = CasadiTrajectoryProjector(simulator, projection, horizon=config.horizon)
        self._progress: _Progress | None = None
        self._paths = self._replans = self._path_failures = self._trimmed_labels = self._goal_labels = self._blocked_replans = 0

    def reset(self) -> None:
        self._progress = None
        self._tracker.reset()

    @property
    def statistics(self) -> dict[str, int]:
        """Cumulative counters: the paths planned (``ompl_paths``), the planning calls that found none
        (``ompl_path_failures``), the paths planned for an episode that already had one (``ompl_replans``: the arm had left
        it), the paths planned because the one the arm followed was blocked at its first row (``blocked_replans``, a part of
        the replans), the labels that had rows trimmed at an obstacle (``trimmed_labels``) and the labels with a part of the
        blend into the goal pose (``goal_labels``)."""
        return {
            "ompl_paths": self._paths,
            "ompl_replans": self._replans,
            "blocked_replans": self._blocked_replans,
            "ompl_path_failures": self._path_failures,
            "trimmed_labels": self._trimmed_labels,
            "goal_labels": self._goal_labels,
        }

    def plan(self, observation: np.ndarray) -> Plan:
        state = np.asarray(self.sim.invert_obs(observation), dtype=float)
        q = state[: self.sim.n]
        weight = path_weight(self.sim.goal_ratio(state))
        target = self._goal_pose_configuration(q)
        if weight > 0.0:
            positions = weight * self._path_reference(observation, q) + (1.0 - weight) * target
        else:
            positions = np.tile(target, (self.config.horizon, 1))
        self._goal_labels += int(weight < 1.0)
        positions = self._trimmed(positions, q)
        self._tracker.begin_tick()
        return Plan(positions=positions, actions=self._tracker.project(state, positions).actions)

    def _goal_pose_configuration(self, q: np.ndarray) -> np.ndarray:
        """The configuration q + dq that closes the pose error of ``q``, within the position limits."""
        error, jacobian, _ = self.sim.goal_task_error(q, self.sim.goal)
        lower, upper = self.sim.state_lower_bounds[: self.sim.n], self.sim.state_upper_bounds[: self.sim.n]
        return np.clip(q + self.sim.goal_step(error, jacobian), lower, upper)

    def _path_reference(self, observation: np.ndarray, q: np.ndarray) -> np.ndarray:
        """The positions (H, n) of the next control periods along the path, planned first where the arm has none or has left it."""
        planned = self._progress is None or self._progress.distance(q) > self.config.replan_distance
        if planned:
            self._progress = self._new_progress(observation)
        reference = self._progress.reference(q, self.config.horizon, self.sim.dt)
        if planned or self.sim.collision.free_prefix(reference) > 0:
            return reference
        try:
            progress = self._new_progress(observation)
        except PlannerSolveError:
            return reference
        self._blocked_replans += 1
        self._progress = progress
        return progress.reference(q, self.config.horizon, self.sim.dt)

    def _new_progress(self, observation: np.ndarray) -> _Progress:
        self._replans += int(self._progress is not None)
        try:
            waypoints = self._ompl.path(observation)
        except PlannerSolveError:
            self._path_failures += 1
            raise
        self._paths += 1
        return _Progress(waypoints, self.sim, self.config)

    def _trimmed(self, reference: np.ndarray, q: np.ndarray) -> np.ndarray:
        free = self.sim.collision.free_prefix(reference)
        if free == len(reference):
            return reference
        self._trimmed_labels += 1
        trimmed = reference.copy()
        trimmed[free:] = reference[free - 1] if free > 0 else q
        return trimmed
