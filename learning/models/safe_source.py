from __future__ import annotations

import warnings

import numpy as np

from planning.casadi_projector import CasadiTrajectoryProjector
from planning.planner import PlannerSolveError
from planning.planner_config import ProjectionConfig
from systems.triple_integrator import TripleIntegratorBase

PATH_SEGMENTS = 10  # the projected path from the source to the target is kept at PATH_SEGMENTS + 1 flow times


class SafeSource:
    """The safe path from a safe source to a safe target of a labeled state, for training the flow on the safe set
    (SafeFlowMPC, Oelerich et al., 2026, Sec. II-B and Eq. 7).

    The source is a chunk of jerks drawn uniformly within the plant's jerk limits, whose positions from the state are made
    safe by the filter; the target is the label made safe by the same filter. The filter is the safety projection of the limits,
    the linear dynamics, the obstacles and the terminal rest: a plan that is not clear of the obstacles is not safe. The
    flow learns the target flow of Eq. 7, the derivative with respect to the flow time s of the projection of the straight
    line between the two, ``P(source + s (target - source))``, kept at ``PATH_SEGMENTS + 1`` values of s and differentiated
    by finite differences as in the authors' dataset. A point of the line that is safe already is its own projection (the
    limits, the linear dynamics and the terminal rest are convex, so only the obstacles can make it unsafe); the others are
    projected by the filter.

    A source is drawn from the index of its frame alone, so the paths of a dataset do not depend on how they are computed. A
    path whose source, target or a point has no safe plan (no plan within the limits that is clear of the obstacles and comes
    to rest) is left out: its row is trained from Gaussian noise towards the expert's label.
    """

    def __init__(self, simulator: TripleIntegratorBase, projection: ProjectionConfig, horizon: int, seed: int) -> None:
        self._sim = simulator
        self._horizon = int(horizon)
        self._seed = int(seed)
        self._filter = CasadiTrajectoryProjector(simulator, projection, horizon=self._horizon)

    def path(self, state: np.ndarray, label: np.ndarray, index: int) -> np.ndarray | None:
        """The projected path (PATH_SEGMENTS + 1, H, n) of the frame ``index`` at ``state`` with the expert's ``label`` (H, n),
        from its safe source to its safe target, or None when any of its points has no safe plan."""
        state = np.asarray(state, dtype=float)
        limit = np.asarray(self._sim.max_action, dtype=float)
        jerks = np.random.default_rng([self._seed, int(index)]).uniform(-limit, limit, (self._horizon, self._sim.nu))
        source = self._make_safe(state, self._sim.rollout_positions(state, jerks)[1:])
        target = self._make_safe(state, np.asarray(label, dtype=float))
        if source is None or target is None:
            return None
        path = [source]
        for segment in range(1, PATH_SEGMENTS):
            point = source + segment / PATH_SEGMENTS * (target - source)
            if not self._sim.collision.trajectory_is_collision_free(point):
                point = self._make_safe(state, point)
                if point is None:
                    return None
            path.append(point)
        path.append(target)
        return np.stack(path)

    def paths(self, states: np.ndarray, labels: np.ndarray, first_index: int) -> np.ndarray:
        """The paths (N, PATH_SEGMENTS + 1, H, n) of the consecutive frames from ``first_index``; NaN where a frame has none."""
        paths = np.full((len(states), PATH_SEGMENTS + 1, self._horizon, self._sim.n), np.nan, dtype=np.float32)
        for row, (state, label) in enumerate(zip(states, labels)):
            found = self.path(state, label, first_index + row)
            if found is not None:
                paths[row] = found
        return paths

    def _make_safe(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
        """The plan of the filter that tracks ``reference`` from ``state``, or None if it has none: a fallback to an earlier
        plan, which would be no projection of the reference, counts as failed."""
        self._filter.reset()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                return self._filter.project(state, reference).positions
            except (PlannerSolveError, UserWarning):
                return None
