from __future__ import annotations

import numpy as np

from learning.models.rest_projection import RestProjection
from learning.models.smoothness import SmoothSpread
from systems.triple_integrator import TripleIntegratorBase

GUIDANCE_DECAY = 10.0  # alpha of the guidance weight (SafeFlowMPC, Eq. 14)
GUIDANCE_ONSET = 0.09  # beta: the weight is 1 once the goal is closer than this fraction of the start distance
JACOBIAN_DAMPING = 1e-3  # of the damped pseudo-inverse of the task Jacobian


class PoseGuidance:
    """Guidance of the flow's proposals toward the goal pose (SafeFlowMPC, Oelerich et al., 2026, Eq. 9, 13, 14).

    The goal is read from the policy observation (the goal in the TCP frame), so, like the projector, the guidance
    reads no goal of its simulator. ``guide`` moves the proposed chunk of positions toward the goal, once per Euler step of
    the flow, before the projection.

    For the positions q_k, k = 1..H, of the proposed chunk, the task error e_k (the logarithm of the pose of the goal in the
    TCP frame, in se(3)) and its Jacobian J_k, with e(q + dq) = e(q) - J dq (the body Jacobian times the left Jacobian of
    SE(3), J_l^-1), give the damped Gauss-Newton step dq_k = (J^T J + a I)^-1 J^T e_k, and the
    positions are moved by w_k dq_k with

        w_k = min(1, exp(-alpha (d_k / d_0 - beta))) k / H,

    d_k the position error of q_k and d_0 that of the start of the episode: the guidance only acts close to the goal
    (relative to the start distance) and more so on the later positions of the horizon.

    The moves of the positions are first spread along the chunk by the smoothness metric of CHOMP (``SmoothSpread``), so
    that the correction is smooth in velocity and acceleration instead of the sum of one step per position. Close to the goal
    the chunk is then brought to rest: ``RestProjection`` scales its final velocity and acceleration by one minus the closeness
    of its last position to the goal, CHOMP's projection onto a constraint (Eq. 39 of Zucker et al. 2013).
    """

    def __init__(self, simulator: TripleIntegratorBase, horizon: int) -> None:
        self.sim = simulator
        self.horizon = int(horizon)
        self._start_distance: float | None = None
        self._spread = SmoothSpread(self.horizon)
        self._rest = RestProjection(self.horizon, simulator.dt)

    def reset(self) -> None:
        """A new episode: the start distance is the one of its first call."""
        self._start_distance = None

    def _closeness(self, distance: np.ndarray | float) -> np.ndarray | float:
        """1 once the goal is closer than ``GUIDANCE_ONSET`` of the start distance, falling off exponentially beyond it
        (of every distance of an array)."""
        return np.minimum(1.0, np.exp(-GUIDANCE_DECAY * (np.asarray(distance, dtype=float) / self._start_distance - GUIDANCE_ONSET)))

    def guide(self, environment_state: np.ndarray, state: np.ndarray, proposal: np.ndarray) -> np.ndarray:
        """The proposed chunk of positions (H, n) moved toward the goal."""
        sim, horizon = self.sim, self.horizon
        goal = sim.observation_from_policy(environment_state, state)[: sim.goal_dim]
        if self._start_distance is None:
            self._start_distance = max(sim.goal_task_error(np.asarray(state, dtype=float)[: sim.n], goal)[2], 1e-6)
        errors, jacobians, distances = (np.array(values) for values in zip(*[sim.goal_task_error(q, goal) for q in proposal]))
        damped = np.einsum("hmi,hmj->hij", jacobians, jacobians) + JACOBIAN_DAMPING * np.eye(sim.n)
        steps = np.linalg.solve(damped, np.einsum("hmi,hm->hi", jacobians, errors)[..., None])[..., 0]
        weights = self._closeness(distances) * np.arange(1, horizon + 1) / horizon
        guided = proposal + self._spread(weights[:, None] * steps)
        return self._rest(guided, float(self._closeness(sim.goal_task_error(guided[-1], goal)[2])))
