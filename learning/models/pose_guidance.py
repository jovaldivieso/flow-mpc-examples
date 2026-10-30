from __future__ import annotations

import numpy as np

from systems.triple_integrator import TripleIntegratorBase

GUIDANCE_DECAY = 10.0  # alpha of the guidance weight (SafeFlowMPC, Eq. 14)
GUIDANCE_ONSET = 0.09  # beta: the weight is 1 once the goal is closer than this fraction of the start distance
JACOBIAN_DAMPING = 1e-3  # of the damped pseudo-inverse of the task Jacobian
JERK_RIDGE = 1e-6  # of the jerk correction, relative to the largest singular value squared of the jerk-to-state map
REST_WEIGHT = 10.0  # of keeping the final velocity and acceleration of the chunk, against the positions it moves


class PoseGuidance:
    """Guidance of the flow's proposals toward the goal pose (SafeFlowMPC, Oelerich et al., 2026, Eq. 9, 13, 14).

    The goal is read from the policy observation (the goal in the TCP frame), so, like the projector, the guidance
    reads no goal of its simulator. ``guide`` turns the proposed chunk of jerks into the chunk of the same plant whose
    positions are moved toward the goal, once per Euler step of the flow, before the projection.

    For the positions q_k, k = 1..H, of the proposed chunk, the task error e_k (position, then orientation) gives the
    damped Gauss-Newton step dq_k = (J^T J + a I)^-1 J^T e_k, and the positions are moved by w_k dq_k with

        w_k = min(1, exp(-alpha (d_k / d_0 - beta))) k / H,

    d_k the position error of q_k and d_0 that of the start of the episode: the guidance only acts close to the goal
    (relative to the start distance) and more so on the later positions of the horizon. The jerks of that move solve

        min_du  ||M du - [w dq; 0; 0]||_W^2 + lambda ||du||^2,

    M the linear map from the jerks of one joint to [q_1..q_H, v_H, a_H] (the exact triple integrator), so that the
    final velocity and acceleration of the chunk, which the policy brings to rest, are kept.
    """

    def __init__(self, simulator: TripleIntegratorBase, horizon: int) -> None:
        self.sim = simulator
        self.horizon = int(horizon)
        self._start_distance: float | None = None
        self._correction = self._correction_map()

    def reset(self) -> None:
        """A new episode: the start distance is the one of its first call."""
        self._start_distance = None

    def _correction_map(self) -> np.ndarray:
        """The (H, H + 2) map from the wanted changes of [q_1..q_H, v_H, a_H] of a joint to its change of jerks."""
        n, horizon = self.sim.n, self.horizon
        jerk_to_state = np.zeros((horizon + 2, horizon))
        for column in range(horizon):
            state = np.zeros(self.sim.nx)
            for step in range(horizon):
                jerk = np.zeros(self.sim.nu)
                jerk[0] = float(step == column)
                state = np.array(self.sim.casadi_dynamics(state, jerk), dtype=float).ravel()
                jerk_to_state[step, column] = state[0]
            jerk_to_state[horizon, column], jerk_to_state[horizon + 1, column] = state[n], state[2 * n]
        weights = np.concatenate([np.ones(horizon), np.full(2, REST_WEIGHT)])
        weighted = jerk_to_state * weights[:, None]
        gram = weighted.T @ weighted
        ridge = JERK_RIDGE * np.linalg.norm(gram, 2)
        return np.linalg.solve(gram + ridge * np.eye(horizon), weighted.T * weights[None, :])

    def _weight(self, distance: float, step: int) -> float:
        closeness = min(1.0, float(np.exp(-GUIDANCE_DECAY * (distance / self._start_distance - GUIDANCE_ONSET))))
        return closeness * (step + 1) / self.horizon

    def guide(self, environment_state: np.ndarray, state: np.ndarray, proposal: np.ndarray) -> np.ndarray:
        """The proposed chunk (H, nu) moved toward the goal."""
        sim, horizon = self.sim, self.horizon
        x0 = sim.state_from_policy(state)
        goal = sim.observation_from_policy(environment_state, state)[: sim.goal_dim]
        if self._start_distance is None:
            self._start_distance = max(sim.goal_task_error(x0[: sim.n], goal)[2], 1e-6)
        moves = np.zeros((horizon + 2, sim.n))
        for step, q in enumerate(sim.rollout_positions(x0, proposal)[1:]):
            error, jacobian, distance = sim.goal_task_error(q, goal)
            damped = jacobian.T @ jacobian + JACOBIAN_DAMPING * np.eye(sim.n)
            moves[step] = self._weight(distance, step) * np.linalg.solve(damped, jacobian.T @ error)
        # The guidance never takes a jerk past its limit; a proposal beyond it is left to the projection
        bound = np.maximum(sim.max_action, np.abs(proposal))
        return np.clip(proposal + self._correction @ moves, -bound, bound)
