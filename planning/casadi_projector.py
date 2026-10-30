import warnings
from typing import Any, Mapping

import casadi as ca
import numpy as np

from planning.casadi_graph import CasadiKinematicGraph, GraphSolution
from planning.planner import PlannerSolveError
from systems.triple_integrator import TripleIntegratorBase


class CasadiTrajectoryProjector:
    """
    CasADi (FATROP) projection of the flow policy's proposed action chunk onto the safe set (SafeFlowMPC, Oelerich et
    al., 2026).

    ``project`` takes the current state (as the policy observes it, see ``SystemSimulator.state_from_policy``) and a
    proposed action chunk ``u_ref`` (H, nu) and returns the closest safe action chunk (H, nu), whose first row is
    executed. Each call keeps its solution as the warm start of the next. If a solve fails, the previous safe solution shifted by one step is revalidated and reused (SafeFlowMPC, Theorem 2) for at
    most ``max_consecutive_fallbacks`` ticks in a row; else it raises ``PlannerSolveError``. The unprojected proposal
    is never returned.

    Problem (Direct Multiple Shooting, Horizon N = H)
    -------------------------------------------------
    State & Action           as in ``CasadiPlanner`` (x_k = [q_k, v_k, a_k], u_k = j_k).
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N), one per sphere shared by all obstacles.
    Parameters               x_init, u_ref, and linearized obstacle half-spaces (A_coll, b_coll) evaluated along the
                             rollout of u_ref from x_init at the first Euler step of a control tick and at its last.

    Cost                     min_{X, U, S}  sum_{k=0..N-1} ||(u_k - u_ref,k) / u_max||_2^2
                                            + beta ||[v_N, a_N]||_2^2 + w_s ||S||_F^2
                             with beta = terminal_velocity_weight (driving the horizon to rest, a soft version of
                             SafeFlowMPC's Eq. 12 so the problem stays feasible from high speed) and w_s = 10^5.

    Constraints              x_0 = x_init
                             x_{k+1} = A_d x_k + B_d u_k                 triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max                      action limits, k = 0 .. N-1
                             x_lower <= x_k <= x_upper                   position, velocity, acceleration limits, k = 1 .. N
                             A_coll,k q_k + s_k >= b_coll,k,  s_k >= 0   linearized sphere-obstacle half-spaces, k = 1 .. N

    The task goal is not part of the problem: the projector never reads the goal of its ``simulator`` (only the dynamics,
    the limits and the obstacles), so it can share the simulator of the rollout instead of holding a copy of its own.

    Solver Choice (``Opti`` + ``fatrop``)
    -------------------------------------
    During flow ODE integration, ``project`` is called at every Euler step, so solver latency dominates inference time.
    The triple-integrator dynamics, state/action box limits, and linearized obstacle half-spaces are affine and the
    tracking + terminal-rest objective is strictly convex quadratic, so the projection is a convex QP with the stage
    structure of an optimal control problem, solved in a few ms by **FATROP** (see ``CasadiKinematicGraph``) started from
    the previous solution across adjacent Euler steps and control ticks.

    Every Euler step of a tick is one such solve to convergence, linearized along the rollout of its own proposal. The
    half-spaces are first-order, so a solution that still penetrates an obstacle is linearized again around itself and
    solved once more (``CasadiKinematicGraph.solve``); if it still penetrates, the projection has failed, as it does in
    SafeFlowMPC when the slack of its filter exceeds a tolerance (see above for what a failed projection does).

    Initial guess strategy
    ----------------------
    The first solve after a reset starts from the current state held still (zero actions: within the limits whatever
    the proposal is, which a guess following a far-off proposal, such as the flow's initial noise, is not). Within a
    control tick (the same ``x_init``; only the proposal is refined between flow steps) the last solution is reused as it
    is; at a new tick it is the previous solution shifted by one step with zero jerk appended (the rest state it ends in
    is a fixed point). A solve that fails is restarted cold
    from the state held still; if that fails too, the previous safe solution is the fallback (see above).
    """

    def __init__(self, simulator: TripleIntegratorBase, config: Mapping[str, Any], *, horizon: int):
        # The chunk length is the flow policy's prediction horizon, not the expert's ``horizon``
        if int(horizon) < 1:
            raise ValueError("'horizon' must be at least 1.")
        # The scalars of the validated config (planning/config.py), used as they are
        self.terminal_velocity_weight = float(config["terminal_velocity_weight"])
        # How close to zero the rest velocity and acceleration of a reused trajectory must be
        self.fallback_terminal_velocity_tol = float(config["fallback_terminal_velocity_tol"])
        self.max_consecutive_fallbacks = int(config["max_consecutive_fallbacks"])
        self.sim = simulator
        self.tracked_horizon = self.N = int(horizon)
        self.rest_idx = list(self.sim.rest_state_indices)
        self._graph, self.u_ref_param = self._problem()
        self.reset()

    def _problem(self) -> tuple[CasadiKinematicGraph, ca.MX]:
        """The projection problem and its ``u_ref``."""
        graph = CasadiKinematicGraph(self.sim, self.N)
        max_action = np.asarray(self.sim.max_action, dtype=float).reshape(-1, 1)
        # Cost: track the proposed chunk (each action divided by its limit) and come to rest at the end of the horizon
        u_ref = graph.parameter(self.sim.nu, self.N)
        tracking = ca.sumsqr((graph.U - u_ref) / ca.repmat(ca.DM(max_action), 1, self.N))
        graph.minimize(tracking + self.terminal_velocity_weight * ca.sumsqr(graph.X[self.rest_idx, self.N]))
        return graph, u_ref

    def reset(self) -> None:
        """Signals the start of a new episode: nothing of the last one is reused."""
        self._prev_X_sol: np.ndarray | None = None
        self._prev_U_sol: np.ndarray | None = None
        self._prev_x0: np.ndarray | None = None
        self._consecutive_fallback_count = 0
        self._graph.reset()

    def next_chunk_guess(self) -> np.ndarray | None:
        """The previous solution's actions shifted by one step, zero actions appended (H, nu): the natural initial
        chunk of the next tick; None right after a reset."""
        if self._prev_U_sol is None:
            return None
        return np.hstack([self._prev_U_sol[:, 1:], np.zeros((self.sim.nu, 1))]).T

    def _hold_still_guess(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stay at the current positions at rest: within the limits whatever the proposal is, which a guess following
        a far-off proposal (e.g. the flow's initial noise) is not, and the solver often fails from there."""
        states = np.zeros((self.sim.nx, self.N + 1))
        states[: self.sim.n, :] = x0[: self.sim.n, None]
        states[:, 0] = x0
        return states, np.zeros((self.sim.nu, self.N))

    def _revalidate_fallback(self, x0: np.ndarray, u_guess: np.ndarray) -> np.ndarray | None:
        """The states of ``u_guess`` applied from the actual ``x0`` if they respect the limits and end at rest (within
        ``fallback_terminal_velocity_tol``), else None: the cached plan was only verified from the state it was solved from."""
        states = np.zeros((self.sim.nx, self.N + 1))
        states[:, 0] = x0
        for step in range(self.N):
            states[:, step + 1] = self.sim.predict_next_state(states[:, step], u_guess[:, step])
        tolerance = 1e-6
        within_limits = np.all(states[:, 1:] >= np.asarray(self.sim.state_lower_bounds)[:, None] - tolerance) and np.all(
            states[:, 1:] <= np.asarray(self.sim.state_upper_bounds)[:, None] + tolerance
        )
        at_rest = np.all(np.abs(states[self.rest_idx, self.N]) <= self.fallback_terminal_velocity_tol)
        collision_free = all(self.sim.collision.is_collision_free(states[: self.sim.n, k]) for k in range(1, self.N + 1))
        return states if within_limits and at_rest and collision_free else None

    def _linearize(self, x0: np.ndarray, u_ref: np.ndarray) -> None:
        """Linearize the obstacles along the rollout of the proposal ``u_ref`` from ``x0``."""
        if self._graph.num_collision_pairs == 0:
            return
        self._graph.set_collision_halfspaces(*self.sim.collision.halfspaces(self.sim.rollout_positions(x0, u_ref)[1:]))

    def _solve(self) -> GraphSolution:
        """The converged solution of the problem set, which has to be clear of the obstacles."""
        solution = self._graph.solve()
        if not all(self.sim.collision.is_collision_free(solution.states[: self.sim.n, k]) for k in range(1, self.N + 1)):
            raise RuntimeError("The projection penetrates an obstacle.")
        return solution

    @staticmethod
    def _set_problem(
        graph: CasadiKinematicGraph, u_ref_param: ca.MX, x0: np.ndarray, u_ref: np.ndarray, states: np.ndarray, actions: np.ndarray
    ) -> None:
        graph.set_state(x0)
        graph.set_value(u_ref_param, u_ref.T)
        graph.set_initial(states, actions)

    def project(self, state: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
        """The projected actions (H, nu) of the proposed chunk ``u_ref`` (H, nu) from ``state``, the observed one of the
        policy (the plant state itself unless the policy cannot see all of it); see the class docstring."""
        u_ref = np.asarray(u_ref, dtype=float)
        if u_ref.shape != (self.N, self.sim.nu):
            raise ValueError(f"u_ref must have shape {(self.N, self.sim.nu)}, got {u_ref.shape}.")
        x0 = self.sim.state_from_policy(state)
        if x0.shape != (self.sim.nx,):
            raise ValueError(f"state must have shape {(self.sim.nx,)}, got {x0.shape}.")
        # Within a control tick (same x0, only the proposal refined between flow steps) the last solution is reused as
        # is; at a new tick it is shifted by one step, with zero jerk appended (the rest state it ends in is a fixed point)
        is_new_control_tick = self._prev_x0 is None or not np.array_equal(x0, self._prev_x0)
        if self._prev_X_sol is None:
            state_guess, u_guess = self._hold_still_guess(x0)
        elif not is_new_control_tick:
            state_guess, u_guess = self._prev_X_sol, self._prev_U_sol
        else:
            state_guess = np.hstack([self._prev_X_sol[:, 1:], self._prev_X_sol[:, -1:]])
            u_guess = np.hstack([self._prev_U_sol[:, 1:], np.zeros((self.sim.nu, 1))])

        try:
            try:
                self._linearize(x0, u_ref)
                self._set_problem(self._graph, self.u_ref_param, x0, u_ref, state_guess, u_guess)
                sol = self._solve()
            except RuntimeError:
                # A warm start can leave the solver stuck at a locally infeasible point once the state has drifted from the
                # prediction: a cold restart resolves most of those
                cold_states, cold_actions = self._hold_still_guess(x0)
                self._linearize(x0, u_ref)
                self._set_problem(self._graph, self.u_ref_param, x0, u_ref, cold_states, cold_actions)
                sol = self._solve()
            self._prev_X_sol = sol.states
            self._prev_U_sol = sol.actions
            self._prev_x0 = x0
            self._consecutive_fallback_count = 0
            return self._prev_U_sol.T.copy()
        except RuntimeError as exc:
            budget_left = not is_new_control_tick or self._consecutive_fallback_count < self.max_consecutive_fallbacks
            if self._prev_U_sol is not None and budget_left:
                revalidated = self._revalidate_fallback(x0, u_guess)
                if revalidated is not None:
                    warnings.warn(
                        f"CasadiTrajectoryProjector solve failed, falling back to the last known-safe trajectory: {exc}",
                        stacklevel=2,
                    )
                    self._prev_X_sol, self._prev_U_sol, self._prev_x0 = revalidated, u_guess, x0
                    self._consecutive_fallback_count += int(is_new_control_tick)
                    return u_guess.T.copy()
                self.reset()  # the cached plan no longer holds
            raise PlannerSolveError(
                f"CasadiTrajectoryProjector solve failed with no revalidated safe trajectory to fall back on: {exc}"
            ) from exc
