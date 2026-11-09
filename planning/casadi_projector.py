import warnings
from typing import NamedTuple

import casadi as ca
import numpy as np

from planning.casadi_graph import CasadiKinematicGraph, GraphSolution
from planning.planner import PlannerSolveError
from planning.planner_config import ProjectionConfig
from systems.triple_integrator import TripleIntegratorBase


class Projection(NamedTuple):
    """The plan a projection finds: the joint positions of the next ``H`` control periods (row k is the configuration k + 1
    periods ahead) and the jerks (``H``, ``nu``) that take the arm there, the first of which is executed."""

    positions: np.ndarray
    actions: np.ndarray


class CasadiTrajectoryProjector:
    """
    CasADi (FATROP) projection of a reference chunk of joint positions onto the safe set (SafeFlowMPC, Oelerich et al.,
    2026): the safety filter that tracks the flow's proposal and returns the jerks that execute it.

    ``project`` takes the current state and a
    reference chunk ``q_ref`` (H, n) of the configurations of the next H control periods, and returns the safe plan that
    tracks it (``Projection``), whose first jerk is executed. Each call keeps its solution as the warm start of the next. If a
    solve fails, the previous safe solution shifted by one step is revalidated and reused (SafeFlowMPC, Theorem 2) for at
    most ``max_consecutive_fallbacks`` ticks in a row; else it raises ``PlannerSolveError``. At the start of an episode
    the safe solution (Assumption 2) is holding still, which holds when the arm is at rest and clear of the obstacles. The
    unprojected reference is never returned.

    Problem (Direct Multiple Shooting, Horizon N = H)
    -------------------------------------------------
    State & Action           x_k = [q_k, v_k, a_k] (positions, velocities, accelerations), u_k = j_k (jerks).
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N), one per sphere shared by its free-space half-spaces.
    Parameters               x_init, q_ref, and the linearized free-space half-spaces (A_coll, b_coll): at every node
                             the set of each bounding sphere around its last step of ``q_ref`` from x_init, linearized
                             along ``q_ref`` (``SphereCollisionModel.halfspaces``).

    Cost                     min_{X, U, S}  sum_{k=1..N} ( ||q_k - q_ref,k||^2 + w_v ||v_k||^2 + w_a ||a_k||^2 )
                                            + w_u sum_{k=0..N-1} ||u_k||^2 + beta ||[v_N, a_N]||^2 + w_1 sum(S) + w_2 ||S||_F^2
                             with the weights of ``ProjectionConfig``: the positions weigh 1 and the derivatives little, as in the
                             authors' safety filter, beta = terminal_velocity_weight (driving the horizon to rest, a soft
                             version of SafeFlowMPC's Eq. 12 so the problem stays feasible from high speed), w_1 = 10^3 and w_2 = 1.

    Constraints              x_0 = x_init
                             x_{k+1} = A_d x_k + B_d u_k                 triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max                      action limits, k = 0 .. N-1
                             x_lower <= x_k <= x_upper                   position, velocity, acceleration limits, k = 1 .. N
                             A_coll,k q_k + s_k >= b_coll,k,  s_k >= 0   linearized free-space half-spaces, k = 1 .. N

    With ``safe=False`` the problem has neither the half-spaces nor the terminal rest: it is the tracking of the reference
    within the plant's limits alone, what a policy without safety filter executes its proposal with. Safe is on or off, with
    nothing in between: a plan is safe when it is clear of the obstacles and comes to rest.

    The task goal is not part of the problem: the projector never reads the goal of its ``simulator`` (only the dynamics,
    the limits and the obstacles), so it can share the simulator of the rollout instead of holding a copy of its own.

    Solver Choice (``Opti`` + ``fatrop``)
    -------------------------------------
    During flow ODE integration, ``project`` is called at every Euler step, so solver latency dominates inference time.
    The triple-integrator dynamics, state/action box limits, and linearized free-space half-spaces are affine and the
    tracking + terminal-rest objective is strictly convex quadratic, so the projection is a convex QP with the stage
    structure of an optimal control problem, solved in a few ms by **FATROP** (see ``CasadiKinematicGraph``) started from
    the previous solution across adjacent Euler steps and control ticks.

    Every Euler step of a tick is one such solve to convergence, linearized along its own reference. The half-spaces are
    first-order, so a solution that still penetrates an obstacle or has a slack above the tolerance is linearized again
    around itself and solved once more (``CasadiKinematicGraph.solve``); if it is still rejected, the projection has
    failed, as it does in SafeFlowMPC when the slack of its filter exceeds a tolerance (see above for what a failed
    projection does).

    Initial guess strategy
    ----------------------
    The first solve after a reset starts from the current state held still (zero actions: within the limits whatever
    the reference is, which a guess following a far-off reference, such as the flow's initial noise, is not). Within a
    control tick (the same ``x_init``; only the reference is refined between flow steps) the last solution is reused as it
    is; at a new tick it is the previous solution shifted by ``tick_steps`` steps (the control periods between two ticks) with zero jerk appended (the rest state it ends in
    is a fixed point). A solve that fails is restarted cold from the state held still; if that fails too, the previous
    safe solution is the fallback (see above).
    """

    def __init__(self, simulator: TripleIntegratorBase, config: ProjectionConfig, *, horizon: int, safe: bool = True, tick_steps: int = 1):
        if int(horizon) < 1:
            raise ValueError("'horizon' must be at least 1.")
        if not 1 <= int(tick_steps) <= int(horizon):
            raise ValueError("'tick_steps' must be between 1 and the horizon.")
        self.tick_steps = int(tick_steps)
        self.config = config
        self.safe = bool(safe)
        self.sim = simulator
        self.tracked_horizon = self.N = int(horizon)
        self.rest_idx = list(self.sim.rest_state_indices)
        self._graph, self.q_ref_param = self._problem()
        self.reset()

    def _problem(self) -> tuple[CasadiKinematicGraph, ca.MX]:
        """The projection problem and its reference parameter."""
        graph = CasadiKinematicGraph(self.sim, self.N, with_obstacles=self.safe)
        n, N, config = self.sim.n, self.N, self.config
        q_ref = graph.parameter(n, N)
        cost = (
            ca.sumsqr(graph.X[:n, 1:] - q_ref)
            + config.velocity_weight * ca.sumsqr(graph.X[n : 2 * n, 1:])
            + config.acceleration_weight * ca.sumsqr(graph.X[2 * n :, 1:])
            + config.jerk_weight * ca.sumsqr(graph.U)
        )
        if self.safe:
            cost += config.terminal_velocity_weight * ca.sumsqr(graph.X[self.rest_idx, N])
        graph.minimize(cost)
        return graph, q_ref

    def reset(self) -> None:
        """Signals the start of a new episode: nothing of the last one is reused."""
        self._prev_X_sol: np.ndarray | None = None
        self._prev_U_sol: np.ndarray | None = None
        self._prev_x0: np.ndarray | None = None
        self._tick_requested = False
        self._consecutive_fallback_count = 0
        self._graph.reset()

    def begin_tick(self) -> None:
        """The next ``project`` call starts a new control tick even if the state is the same as in the last one: an arm that
        is held at rest has the same state tick after tick, which the state alone does not tell from the flow steps of one
        tick (calls with the same state, only the reference refined)."""
        self._tick_requested = True

    def next_chunk_guess(self) -> np.ndarray | None:
        """The previous solution's positions ``tick_steps`` control periods later (H, n), the last one held: the natural
        initial chunk of the next tick; None right after a reset."""
        if self._prev_X_sol is None:
            return None
        n, k = self.sim.n, self.tick_steps
        return np.hstack([self._prev_X_sol[:n, k + 1 :], np.repeat(self._prev_X_sol[:n, -1:], k, axis=1)]).T

    def _hold_still_guess(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stay at the current positions at rest: within the limits whatever the reference is, which a guess following
        a far-off reference (e.g. the flow's initial noise) is not, and the solver often fails from there."""
        states = np.zeros((self.sim.nx, self.N + 1))
        states[: self.sim.n, :] = x0[: self.sim.n, None]
        states[:, 0] = x0
        return states, np.zeros((self.sim.nu, self.N))

    def _revalidate_fallback(self, x0: np.ndarray, u_guess: np.ndarray) -> np.ndarray | None:
        """The states of ``u_guess`` applied from the actual ``x0`` if they respect the limits and, with the safety filter, are
        clear of the obstacles, else None: the cached plan was only verified from the state it was solved from."""
        states = np.zeros((self.sim.nx, self.N + 1))
        states[:, 0] = x0
        for step in range(self.N):
            states[:, step + 1] = self.sim.predict_next_state(states[:, step], u_guess[:, step])
        tolerance = 1e-6
        within_limits = np.all(states[:, 1:] >= np.asarray(self.sim.state_lower_bounds)[:, None] - tolerance) and np.all(
            states[:, 1:] <= np.asarray(self.sim.state_upper_bounds)[:, None] + tolerance
        )
        if not self.safe:
            return states if within_limits else None
        collision_free = self.sim.collision.trajectory_is_collision_free(states[: self.sim.n, 1:].T)
        return states if within_limits and collision_free else None

    def _linearize(self, x0: np.ndarray, q_ref: np.ndarray) -> None:
        """Linearize the obstacles along the reference ``q_ref`` from ``x0``, which the graph already holds."""
        self._graph.update_collision_halfspaces(q_ref)

    def _solve(self) -> GraphSolution:
        """The converged solution of the problem set, which with the safety filter has to be clear of the obstacles."""
        solution = self._graph.solve()
        if self.safe and solution.penetrates:
            raise RuntimeError("The projection penetrates an obstacle.")
        return solution

    @staticmethod
    def _set_problem(
        graph: CasadiKinematicGraph, q_ref_param: ca.MX, x0: np.ndarray, q_ref: np.ndarray, states: np.ndarray, actions: np.ndarray
    ) -> None:
        graph.set_state(x0)
        graph.set_value(q_ref_param, q_ref.T)
        graph.set_initial(states, actions)

    def project(self, state: np.ndarray, q_ref: np.ndarray) -> Projection:
        """The plan that tracks the reference chunk ``q_ref`` (H, n) from ``state``; see the class docstring."""
        q_ref = np.asarray(q_ref, dtype=float)
        if q_ref.shape != (self.N, self.sim.n):
            raise ValueError(f"q_ref must have shape {(self.N, self.sim.n)}, got {q_ref.shape}.")
        x0 = np.asarray(state, dtype=float)
        if x0.shape != (self.sim.nx,):
            raise ValueError(f"state must have shape {(self.sim.nx,)}, got {x0.shape}.")
        # Within a control tick (same x0, only the reference refined between flow steps) the last solution is reused as
        # is; at a new tick (a new state, or ``begin_tick``) it is shifted by ``tick_steps`` steps, with zero jerk appended
        # (the rest state it ends in is a fixed point)
        is_new_control_tick = self._tick_requested or self._prev_x0 is None or not np.array_equal(x0, self._prev_x0)
        self._tick_requested = False
        if self._prev_X_sol is None:
            state_guess, u_guess = self._hold_still_guess(x0)
        elif not is_new_control_tick:
            state_guess, u_guess = self._prev_X_sol, self._prev_U_sol
        else:
            k = self.tick_steps
            state_guess = np.hstack([self._prev_X_sol[:, k:], np.repeat(self._prev_X_sol[:, -1:], k, axis=1)])
            u_guess = np.hstack([self._prev_U_sol[:, k:], np.zeros((self.sim.nu, k))])

        try:
            try:
                self._set_problem(self._graph, self.q_ref_param, x0, q_ref, state_guess, u_guess)
                self._linearize(x0, q_ref)
                sol = self._solve()
            except RuntimeError:
                # A warm start can leave the solver stuck at a locally infeasible point once the state has drifted from the
                # prediction: a cold restart resolves most of those, unless the attempt already started from holding still
                cold_states, cold_actions = self._hold_still_guess(x0)
                if np.allclose(state_guess, cold_states, rtol=0.0, atol=1e-9) and np.allclose(u_guess, cold_actions, rtol=0.0, atol=1e-9):
                    raise
                self._set_problem(self._graph, self.q_ref_param, x0, q_ref, cold_states, cold_actions)
                self._linearize(x0, q_ref)
                sol = self._solve()
            self._prev_X_sol = sol.states
            self._prev_U_sol = sol.actions
            self._prev_x0 = x0
            self._consecutive_fallback_count = 0
            return Projection(sol.states[: self.sim.n, 1:].T.copy(), sol.actions.T.copy())
        except RuntimeError as exc:
            budget_left = not is_new_control_tick or self._consecutive_fallback_count < self.config.max_consecutive_fallbacks
            if budget_left:  # without a previous solution u_guess holds still
                revalidated = self._revalidate_fallback(x0, u_guess)
                if revalidated is not None:
                    warnings.warn(
                        f"CasadiTrajectoryProjector solve failed, falling back to the last known-safe trajectory: {exc}",
                        stacklevel=2,
                    )
                    self._prev_X_sol, self._prev_U_sol, self._prev_x0 = revalidated, u_guess, x0
                    self._consecutive_fallback_count += int(is_new_control_tick)
                    return Projection(revalidated[: self.sim.n, 1:].T.copy(), u_guess.T.copy())
                self.reset()  # the cached plan no longer holds
            raise PlannerSolveError(
                f"CasadiTrajectoryProjector solve failed with no revalidated safe trajectory to fall back on: {exc}"
            ) from exc
