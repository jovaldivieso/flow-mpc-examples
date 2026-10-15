import warnings
from typing import Any, Mapping

import casadi as ca
import numpy as np

from planning.planner import PlannerSolveError
from dynamics.dynamics import DynamicsProtocol


class CasadiTrajectoryProjector:
    """
    CasADi (IPOPT) projection of the flow policy's proposed action chunk onto the safe set (SafeFlowMPC, Oelerich et
    al., 2026).

    ``project`` takes an observation and a chunk ``u_ref`` (H, nu), row k the proposed action at step k, and returns
    the projected actions (H, nu): the first row is what the system executes. Each call keeps its solution as the
    warm start of the next. If a solve fails, the previous solution is reused (SafeFlowMPC, Theorem 2) as long as it
    still respects the limits and ends at rest from the current state, at most ``max_consecutive_fallbacks`` ticks in
    a row; else it raises ``PlannerSolveError``. The solution is never the unprojected proposal.

    Problem
    -------
    State (nx), action (nu)  as in ``CasadiPlanner`` (x = [q, v, a], u = jerk).
    Optimization variables   X = [x_0 .. x_N] (nx, N + 1) and U = [u_0 .. u_{N-1}] (nu, N), horizon N = H.
    Parameters               x_init (the current state) and u_ref.

    Cost                     J = sum_{k=0..N-1} |(u_k - u_ref_k) / u_max|^2 + beta |x_N[v, a]|^2
                             with u_max the action limit and beta = ``terminal_velocity_weight``: the horizon ends at
                             rest, a soft version of SafeFlowMPC's Eq. 12 (a hard condition can be infeasible from
                             high speed).
    Constraints              x_0 = x_init
                             x_{k+1} = f(x_k, u_k)             triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max             action limits
                             x_lower <= x_k <= x_upper, k >= 1  position, velocity and acceleration limits

    The task goal and collisions are not part of the problem.
    """

    def __init__(self, local_sim: DynamicsProtocol, config: Mapping[str, Any], *, horizon: int):
        self.sim = local_sim
        # The chunk length is the flow policy's prediction horizon, not the expert's ``horizon``
        self.tracked_horizon = self.N = int(horizon)
        if self.N < 1:
            raise ValueError("'horizon' must be at least 1.")
        self.terminal_velocity_weight = float(config.get("terminal_velocity_weight", 1.0))
        if self.terminal_velocity_weight <= 0:
            raise ValueError("'terminal_velocity_weight' must be positive.")
        # How close to zero the rest velocity and acceleration of a reused trajectory must be
        self.fallback_terminal_velocity_tol = float(config.get("fallback_terminal_velocity_tol", 1e-2))
        if self.fallback_terminal_velocity_tol < 0:
            raise ValueError("'fallback_terminal_velocity_tol' must be non-negative.")
        self.max_consecutive_fallbacks = int(config.get("max_consecutive_fallbacks", 5))
        if self.max_consecutive_fallbacks <= 0:
            raise ValueError("'max_consecutive_fallbacks' must be positive.")
        self.rest_idx = list(self.sim.rest_state_indices)
        max_action = np.asarray(self.sim.max_action, dtype=float).reshape(-1, 1)

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)
        self.x0_param = self.opti.parameter(self.sim.nx)
        self.u_ref_param = self.opti.parameter(self.sim.nu, self.N)

        x_sym, u_sym = ca.SX.sym("x", self.sim.nx), ca.SX.sym("u", self.sim.nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [self.sim.casadi_dynamics(x_sym, u_sym)])
        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        self.opti.subject_to(self.X[:, 1:] == step_fn.map(self.N)(self.X[:, :-1], self.U))
        # vec() makes the comparisons element-wise; the per-dimension limits are repeated across the horizon
        limits = np.repeat(max_action, self.N, axis=1)
        self.opti.subject_to(ca.vec(self.U) >= ca.vec(-limits))
        self.opti.subject_to(ca.vec(self.U) <= ca.vec(limits))
        lower = ca.repmat(ca.DM(np.asarray(self.sim.state_lower_bounds, dtype=float)), 1, self.N)
        upper = ca.repmat(ca.DM(np.asarray(self.sim.state_upper_bounds, dtype=float)), 1, self.N)
        self.opti.subject_to(ca.vec(self.X[:, 1:]) >= ca.vec(lower))
        self.opti.subject_to(ca.vec(self.X[:, 1:]) <= ca.vec(upper))

        tracking = ca.sumsqr((self.U - self.u_ref_param) / ca.repmat(ca.DM(max_action), 1, self.N))
        self.opti.minimize(tracking + self.terminal_velocity_weight * ca.sumsqr(self.X[self.rest_idx, self.N]))

        # Consecutive solves differ little: the multipliers are warm started too (see CasadiPlanner)
        self.opti.solver(
            "ipopt",
            {
                "expand": True,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.sb": "yes",
                "ipopt.warm_start_init_point": "yes",
                "ipopt.warm_start_bound_push": 1e-9,
                "ipopt.warm_start_bound_frac": 1e-9,
                "ipopt.warm_start_slack_bound_push": 1e-9,
                "ipopt.warm_start_slack_bound_frac": 1e-9,
                "ipopt.warm_start_mult_bound_push": 1e-9,
                "ipopt.mu_init": 1e-6,
            },
        )
        self.reset()

    def reset(self) -> None:
        """Signals the start of a new episode: nothing of the last one is reused."""
        self._prev_lam_g: np.ndarray | None = None
        self._prev_X_sol: np.ndarray | None = None
        self._prev_U_sol: np.ndarray | None = None
        self._prev_x0: np.ndarray | None = None
        self._consecutive_fallback_count = 0
        # set_initial(..., lam_g, ...) is sticky on the Opti object
        self.opti.set_initial(self.opti.lam_g, 0.0)

    def next_chunk_guess(self) -> np.ndarray | None:
        """The previous solution's actions shifted by one step, zero actions appended (H, nu): the natural initial
        chunk of the next tick; None right after a reset."""
        if self._prev_U_sol is None:
            return None
        return np.hstack([self._prev_U_sol[:, 1:], np.zeros((self.sim.nu, 1))]).T

    def _hold_still_guess(self, x0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Stay at the current positions at rest: within the limits whatever the proposal is, which a guess following
        a far-off proposal (e.g. the flow's initial noise) is not, and IPOPT often fails from there."""
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
            states[:, step + 1] = self.sim.predict_next_state(states[:, step], u_guess[:, step], validate=False)
        tolerance = 1e-6
        within_limits = np.all(states[:, 1:] >= np.asarray(self.sim.state_lower_bounds)[:, None] - tolerance) and np.all(
            states[:, 1:] <= np.asarray(self.sim.state_upper_bounds)[:, None] + tolerance
        )
        at_rest = np.all(np.abs(states[self.rest_idx, self.N]) <= self.fallback_terminal_velocity_tol)
        return states if within_limits and at_rest else None

    def project(self, obs: np.ndarray, u_ref: np.ndarray) -> np.ndarray:
        """The projected actions (H, nu) of the proposed chunk ``u_ref`` (H, nu); see the class docstring."""
        u_ref = np.asarray(u_ref, dtype=float)
        if u_ref.shape != (self.N, self.sim.nu):
            raise ValueError(f"u_ref must have shape {(self.N, self.sim.nu)}, got {u_ref.shape}.")
        x0 = self.sim.invert_obs(obs)
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

        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.u_ref_param, u_ref.T)
        self.opti.set_initial(self.X, state_guess)
        self.opti.set_initial(self.U, u_guess)
        if self._prev_lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self._prev_lam_g)

        try:
            try:
                sol = self.opti.solve()
            except RuntimeError:
                # A warm start can leave IPOPT stuck at a locally infeasible point once the state has drifted from the
                # prediction: a cold restart resolves most of those
                cold_states, cold_actions = self._hold_still_guess(x0)
                self.opti.set_initial(self.X, cold_states)
                self.opti.set_initial(self.U, cold_actions)
                self.opti.set_initial(self.opti.lam_g, 0.0)
                sol = self.opti.solve()
            self._prev_lam_g = sol.value(self.opti.lam_g)
            self._prev_X_sol = sol.value(self.X)
            self._prev_U_sol = sol.value(self.U)
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
