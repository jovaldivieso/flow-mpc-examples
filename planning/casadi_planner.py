import casadi as ca
import numpy as np
from typing import Any, Mapping

from dynamics.dynamics import DynamicsProtocol
from .planner import Planner, PlannerSolveError


class CasadiPlanner(Planner):
    """
    CasADi/IPOPT finite-horizon optimal control for a single robot.

    Modes:
        - "mpc" (Receding Horizon): Solves the NLP at every step and executes
          only the first action. Warm-started from the previous solution.
        - "open_loop" (Trajectory Optimization): Solves the NLP once at the
          start of the episode, caches the trajectory, and executes it blindly.

    Mathematical Formulation
    ------------------------
    With state x, action u, goal x_goal and horizon N, the optimizer solves:

        min  sum_{k=0..N-1} (x[k] - x_goal)^T Q (x[k] - x_goal) + u[k]^T R u[k]
             + (x[N] - x_goal)^T (alpha * Q) (x[N] - x_goal)

        s.t. x[0] = x_init
             x[k + 1] = f(x[k], u[k])         k = 0..N-1
             -max_action <= u[k] <= max_action
             -max_torque <= casadi_torque(x[k], u[k]) <= max_torque   (if the dynamics defines it)
             state_lower_bounds <= x[k] <= state_upper_bounds   k = 1..N

    where alpha is ``terminal_cost_multiplier``. Angular state components
    (``angular_state_indices``) are penalized as q * (1 - cos(error)) so that
    errors wrap correctly.

    Decision variables: X (nx x N+1) and U (nu x N).
    Parameters: x_init and x_goal.

    Control penalty configuration
    -----------------------------
    ``R`` is diagonal and set by ``R_diag`` (list of length nu) or, as a
    fallback, by the positive scalar ``R_weight``.
    """

    def __init__(self, dynamics: DynamicsProtocol, config: Mapping[str, Any]):
        self.dyn = dynamics
        self.N = config.get("horizon", 20)
        self.mode = config.get("mode", "mpc")
        self.terminal_cost_multiplier = float(config.get("terminal_cost_multiplier", 10.0))

        if self.N <= 0:
            raise ValueError("'horizon' must be a positive integer.")
        if self.mode not in {"mpc", "open_loop"}:
            raise ValueError("'mode' must be one of {'mpc', 'open_loop'}.")
        if self.terminal_cost_multiplier <= 0:
            raise ValueError("'terminal_cost_multiplier' must be positive.")

        nx, nu = self.dyn.nx, self.dyn.nu

        # State variables for open-loop planning and MPC warm starts
        self.cached_plan = None
        self.step_idx = 0
        self.last_X_sol = None
        self.last_U_sol = None
        self._prev_lam_g: np.ndarray | None = None

        q_diag = np.asarray(config.get("Q_diag", [10.0] * nx), dtype=float)
        if q_diag.shape != (nx,):
            raise ValueError(f"'Q_diag' length must equal state dimension nx={nx}, got {q_diag.size}.")
        r_diag = np.asarray(config.get("R_diag", [config.get("R_weight", 0.1)] * nu), dtype=float)
        if r_diag.shape != (nu,):
            raise ValueError(f"'R_diag' length must equal action dimension nu={nu}, got {r_diag.size}.")
        if np.any(q_diag < 0) or np.any(r_diag <= 0):
            raise ValueError("'Q_diag' must be non-negative and 'R_diag'/'R_weight' positive.")

        angular_idx = tuple(getattr(self.dyn, "angular_state_indices", ()))
        if any(i < 0 or i >= nx for i in angular_idx):
            raise ValueError(f"Invalid angular_state_indices {angular_idx} for state dimension {nx}.")

        self.opti = ca.Opti()
        self.X = self.opti.variable(nx, self.N + 1)
        self.U = self.opti.variable(nu, self.N)
        self.x0_param = self.opti.parameter(nx)
        self.goal_param = self.opti.parameter(nx)
        # Optional task-space goal (e.g. end-effector pose) handled by the dynamics
        self.task_cost = getattr(self.dyn, "casadi_task_cost", None)
        if self.task_cost is not None:
            self.task_goal_param = self.opti.parameter(*self.dyn.task_goal.shape)

        # Dynamics over the whole horizon, vectorized with map().
        x_sym = ca.SX.sym("x", nx)
        u_sym = ca.SX.sym("u", nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [self.dyn.casadi_dynamics(x_sym, u_sym)])
        self.opti.subject_to(self.X[:, 1:] == step_fn.map(self.N)(self.X[:, :-1], self.U))
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # max_action may be per-action; repeat it across the horizon explicitly,
        # since vec() flattens column-major.
        max_action = np.broadcast_to(
            np.asarray(self.dyn.max_action, dtype=float).reshape(-1, 1), (nu, self.N)
        )
        self.opti.subject_to(ca.vec(self.U) >= ca.vec(-max_action))
        self.opti.subject_to(ca.vec(self.U) <= ca.vec(max_action))

        # Optional torque limit when actions are accelerations (inverse dynamics)
        if hasattr(self.dyn, "casadi_torque"):
            torque_fn = ca.Function("torque", [x_sym, u_sym], [self.dyn.casadi_torque(x_sym, u_sym)])
            torque = ca.vec(torque_fn.map(self.N)(self.X[:, :-1], self.U))
            max_torque = ca.vec(np.broadcast_to(np.asarray(self.dyn.max_torque).reshape(-1, 1), (nu, self.N)))
            self.opti.subject_to(torque >= -max_torque)
            self.opti.subject_to(torque <= max_torque)

        # Optional state bounds over steps 1..N; only finite entries become constraints.
        for bounds, sign in ((getattr(self.dyn, "state_lower_bounds", None), 1),
                             (getattr(self.dyn, "state_upper_bounds", None), -1)):
            if bounds is None:
                continue
            bounds = np.asarray(bounds, dtype=float)
            idx = np.flatnonzero(np.isfinite(bounds)).tolist()
            if idx:
                values = ca.repmat(ca.DM(bounds[idx]), 1, self.N)
                # vec() forces element-wise inequality on non-square matrices.
                self.opti.subject_to(sign * ca.vec(self.X[idx, 1:]) >= sign * ca.vec(values))

        # Costs: angular components use 1 - cos(error), the rest are quadratic.
        q_quad = q_diag.copy()
        q_quad[list(angular_idx)] = 0.0
        Q_quad = np.diag(q_quad)
        error = self.X - ca.repmat(self.goal_param, 1, self.N + 1)

        def stage_cost(e: ca.MX, q_weight: float) -> ca.MX:
            cost = q_weight * ca.sum2(ca.sum1(ca.mtimes(Q_quad, e) * e))
            for i in angular_idx:
                cost += q_weight * q_diag[i] * ca.sum2(1.0 - ca.cos(e[i, :]))
            return cost

        cost = stage_cost(error[:, :-1], 1.0)
        cost += stage_cost(error[:, -1:], self.terminal_cost_multiplier)
        if self.task_cost is not None:
            for k in range(self.N):
                cost += self.task_cost(self.X[:, k], self.task_goal_param)
            cost += self.terminal_cost_multiplier * self.task_cost(self.X[:, -1], self.task_goal_param)
        cost += ca.sum2(ca.sum1(ca.mtimes(np.diag(r_diag), self.U) * self.U))
        self.opti.minimize(cost)

        # In "mpc" mode this Opti problem is re-solved every timestep with
        # slightly perturbed parameters, so the duals are warm-started too.
        # That needs small bound-push/frac values and mu_init, otherwise IPOPT
        # pushes iterates away from bounds and restarts the barrier parameter
        # high anyway.
        opts = {
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
        }
        self.opti.solver("ipopt", opts)

    def reset(self) -> None:
        """Signals the start of a new episode."""
        self.last_X_sol = None
        self.last_U_sol = None
        self._prev_lam_g = None
        self.opti.set_initial(self.X, 0.0)
        self.opti.set_initial(self.U, 0.0)
        # set_initial on lam_g is sticky on the Opti object; clearing the
        # Python-side cache is not enough, or the first solve of the new
        # episode would warm-start from the previous episode's duals.
        self.opti.set_initial(self.opti.lam_g, 0.0)
        if self.mode == "open_loop":
            self.cached_plan = None
            self.step_idx = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if self.mode == "mpc":
            x0 = self.dyn.invert_obs(obs)
            self.opti.set_value(self.x0_param, x0)
            self.opti.set_value(self.goal_param, self.dyn.goal_state)
            if self.task_cost is not None:
                self.opti.set_value(self.task_goal_param, self.dyn.task_goal)

            if self.last_X_sol is not None and self.last_U_sol is not None:
                shifted_X = np.hstack([self.last_X_sol[:, 1:], self.last_X_sol[:, -1:]])
                shifted_U = np.hstack([self.last_U_sol[:, 1:], self.last_U_sol[:, -1:]])
                self.opti.set_initial(self.X, shifted_X)
                self.opti.set_initial(self.U, shifted_U)
            if self._prev_lam_g is not None:
                self.opti.set_initial(self.opti.lam_g, self._prev_lam_g)

            try:
                sol = self.opti.solve()
            except RuntimeError:
                # The shifted warm start can leave IPOPT stuck at a locally
                # infeasible point; a cold restart recovers most such states.
                self.opti.set_initial(self.X, 0.0)
                self.opti.set_initial(self.U, 0.0)
                self.opti.set_initial(self.opti.lam_g, 0.0)
                try:
                    sol = self.opti.solve()
                except RuntimeError as exc:
                    raise PlannerSolveError(
                        "CasADi planner solve failed in MPC mode (both warm-started and "
                        f"cold-restarted). Current state estimate: {x0.tolist()}, goal: "
                        f"{self.dyn.goal_state.tolist()}."
                    ) from exc

            self.last_X_sol = sol.value(self.X)
            self.last_U_sol = sol.value(self.U)
            self._prev_lam_g = sol.value(self.opti.lam_g)
            return self.last_U_sol[:, 0]

        elif self.mode == "open_loop":
            # Plan once on the first step
            if self.cached_plan is None:
                x0 = self.dyn.invert_obs(obs)
                self.opti.set_value(self.x0_param, x0)
                self.opti.set_value(self.goal_param, self.dyn.goal_state)
                if self.task_cost is not None:
                    self.opti.set_value(self.task_goal_param, self.dyn.task_goal)

                try:
                    sol = self.opti.solve()
                    self.cached_plan = sol.value(self.U)
                except RuntimeError as exc:
                    raise PlannerSolveError(
                        "CasADi planner solve failed in open-loop mode. "
                        f"Initial state estimate: {x0.tolist()}, goal: {self.dyn.goal_state.tolist()}."
                    ) from exc

            if self.step_idx < self.cached_plan.shape[1]:
                action = self.cached_plan[:, self.step_idx]
                self.step_idx += 1
                return action
            else:
                return np.zeros(self.dyn.nu)  # Ran out of planned steps

        else:
            raise ValueError(f"Unknown CasadiPlanner mode: {self.mode}")
