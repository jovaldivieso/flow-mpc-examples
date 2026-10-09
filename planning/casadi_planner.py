from typing import Any, Mapping

import casadi as ca
import numpy as np

from dynamics.franka_dynamics import FrankaDynamics
from .planner import Planner, PlannerSolveError


class CasadiPlanner(Planner):
    """
    CasADi/IPOPT receding-horizon (MPC) planner for the Franka dynamics.

    At every call the NLP is solved from the current state and only the first
    action is returned. The previous solution is shifted to warm-start primals
    and duals.

    With state x, action u (joint acceleration), horizon N and goal state x_goal:

        min  sum_{k<N} (x[k] - x_goal)^T Q (x[k] - x_goal) + u[k]^T R u[k] + task(x[k])
             + alpha * [ (x[N] - x_goal)^T Q (x[N] - x_goal) + task(x[N]) ]

        s.t. x[0] = x_init
             x[k + 1] = f(x[k], u[k])
             -max_action <= u[k] <= max_action
             -max_torque <= torque(x[k], u[k]) <= max_torque
             state_lower_bounds <= x[k] <= state_upper_bounds     k = 1..N

    where ``task`` is the TCP pose error of the dynamics, alpha is
    ``terminal_cost_multiplier``, and Q and R are diagonal (``Q_diag``, ``R_diag``).
    Decision variables: X (nx x N+1) and U (nu x N). Parameters: x_init, x_goal
    and the task goal.
    """

    def __init__(self, dynamics: FrankaDynamics, config: Mapping[str, Any]):
        self.dyn = dynamics
        self.N = int(config.get("horizon", 20))
        alpha = float(config.get("terminal_cost_multiplier", 10.0))
        nx, nu = dynamics.nx, dynamics.nu

        q_diag = np.asarray(config.get("Q_diag", [10.0] * nx), dtype=float)
        r_diag = np.asarray(config.get("R_diag", [0.1] * nu), dtype=float)
        if q_diag.shape != (nx,) or r_diag.shape != (nu,):
            raise ValueError(f"'Q_diag' must have length {nx} and 'R_diag' length {nu}.")

        self.opti = ca.Opti()
        self.X = self.opti.variable(nx, self.N + 1)
        self.U = self.opti.variable(nu, self.N)
        self.x0_param = self.opti.parameter(nx)
        self.goal_param = self.opti.parameter(nx)
        self.task_goal_param = self.opti.parameter(*dynamics.task_goal.shape)

        x_sym, u_sym = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [dynamics.casadi_dynamics(x_sym, u_sym)])
        torque_fn = ca.Function("torque", [x_sym, u_sym], [dynamics.casadi_torque(x_sym, u_sym)])
        g_sym = ca.SX.sym("g", *dynamics.task_goal.shape)
        task_fn = ca.Function("task", [x_sym, g_sym], [dynamics.casadi_task_cost(x_sym, g_sym)])
        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        self.opti.subject_to(self.X[:, 1:] == step_fn.map(self.N)(self.X[:, :-1], self.U))

        # vec() flattens column-major, so repeat per-joint limits across the horizon
        def per_step(limit: np.ndarray) -> ca.DM:
            return ca.vec(np.broadcast_to(np.asarray(limit, dtype=float).reshape(-1, 1), (nu, self.N)))

        self.opti.subject_to(ca.vec(self.U) >= -per_step(dynamics.max_action))
        self.opti.subject_to(ca.vec(self.U) <= per_step(dynamics.max_action))
        torque = ca.vec(torque_fn.map(self.N)(self.X[:, :-1], self.U))
        self.opti.subject_to(torque >= -per_step(dynamics.max_torque))
        self.opti.subject_to(torque <= per_step(dynamics.max_torque))

        lower = ca.repmat(ca.DM(dynamics.state_lower_bounds), 1, self.N)
        upper = ca.repmat(ca.DM(dynamics.state_upper_bounds), 1, self.N)
        self.opti.subject_to(ca.vec(self.X[:, 1:]) >= ca.vec(lower))
        self.opti.subject_to(ca.vec(self.X[:, 1:]) <= ca.vec(upper))

        error = self.X - ca.repmat(self.goal_param, 1, self.N + 1)
        Q, R = np.diag(q_diag), np.diag(r_diag)

        def state_cost(e: ca.MX) -> ca.MX:
            return ca.sum2(ca.sum1(ca.mtimes(Q, e) * e))

        cost = state_cost(error[:, :-1]) + alpha * state_cost(error[:, -1:])
        cost += ca.sum2(ca.sum1(ca.mtimes(R, self.U) * self.U))
        task_costs = task_fn.map(self.N + 1)(self.X, ca.repmat(self.task_goal_param, 1, self.N + 1))
        cost += ca.sum2(task_costs[:, :-1]) + alpha * task_costs[:, -1]
        self.opti.minimize(cost)

        # The NLP is re-solved every step from a slightly perturbed problem, so
        # duals are warm-started too; small bound-push/mu_init values keep IPOPT
        # from pushing iterates away from the bounds and restarting the barrier high.
        self.opti.solver("ipopt", {
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
        })
        self.reset()

    def reset(self) -> None:
        """Forget the previous solution; call at the start of each episode."""
        self.last_X = None
        self.last_U = None
        self.last_lam_g = None
        self._cold_start()

    def _cold_start(self) -> None:
        self.opti.set_initial(self.X, 0.0)
        self.opti.set_initial(self.U, 0.0)
        # set_initial on lam_g is sticky on the Opti object, so it must be
        # cleared explicitly or the next episode would reuse stale duals.
        self.opti.set_initial(self.opti.lam_g, 0.0)

    def __call__(self, state: np.ndarray) -> np.ndarray:
        self.opti.set_value(self.x0_param, state)
        self.opti.set_value(self.goal_param, self.dyn.goal_state)
        self.opti.set_value(self.task_goal_param, self.dyn.task_goal)

        if self.last_X is not None:
            self.opti.set_initial(self.X, np.hstack([self.last_X[:, 1:], self.last_X[:, -1:]]))
            self.opti.set_initial(self.U, np.hstack([self.last_U[:, 1:], self.last_U[:, -1:]]))
            self.opti.set_initial(self.opti.lam_g, self.last_lam_g)

        try:
            sol = self.opti.solve()
        except RuntimeError:
            # A shifted warm start can leave IPOPT stuck at a locally infeasible point
            self._cold_start()
            try:
                sol = self.opti.solve()
            except RuntimeError as exc:
                raise PlannerSolveError(
                    f"MPC solve failed (warm and cold start) from state {state.tolist()}."
                ) from exc

        self.last_X = sol.value(self.X)
        self.last_U = sol.value(self.U)
        self.last_lam_g = sol.value(self.opti.lam_g)
        return self.last_U[:, 0]
