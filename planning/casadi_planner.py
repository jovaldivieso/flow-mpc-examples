import casadi as ca
import numpy as np
from typing import Any, Mapping

from .planner import Plan, Planner, PlannerSolveError
from dynamics.dynamics import DynamicsProtocol


class CasadiPlanner(Planner):
    """
    CasADi model predictive control (IPOPT) of the Franka arm in joint space.

    ``plan`` solves the NLP below from the current state, returns the planned joint positions over the horizon and
    keeps the solution as the warm start of the next call (shifted by one step). The caller executes the first step.

    Problem
    -------
    State (nx = 21)          x = [q, v, a]: joint positions, velocities and accelerations.
    Action (nu = 7)          u: joint jerks.
    Goal                     x_goal = [q_goal, 0, 0]: the goal configuration at rest.
    Optimization variables   X = [x_0 .. x_N] (nx, N + 1) and U = [u_0 .. u_{N-1}] (nu, N), horizon N.
    Parameters               x_init (the current state) and x_goal.

    Cost                     J = sum_{k=0..N-1} (x_k - x_goal)^T Q (x_k - x_goal) + u_k^T R u_k
                                 + alpha (x_N - x_goal)^T Q (x_N - x_goal)
                             with Q = diag(``Q_diag``), R = diag(``R_diag``) and alpha = ``terminal_cost_multiplier``.
    Constraints              x_0 = x_init
                             x_{k+1} = f(x_k, u_k)              triple integrator (``sim.casadi_dynamics``)
                             -j_max <= u_k <= j_max             joint jerk limits
                             x_lower <= x_k <= x_upper, k >= 1  joint position, velocity and acceleration limits

    Collisions are not part of the problem: the global planner (``OmplPlanner``) is the only one that checks them.

    A solve starts from ``initial_plan`` (a joint-position plan, e.g. a path from a global planner), else from the
    previous solution shifted by one step, else from the current state held still.
    """

    def _resolve_r_diag(self, config: Mapping[str, Any]) -> np.ndarray:
        raw_r_diag = config.get("R_diag")

        if raw_r_diag is not None:
            if not isinstance(raw_r_diag, list):
                raise ValueError("'R_diag' must be a list.")
            if len(raw_r_diag) != self.sim.nu:
                raise ValueError(
                    f"'R_diag' length must equal simulator action dimension nu={self.sim.nu}, got {len(raw_r_diag)}."
                )
            diag_values = []
            for idx, value in enumerate(raw_r_diag):
                if not isinstance(value, (int, float)):
                    raise ValueError(f"'R_diag[{idx}]' must be numeric.")
                value_f = float(value)
                if value_f <= 0:
                    raise ValueError(f"'R_diag[{idx}]' must be positive.")
                diag_values.append(value_f)
            return np.asarray(diag_values, dtype=float)

        r_weight = config.get("R_weight", 0.1)
        if not isinstance(r_weight, (int, float)):
            raise ValueError("'R_weight' must be numeric.")
        if float(r_weight) <= 0:
            raise ValueError("'R_weight' must be positive.")
        return np.asarray([float(r_weight)] * self.sim.nu, dtype=float)

    def __init__(self, simulator: DynamicsProtocol, config: Mapping[str, Any]):
        self.sim = simulator
        self.N = config.get("horizon", 20)
        self.n_joints = simulator.n
        self.terminal_cost_multiplier = float(config.get("terminal_cost_multiplier", 10.0))

        if self.N <= 0:
            raise ValueError("'horizon' must be a positive integer.")
        if self.terminal_cost_multiplier <= 0:
            raise ValueError("'terminal_cost_multiplier' must be positive.")

        q_diag = config.get("Q_diag", [10.0] * self.sim.nx)
        if len(q_diag) != self.sim.nx:
            raise ValueError(
                f"'Q_diag' length must equal simulator state dimension nx={self.sim.nx}, got {len(q_diag)}."
            )
        self.Q = np.diag(q_diag)
        self.R = np.diag(self._resolve_r_diag(config))

        # The last solution (the warm start of the next solve) and what the current goal needs
        self.last_X_sol: np.ndarray | None = None
        self.last_U_sol: np.ndarray | None = None
        self._prev_lam_g: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self.goal_configuration: np.ndarray | None = None
        self.optimizations = 0  # NLP solves since the planner was built

        self.opti = ca.Opti()
        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)
        self.x0_param = self.opti.parameter(self.sim.nx)
        self.goal_param = self.opti.parameter(self.sim.nx)

        # Dynamics: X[:, 1:] = f(X[:, :-1], U), the step function mapped over the horizon
        x_sym = ca.SX.sym("x", self.sim.nx)
        u_sym = ca.SX.sym("u", self.sim.nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [self.sim.casadi_dynamics(x_sym, u_sym)])
        self.opti.subject_to(self.X[:, 1:] == step_fn.map(self.N)(self.X[:, :-1], self.U))
        self.opti.subject_to(self.X[:, 0] == self.x0_param)

        # Joint jerk limits. vec() forces element-wise inequality (CasADi reads a non-square matrix comparison
        # as a PSD constraint); the per-joint limits are repeated across the horizon to match its column-major order.
        max_action = np.broadcast_to(np.asarray(self.sim.max_action, dtype=float).reshape(-1, 1), (self.sim.nu, self.N))
        self.opti.subject_to(ca.vec(self.U) >= ca.vec(-max_action))
        self.opti.subject_to(ca.vec(self.U) <= ca.vec(max_action))

        # State limits at the nodes 1 .. N (node 0 is the current state)
        lower = ca.repmat(ca.DM(np.asarray(self.sim.state_lower_bounds, dtype=float)), 1, self.N)
        upper = ca.repmat(ca.DM(np.asarray(self.sim.state_upper_bounds, dtype=float)), 1, self.N)
        self.opti.subject_to(ca.vec(self.X[:, 1:]) >= ca.vec(lower))
        self.opti.subject_to(ca.vec(self.X[:, 1:]) <= ca.vec(upper))

        # Cost: distance to the goal state and effort over the horizon, and the terminal distance
        error = self.X - ca.repmat(self.goal_param, 1, self.N + 1)
        running = ca.sum2(ca.sum1(ca.mtimes(self.Q, error[:, :-1]) * error[:, :-1])) + ca.sum2(
            ca.sum1(ca.mtimes(self.R, self.U) * self.U)
        )
        terminal = ca.mtimes(error[:, -1].T, ca.mtimes(self.Q * self.terminal_cost_multiplier, error[:, -1]))
        self.opti.minimize(running + terminal)

        # The same problem is re-solved at every step with slightly perturbed parameters, so the multipliers are warm
        # started too: with the warm_start_init_point option alone IPOPT still pushes the iterates away from the bounds
        # and restarts the barrier parameter high, so the push/frac options and a small mu_init are part of the recipe.
        # The mapped dynamics expand to SX, which makes every solve faster.
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

    def reset(self) -> None:
        """Signals the start of a new episode."""
        self.last_X_sol = None
        self.last_U_sol = None
        self._prev_lam_g = None
        self._goal = None
        self.goal_configuration = None
        self.opti.set_initial(self.X, 0.0)
        self.opti.set_initial(self.U, 0.0)
        # set_initial(..., lam_g, ...) is sticky on the Opti object: without this the first solve of the new episode
        # would warm start from the final multipliers of the previous, unrelated one.
        self.opti.set_initial(self.opti.lam_g, 0.0)

    @property
    def statistics(self) -> dict[str, int]:
        return {"trajectory_optimizations": self.optimizations}

    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        """The plan from the state of ``observation``: the planned joint positions (N + 1 rows, row 0 the current
        configuration) and the N planned jerks."""
        x0 = self.sim.invert_obs(observation)
        goal = np.asarray(observation[: self.sim.goal_dim], dtype=float)
        if self._goal is None or not np.allclose(goal, self._goal):
            self._begin_goal(goal)
        if initial_plan is None:
            initial_plan = self._initial_plan(observation, x0)
        if self.goal_configuration is None:
            self.goal_configuration = self._goal_configuration(x0)
        self._solve(x0, initial_plan)
        return Plan(positions=self.last_X_sol[: self.n_joints].T.copy(), actions=self.last_U_sol.T.copy())

    def _begin_goal(self, goal: np.ndarray) -> None:
        """A new goal: what was planned for the last one is no use."""
        self._goal = goal.copy()
        self.goal_configuration = None
        self.last_X_sol = None
        self.last_U_sol = None

    def _initial_plan(self, observation: np.ndarray, x0: np.ndarray) -> Plan | None:
        """A plan the next solve starts from; none means the previous solution, or the state held still. A subclass can
        supply one of its own."""
        return None

    def _goal_configuration(self, x0: np.ndarray) -> np.ndarray:
        """The goal configuration."""
        return self._goal.copy()

    def _solve(self, x0: np.ndarray, initial_plan: Plan | None) -> None:
        """Solve the NLP from ``x0`` and keep the solution."""
        self.opti.set_value(self.x0_param, x0)
        self.opti.set_value(self.goal_param, np.concatenate([self.goal_configuration, np.zeros(2 * self.n_joints)]))

        if initial_plan is not None:
            self._set_initial(*self._states_through(x0, initial_plan))
        elif self.last_X_sol is not None:
            self._set_initial(
                np.hstack([self.last_X_sol[:, 1:], self.last_X_sol[:, -1:]]),
                np.hstack([self.last_U_sol[:, 1:], self.last_U_sol[:, -1:]]),
            )
        else:
            self._set_initial(*self._states_through(x0, Plan(positions=np.tile(x0[: self.n_joints], (2, 1)))))
        if self._prev_lam_g is not None:
            self.opti.set_initial(self.opti.lam_g, self._prev_lam_g)

        try:
            sol = self.opti.solve()
        except RuntimeError:
            # A warm start can leave IPOPT stuck at a locally infeasible point once the actual state has drifted from
            # what the previous solve predicted (execution noise, or during DAgger the actions of a still untrained
            # policy). A cold restart from the current state held still resolves far more of those than giving up.
            self._set_initial(*self._states_through(x0, Plan(positions=np.tile(x0[: self.n_joints], (2, 1)))))
            self.opti.set_initial(self.opti.lam_g, 0.0)
            try:
                sol = self.opti.solve()
            except RuntimeError as exc:
                raise PlannerSolveError(
                    "CasADi planner solve failed (both warm-started and cold-restarted). "
                    f"Current state estimate: {x0.tolist()}, goal configuration: {self.goal_configuration.tolist()}."
                ) from exc

        self.last_X_sol = sol.value(self.X)
        self.last_U_sol = sol.value(self.U)
        self._prev_lam_g = sol.value(self.opti.lam_g)
        self.optimizations += 1

    def _set_initial(self, states: np.ndarray, actions: np.ndarray) -> None:
        self.opti.set_initial(self.X, states)
        self.opti.set_initial(self.U, actions)

    def _states_through(self, x0: np.ndarray, plan: Plan) -> tuple[np.ndarray, np.ndarray]:
        """The states (nx, N + 1) and actions (nu, N) of the plant moving through the configurations of ``plan`` (its
        rows 1 .. N, held at the last one if it is shorter) from the state ``x0``."""
        positions = np.asarray(plan.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != self.n_joints or len(positions) < 2:
            raise ValueError(
                f"The initial plan needs at least 2 rows of {self.n_joints} joint positions, got {positions.shape}."
            )
        ahead = positions[1 : self.N + 1]
        q = np.vstack([x0[: self.n_joints], ahead, np.repeat(ahead[-1:], self.N - len(ahead), axis=0)]).T
        n, dt = self.n_joints, self.sim.dt
        v = np.hstack([x0[n : 2 * n, None], np.diff(q, axis=1) / dt])
        a = np.hstack([x0[2 * n :, None], np.diff(v, axis=1) / dt])
        return np.vstack([q, v, a]), np.diff(a, axis=1) / dt
