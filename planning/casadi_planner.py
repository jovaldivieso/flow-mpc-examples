import casadi as ca
import numpy as np

from .casadi_graph import CasadiKinematicGraph
from .initial_guess import InitialGuessGenerator
from .planner_config import PlannerConfig
from .planner import Plan, Planner, PlannerSolveError, goal_configuration
from systems.triple_integrator import TripleIntegratorBase


class CasadiPlanner(Planner):
    """
    CasADi model predictive control (FATROP) in joint space.

    ``plan`` solves the quadratic program (QP) below from the current state, returns the planned joint positions over
    the horizon, and keeps the solution as the warm start of the next call.

    Problem (Direct Multiple Shooting, Horizon N)
    ---------------------------------------------
    State (nx = 3n)          x_k = [q_k, v_k, a_k]: joint positions, velocities, and accelerations.
    Action (nu = n)          u_k = j_k: joint jerks.
    Goal                     x_goal = [q_goal, 0, 0]: goal configuration at rest (from ``FrankaKinematics`` IK).
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N), one per sphere shared by all obstacles.
    Parameters               x_init, x_goal, and linearized obstacle half-spaces (A_coll, b_coll).

    Cost                     min_{X, U, S}  sum_{k=0..N-1} ( ||x_k - x_goal||_Q^2 + ||u_k||_R^2 )
                                            + alpha ||x_N - x_goal||_Q^2 + w_s ||S||_F^2
                             with Q = diag(Q_diag), R = diag(R_diag), alpha = terminal_cost_multiplier, w_s = 10^5.

    Constraints              as in ``CasadiKinematicGraph``: the initial state, the triple-integrator dynamics, the
                             jerk and state limits, and the linearized sphere-obstacle half-spaces.

    Solver Choice (``Opti`` + ``fatrop``)
    -------------------------------------
    The triple-integrator dynamics, box limits, and 1st-order Taylor obstacle half-spaces ``n^T J_s(q_ref,k) q_k >= b_k``
    are all affine in ``(X, U, S)`` and the cost is convex quadratic, so the problem is a convex QP with the stage
    structure of an optimal control problem. **FATROP** (an interior point method with a Riccati recursion over the
    stages) solves it in time linear in the horizon, and in the number of obstacles: the slack is one per sphere, so
    more obstacles only add half-space rows to a stage. See ``CasadiKinematicGraph`` for the layout FATROP needs. If the first pass penetrates a spherical obstacle, one re-linearization pass around the solved
    trajectory refines the half-spaces.

    Initial guess strategy
    ----------------------
    A solve starts from ``initial_plan`` (a joint-position plan), else from the plan of the optional
    ``initial_guess_generator`` (a global planner, e.g. an OMPL path), else from the previous solution shifted by one
    step, else from the current state held still. A plan of the generator ends at the goal configuration, which then
    replaces the one of the inverse kinematics; and if the solve that started from a plan fails, the plan is the
    solution (counted in ``trajectory_fallbacks``).
    """

    def _resolve_r_diag(self, config: PlannerConfig) -> np.ndarray:
        """``r_diag`` (nu entries), else ``r_weight`` for every action dimension."""
        if not config.r_diag:
            return np.full(self.sim.nu, config.r_weight)
        if len(config.r_diag) != self.sim.nu:
            raise ValueError(f"'R_diag' length must equal simulator action dimension nu={self.sim.nu}, got {len(config.r_diag)}.")
        return np.asarray(config.r_diag, dtype=float)

    def __init__(
        self,
        simulator: TripleIntegratorBase,
        config: PlannerConfig,
        initial_guess_generator: InitialGuessGenerator | None = None,
    ):
        # The scalars of the config are used as they are; only what depends on the system's dimensions is checked (and
        # completed) here
        self.terminal_cost_multiplier = config.terminal_cost_multiplier
        q_diag = config.q_diag or [10.0] * simulator.nx
        if len(q_diag) != simulator.nx:
            raise ValueError(
                f"'Q_diag' length must equal simulator state dimension nx={simulator.nx}, got {len(q_diag)}."
            )
        self.sim = simulator
        self._graph = CasadiKinematicGraph(simulator, config.horizon)
        self.N = self._graph.N
        self.n_joints = simulator.n
        self.Q = np.diag(q_diag)
        self.R = np.diag(self._resolve_r_diag(config))
        self.initial_guess_generator = initial_guess_generator

        # The last solution (the warm start of the next solve) and what the current goal needs
        self.last_X_sol: np.ndarray | None = None
        self.last_U_sol: np.ndarray | None = None
        self._goal: np.ndarray | None = None
        self.goal_configuration: np.ndarray | None = None
        self.optimizations = 0  # NLP solves since the planner was built
        self.fallbacks = 0  # solves that failed from a plan, for which the plan was the solution

        # Cost: distance to the goal state and effort over the horizon, and the terminal distance
        self.goal_param = self._graph.parameter(self.sim.nx)
        error = self._graph.X - ca.repmat(self.goal_param, 1, self.N + 1)
        running = ca.sum2(ca.sum1(ca.mtimes(self.Q, error[:, :-1]) * error[:, :-1])) + ca.sum2(
            ca.sum1(ca.mtimes(self.R, self._graph.U) * self._graph.U)
        )
        terminal = ca.mtimes(error[:, -1].T, ca.mtimes(self.Q * self.terminal_cost_multiplier, error[:, -1]))
        self._graph.minimize(running + terminal)

    def reset(self) -> None:
        """Signals the start of a new episode."""
        self.last_X_sol = None
        self.last_U_sol = None
        self._goal = None
        self.goal_configuration = None
        if self.initial_guess_generator is not None:
            self.initial_guess_generator.reset()
        self._graph.reset()

    @property
    def statistics(self) -> dict[str, int]:
        """The NLP solves since the planner was built (``trajectory_optimizations``) and, with an initial guess
        generator, its own counters and the solves that failed from its plan (``trajectory_fallbacks``)."""
        if self.initial_guess_generator is None:
            return {"trajectory_optimizations": self.optimizations}
        return {
            **self.initial_guess_generator.statistics,
            "trajectory_optimizations": self.optimizations,
            "trajectory_fallbacks": self.fallbacks,
        }

    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        """The plan from the state of ``observation``: the planned joint positions (N + 1 rows, row 0 the current
        configuration) and the N planned jerks."""
        x0 = self.sim.invert_obs(observation)
        goal = np.asarray(observation[: self.sim.goal_dim], dtype=float)
        if self._goal is None or not np.allclose(goal, self._goal):
            self._begin_goal(goal)
        if initial_plan is None and self.initial_guess_generator is not None:
            last_positions = None if self.last_X_sol is None else self.last_X_sol[: self.n_joints].T
            initial_plan = self.initial_guess_generator.initial_plan(observation, x0[: self.n_joints], last_positions)
            if initial_plan is not None:
                self.goal_configuration = initial_plan.positions[-1].copy()  # the plan ends at the goal configuration
        if self.goal_configuration is None:
            self.goal_configuration = goal_configuration(self.sim, self._goal, x0[: self.n_joints])
        try:
            self._solve(x0, initial_plan)
        except PlannerSolveError:
            if self.initial_guess_generator is None or initial_plan is None:
                raise
            self.fallbacks += 1
            self.last_X_sol, self.last_U_sol = self._states_through(x0, initial_plan)
        return Plan(positions=self.last_X_sol[: self.n_joints].T.copy(), actions=self.last_U_sol.T.copy())

    def _begin_goal(self, goal: np.ndarray) -> None:
        """A new goal: what was planned for the last one is no use."""
        self._goal = goal.copy()
        self.goal_configuration = None
        self.last_X_sol = None
        self.last_U_sol = None
        if self.initial_guess_generator is not None:
            self.initial_guess_generator.begin_goal()

    def _solve(self, x0: np.ndarray, initial_plan: Plan | None) -> None:
        """Solve the NLP from ``x0`` and keep the solution."""
        self._graph.set_state(x0)
        self._graph.set_value(self.goal_param, np.concatenate([self.goal_configuration, np.zeros(2 * self.n_joints)]))

        if initial_plan is not None:
            states_guess, actions_guess = self._states_through(x0, initial_plan)
        elif self.last_X_sol is not None:
            states_guess = np.hstack([self.last_X_sol[:, 1:], self.last_X_sol[:, -1:]])
            actions_guess = np.hstack([self.last_U_sol[:, 1:], self.last_U_sol[:, -1:]])
        else:
            states_guess, actions_guess = self._states_through(
                x0, Plan(positions=np.tile(x0[: self.n_joints], (2, 1)))
            )
        self._graph.set_initial(states_guess, actions_guess)
        self._graph.update_collision_halfspaces(states_guess[: self.n_joints, 1:].T)

        try:
            sol = self._graph.solve()
        except RuntimeError:
            # A warm start can leave the solver stuck at a locally infeasible point once the actual state has drifted from
            # what the previous solve predicted (execution noise, or during DAgger the actions of a still untrained
            # policy). A cold restart from the current state held still resolves far more of those than giving up.
            cold_states, cold_actions = self._states_through(x0, Plan(positions=np.tile(x0[: self.n_joints], (2, 1))))
            self._graph.set_initial(cold_states, cold_actions)
            self._graph.update_collision_halfspaces(cold_states[: self.n_joints, 1:].T)
            try:
                sol = self._graph.solve()
            except RuntimeError as exc:
                raise PlannerSolveError(
                    "CasADi planner solve failed (both warm-started and cold-restarted). "
                    f"Current state estimate: {x0.tolist()}, goal configuration: {self.goal_configuration.tolist()}."
                ) from exc

        self.last_X_sol = sol.states
        self.last_U_sol = sol.actions
        self.optimizations += 1

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
