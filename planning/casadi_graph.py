from typing import NamedTuple

import casadi as ca
import numpy as np

from systems.triple_integrator import TripleIntegratorBase

# structure_detection "auto" needs the stage-wise layout of ``CasadiKinematicGraph``; the mapped dynamics are expanded to SX.
# FATROP has no warm start option: the initial barrier parameter mu_init is what makes it start close to the solution it is
# given (the previous one, shifted) instead of centering the iterates first.
FATROP_OPTIONS: dict = {
    "structure_detection": "auto",
    "expand": True,
    "print_time": False,
    "fatrop": {
        "print_level": 0,
        "tol": 1e-6,
        "mu_init": 1e-6,
    },
}
# The slack of the obstacle half-spaces costs w_1 s + w_2 s^2 (the linear term makes the penalty exact: a half-space is only
# violated when keeping it costs more than w_1 per meter) and a solution is rejected when a slack exceeds the tolerance.
# SafeFlowMPC's filter uses w_1 = 1000 with a tracking cost of unit scale; the MPC's goal cost (Q_diag, terminal multiplier)
# is far larger and outweighs that, so w_1 is 10 times higher.
SLACK_LINEAR_WEIGHT = 1e4
SLACK_QUADRATIC_WEIGHT = 1.0
SLACK_TOLERANCE = 1e-3  # [m]


class GraphSolution(NamedTuple):
    """The states (nx, N + 1) and actions (nu, N) of a solve, and the largest obstacle slack [m] it uses."""

    states: np.ndarray
    actions: np.ndarray
    max_slack: float = 0.0


class CasadiKinematicGraph:
    """
    The CasADi problem skeleton shared by ``CasadiPlanner`` and ``CasadiTrajectoryProjector``, which each hold one.

    It builds everything of their quadratic programs (QP) but the objective: its owner adds its parameters and its
    cost to ``opti`` and ends with ``minimize``.

    Problem (Direct Multiple Shooting, Horizon N)
    ---------------------------------------------
    State (nx = 3n)          x_k = [q_k, v_k, a_k]: positions, velocities, and accelerations.
    Action (nu = n)          u_k = j_k: jerks.
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N): one per sphere and node, shared by
                             its half-spaces.
    Parameters               x_init, and linearized free-space half-spaces (A_coll, b_coll), P per node.

    Cost                     none here: ``minimize(objective)`` adds w_1 sum(S) + w_2 ||S||_F^2 (w_1 = 10^4, w_2 = 1)
                             to its owner's.

    Constraints              x_0 = x_init
                             x_{k+1} = f(x_k, u_k)                       triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max                      action limits, k = 0 .. N-1
                             x_lower <= x_k <= x_upper                   the finite state limits, k = 1 .. N
                             A_coll,k q_k + s_k >= b_coll,k,  s_k >= 0   the P linearized free-space half-spaces of
                                                                         the bounding spheres (``SphereCollisionModel``),
                                                                         each with the slack of its sphere, k = 1 .. N

    Solver Choice (``Opti`` + ``fatrop``)
    -------------------------------------
    The problem is an optimal control problem (OCP): each stage k couples only x_k, u_k, s_k and x_{k+1}. **FATROP**
    (an interior point method that factorizes the KKT system stage by stage with a Riccati recursion, so its time is
    linear in N) exploits that, provided the variables and constraints are laid out per stage: the variables of
    stage k are [x_k, u_k, s_k] and its constraints are the dynamics x_k -> x_{k+1} first, then its path constraints
    (the initial state, the limits, the half-spaces). CasADi detects the structure from that order
    (``structure_detection: auto``).
    The slack is part of the stage's variables, so a slack per half-space would widen every stage with the set size and
    slow FATROP down (about twice as slow on the table scene); one slack per sphere, shared by its half-spaces, keeps
    the stage at one slack per sphere (the penalty is then on the deepest violation of a sphere's set instead of the sum
    over its half-spaces). A free-space set holds at most ``MAX_SET_SIZE`` half-spaces per bounding sphere however many
    obstacles there are, so the stage does not grow with the obstacles either. The first-order Taylor
    half-spaces ``n^T J_s(q_ref,k) q_k >= b_k`` are affine, so the problem is convex quadratic; FATROP has no feasibility
    restoration, which the soft slack constraint makes unnecessary. The slack is how SafeFlowMPC's filter softens its
    obstacle constraints too (see ``SLACK_LINEAR_WEIGHT``); a solution whose slack exceeds the tolerance of the
    owner (``SLACK_TOLERANCE`` unless it asks for another) is rejected, as it is there.

    Initial guess strategy
    ----------------------
    Its owner sets the initial values of ``X`` and ``U`` before a solve (``set_initial``) and linearizes the half-spaces
    along its guess (``update_collision_halfspaces``, after ``set_state``). FATROP has no warm start options and ignores
    the multipliers it is given: a small initial barrier parameter (``mu_init``) makes it start close to the guess instead
    of centering the iterates first, which halves the iterations of a solve started from the previous solution. If the
    solution of ``solve`` still penetrates an obstacle or has a slack above the tolerance, it linearizes the kinematics
    again around it, within the same sets, and solves once more from there; the first solution stays when that solve
    fails. A solution that is still rejected raises ``RuntimeError``.

    ``X`` and ``U`` are the symbolic expressions its owner builds its cost from; everything else of ``opti`` stays here.
    """

    def __init__(self, simulator: TripleIntegratorBase, horizon: int) -> None:
        self.sim = simulator
        self.N = int(horizon)
        self.n_joints = simulator.n
        nx, nu, N, n = self.sim.nx, self.sim.nu, self.N, self.n_joints

        self.opti = ca.Opti()
        self.x0_param = self.opti.parameter(nx)

        num_spheres, set_size = self.sim.collision.num_spheres, self.sim.collision.free_space.size
        self.num_halfspaces = P = num_spheres * set_size
        self._q_current: np.ndarray | None = None
        self._sets: tuple[np.ndarray, np.ndarray] | None = None  # the free-space sets of the last ``update``

        # The variables stage by stage, in the order FATROP reads them: x_k (k = 0 .. N), u_k (k < N), s_k (k >= 1)
        self._x, self._u, self._s = [], [], []
        for k in range(N + 1):
            self._x.append(self.opti.variable(nx))
            if k < N:
                self._u.append(self.opti.variable(nu))
            if P > 0 and k >= 1:
                self._s.append(self.opti.variable(num_spheres))
        self.X = ca.horzcat(*self._x)
        self.U = ca.horzcat(*self._u)
        if P > 0:
            self.slack = ca.vertcat(*self._s)
            self.A_coll = self.opti.parameter(P * N, n)
            self.b_coll = self.opti.parameter(P * N)
            # Each sphere's slack serves its half-spaces (row index = sphere * set_size + i)
            sphere_of_row = ca.DM(np.kron(np.eye(num_spheres), np.ones((set_size, 1))))

        # Dynamics: x_{k+1} = f(x_k, u_k), the step function of the system
        x_sym, u_sym = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [self.sim.casadi_dynamics(x_sym, u_sym)])

        # The infinite state bounds (e.g. the point mass positions) are no constraint
        lower_bounds = np.asarray(self.sim.state_lower_bounds, dtype=float)
        upper_bounds = np.asarray(self.sim.state_upper_bounds, dtype=float)
        finite_lower = np.flatnonzero(np.isfinite(lower_bounds))
        finite_upper = np.flatnonzero(np.isfinite(upper_bounds))
        max_action = np.asarray(self.sim.max_action, dtype=float) * np.ones(nu)

        for k in range(N + 1):
            if k < N:  # the dynamics come before the path constraints of the stage, the order FATROP detects the structure by
                self.opti.subject_to(self._x[k + 1] == step_fn(self._x[k], self._u[k]))
            if k == 0:
                self.opti.subject_to(self._x[0] == self.x0_param)
            else:
                if len(finite_lower):
                    self.opti.subject_to(self._x[k][finite_lower] >= lower_bounds[finite_lower])
                if len(finite_upper):
                    self.opti.subject_to(self._x[k][finite_upper] <= upper_bounds[finite_upper])
                if P > 0:
                    self.opti.subject_to(self._s[k - 1] >= 0.0)
                    rows = slice((k - 1) * P, k * P)
                    self.opti.subject_to(ca.mtimes(self.A_coll[rows, :], self._x[k][:n]) + ca.mtimes(sphere_of_row, self._s[k - 1]) >= self.b_coll[rows])
            if k < N:
                self.opti.subject_to(self._u[k] >= -max_action)
                self.opti.subject_to(self._u[k] <= max_action)

    def minimize(self, objective: ca.MX) -> None:
        """Minimize ``objective`` plus the penalty on the obstacle slack, and set the solver."""
        if self.num_halfspaces > 0:
            objective = objective + SLACK_LINEAR_WEIGHT * ca.sum1(self.slack) + SLACK_QUADRATIC_WEIGHT * ca.sumsqr(self.slack)
        self.opti.minimize(objective)
        self.opti.solver("fatrop", FATROP_OPTIONS)

    def parameter(self, rows: int, cols: int = 1) -> ca.MX:
        """A parameter of its owner's cost, set with ``set_value``."""
        return self.opti.parameter(rows, cols)

    def set_value(self, parameter: ca.MX, value: np.ndarray) -> None:
        self.opti.set_value(parameter, value)

    def set_state(self, x0: np.ndarray) -> None:
        """The state the horizon starts from."""
        self._q_current = np.asarray(x0, dtype=float)[: self.n_joints]
        self.opti.set_value(self.x0_param, x0)

    def set_initial(self, states: np.ndarray, actions: np.ndarray) -> None:
        """The initial values of the variables of the next solve."""
        states = np.broadcast_to(states, (self.sim.nx, self.N + 1))
        actions = np.broadcast_to(actions, (self.sim.nu, self.N))
        for k in range(self.N + 1):
            self.opti.set_initial(self._x[k], states[:, k])
        for k in range(self.N):
            self.opti.set_initial(self._u[k], actions[:, k])

    def reset(self) -> None:
        """Nothing of the last episode is reused: the initial values are sticky on the ``Opti`` object."""
        self.set_initial(0.0, 0.0)
        for stage_slack in self._s:
            self.opti.set_initial(stage_slack, 0.0)

    def update_collision_halfspaces(self, q_ref: np.ndarray) -> None:
        """The free-space sets along ``q_ref`` (N, n_joints), the set of each node being the one around the motion of the
        spheres from the previous node (the state of ``set_state`` for the first), linearized along it."""
        if self.num_halfspaces > 0:
            self._sets = self.sim.collision.sets_along(self._q_current, q_ref)
            self._linearize_sets(q_ref)

    def _linearize_sets(self, q_ref: np.ndarray) -> None:
        A, b = self.sim.collision.linearize(self._sets, q_ref)
        self.opti.set_value(self.A_coll, A.reshape(-1, self.n_joints))
        self.opti.set_value(self.b_coll, b.reshape(-1))

    def solve(self, slack_tolerance: float = SLACK_TOLERANCE) -> GraphSolution:
        """Solve from the initial values set; raises ``RuntimeError`` if the solver fails or the slack of the solution
        exceeds ``slack_tolerance`` (the relinearization of a rejected first solution always judges by
        ``SLACK_TOLERANCE``)."""
        sol = self._refine_if_rejected(self.opti.solve())
        max_slack = self._max_slack(sol)
        if max_slack > slack_tolerance:
            raise RuntimeError(f"The obstacle slack of the solution exceeds {slack_tolerance} m.")
        return GraphSolution(sol.value(self.X), sol.value(self.U), max_slack)

    def _max_slack(self, sol: ca.OptiSol) -> float:
        return float(np.max(sol.value(self.slack))) if self.num_halfspaces > 0 else 0.0

    def _is_rejected(self, sol: ca.OptiSol) -> bool:
        positions = sol.value(self.X)[: self.n_joints]
        penetrates = not all(self.sim.collision.is_collision_free(positions[:, k]) for k in range(1, self.N + 1))
        return penetrates or self._max_slack(sol) > SLACK_TOLERANCE

    def _refine_if_rejected(self, sol: ca.OptiSol) -> ca.OptiSol:
        """The half-spaces are only a linearization: if the solution still penetrates an obstacle or has too much slack,
        linearize the kinematics again around it (within the same sets) and solve once more. The first solution stays
        when that solve fails."""
        if self.num_halfspaces == 0 or not self._is_rejected(sol):
            return sol
        X_first = sol.value(self.X)
        self.set_initial(X_first, sol.value(self.U))
        self._linearize_sets(X_first[: self.n_joints, 1:].T)
        try:
            return self.opti.solve()
        except RuntimeError:
            return sol
