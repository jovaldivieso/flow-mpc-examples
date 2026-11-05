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
# violated when keeping it costs more than w_1 per meter) and a solution is rejected when a slack exceeds the tolerance, as
# in SafeFlowMPC's filter (w_1 = 1000 with a tracking cost of unit scale).
SLACK_LINEAR_WEIGHT = 1e3
SLACK_QUADRATIC_WEIGHT = 1.0
SLACK_TOLERANCE = 1e-3  # [m]


class GraphSolution(NamedTuple):
    """The states (nx, N + 1) and actions (nu, N) of a solve, the largest obstacle slack [m] it uses, and whether a bounding
    sphere still penetrates an obstacle at one of its nodes."""

    states: np.ndarray
    actions: np.ndarray
    max_slack: float = 0.0
    penetrates: bool = False


class CasadiKinematicGraph:
    """
    The CasADi problem skeleton of ``CasadiTrajectoryProjector``, which holds one.

    It builds everything of its quadratic program (QP) but the objective: its owner adds its parameters and its
    cost to ``opti`` and ends with ``minimize``. Without obstacles (``with_obstacles=False``) the problem has no
    half-spaces and no slack: the plant's limits are its only constraints.

    Problem (Direct Multiple Shooting, Horizon N)
    ---------------------------------------------
    State (nx = 3n)          x_k = [q_k, v_k, a_k]: positions, velocities, and accelerations.
    Action (nu = n)          u_k = j_k: jerks.
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N): one per sphere and node, shared by
                             its half-spaces.
    Parameters               x_init, and linearized free-space half-spaces (A_coll, b_coll), P per node.

    Cost                     none here: ``minimize(objective)`` adds w_1 sum(S) + w_2 ||S||_F^2 (w_1 = 10^3, w_2 = 1)
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
    ``Opti`` only describes the problem: a solve calls the CasADi function ``Opti.to_function`` makes of it, with the
    values of the parameters and the initial values of the variables as its inputs, and a solve the solver reports as
    failed raises ``RuntimeError``.
    """

    def __init__(self, simulator: TripleIntegratorBase, horizon: int, with_obstacles: bool = True) -> None:
        self.sim = simulator
        self.N = int(horizon)
        self.n_joints = simulator.n
        nx, nu, N, n = self.sim.nx, self.sim.nu, self.N, self.n_joints

        self.opti = ca.Opti()
        self._parameters: list[ca.MX] = []  # the parameters, with their values, in the order of the inputs of the solver
        self._values: list[np.ndarray] = []
        self.x0_param = self.parameter(nx)

        num_spheres = self.sim.collision.num_spheres if with_obstacles else 0
        set_size = self.sim.collision.free_space.size if with_obstacles else 0
        self.num_halfspaces = P = num_spheres * set_size
        self._q_current: np.ndarray | None = None
        self._sets: tuple[np.ndarray, np.ndarray] | None = None  # the free-space sets of the last ``update``

        # The variables stage by stage, in the order FATROP reads them: x_k (k = 0 .. N), u_k (k < N), s_k (k >= 1)
        self._x, self._u, self._s = [], [], []
        self._state_index = np.zeros((N + 1, nx), dtype=int)  # where x_k and u_k are in the vector of all variables
        self._action_index = np.zeros((N, nu), dtype=int)
        self._num_variables = 0
        for k in range(N + 1):
            self._x.append(self.opti.variable(nx))
            self._state_index[k] = self._num_variables + np.arange(nx)
            self._num_variables += nx
            if k < N:
                self._u.append(self.opti.variable(nu))
                self._action_index[k] = self._num_variables + np.arange(nu)
                self._num_variables += nu
            if P > 0 and k >= 1:
                self._s.append(self.opti.variable(num_spheres))
                self._num_variables += num_spheres
        self.X = ca.horzcat(*self._x)
        self.U = ca.horzcat(*self._u)
        self.set_initial(0.0, 0.0)
        if P > 0:
            self.slack = ca.vertcat(*self._s)
            self.A_coll = self.parameter(P * N, n)
            self.b_coll = self.parameter(P * N)
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
        outputs = [self.X, self.U] + ([self.slack] if self.num_halfspaces > 0 else [])
        self._solver = self.opti.to_function("solve", [*self._parameters, self.opti.x], outputs)

    def parameter(self, rows: int, cols: int = 1) -> ca.MX:
        """A parameter of its owner's cost, set with ``set_value``."""
        parameter = self.opti.parameter(rows, cols)
        self._parameters.append(parameter)
        self._values.append(np.zeros((rows, cols)))
        return parameter

    def _slot(self, parameter: ca.MX) -> int:
        return next(slot for slot, known in enumerate(self._parameters) if known is parameter)

    def set_value(self, parameter: ca.MX, value: np.ndarray) -> None:
        """The value of ``parameter`` for the next solves: a matrix of its shape, or a vector for a column."""
        slot = self._slot(parameter)
        shape, array = self._values[slot].shape, np.asarray(value, dtype=float)
        if array.shape != shape and not (shape[1] == 1 and array.shape == (shape[0],)):
            raise ValueError(f"A parameter of shape {shape} cannot take a value of shape {array.shape}.")
        self._values[slot] = array.reshape(shape)

    def value(self, parameter: ca.MX) -> np.ndarray:
        """The value of ``parameter`` the next solve uses (a vector for a column)."""
        value = self._values[self._slot(parameter)]
        return value[:, 0] if value.shape[1] == 1 else value

    def set_state(self, x0: np.ndarray) -> None:
        """The state the horizon starts from."""
        self._q_current = np.asarray(x0, dtype=float)[: self.n_joints]
        self.set_value(self.x0_param, x0)

    def set_initial(self, states: np.ndarray, actions: np.ndarray) -> None:
        """The initial values of the states and actions of the next solves, kept until they are set again (the slack
        starts at zero)."""
        self._initial_states = np.array(np.broadcast_to(states, (self.sim.nx, self.N + 1)), dtype=float)
        self._initial_actions = np.array(np.broadcast_to(actions, (self.sim.nu, self.N)), dtype=float)

    @property
    def initial_guess(self) -> tuple[np.ndarray, np.ndarray]:
        """The initial states (nx, N + 1) and actions (nu, N) of the next solve."""
        return self._initial_states.copy(), self._initial_actions.copy()

    @property
    def iterations(self) -> int:
        """The iterations of the last solve."""
        return int(self._solver.stats()["iter_count"])

    def reset(self) -> None:
        """Nothing of the last episode is reused: the initial values go back to zero."""
        self.set_initial(0.0, 0.0)

    def update_collision_halfspaces(self, q_ref: np.ndarray) -> None:
        """The free-space sets along ``q_ref`` (N, n_joints), the set of each node being the one around the motion of the
        spheres from the previous node (the state of ``set_state`` for the first), linearized along it."""
        if self.num_halfspaces > 0:
            self._sets = self.sim.collision.sets_along(self._q_current, q_ref)
            self._linearize_sets(q_ref)

    def _linearize_sets(self, q_ref: np.ndarray) -> None:
        A, b = self.sim.collision.linearize(self._sets, q_ref)
        self.set_value(self.A_coll, A.reshape(-1, self.n_joints))
        self.set_value(self.b_coll, b.reshape(-1))

    def solve(self, slack_tolerance: float = SLACK_TOLERANCE) -> GraphSolution:
        """Solve from the initial values set; raises ``RuntimeError`` if the solver fails or the slack of the solution
        exceeds ``slack_tolerance`` (the relinearization of a rejected first solution always judges by
        ``SLACK_TOLERANCE``)."""
        states, actions, max_slack, penetrates = self._refine_if_rejected(*self._solve_once())
        if max_slack > slack_tolerance:
            raise RuntimeError(f"The obstacle slack of the solution exceeds {slack_tolerance} m.")
        return GraphSolution(states, actions, max_slack, penetrates)

    def _solve_once(self) -> tuple[np.ndarray, np.ndarray, float]:
        """One solve from the initial values: the states (nx, N + 1), the actions (nu, N) and the largest slack of the
        solution. Raises ``RuntimeError`` if the solver reports a failure."""
        initial = np.zeros(self._num_variables)
        initial[self._state_index] = self._initial_states.T
        initial[self._action_index] = self._initial_actions.T
        outputs = self._solver(*self._values, initial)
        statistics = self._solver.stats()
        if not statistics["success"]:
            raise RuntimeError(f"The solver failed (return status {statistics['return_status']}).")
        max_slack = float(np.max(np.array(outputs[2]))) if self.num_halfspaces > 0 else 0.0
        return np.array(outputs[0]), np.array(outputs[1]), max_slack

    def _penetrates(self, states: np.ndarray) -> bool:
        return not self.sim.collision.trajectory_is_collision_free(states[: self.n_joints, 1:].T)

    def _refine_if_rejected(
        self, states: np.ndarray, actions: np.ndarray, max_slack: float
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """The half-spaces are only a linearization: if the solution still penetrates an obstacle or has too much slack,
        linearize the kinematics again around it (within the same sets) and solve once more. The first solution stays
        when that solve fails. Returns the solution that stays, its largest slack and whether it penetrates an obstacle."""
        penetrates = self._penetrates(states)
        if self.num_halfspaces == 0 or not (penetrates or max_slack > SLACK_TOLERANCE):
            return states, actions, max_slack, penetrates
        self.set_initial(states, actions)
        self._linearize_sets(states[: self.n_joints, 1:].T)
        try:
            refined_states, refined_actions, refined_slack = self._solve_once()
        except RuntimeError:
            return states, actions, max_slack, penetrates
        return refined_states, refined_actions, refined_slack, self._penetrates(refined_states)
