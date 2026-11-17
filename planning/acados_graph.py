import contextlib
import ctypes
import fcntl
import hashlib
import io
import os
import sys
from pathlib import Path
from typing import NamedTuple

import casadi as ca
import numpy as np

from systems.triple_integrator import TripleIntegratorBase

# acados is the submodule third_party/acados, built in place. Its libraries refer to each other through the run-time
# search path, which nothing sets for them: loading them here first (dependencies before dependents) lets the solvers
# acados builds find them without DYLD_LIBRARY_PATH / LD_LIBRARY_PATH.
ACADOS_DIRECTORY = Path(__file__).resolve().parents[1] / "third_party" / "acados"
os.environ.setdefault("ACADOS_SOURCE_DIR", str(ACADOS_DIRECTORY))
for _library in ("blasfeo", "hpipm", "qpOASES_e", "acados"):
    _path = Path(os.environ["ACADOS_SOURCE_DIR"]) / "lib" / f"lib{_library}{'.dylib' if sys.platform == 'darwin' else '.so'}"
    if _path.exists():
        ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL)

from acados_template import ACADOS_INFTY, AcadosModel, AcadosOcp, AcadosOcpSolver  # noqa: E402
from acados_template.utils import get_acados_path  # noqa: E402

# The slack of the obstacle half-spaces costs w_1 s + w_2 s^2 (the linear term makes the penalty exact: a half-space is only
# violated when keeping it costs more than w_1 per meter) and a solution is rejected when a slack exceeds the tolerance, as
# in SafeFlowMPC's filter (w_1 = 1000 with a tracking cost of unit scale).
SLACK_LINEAR_WEIGHT = 1e3
SLACK_QUADRATIC_WEIGHT = 1.0
SLACK_TOLERANCE = 1e-3  # [m]
QP_ITERATION_LIMIT = 50  # HPIPM iterations of a solve; a QP that needs more is a failed solve
# acados' QP statuses of a usable solve: converged (ACADOS_SUCCESS, 0), or at the iteration limit (ACADOS_MAXITER, 2),
# where HPIPM's speed mode stalls on a few QPs close to the solution; the plan is checked against the limits and the
# half-spaces either way
ACCEPTED_QP_STATUS = (0, 2)
LIMIT_TOLERANCE = 1e-6  # of a state limit by the plan rolled out from the solved actions
SOLVER_OPTIONS: dict = {
    "integrator_type": "DISCRETE",
    "nlp_solver_type": "SQP_RTI",
    "hessian_approx": "GAUSS_NEWTON",
    "qp_solver": "PARTIAL_CONDENSING_HPIPM",
    "hpipm_mode": "SPEED",
    "qp_solver_warm_start": 2,
    "qp_solver_iter_max": QP_ITERATION_LIMIT,
    "print_level": 0,
}
CACHE_DIRECTORY = Path(__file__).resolve().parents[1] / ".acados_cache"


class GraphSolution(NamedTuple):
    """The states (nx, N + 1) and actions (nu, N) of a solve, the largest obstacle slack [m] it uses, and whether a bounding
    sphere still penetrates an obstacle at one of its nodes."""

    states: np.ndarray
    actions: np.ndarray
    max_slack: float = 0.0
    penetrates: bool = False


class TrackingCost(NamedTuple):
    """The weights of the tracking cost (the positions weigh 1) and of the rest at the end of the horizon (0: no rest)."""

    velocity_weight: float
    acceleration_weight: float
    jerk_weight: float
    terminal_weight: float


class AcadosKinematicGraph:
    """
    The projection problem of ``TrajectoryProjector``, which holds one: the tracking of a reference chunk of
    joint positions within the plant's limits and, with obstacles, clear of them.

    Without obstacles (``with_obstacles=False``) the problem has no half-spaces and no slack: the plant's limits are its
    only constraints.

    Problem (Direct Multiple Shooting, Horizon N)
    ---------------------------------------------
    State (nx = 3n)          x_k = [q_k, v_k, a_k]: positions, velocities, and accelerations.
    Action (nu = n)          u_k = j_k: jerks.
    Decision variables       X = [x_0 .. x_N] (nx, N + 1), U = [u_0 .. u_{N-1}] (nu, N), and (when obstacles are
                             present) slack S = [s_1 .. s_N] (n_spheres, N): one per sphere and node, shared by
                             its half-spaces.
    Parameters               x_init, the reference q_ref (N, n), and the linearized free-space half-spaces
                             (A_coll,k, b_coll,k), P per node.

    Cost                     sum_{k=1..N} ( ||q_k - q_ref,k||^2 + w_v ||v_k||^2 + w_a ||a_k||^2 ) + w_u sum_{k=0..N-1} ||u_k||^2
                             + beta ||x_N[rest]||^2 + w_1 sum(S) + w_2 ||S||_F^2
                             with the weights of ``TrackingCost`` (beta = its terminal weight, on the velocities and
                             accelerations of node N) and w_1 = 10^3, w_2 = 1.

    Constraints              x_0 = x_init
                             x_{k+1} = A_d x_k + B_d u_k + c             triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max                      action limits, k = 0 .. N-1
                             x_lower <= x_k <= x_upper                   the finite state limits, k = 1 .. N
                             A_coll,k q_k + s_k >= b_coll,k,  s_k >= 0   the P linearized free-space half-spaces of
                                                                         the bounding spheres (``SphereCollisionModel``),
                                                                         each with the slack of its sphere, k = 1 .. N

    Solver Choice (acados, SQP-RTI with HPIPM)
    ------------------------------------------
    The dynamics, the limits and the first-order Taylor half-spaces ``n^T J_s(q_ref,k) q_k >= b_k`` are affine and the
    cost is a positive definite quadratic, so the problem is a convex QP with the stage structure of an optimal control
    problem. acados solves it with one real-time iteration (one QP, which is the solution of a QP) by **HPIPM**, an
    interior point method that factorizes the KKT system stage by stage (a Riccati recursion, linear in N), in its speed
    mode. The half-spaces are acados' linear constraints ``C x_k >= lg``, whose rows and bounds are set before every
    solve; the slack of a sphere is shared by its half-spaces (acados' ``idxs_rev``) with the cost w_1 s + w_2 s^2, as in
    SafeFlowMPC's filter. A free-space set holds at most ``MAX_SET_SIZE`` half-spaces per bounding sphere however many
    obstacles there are, so the problem does not grow with the obstacles. A solution whose slack exceeds the tolerance of
    the owner (``SLACK_TOLERANCE`` unless it asks for another) is rejected, as it is there. acados reports a QP at its
    iteration limit as a success; the graph reads the QP status itself (``ACCEPTED_QP_STATUS``) and checks the plan it
    returns against the state limits, and its slack is the deepest violation of a half-space by that plan.

    acados generates and compiles C code for the problem; that code depends only on the dimensions, the dynamics and
    the solver options (the weights, the limits and everything set per solve are set at run time). It is kept in
    ``CACHE_DIRECTORY`` under a name derived from those, built once by the first process that needs it (a file lock keeps
    parallel processes from building it twice) and loaded by every later one.

    Initial guess strategy
    ----------------------
    Its owner sets the initial values of ``X`` and ``U`` before a solve (``set_initial``) and linearizes the half-spaces
    along its reference (``update_collision_halfspaces``, after ``set_state``). HPIPM starts from the previous solve's
    primal and dual solution (``qp_solver_warm_start`` 2): the flow steps of a tick solve nearly the same QP, which halves
    the iterations. A failed solve, and ``reset``, clear that memory. If the solution of ``solve`` still penetrates an
    obstacle or has a slack above the tolerance, it linearizes the kinematics again around it, within the same sets, and
    solves once more; the first solution stays when that solve fails. A solution that is still rejected, or a failed
    solve, raises ``RuntimeError``.

    The plan returned is the exact consequence of its actions: the actions are clipped to their limits and the states
    rolled out from x_init through the dynamics (the solver meets the dynamics only up to its tolerance).
    """

    def __init__(self, simulator: TripleIntegratorBase, horizon: int, cost: TrackingCost, with_obstacles: bool = True) -> None:
        self.sim = simulator
        self.N = int(horizon)
        self.n_joints = n = simulator.n
        nx, nu, N = self.sim.nx, self.sim.nu, self.N

        num_spheres = self.sim.collision.num_spheres if with_obstacles else 0
        set_size = self.sim.collision.free_space.size if with_obstacles else 0
        self.num_halfspaces = P = num_spheres * set_size
        self.num_slacks = num_spheres if P > 0 else 0
        self._q_current: np.ndarray | None = None
        self._sets: tuple[np.ndarray, np.ndarray] | None = None  # the free-space sets of the last ``update``

        x_sym, u_sym = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
        step = self.sim.casadi_dynamics(x_sym, u_sym)
        affine = ca.Function("affine", [x_sym, u_sym], [ca.jacobian(step, x_sym), ca.jacobian(step, u_sym), step])
        A_d, B_d, c = (np.array(value, dtype=float) for value in affine(np.zeros(nx), np.zeros(nu)))
        self._A_d, self._B_d, self._c = A_d, B_d, c.ravel()

        # The infinite state bounds (e.g. the point mass positions) are no constraint
        lower = np.asarray(self.sim.state_lower_bounds, dtype=float)
        upper = np.asarray(self.sim.state_upper_bounds, dtype=float)
        self._bounded = np.flatnonzero(np.isfinite(lower) | np.isfinite(upper))
        self._lower = np.where(np.isfinite(lower), lower, -ACADOS_INFTY)[self._bounded]
        self._upper = np.where(np.isfinite(upper), upper, ACADOS_INFTY)[self._bounded]
        self._max_action = np.asarray(self.sim.max_action, dtype=float) * np.ones(nu)
        self._sphere_of_row = np.repeat(np.arange(num_spheres), set_size)

        self._solver = self._load(self._formulation(set_size))
        self._set_cost(cost)
        for k in range(N + 1):
            if k >= 1:
                self._solver.constraints_set(k, "lbx", self._lower)
                self._solver.constraints_set(k, "ubx", self._upper)
            if k < N:
                self._solver.constraints_set(k, "lbu", -self._max_action)
                self._solver.constraints_set(k, "ubu", self._max_action)
        self._x0 = np.zeros(nx)
        self._halfspace_rows = np.zeros((P, nx))
        self._A_coll, self._b_coll = np.zeros((N, P, n)), np.zeros((N, P))
        self.set_initial(0.0, 0.0)

    def _key(self, set_size: int) -> str:
        """The name of the generated code: everything it depends on."""
        acados_commit = Path(get_acados_path()) / "lib" / "git_commit_hash"
        parts = (
            self.sim.nx, self.sim.nu, self.N, self.num_halfspaces, self.num_slacks, set_size, self._bounded, self._A_d,
            self._B_d, self._c, sorted(SOLVER_OPTIONS.items()), SLACK_LINEAR_WEIGHT, SLACK_QUADRATIC_WEIGHT,
            acados_commit.read_text() if acados_commit.exists() else "",
        )
        digest = hashlib.sha1()
        for part in parts:
            digest.update(repr(part.tolist() if isinstance(part, np.ndarray) else part).encode())
        return digest.hexdigest()[:16]

    def _formulation(self, set_size: int) -> AcadosOcp:
        """The OCP of the problem, with placeholders for everything set at run time."""
        nx, nu, N, P, S = self.sim.nx, self.sim.nu, self.N, self.num_halfspaces, self.num_slacks
        key = self._key(set_size)
        model = AcadosModel()
        model.name = f"projection_{key}"
        model.x, model.u = ca.SX.sym("x", nx), ca.SX.sym("u", nu)
        model.disc_dyn_expr = ca.DM(self._A_d) @ model.x + ca.DM(self._B_d) @ model.u + ca.DM(self._c)

        ocp = AcadosOcp()
        ocp.model = model
        ocp.solver_options.N_horizon = N
        ocp.solver_options.tf = float(N)
        for name, value in SOLVER_OPTIONS.items():
            setattr(ocp.solver_options, name, value)
        ocp.solver_options.cost_scaling = np.ones(N + 1)  # acados would integrate the stage cost over the step

        # acados' cost is 1/2 ||y - y_ref||_W^2 + z_l s + 1/2 Z_l s^2: W and Z_l are twice the weights
        ocp.cost.cost_type = ocp.cost.cost_type_0 = ocp.cost.cost_type_e = "LINEAR_LS"
        ocp.cost.Vx = ocp.cost.Vx_0 = np.vstack([np.eye(nx), np.zeros((nu, nx))])
        ocp.cost.Vu = ocp.cost.Vu_0 = np.vstack([np.zeros((nx, nu)), np.eye(nu)])
        ocp.cost.W = ocp.cost.W_0 = np.eye(nx + nu)
        ocp.cost.yref = ocp.cost.yref_0 = np.zeros(nx + nu)
        ocp.cost.Vx_e, ocp.cost.W_e, ocp.cost.yref_e = np.eye(nx), np.eye(nx), np.zeros(nx)

        constraints = ocp.constraints
        constraints.x0 = np.zeros(nx)
        constraints.idxbu, constraints.lbu, constraints.ubu = np.arange(nu), -np.ones(nu), np.ones(nu)
        nb = len(self._bounded)
        constraints.idxbx, constraints.lbx, constraints.ubx = self._bounded, -np.ones(nb), np.ones(nb)
        constraints.idxbx_e, constraints.lbx_e, constraints.ubx_e = self._bounded, -np.ones(nb), np.ones(nb)
        if P > 0:
            constraints.C, constraints.D = np.zeros((P, nx)), np.zeros((P, nu))
            constraints.lg, constraints.ug = -ACADOS_INFTY * np.ones(P), ACADOS_INFTY * np.ones(P)
            constraints.C_e, constraints.lg_e, constraints.ug_e = np.zeros((P, nx)), -ACADOS_INFTY * np.ones(P), ACADOS_INFTY * np.ones(P)
            # The slack of each constraint row (-1: none), in acados' order of the constraints: bounds on u, on x, then C
            constraints.idxs_rev = np.concatenate([-np.ones(nu + nb, dtype=int), self._sphere_of_row])
            constraints.idxs_rev_e = np.concatenate([-np.ones(nb, dtype=int), self._sphere_of_row])
            constraints.ls, constraints.us, constraints.ls_e, constraints.us_e = (np.zeros(S) for _ in range(4))
            for suffix in ("", "_e"):
                for side in ("l", "u"):
                    setattr(ocp.cost, f"z{side}{suffix}", SLACK_LINEAR_WEIGHT * np.ones(S))
                    setattr(ocp.cost, f"Z{side}{suffix}", 2.0 * SLACK_QUADRATIC_WEIGHT * np.ones(S))

        directory = CACHE_DIRECTORY / key
        ocp.code_gen_options.code_export_directory = str(directory)
        ocp.code_gen_options.json_file = str(directory / "acados_ocp.json")
        return ocp

    @staticmethod
    def _load(ocp: AcadosOcp) -> AcadosOcpSolver:
        """The solver of ``ocp``: its code is generated and built unless an identical one is in the cache."""
        directory = Path(ocp.code_gen_options.code_export_directory)
        directory.parent.mkdir(parents=True, exist_ok=True)
        with open(directory.parent / f"{directory.name}.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with contextlib.redirect_stdout(io.StringIO()):
                return AcadosOcpSolver(ocp, generate=False, build=False, verbose=False)

    def _set_cost(self, cost: TrackingCost) -> None:
        n, nu, N = self.n_joints, self.sim.nu, self.N
        state = np.concatenate([np.ones(n), cost.velocity_weight * np.ones(n), cost.acceleration_weight * np.ones(n)])
        stage = 2.0 * np.diag(np.concatenate([state, cost.jerk_weight * np.ones(nu)]))
        terminal = state.copy()
        terminal[list(self.sim.rest_state_indices)] += cost.terminal_weight
        for k in range(N):
            self._solver.cost_set(k, "W", stage)
        self._solver.cost_set(N, "W", 2.0 * np.diag(terminal))

    def set_state(self, x0: np.ndarray) -> None:
        """The state the horizon starts from."""
        self._x0 = np.array(x0, dtype=float)
        self._q_current = self._x0[: self.n_joints]
        self._solver.constraints_set(0, "lbx", self._x0)
        self._solver.constraints_set(0, "ubx", self._x0)

    def set_reference(self, q_ref: np.ndarray) -> None:
        """The positions (N, n) the nodes 1 .. N track."""
        q_ref = np.asarray(q_ref, dtype=float)
        if q_ref.shape != (self.N, self.n_joints):
            raise ValueError(f"The reference must have shape {(self.N, self.n_joints)}, got {q_ref.shape}.")
        rest = self.sim.nx - self.n_joints
        for k in range(1, self.N):
            self._solver.cost_set(k, "yref", np.concatenate([q_ref[k - 1], np.zeros(rest + self.sim.nu)]))
        self._solver.cost_set(self.N, "yref", np.concatenate([q_ref[-1], np.zeros(rest)]))

    def set_initial(self, states: np.ndarray, actions: np.ndarray) -> None:
        """The initial values of the states and actions of the next solves, kept until they are set again."""
        self._initial_states = np.array(np.broadcast_to(states, (self.sim.nx, self.N + 1)), dtype=float)
        self._initial_actions = np.array(np.broadcast_to(actions, (self.sim.nu, self.N)), dtype=float)

    @property
    def initial_guess(self) -> tuple[np.ndarray, np.ndarray]:
        """The initial states (nx, N + 1) and actions (nu, N) of the next solve."""
        return self._initial_states.copy(), self._initial_actions.copy()

    def reset(self) -> None:
        """Nothing of the last episode is reused: the initial values go back to zero and the solver forgets its last
        solution."""
        self.set_initial(0.0, 0.0)
        self._solver.reset()

    def update_collision_halfspaces(self, q_ref: np.ndarray) -> None:
        """The free-space sets along ``q_ref`` (N, n_joints), the set of each node being the one around the motion of the
        spheres from the previous node (the state of ``set_state`` for the first), linearized along it."""
        if self.num_halfspaces > 0:
            self._sets, A, b = self.sim.collision.linearized_sets(self._q_current, q_ref)
            self._set_linearization(A, b)

    def _linearize_sets(self, q_ref: np.ndarray) -> None:
        self._set_linearization(*self.sim.collision.linearize(self._sets, q_ref))

    def _set_linearization(self, A: np.ndarray, b: np.ndarray) -> None:
        """The half-spaces ``A[k] q_{k+1} >= b[k]`` (A (N, P, n), b (N, P)) of the nodes 1 .. N."""
        A = self._A_coll = np.asarray(A, dtype=float).reshape(self.N, self.num_halfspaces, self.n_joints)
        b = self._b_coll = np.asarray(b, dtype=float).reshape(self.N, self.num_halfspaces)
        rows = self._halfspace_rows
        for k in range(1, self.N + 1):
            rows[:, : self.n_joints] = A[k - 1]
            self._solver.constraints_set(k, "C", rows, api="new")
            self._solver.constraints_set(k, "lg", b[k - 1])

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
        solution. Raises ``RuntimeError`` if the solver reports a failure or its QP does not converge."""
        solver = self._solver
        for k in range(self.N + 1):
            solver.set(k, "x", self._initial_states[:, k])
            if k < self.N:
                solver.set(k, "u", self._initial_actions[:, k])
        status = solver.solve()
        qp_status = int(np.atleast_1d(solver.get_stats("qp_stat"))[-1])
        actions = np.stack([solver.get(k, "u") for k in range(self.N)], axis=1)
        if status != 0 or qp_status not in ACCEPTED_QP_STATUS or not np.all(np.isfinite(actions)):
            solver.reset()  # a failed solve leaves no warm start for the next one
            raise RuntimeError(f"The solver failed (return status {status}, QP status {qp_status}).")
        actions = np.clip(actions, -self._max_action[:, None], self._max_action[:, None])
        states = self._rollout(self._x0, actions)
        bounded = states[self._bounded, 1:]
        if np.any(bounded < self._lower[:, None] - LIMIT_TOLERANCE) or np.any(bounded > self._upper[:, None] + LIMIT_TOLERANCE):
            solver.reset()
            raise RuntimeError(f"The solver failed: its plan leaves the state limits (QP status {qp_status}).")
        return states, actions, self._max_violation(states)

    def _max_violation(self, states: np.ndarray) -> float:
        """The largest slack the plan ``states`` needs: the deepest violation of a linearized half-space (0 if none)."""
        if self.num_halfspaces == 0:
            return 0.0
        q = states[: self.n_joints, 1:].T  # (N, n)
        return max(0.0, float(np.max(self._b_coll - np.einsum("kpn,kn->kp", self._A_coll, q))))

    def _rollout(self, x0: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """The states (nx, N + 1) the actions (nu, N) take the plant through from ``x0``."""
        states = np.empty((len(x0), actions.shape[1] + 1))
        states[:, 0] = x0
        for k in range(actions.shape[1]):
            states[:, k + 1] = self._A_d @ states[:, k] + self._B_d @ actions[:, k] + self._c
        return states

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
