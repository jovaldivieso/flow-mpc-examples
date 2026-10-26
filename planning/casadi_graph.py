from typing import NamedTuple

import casadi as ca
import numpy as np

from systems.kinematic_collision_system import KinematicCollisionSystem

# Consecutive MPC/projection solves differ little: ProxQP warm-starts both primal and dual variables by default,
# and expanding the mapped dynamics to SX keeps every solve fast.
PROXQP_OPTIONS: dict = {
    "expand": True,
    "print_time": False,
    "error_on_fail": False,
    "proxqp": {
        "verbose": False,
        "eps_abs": 1e-8,
        "eps_rel": 0.0,
    },
}
SLACK_WEIGHT = 1e5  # weight of the squared slack of the obstacle half-spaces in the cost


class GraphSolution(NamedTuple):
    """The states (nx, N + 1) and actions (nu, N) of a solve."""

    states: np.ndarray
    actions: np.ndarray


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
                             present) slack S = [s_1 .. s_N] (P, N) with P = n_spheres * n_obstacles.
    Parameters               x_init, and linearized obstacle half-spaces (A_coll, b_coll).

    Cost                     none here: ``minimize(objective)`` adds w_s ||S||_F^2 (w_s = 10^5) to its owner's.

    Constraints              x_0 = x_init
                             x_{k+1} = f(x_k, u_k)                       triple integrator (``sim.casadi_dynamics``)
                             -u_max <= u_k <= u_max                      action limits, k = 0 .. N-1
                             x_lower <= x_k <= x_upper                   the finite state limits, k = 1 .. N
                             A_coll,k q_k + s_k >= b_coll,k,  s_k >= 0   linearized sphere-obstacle half-spaces, k = 1 .. N

    Solver Choice (``Opti("conic")`` + ``proxqp``)
    ----------------------------------------------
    The dynamics, the box limits and the first-order Taylor obstacle half-spaces ``n^T J_s(q_ref,k) q_k >= b_k`` are
    affine in ``(X, U, S)``, so with a convex quadratic cost CasADi transcribes the problem into a sparse conic QP solved
    by **ProxQP** (augmented-Lagrangian primal-dual QP solver), with the mapped dynamics expanded to ``SX``
    (``PROXQP_OPTIONS``).

    Initial guess strategy
    ----------------------
    Its owner sets the initial values of ``X`` and ``U`` before a solve (``set_initial``) and linearizes the half-spaces
    along its guess (``update_collision_halfspaces``). The multipliers ``lam_g`` of the last solve are kept here and
    warm start the next one, unless ``set_initial`` is asked for a cold start (zero multipliers). If the solution of
    ``solve`` still penetrates an obstacle, it linearizes again around it and solves once more from there; the first
    solution stays when that solve fails.

    ``X`` and ``U`` are the symbolic variables its owner builds its cost from; everything else of ``opti`` stays here.
    """

    def __init__(self, simulator: KinematicCollisionSystem, horizon: int) -> None:
        self.sim = simulator
        self.N = int(horizon)
        self.n_joints = simulator.n

        self._multipliers: np.ndarray | None = None  # lam_g of the last solve, the warm start of the next
        self.opti = ca.Opti("conic")
        self.X = self.opti.variable(self.sim.nx, self.N + 1)
        self.U = self.opti.variable(self.sim.nu, self.N)
        self.x0_param = self.opti.parameter(self.sim.nx)

        # Dynamics: X[:, 1:] = f(X[:, :-1], U), the step function mapped over the horizon
        x_sym, u_sym = ca.SX.sym("x", self.sim.nx), ca.SX.sym("u", self.sim.nu)
        step_fn = ca.Function("step", [x_sym, u_sym], [self.sim.casadi_dynamics(x_sym, u_sym)])
        self.opti.subject_to(self.X[:, 0] == self.x0_param)
        self.opti.subject_to(self.X[:, 1:] == step_fn.map(self.N)(self.X[:, :-1], self.U))

        # Action limits. vec() forces element-wise inequality (CasADi reads a non-square matrix comparison as a PSD
        # constraint); the per-dimension limits are repeated across the horizon to match its column-major order.
        max_action = np.broadcast_to(np.asarray(self.sim.max_action, dtype=float).reshape(-1, 1), (self.sim.nu, self.N))
        self.opti.subject_to(ca.vec(self.U) >= ca.vec(-max_action))
        self.opti.subject_to(ca.vec(self.U) <= ca.vec(max_action))

        # State limits at the nodes 1 .. N (node 0 is the current state); infinite bounds (e.g. the point mass
        # positions) are no constraint
        lower_bounds = np.asarray(self.sim.state_lower_bounds, dtype=float)
        finite_lower = np.flatnonzero(np.isfinite(lower_bounds)).tolist()
        if finite_lower:
            lower = ca.repmat(ca.DM(lower_bounds[finite_lower]), 1, self.N)
            self.opti.subject_to(ca.vec(self.X[finite_lower, 1:]) >= ca.vec(lower))
        upper_bounds = np.asarray(self.sim.state_upper_bounds, dtype=float)
        finite_upper = np.flatnonzero(np.isfinite(upper_bounds)).tolist()
        if finite_upper:
            upper = ca.repmat(ca.DM(upper_bounds[finite_upper]), 1, self.N)
            self.opti.subject_to(ca.vec(self.X[finite_upper, 1:]) <= ca.vec(upper))

        # Linearized spherical-obstacle half-spaces A_coll @ q_k + slack >= b_coll at nodes 1 .. N
        obstacles = self.sim.obstacles
        if obstacles:
            _, _, radii = self.sim.collision_spheres(np.zeros(self.n_joints))
            self.num_collision_pairs = len(radii) * len(obstacles)
        else:
            self.num_collision_pairs = 0
        if self.num_collision_pairs > 0:
            P = self.num_collision_pairs
            self.A_coll = self.opti.parameter(P * self.N, self.n_joints)
            self.b_coll = self.opti.parameter(P * self.N)
            self.slack = self.opti.variable(P * self.N)
            self.opti.subject_to(self.slack >= 0.0)
            lhs = ca.vertcat(
                *[ca.mtimes(self.A_coll[k * P : (k + 1) * P, :], self.X[: self.n_joints, k + 1]) for k in range(self.N)]
            )
            self.opti.subject_to(lhs + self.slack >= self.b_coll)

    def minimize(self, objective: ca.MX) -> None:
        """Minimize ``objective`` plus the penalty on the obstacle slack, and set the solver."""
        if self.num_collision_pairs > 0:
            objective = objective + SLACK_WEIGHT * ca.sumsqr(self.slack)
        self.opti.minimize(objective)
        self.opti.solver("proxqp", PROXQP_OPTIONS)

    def parameter(self, rows: int, cols: int = 1) -> ca.MX:
        """A parameter of its owner's cost, set with ``set_value``."""
        return self.opti.parameter(rows, cols)

    def set_value(self, parameter: ca.MX, value: np.ndarray) -> None:
        self.opti.set_value(parameter, value)

    def set_state(self, x0: np.ndarray) -> None:
        """The state the horizon starts from."""
        self.opti.set_value(self.x0_param, x0)

    def set_initial(self, states: np.ndarray, actions: np.ndarray, *, cold: bool = False) -> None:
        """The initial values of the variables of the next solve, and the multipliers of the last one (zero if ``cold``)."""
        self.opti.set_initial(self.X, states)
        self.opti.set_initial(self.U, actions)
        if cold:
            self.opti.set_initial(self.opti.lam_g, 0.0)
        elif self._multipliers is not None:
            self.opti.set_initial(self.opti.lam_g, self._multipliers)

    def reset(self) -> None:
        """Nothing of the last episode is reused. The multipliers are sticky on the ``Opti`` object: without zeroing
        them the first solve of the next episode would warm start from the final ones of the last."""
        self._multipliers = None
        self.set_initial(0.0, 0.0, cold=True)
        if self.num_collision_pairs > 0:
            self.opti.set_initial(self.slack, 0.0)

    def update_collision_halfspaces(self, q_ref: np.ndarray) -> None:
        """Linearize the obstacle half-spaces along ``q_ref`` (N, n_joints)."""
        if self.num_collision_pairs > 0:
            A, b = self.sim.collision_halfspaces(q_ref)
            self.opti.set_value(self.A_coll, A.reshape(-1, self.n_joints))
            self.opti.set_value(self.b_coll, b.reshape(-1))

    def solve(self) -> GraphSolution:
        """Solve from the initial values set; raises ``RuntimeError`` if the solver fails."""
        sol = self._relinearize_if_colliding(self.opti.solve())
        self._multipliers = sol.value(self.opti.lam_g)
        return GraphSolution(sol.value(self.X), sol.value(self.U))

    def _relinearize_if_colliding(self, sol: ca.OptiSol) -> ca.OptiSol:
        """The half-spaces are only a linearization: if the solution still penetrates an obstacle, linearize again
        around it and solve once more. The first solution stays when that solve fails."""
        if self.num_collision_pairs == 0:
            return sol
        X_first = sol.value(self.X)
        if all(self.sim.is_collision_free(X_first[: self.n_joints, k]) for k in range(1, self.N + 1)):
            return sol
        self.set_initial(X_first, sol.value(self.U))
        self.update_collision_halfspaces(X_first[: self.n_joints, 1:].T)
        self.opti.set_initial(self.opti.lam_g, sol.value(self.opti.lam_g))
        try:
            return self.opti.solve()
        except RuntimeError:
            return sol
