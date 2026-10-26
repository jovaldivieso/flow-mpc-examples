"""CasADi kinematics of the FR3 arm, generated from its Pinocchio model (``franka_model``) and built once (see
``franka_kinematics``).

``fk`` maps a configuration to the TCP position (3) and rotation matrix (3 x 3) and is symbolic-friendly, so planners
and projectors can use it inside their own problems (e.g. for obstacle constraints). ``inverse_kinematics`` is SafeFlowMPC's
IK problem: minimize ``||p - p_d||^2 + ||R R_d^T - I||_F^2`` over the configuration within the joint limits (IPOPT).
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import casadi as ca
import numpy as np
import pinocchio as pin

from .collision_spheres import fit_collision_spheres
from .franka_model import Q_READY, TCP_FRAME, build_model, build_robot

# The bounding spheres of the links are fitted to the collision meshes with this radius at most: smaller ones fit the arm
# more tightly but cost more constraints in the planners
COLLISION_SPHERE_MAX_RADIUS = 0.08

IK_TOLERANCE = 1e-6  # of the residual norm (position in m, rotation matrix difference) for a solution
IK_MAX_ITERATIONS = 200
IK_ATTEMPTS = 30
IK_SOLUTIONS = 5
IK_IPOPT_OPTIONS = {
    "print_time": False,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-12,
    "ipopt.max_iter": IK_MAX_ITERATIONS,
}


@functools.lru_cache(maxsize=None)
def franka_kinematics() -> FrankaKinematics:
    """The CasADi kinematics of the TCP (forward kinematics, link spheres, and IK), built once per process, the spheres
    fitted to the collision meshes."""
    model = build_model()
    spheres = fit_collision_spheres(build_robot().collision_model, COLLISION_SPHERE_MAX_RADIUS)
    return FrankaKinematics(model, model.getFrameId(TCP_FRAME), spheres)


def _axis_rotation(axis: str, angle: ca.SX) -> ca.SX:
    cosine, sine = ca.cos(angle), ca.sin(angle)
    rotation = ca.SX.eye(4)
    first, second = {"X": (1, 2), "Y": (2, 0), "Z": (0, 1)}[axis]
    rotation[first, first], rotation[first, second] = cosine, -sine
    rotation[second, first], rotation[second, second] = sine, cosine
    return rotation


def _homogeneous(placement: pin.SE3) -> ca.DM:
    return ca.DM(placement.homogeneous)


class FrankaKinematics:
    """
    CasADi forward kinematics, link bounding spheres, and inverse kinematics (IPOPT) of the FR3 arm.

    Built once from the reduced Pinocchio model (see ``franka_kinematics``) for a serial chain of revolute joints
    about their X, Y, or Z axes.

    Exports
    -------
    - ``fk(q) -> (p, R)``: TCP position ``p`` (3,) and rotation matrix ``R`` (3, 3) as a symbolic-friendly ``SX`` function.
    - ``collision_spheres(q) -> (centers, jacobians, radii)``: centers (S, 3), analytical position Jacobians
      ``J_s = dp_s/dq`` (S, 3, n), and radii (S,) of the bounding spheres of the links.
    - ``inverse_kinematics(p_d, R_d, q_init) -> q``: solves the nonlinear program (NLP) below from ``q_init``.

    Problem (Inverse Kinematics NLP)
    --------------------------------
    Decision variable        q in R^n (n = 7): joint angles.
    Parameters               p_d in R^3 (target TCP position), R_d in SO(3) (target TCP rotation matrix).

    Cost                     min_q  ||p(q) - p_d||_2^2 + ||R(q) R_d^T - I_3||_F^2
    Constraints              q_lower <= q <= q_upper            joint position limits

    Solver Choice (``nlpsol`` + ``ipopt``)
    --------------------------------------
    Forward kinematics ``p(q), R(q)`` is a product of trigonometric joint transforms, making the pose residual
    non-convex in ``q``. Because there is only one 7-D decision variable ``q`` with simple box bounds and no general
    constraints, the problem is formulated directly on the ``SX`` kinematics graph via ``ca.nlpsol("ik", "ipopt", ...)``
    (interior-point NLP solver) without ``Opti`` wrapper state, allowing a single stateless solver instance to be shared
    across all simulators.

    Initial guess strategy
    ----------------------
    ``inverse_kinematics`` starts IPOPT from ``q_init`` (clipped into the joint limits) and returns ``q_init`` itself,
    without a solve, if it already reaches the pose within ``IK_TOLERANCE``. A solve that ends in a local minimum against
    the joint limits raises ``RuntimeError``. ``solve_ik`` is the multi-start around it: it tries ``Q_READY``, then the
    reference configuration, then up to ``IK_ATTEMPTS`` uniformly random starts (a fixed seed), keeps up to
    ``IK_SOLUTIONS`` valid solutions and returns the one nearest ``Q_READY``.
    """

    def __init__(self, model: pin.Model, frame_id: int, spheres: list[tuple[int, np.ndarray, float]]) -> None:
        """``spheres``: the bounding spheres of the links as ``(joint, center in its frame, radius)``."""
        frame = model.frames[frame_id]
        if frame.parentJoint != model.njoints - 1 or any(model.parents[joint] != joint - 1 for joint in range(1, model.njoints)):
            raise ValueError("The kinematics need a serial chain with the frame on its last joint.")
        self.n = model.nq
        self.lower, self.upper = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()

        q = ca.SX.sym("q", self.n)
        transform = ca.SX.eye(4)
        joint_frames: list[ca.SX] = []
        for joint in range(1, model.njoints):
            axis = model.joints[joint].shortname()[-1]
            if model.joints[joint].shortname() != f"JointModelR{axis}":
                raise ValueError(f"Joint {model.names[joint]} is not a revolute joint about a coordinate axis.")
            transform = transform @ _homogeneous(model.jointPlacements[joint]) @ _axis_rotation(axis, q[model.joints[joint].idx_q])
            joint_frames.append(transform)
        transform = transform @ _homogeneous(frame.placement)
        tcp_origin = transform[:3, 3]
        self.fk = ca.Function("fk", [q], [tcp_origin, transform[:3, :3]])

        # Bounding spheres covering the collision meshes of the moving links, fixed in the frame of their joint
        sphere_centers = [(joint_frames[joint - 1] @ ca.DM([*center, 1.0]))[:3] for joint, center, _ in spheres]
        self.sphere_radii = np.array([radius for _, _, radius in spheres], dtype=float)
        centers_mat = ca.vertcat(*[c.T for c in sphere_centers])
        jacobians_mat = ca.vertcat(*[ca.jacobian(c, q) for c in sphere_centers])
        self._spheres_fn = ca.Function("spheres", [q], [centers_mat, jacobians_mat])

        position_goal, rotation_goal = ca.SX.sym("p_d", 3), ca.SX.sym("R_d", 3, 3)
        position, rotation = self.fk(q)
        cost = ca.sumsqr(position - position_goal) + ca.sumsqr(rotation @ rotation_goal.T - ca.SX.eye(3))
        goal = ca.vertcat(position_goal, ca.vec(rotation_goal))
        self._ik_cost = ca.Function("ik_cost", [q, goal], [cost])
        self._ik_solver = ca.nlpsol("ik", "ipopt", {"x": q, "f": cost, "p": goal}, IK_IPOPT_OPTIONS)

    def __deepcopy__(self, memo: dict) -> FrankaKinematics:
        """Return this very instance instead of a copy.

        This is deliberate. Building the kinematics is expensive (the CasADi function graphs and the IPOPT solver), and
        those objects cannot be copied. The instance is immutable once built and is shared by the whole process through
        ``franka_kinematics`` (a singleton, ``functools.lru_cache``). Systems are deep-copied (the safe flow policy keeps a
        private copy of the simulator it is built from, rollouts copy it), and a copy has to share the kinematics, not
        rebuild or duplicate them.
        """
        return self

    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (S, 3), position Jacobians (S, 3, n), and radii (S,) at configuration ``q``."""
        centers, jacobians = self._spheres_fn(q)
        s = len(self.sphere_radii)
        return np.array(centers, dtype=float), np.array(jacobians, dtype=float).reshape(s, 3, self.n), self.sphere_radii

    def reaches_pose(self, q: np.ndarray, position: np.ndarray, rotation: np.ndarray) -> bool:
        """True if ``q`` already achieves the target TCP ``(position, rotation)`` within ``IK_TOLERANCE``."""
        q_clipped = np.clip(q, self.lower, self.upper)
        goal = np.concatenate([position, np.asarray(rotation).ravel(order="F")])
        return float(self._ik_cost(q_clipped, goal)) <= IK_TOLERANCE**2

    def inverse_kinematics(self, position: np.ndarray, rotation: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """The configuration of the TCP pose, found by IPOPT from ``q_init`` (which is returned as it is if it already
        is one). Raises ``RuntimeError`` if it ends in a local minimum (``IK_TOLERANCE``) instead of the pose."""
        q_init = np.clip(q_init, self.lower, self.upper)
        goal = np.concatenate([position, np.asarray(rotation).ravel(order="F")])
        if float(self._ik_cost(q_init, goal)) <= IK_TOLERANCE**2:
            return q_init
        solution = self._ik_solver(x0=q_init, lbx=self.lower, ubx=self.upper, p=goal)
        if float(solution["f"]) > IK_TOLERANCE**2:
            raise RuntimeError("Inverse kinematics did not converge; the pose may be unreachable.")
        return np.array(solution["x"]).ravel()

    def solve_ik(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        q_reference: np.ndarray = Q_READY,
        is_valid: Callable[[np.ndarray], bool] | None = None,
    ) -> np.ndarray:
        """Multi-start inverse kinematics: tries ``Q_READY``, ``q_reference``, and up to ``IK_ATTEMPTS`` random starts,
        returning the valid solution nearest to ``Q_READY``."""
        valid = (lambda q: True) if is_valid is None else is_valid
        q_ref = np.clip(q_reference, self.lower, self.upper)
        for q_seed in (Q_READY, q_ref):
            try:
                solution = self.inverse_kinematics(position, rotation, q_seed)
                if valid(solution):
                    return solution
            except RuntimeError:
                pass
        rng = np.random.default_rng(0)
        found: list[np.ndarray] = []
        for _ in range(IK_ATTEMPTS):
            try:
                q_start = rng.uniform(self.lower, self.upper)
                candidate = self.inverse_kinematics(position, rotation, q_start)
                if valid(candidate):
                    found.append(candidate)
            except RuntimeError:
                continue
            if len(found) == IK_SOLUTIONS:
                break
        if not found:
            raise RuntimeError("Inverse kinematics did not converge from any start; the pose may be unreachable.")
        return min(found, key=lambda q: float(np.linalg.norm(q - Q_READY)))
