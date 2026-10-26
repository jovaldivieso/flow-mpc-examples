"""CasADi kinematics of the FR3 arm, generated from its Pinocchio model and built once (see ``franka_kinematics``).

``fk`` maps a configuration to the TCP position (3) and rotation matrix (3 x 3) and is symbolic-friendly, so planners
and projectors can use it inside their own problems (e.g. for obstacle constraints). ``inverse_kinematics`` is SafeFlowMPC's
IK problem: minimize ``||p - p_d||^2 + ||R R_d^T - I||_F^2`` over the configuration within the joint limits (IPOPT).
"""

from __future__ import annotations

import casadi as ca
import numpy as np
import pinocchio as pin

IK_TOLERANCE = 1e-6  # of the residual norm (position in m, rotation matrix difference) for a solution
IK_MAX_ITERATIONS = 200
IK_IPOPT_OPTIONS = {
    "print_time": False,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-12,
    "ipopt.max_iter": IK_MAX_ITERATIONS,
}


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
    """Forward kinematics ``fk(q) -> (position, rotation)`` of a frame of a serial chain of revolute joints about
    their X, Y or Z axes, and the IK solver of its pose."""

    def __init__(self, model: pin.Model, frame_id: int) -> None:
        frame = model.frames[frame_id]
        if frame.parentJoint != model.njoints - 1 or any(model.parents[joint] != joint - 1 for joint in range(1, model.njoints)):
            raise ValueError("The kinematics need a serial chain with the frame on its last joint.")
        self.n = model.nq
        self.lower, self.upper = model.lowerPositionLimit.copy(), model.upperPositionLimit.copy()

        q = ca.SX.sym("q", self.n)
        transform = ca.SX.eye(4)
        for joint in range(1, model.njoints):
            axis = model.joints[joint].shortname()[-1]
            if model.joints[joint].shortname() != f"JointModelR{axis}":
                raise ValueError(f"Joint {model.names[joint]} is not a revolute joint about a coordinate axis.")
            transform = transform @ _homogeneous(model.jointPlacements[joint]) @ _axis_rotation(axis, q[model.joints[joint].idx_q])
        transform = transform @ _homogeneous(frame.placement)
        self.fk = ca.Function("fk", [q], [transform[:3, 3], transform[:3, :3]])

        position_goal, rotation_goal = ca.SX.sym("p_d", 3), ca.SX.sym("R_d", 3, 3)
        position, rotation = self.fk(q)
        cost = ca.sumsqr(position - position_goal) + ca.sumsqr(rotation @ rotation_goal.T - ca.SX.eye(3))
        goal = ca.vertcat(position_goal, ca.vec(rotation_goal))
        self._ik_cost = ca.Function("ik_cost", [q, goal], [cost])
        self._ik_solver = ca.nlpsol("ik", "ipopt", {"x": q, "f": cost, "p": goal}, IK_IPOPT_OPTIONS)

    def __deepcopy__(self, memo: dict) -> FrankaKinematics:
        return self  # immutable once built, so the copies of a simulator share it

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
