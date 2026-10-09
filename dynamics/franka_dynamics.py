"""Franka FR3 arm with gripper (9 DoF: 7 arm joints + 2 prismatic fingers).

State  x = [q, v]  (joint positions, joint velocities), nx = 18
Action u = a       (joint accelerations, fingers included), nu = 9

Inverse-dynamics formulation: the planner chooses accelerations and the torques
tau = RNEA(q, v, a) are constrained to the effort limits (``casadi_torque``).
This needs one RNEA pass per step instead of the mass matrix and a linear solve.

The model comes from the franka_description xacro via Pinocchio. Brew's
Pinocchio has no ``pinocchio.casadi``, so the symbolic dynamics are a CasADi
RNEA built from the Pinocchio model's inertias, placements and joint axes;
the numeric ``step`` uses Pinocchio's ABA (same semi-implicit Euler scheme).
"""

from pathlib import Path
from typing import Any, Mapping

import casadi as ca
import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

from .dynamics import DynamicsSimulator

# Official franka_description, pinned as a git submodule
FRANKA_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"
XACRO_PATH = FRANKA_DESCRIPTION / "robots" / "fr3" / "fr3.urdf.xacro"

# xacro resolves $(find pkg) via ament_index_python (ROS-only, not on PyPI)
substitution_args._eval_find = lambda pkg: str(FRANKA_DESCRIPTION)

FINGER_OPEN = 0.04
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4, FINGER_OPEN, FINGER_OPEN])
TCP_FRAME = "fr3_hand_tcp"


def build_robot() -> RobotWrapper:
    """Full 9-DoF FR3 with gripper, from the franka_description xacro."""
    urdf = xacro.process_file(XACRO_PATH, mappings={"hand": "true"}).toxml()
    # meshes are package://franka_description/..., resolved from the clone's parent
    pkg = [str(FRANKA_DESCRIPTION.parent)]
    model = pin.buildModelFromXML(urdf)
    visual = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.VISUAL, package_dirs=pkg)
    collision = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.COLLISION, package_dirs=pkg)
    return RobotWrapper(model, collision, visual)


def _rotation(axis: np.ndarray, angle: Any) -> Any:
    """Rodrigues rotation about a constant unit axis; ``angle`` may be symbolic."""
    ax = ca.DM(axis)
    skew = ca.skew(ax)
    return ca.DM.eye(3) + ca.sin(angle) * skew + (1 - ca.cos(angle)) * ca.mtimes(skew, skew)


def _crm(v: Any) -> Any:
    """Spatial motion cross-product matrix (linear; angular ordering)."""
    lin, ang = v[:3], v[3:]
    return ca.blockcat([[ca.skew(ang), ca.skew(lin)], [ca.DM.zeros(3, 3), ca.skew(ang)]])


class FrankaDynamics(DynamicsSimulator):
    """Joint-space torque-controlled FR3 + gripper; the goal is a joint configuration (fingers included)."""

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.robot = build_robot()
        self.model = self.robot.model
        self.data = self.robot.data
        self.tcp_id = self.model.getFrameId(TCP_FRAME)
        self.n = self.model.nv

        self.nx = 2 * self.n
        self.nu = self.n
        self.max_torque = self.model.effortLimit.copy()
        self.max_action = np.asarray(config.get("max_acceleration", [15.0] * 7 + [20.0] * 2), dtype=float)
        self.state_lower_bounds = np.concatenate([self.model.lowerPositionLimit, -self.model.velocityLimit])
        self.state_upper_bounds = np.concatenate([self.model.upperPositionLimit, self.model.velocityLimit])

        # Goal TCP pose as [position(3), rotation matrix row-major(9)]; weights = (position, orientation)
        self.set_task_goal(self.tcp_placement(Q_READY))
        self.task_weights = config.get("task_weights", (1000.0, 100.0))
        self._build_casadi_model()

    # --- symbolic dynamics -------------------------------------------------

    def _build_casadi_model(self) -> None:
        """Extract constant link data for the CasADi RNEA."""
        model = self.model
        q0 = pin.neutral(model)
        data = model.createData()
        pin.computeJointJacobians(model, data, q0)

        self._parents = list(model.parents)
        self._placements = []  # (R, p) of joint frame in parent frame at q=0
        self._axes = []
        self._prismatic = []
        self._inertias = []  # 6x6 spatial inertia at the joint frame (linear; angular)
        for i in range(1, model.njoints):
            joint = model.joints[i]
            S = pin.getJointJacobian(model, data, i, pin.ReferenceFrame.LOCAL)[:, joint.idx_v]
            prismatic = np.allclose(S[3:], 0.0)
            if joint.nv != 1 or not (prismatic or np.allclose(S[:3], 0.0)):
                raise ValueError("Only 1-DoF revolute/prismatic joints are supported by the CasADi dynamics.")
            self._prismatic.append(prismatic)
            self._axes.append(S[:3].copy() if prismatic else S[3:].copy())
            placement = model.jointPlacements[i]
            self._placements.append((placement.rotation.copy(), placement.translation.copy()))
            self._inertias.append(model.inertias[i].matrix().copy())
        self._gravity_acc = -model.gravity.vector.copy()  # base acceleration (linear; angular)

    def _joint_transform(self, i: int, q_i: Any) -> tuple[Any, Any]:
        """Rotation and translation of joint i's frame in its parent's frame."""
        R0, p0 = self._placements[i - 1]
        axis = self._axes[i - 1]
        if self._prismatic[i - 1]:
            R = ca.DM(R0)
            return R, ca.DM(p0) + ca.mtimes(R, ca.DM(axis)) * q_i
        return ca.mtimes(ca.DM(R0), _rotation(axis, q_i)), ca.DM(p0)

    def _fk(self, q: Any) -> tuple[Any, Any]:
        """TCP rotation (3x3) and position (3) in the world frame."""
        R_w = [ca.DM.eye(3)] * self.model.njoints
        p_w = [ca.DM.zeros(3)] * self.model.njoints
        for i in range(1, self.model.njoints):
            R, p = self._joint_transform(i, q[i - 1])
            parent = self._parents[i]
            R_w[i] = ca.mtimes(R_w[parent], R)
            p_w[i] = p_w[parent] + ca.mtimes(R_w[parent], p)
        frame = self.model.frames[self.tcp_id]
        R_f, p_f = frame.placement.rotation, frame.placement.translation
        j = frame.parentJoint
        return ca.mtimes(R_w[j], ca.DM(R_f)), p_w[j] + ca.mtimes(R_w[j], ca.DM(p_f))

    def casadi_task_cost(self, x: Any, task_goal: Any) -> Any:
        """TCP SE(3) error to ``task_goal`` = [position(3), rotation matrix row-major(9)].

        Orientation error is 3 - tr(R_goal^T R) = 2 (1 - cos(angle)): smooth, no wrapping.
        """
        R, p = self._fk(x[: self.n])
        R_goal = ca.reshape(task_goal[3:12], 3, 3).T
        position = ca.sumsqr(p - task_goal[:3])
        orientation = 3 - ca.trace(ca.mtimes(R_goal.T, R))
        return self.task_weights[0] * position + self.task_weights[1] * orientation

    def _rnea(self, q: Any, v: Any, a: Any, gravity: bool = True) -> Any:
        """Joint torques for (q, v, a) via recursive Newton-Euler in CasADi."""
        n_j = self.model.njoints
        v_s = [ca.DM.zeros(6)] * n_j
        a_s = [ca.DM(self._gravity_acc) if gravity else ca.DM.zeros(6)] + [None] * (n_j - 1)
        f_s = [None] * n_j
        X = [None] * n_j  # motion transforms parent -> child

        for i in range(1, n_j):
            axis = self._axes[i - 1]
            R, p = self._joint_transform(i, q[i - 1])
            S = ca.DM(np.concatenate([axis, np.zeros(3)]) if self._prismatic[i - 1]
                      else np.concatenate([np.zeros(3), axis]))
            Rt = R.T
            X[i] = ca.blockcat([[Rt, -ca.mtimes(Rt, ca.skew(p))], [ca.DM.zeros(3, 3), Rt]])

            parent = self._parents[i]
            v_joint = S * v[i - 1]
            v_s[i] = ca.mtimes(X[i], v_s[parent]) + v_joint
            a_s[i] = ca.mtimes(X[i], a_s[parent]) + S * a[i - 1] + ca.mtimes(_crm(v_s[i]), v_joint)

            inertia = ca.DM(self._inertias[i - 1])
            momentum = ca.mtimes(inertia, v_s[i])
            f_s[i] = ca.mtimes(inertia, a_s[i]) - ca.mtimes(_crm(v_s[i]).T, momentum)

        tau = [None] * (n_j - 1)
        for i in range(n_j - 1, 0, -1):
            axis = self._axes[i - 1]
            S = np.concatenate([axis, np.zeros(3)]) if self._prismatic[i - 1] else np.concatenate([np.zeros(3), axis])
            tau[i - 1] = ca.dot(ca.DM(S), f_s[i])
            parent = self._parents[i]
            if parent > 0:
                f_s[parent] = f_s[parent] + ca.mtimes(X[i].T, f_s[i])
        return ca.vertcat(*tau)

    def casadi_torque(self, x: Any, u: Any) -> Any:
        """Joint torques needed to realise acceleration ``u`` in state ``x``."""
        return self._rnea(x[: self.n], x[self.n :], u)

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Semi-implicit Euler step, matching ``predict_next_state``."""
        q, v = x[: self.n], x[self.n :]
        v_next = v + self.dt * u
        q_next = q + self.dt * v_next
        return ca.vertcat(q_next, v_next)

        # --- numeric simulation ------------------------------------------------

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        q, v = state[: self.n], state[self.n :]
        tau = np.clip(pin.rnea(self.model, self.data, q, v, action), -self.max_torque, self.max_torque)
        acc = pin.aba(self.model, self.data, q, v, tau)
        v_next = v + self.dt * acc
        return np.concatenate([q + self.dt * v_next, v_next])

    def observe(self, state: np.ndarray) -> np.ndarray:
        return state

    @property
    def goal_state(self) -> np.ndarray:
        """Reference joint state; the planner's Q_diag decides which entries matter."""
        return np.concatenate([Q_READY, np.zeros(self.n)])

    def is_done(self, state: np.ndarray) -> bool:
        """TCP at the goal pose (within tolerances) and the arm at rest."""
        pose = self.tcp_placement(state[: self.n])
        goal = pin.SE3(self.task_goal[3:].reshape(3, 3), self.task_goal[:3])
        position_error = np.linalg.norm(pose.translation - goal.translation)
        rotation_error = np.linalg.norm(pin.log3(pose.rotation.T @ goal.rotation))
        return bool(
            position_error < self.config.get("position_tolerance", 2e-3)
            and rotation_error < self.config.get("rotation_tolerance", 2e-2)
            and np.abs(state[self.n :]).max() < self.config.get("velocity_tolerance", 2e-2)
        )

    # --- kinematics and dataset

    def set_task_goal(self, pose: pin.SE3) -> None:
        self.task_goal = np.concatenate([pose.translation, pose.rotation.ravel()])

    def inverse_kinematics(self, pose: pin.SE3, q_init: np.ndarray, iters: int = 500) -> np.ndarray:
        """Damped least-squares IK for the TCP, used to get a joint-space start state."""
        q = q_init.copy()
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        for _ in range(iters):
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = pin.log6(self.data.oMf[self.tcp_id].actInv(pose)).vector
            if np.linalg.norm(err) < 1e-6:
                return q
            J = pin.getFrameJacobian(self.model, self.data, self.tcp_id, pin.ReferenceFrame.LOCAL)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
            q = np.clip(pin.integrate(self.model, q, dq), lower, upper)
        raise RuntimeError("IK did not converge; pose may be unreachable.")

    def sample_start_goal(self, rng: np.random.Generator) -> tuple[np.ndarray, pin.SE3]:
        """Random rest state and goal TCP pose; both come from FK of sampled joint
        configurations, so the goal is reachable by construction.

        Poses must be in front of the base, above the table plane and at least
        ``min_start_goal_distance`` apart. Fingers start at the open width.
        """
        lower, upper = self.model.lowerPositionLimit[:7], self.model.upperPositionLimit[:7]
        margin = 0.1 * (upper - lower)
        min_distance = self.config.get("min_start_goal_distance", 0.25)

        def sample_config() -> tuple[np.ndarray, pin.SE3]:
            while True:
                q = np.concatenate([rng.uniform(lower + margin, upper - margin), Q_READY[7:]])
                pose = self.tcp_placement(q)
                if pose.translation[0] > 0.2 and 0.1 < pose.translation[2] < 0.8:
                    return q, pose

        q_start, start_pose = sample_config()
        while True:
            _, goal_pose = sample_config()
            if np.linalg.norm(goal_pose.translation - start_pose.translation) > min_distance:
                return np.concatenate([q_start, np.zeros(self.n)]), goal_pose

    def tcp_placement(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].copy()

    def get_dataset_features(self) -> dict[str, Any]:
        """LeRobot features: state [q, v], goal TCP pose as environment state, acceleration action."""
        joints = list(self.model.names[1:])
        return {
            "observation.state": {
                "dtype": "float32",
                "shape": (self.nx,),
                "names": [f"{j}.pos" for j in joints] + [f"{j}.vel" for j in joints],
            },
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (12,),
                "names": ["goal.x", "goal.y", "goal.z"] + [f"goal.R{i}{j}" for i in range(3) for j in range(3)],
            },
            "action": {"dtype": "float32", "shape": (self.nu,), "names": [f"{j}.acc" for j in joints]},
        }

    def format_dataset_frame(self, obs: np.ndarray, action: np.ndarray) -> dict[str, Any]:
        """One frame without ``task``, which LeRobot's add_frame requires from the caller."""
        return {
            "observation.state": np.asarray(obs, dtype=np.float32),
            "observation.environment_state": self.task_goal.astype(np.float32),
            "action": np.asarray(action, dtype=np.float32),
        }
