"""Franka FR3 arm (7 joints) with the gripper as fixed geometry.

State  x = [q, v, a]  (joint positions, velocities, accelerations), nx = 21
Action u = j          (joint jerks), nu = 7
Goal   g = TCP pose [position(3), rotation matrix row-major(9)]
Observation o = [g, x] (obs_dim = 33); the LeRobot frame stores g as ``observation.environment_state`` and x as
``observation.state``.

Kinematic model: every joint is a triple integrator with piecewise constant jerk, with SafeFlowMPC's limits (the
joint position limits narrowed by ``JOINT_LIMIT_MARGIN``, the velocity limits scaled by ``VELOCITY_LIMIT_SCALE``,
acceleration and jerk limits from the config). No masses or torques: a joint tracking controller is assumed to
follow the motion.

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. The only collision
checked is the floor (z = 0, the plane the base stands on): the joint frames and the TCP keep ``floor_clearance``
above it. Start and goal configurations are sampled well inside the joint limits, where the arm hardly ever hits
itself, so self-collision is not checked.
"""

import functools
from pathlib import Path
from typing import Any, Mapping

import casadi as ca
import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

from .dynamics import DynamicsSimulator, constant_jerk_step, saturated_step

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7
# Goal = TCP pose [position(3), rotation matrix row-major(9)]
FRANKA_GOAL_DIM = 12
# Cartesian TCP box [[x_min, x_max], [y_min, y_max], [z_min, z_max]] both random goals and random start poses are drawn from
FRANKA_DEFAULT_WORKSPACE_BOUNDS = ((0.2, 1.0), (-1.0, 1.0), (0.1, 0.8))
FRANKA_DEFAULT_FLOOR_CLEARANCE = 0.05

# SafeFlowMPC's joint limits: positions 3 degrees inside the description's, velocities at 92 %
JOINT_LIMIT_MARGIN = np.deg2rad(3.0)
VELOCITY_LIMIT_SCALE = 0.92
# Random start/goal configurations keep this fraction of every joint range away from the limits
JOINT_SAMPLING_MARGIN = 0.1
SAMPLING_ATTEMPTS = 256

# Official franka_description, pinned as a git submodule
FRANKA_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"
XACRO_PATH = FRANKA_DESCRIPTION / "robots" / "fr3" / "fr3.urdf.xacro"

# xacro resolves $(find pkg) via ament_index_python (ROS-only, not on PyPI)
substitution_args._eval_find = lambda pkg: str(FRANKA_DESCRIPTION)

# The finger joints are locked at this opening, the hand and fingers are fixed geometry
FINGER_OPEN = 0.04
FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
TCP_FRAME = "fr3_hand_tcp"


@functools.lru_cache(maxsize=None)
def _urdf() -> str:
    return xacro.process_file(XACRO_PATH, mappings={"hand": "true"}).toxml()


def _finger_locks(full_model: pin.Model) -> tuple[list[int], np.ndarray]:
    """Ids of the finger joints and the full 9-D configuration (arm at its ready pose, fingers open) they are locked at."""
    ids = [full_model.getJointId(name) for name in FINGER_JOINTS]
    return ids, np.concatenate([Q_READY, [FINGER_OPEN] * len(FINGER_JOINTS)])


def build_model() -> pin.Model:
    """The 7-DoF FR3 with the finger joints locked open and the limits of SafeFlowMPC (kinematics only)."""
    full = pin.buildModelFromXML(_urdf())
    ids, reference = _finger_locks(full)
    model = pin.buildReducedModel(full, ids, reference)
    model.lowerPositionLimit = model.lowerPositionLimit + JOINT_LIMIT_MARGIN
    model.upperPositionLimit = model.upperPositionLimit - JOINT_LIMIT_MARGIN
    model.velocityLimit = VELOCITY_LIMIT_SCALE * model.velocityLimit
    return model


def build_robot() -> RobotWrapper:
    """The 7-DoF FR3 with collision and visual geometry (for Meshcat), the hand and fingers fixed on the last joint."""
    urdf = _urdf()
    # meshes are package://franka_description/..., resolved from the clone's parent
    pkg = [str(FRANKA_DESCRIPTION.parent)]
    full = pin.buildModelFromXML(urdf)
    visual = pin.buildGeomFromUrdfString(full, urdf, pin.GeometryType.VISUAL, package_dirs=pkg)
    collision = pin.buildGeomFromUrdfString(full, urdf, pin.GeometryType.COLLISION, package_dirs=pkg)
    ids, reference = _finger_locks(full)
    model, (collision, visual) = pin.buildReducedModel(full, [collision, visual], ids, reference)
    return RobotWrapper(model, collision, visual)


class FrankaDynamics(DynamicsSimulator):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a TCP pose, the floor is the only obstacle."""

    task_description = "reach target pose"

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.model = build_model()
        self.data = self.model.createData()
        self._robot: RobotWrapper | None = None
        self.tcp_id = self.model.getFrameId(TCP_FRAME)
        self.n = self.model.nv
        if self.n != FRANKA_NUM_JOINTS:
            raise ValueError(f"Expected {FRANKA_NUM_JOINTS} joints, the reduced franka_description has {self.n}.")

        self.nx = 3 * self.n
        self.nu = self.n
        self.obs_dim = FRANKA_GOAL_DIM + self.nx
        per_joint = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (self.n,)).copy()
        self.max_acceleration = per_joint("max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION)
        self.max_action = per_joint("max_jerk", FRANKA_DEFAULT_MAX_JERK)
        self.state_lower_bounds = np.concatenate([self.model.lowerPositionLimit, -self.model.velocityLimit, -self.max_acceleration])
        self.state_upper_bounds = np.concatenate([self.model.upperPositionLimit, self.model.velocityLimit, self.max_acceleration])

        # Goal = TCP pose as [position(3), rotation matrix row-major(9)]
        self.goal = np.asarray(config["goal"], dtype=float) if "goal" in config else self.pose_to_goal(self.tcp_placement(Q_READY))
        self.randomize_goal = config.get("randomize_goal", "goal" not in config)
        self.workspace_bounds = np.asarray(config.get("workspace_bounds", FRANKA_DEFAULT_WORKSPACE_BOUNDS), dtype=float)
        self.min_start_goal_distance = float(config.get("min_start_goal_distance", 0.25))
        self.pos_tol = float(config.get("pos_tol", 2e-3))
        self.rot_tol = float(config.get("rot_tol", 2e-2))
        self.vel_tol = float(config.get("vel_tol", 2e-2))
        self.floor_clearance = float(config.get("floor_clearance", FRANKA_DEFAULT_FLOOR_CLEARANCE))

    @property
    def rest_state_indices(self) -> tuple[int, ...]:
        return tuple(range(self.n, self.nx))  # velocity and acceleration

    @property
    def robot(self) -> RobotWrapper:
        """Model with collision and visual geometry, built on first use (for Meshcat)."""
        if self._robot is None:
            self._robot = build_robot()
        return self._robot

    def casadi_dynamics(self, x: Any, u: Any) -> Any:
        """Exact step under constant jerk, matching ``predict_next_state`` away from the limits."""
        return ca.vertcat(*constant_jerk_step(x[: self.n], x[self.n : 2 * self.n], x[2 * self.n :], u, self.dt))

    # --- numeric simulation ------------------------------------------------

    def predict_next_state(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        action_array = self.validate_action(action) if validate else np.asarray(action, dtype=float)
        return saturated_step(
            state_array, action_array, self.dt, self.state_lower_bounds, self.state_upper_bounds, self.max_action
        )

    def step(self, state: np.ndarray, action: np.ndarray, validate: bool = True) -> np.ndarray:
        return self.predict_next_state(state, action, validate=validate)

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """TCP position error [m] and orientation error [rad] of ``state`` with respect to the goal pose."""
        pose = self.tcp_placement(np.asarray(state, dtype=float)[: self.n])
        goal = self.goal_pose
        return float(np.linalg.norm(pose.translation - goal.translation)), float(np.linalg.norm(pin.log3(pose.rotation.T @ goal.rotation)))

    def is_done(self, state: np.ndarray, validate: bool = True) -> bool:
        """TCP at the goal pose (within tolerances) and the arm at rest."""
        state_array = self.validate_state(state) if validate else np.asarray(state, dtype=float)
        position_error, rotation_error = self.goal_errors(state_array)
        return bool(
            position_error < self.pos_tol
            and rotation_error < self.rot_tol
            and np.abs(state_array[self.n : 2 * self.n]).max() < self.vel_tol
        )

    def clearance(self, state: np.ndarray) -> float:
        """Height above the floor of the lowest joint frame or the TCP, for a state or just the joint positions."""
        q = np.asarray(state, dtype=float)[: self.n]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        heights = [placement.translation[2] for placement in self.data.oMi[1:]]
        return float(min(heights + [self.data.oMf[self.tcp_id].translation[2]]))

    def is_collision(self, state: np.ndarray) -> bool:
        return self.clearance(state) < self.floor_clearance

    # --- kinematics, goal and dataset -----------------------------------------

    @staticmethod
    def pose_to_goal(pose: pin.SE3) -> np.ndarray:
        return np.concatenate([pose.translation, pose.rotation.ravel()])

    @staticmethod
    def goal_to_pose(goal: np.ndarray) -> pin.SE3:
        goal = np.asarray(goal, dtype=float)
        return pin.SE3(goal[3:].reshape(3, 3), goal[:3])

    @property
    def goal_pose(self) -> pin.SE3:
        return self.goal_to_pose(self.goal)

    def tcp_placement(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position of the TCP in configuration ``q`` (what the viewers draw)."""
        return self.tcp_placement(q).translation

    def inverse_kinematics(self, pose: pin.SE3, q_init: np.ndarray, iters: int = 500) -> np.ndarray:
        """Damped least-squares IK for the TCP."""
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

    def nearest_inverse_kinematics(
        self, pose: pin.SE3, q_reference: np.ndarray, attempts: int = 30, solutions: int = 5
    ) -> np.ndarray:
        """The joint configuration of ``pose`` nearest to ``q_reference``: the solution from ``q_reference`` itself if
        it converges, else the nearest of the first few solutions from random configurations within the limits.
        Raises ``RuntimeError`` if none converges."""
        try:
            return self.inverse_kinematics(pose, q_reference)
        except RuntimeError:
            pass
        rng = np.random.default_rng(0)
        found: list[np.ndarray] = []
        for _ in range(attempts):
            try:
                found.append(self.inverse_kinematics(pose, rng.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit)))
            except RuntimeError:
                continue
            if len(found) == solutions:
                break
        if not found:
            raise RuntimeError("IK did not converge from any start; pose may be unreachable.")
        return min(found, key=lambda q: float(np.linalg.norm(q - q_reference)))

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        """LeRobot features: state [q, v, a], goal TCP pose as environment state, the expert's planned jerks as action
        (its next ``prediction_horizon`` jerks, flattened row by row)."""
        joints = list(self.model.names[1:])
        return {
            "observation.environment_state": {
                "dtype": "float32",
                "shape": (FRANKA_GOAL_DIM,),
                "names": ["goal.x", "goal.y", "goal.z"] + [f"goal.R{i}{j}" for i in range(3) for j in range(3)],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (self.nx,),
                "names": [f"{j}.{kind}" for kind in ("pos", "vel", "acc") for j in joints],
            },
            "action": {
                "dtype": "float32",
                "shape": (prediction_horizon * self.nu,),
                "names": [f"{j}.jerk[t+{k}]" for k in range(prediction_horizon) for j in joints],
            },
        }

    def _sample_configuration(self, rng: np.random.Generator) -> tuple[np.ndarray, pin.SE3]:
        """Random arm configuration inside the joint limits (by ``JOINT_SAMPLING_MARGIN``), with its TCP in the
        workspace box and clear of the floor."""
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        margin = JOINT_SAMPLING_MARGIN * (upper - lower)
        for _ in range(SAMPLING_ATTEMPTS):
            q = rng.uniform(lower + margin, upper - margin)
            pose = self.tcp_placement(q)
            inside = np.all(self.workspace_bounds[:, 0] <= pose.translation) and np.all(pose.translation <= self.workspace_bounds[:, 1])
            if inside and not self.is_collision(q):
                return q, pose
        raise RuntimeError(f"Unable to sample an arm configuration in the workspace in {SAMPLING_ATTEMPTS} attempts.")

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        """Rest state from a random configuration at least ``min_start_goal_distance`` from the goal."""
        for _ in range(SAMPLING_ATTEMPTS):
            q, pose = self._sample_configuration(rng)
            if np.linalg.norm(pose.translation - self.goal[:3]) > self.min_start_goal_distance:
                return np.concatenate([q, np.zeros(2 * self.n)])
        raise RuntimeError(
            f"Unable to sample a start configuration {self.min_start_goal_distance} m away from the goal "
            f"in {SAMPLING_ATTEMPTS} attempts."
        )

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        """Goal TCP pose from the forward kinematics of a random configuration, so it is reachable by construction."""
        if self.randomize_goal:
            _, pose = self._sample_configuration(rng)
            self.set_goal(self.pose_to_goal(pose))
