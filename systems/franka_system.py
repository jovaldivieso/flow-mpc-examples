"""Franka FR3 arm (7 joints) with the gripper as fixed geometry.

State  x = [q, v, a]  (joint positions, velocities, accelerations), nx = 21
Action u = j          (joint jerks), nu = 7
Goal   g = target TCP pose [position (3), rotation matrix row-major (9)], 12 numbers
Observation o = [g, x] (obs_dim = 33), what the planners read. The policy sees the goal in the TCP frame instead,
T_rel = T_tcp^-1 T_goal as [p_rel, R_rel] with p_rel = R^T (p_d - p) and R_rel = R^T R_d (``policy_observation``, the same
12-number encoding): the LeRobot frame stores it as ``observation.environment_state`` and x as ``observation.state``.

Kinematic model: inherits the triple-integrator plant from ``TripleIntegratorSystem`` with SafeFlowMPC's FR3 limits (the
joint position limits narrowed by ``JOINT_LIMIT_MARGIN``, the velocity limits scaled by ``VELOCITY_LIMIT_SCALE``,
acceleration and jerk limits from the config). No masses or torques: a joint tracking controller is assumed to
follow the motion.

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. Start configurations
are sampled well inside the joint limits and free of spherical obstacles, goals as the TCP pose of such a configuration.
The goal is reached when the TCP is within ``pos_tol`` of the goal position and the arm is at rest: the orientation is
not tested (``goal_errors`` reports it).
"""

import functools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

from .franka_kinematics import IK_TOLERANCE, FrankaKinematics
from .triple_integrator_system import SAMPLING_ATTEMPTS, TripleIntegratorSystem

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7

# SafeFlowMPC's joint limits: positions 3 degrees inside the description's, velocities at 92 %
JOINT_LIMIT_MARGIN = np.deg2rad(3.0)
VELOCITY_LIMIT_SCALE = 0.92
# Random start/goal configurations stay inside the joint margins, within SAMPLING_RADIUS of Q_READY, with the TCP in
# WORKSPACE_BOUNDS in front of the base and every joint frame at least FLOOR_CLEARANCE above the floor (z = 0)
JOINT_SAMPLING_MARGIN = 0.1
SAMPLING_RADIUS = 1.0
WORKSPACE_BOUNDS = ((0.2, 0.8), (-0.6, 0.6), (0.1, 0.8))
FLOOR_CLEARANCE = 0.1

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
# A pose the inverse kinematics does not reach from the first start is tried from random configurations
IK_ATTEMPTS = 30
IK_SOLUTIONS = 5
POSE_COMPONENTS = ["x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)]
POSE_DIM = len(POSE_COMPONENTS)


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


@functools.lru_cache(maxsize=None)
def franka_kinematics() -> FrankaKinematics:
    """The CasADi kinematics of the TCP (forward kinematics and IK), built once per process for every user."""
    model = build_model()
    return FrankaKinematics(model, model.getFrameId(TCP_FRAME))


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


class FrankaSystem(TripleIntegratorSystem):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a TCP pose."""

    task_description = "reach target pose"

    def __init__(self, config: Mapping[str, Any]):
        # Call SystemSimulator.__init__ first, then configure the 7-DoF FR3 bounds on TripleIntegratorSystem
        super(TripleIntegratorSystem, self).__init__(config)
        self.model = build_model()
        self.data = self.model.createData()
        self._robot: RobotWrapper | None = None
        self.tcp_id = self.model.getFrameId(TCP_FRAME)
        self.kinematics = franka_kinematics()
        n = self.model.nv
        if n != FRANKA_NUM_JOINTS:
            raise ValueError(f"Expected {FRANKA_NUM_JOINTS} joints, the reduced franka_description has {n}.")

        per_joint = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (n,)).copy()
        self._init_triple_integrator(
            n=n,
            lower_position=self.model.lowerPositionLimit,
            upper_position=self.model.upperPositionLimit,
            max_velocity=self.model.velocityLimit,
            max_acceleration=per_joint("max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION),
            max_jerk=per_joint("max_jerk", FRANKA_DEFAULT_MAX_JERK),
            default_goal=self.pose_vector(self.tcp_placement(Q_READY)),
            default_min_distance=0.25,
            default_pos_tol=0.03,
            default_vel_tol=0.05,
        )
        self.obs_dim = POSE_DIM + self.nx
        self._cached_goal: np.ndarray | None = None
        self._cached_q_goal: np.ndarray | None = None

    @property
    def environment_dim(self) -> int:
        return POSE_DIM + self.n

    @property
    def robot(self) -> RobotWrapper:
        """Model with collision and visual geometry, built on first use (for Meshcat)."""
        if self._robot is None:
            self._robot = build_robot()
        return self._robot

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """The goal pose in the TCP frame (``T_tcp^-1 T_goal``), the joint configuration error ``q_goal - q``, and the
        state."""
        obs_array = np.asarray(obs, dtype=float)
        goal, state = obs_array[:POSE_DIM], obs_array[POSE_DIM:]
        q = state[: self.n]
        relative = self.tcp_placement(q).actInv(self.vector_pose(goal))
        q_rel = self.goal_configuration(goal, q) - q
        return {
            "observation.environment_state": np.concatenate([self.pose_vector(relative), q_rel]).astype(np.float32),
            "observation.state": state.astype(np.float32),
        }

    def observation_from_policy(self, environment_state: np.ndarray, state: np.ndarray) -> np.ndarray:
        """The observation ``[goal, state]`` of a policy observation: the goal in the TCP frame in the given state
        gives the goal pose, ``T_goal = T_tcp T_rel``."""
        state_array = np.asarray(state, dtype=float)
        goal = self.tcp_placement(state_array[: self.n]) * self.vector_pose(
            np.asarray(environment_state, dtype=float)[:POSE_DIM]
        )
        return np.concatenate([self.pose_vector(goal), state_array])

    def goal_errors(self, state: np.ndarray) -> tuple[float, float]:
        """TCP position error [m] and orientation error [rad] of ``state`` with respect to the goal pose."""
        pose, goal = self.tcp_placement(np.asarray(state, dtype=float)[: self.n]), self.vector_pose(self.goal)
        position_error = np.linalg.norm(pose.translation - goal.translation)
        return float(position_error), float(np.linalg.norm(pin.log3(pose.rotation.T @ goal.rotation)))

    # --- kinematics ----------------------------------------------------------

    @staticmethod
    def pose_vector(pose: pin.SE3) -> np.ndarray:
        """A pose as ``[position, rotation matrix row-major]``."""
        return np.concatenate([pose.translation, pose.rotation.ravel()])

    @staticmethod
    def vector_pose(vector: np.ndarray) -> pin.SE3:
        """The pose of a ``[position, rotation matrix row-major]`` vector."""
        vector = np.asarray(vector, dtype=float)
        return pin.SE3(vector[3:].reshape(3, 3), vector[:3])

    def tcp_placement(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position of the TCP in configuration ``q`` (what the viewers draw)."""
        return self.tcp_placement(q).translation

    def goal_transform(self, goal: np.ndarray) -> np.ndarray:
        """The 4x4 homogeneous transform of the goal TCP pose."""
        return self.vector_pose(goal).homogeneous

    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (S, 3), position Jacobians (S, 3, n), and radii (S,) at configuration ``q``."""
        return self.kinematics.collision_spheres(np.asarray(q, dtype=float)[: self.n])

    def goal_configuration(self, goal: np.ndarray, q_reference: np.ndarray = Q_READY) -> np.ndarray:
        """The joint configuration of the ``goal`` pose: ``q_reference`` if it already is one, else the inverse
        kinematics solution from ``Q_READY`` (or the nearest to ``Q_READY`` from ``q_reference`` and random starts).
        Raises ``RuntimeError`` if none converges."""
        pose = self.vector_pose(goal)
        q_ref = np.clip(q_reference, self.kinematics.lower, self.kinematics.upper)
        goal_vec = np.concatenate([pose.translation, pose.rotation.ravel(order="F")])
        if float(self.kinematics._ik_cost(q_ref, goal_vec)) <= IK_TOLERANCE**2 and self.is_collision_free(q_ref):
            return q_ref
        if self._cached_goal is not None and np.array_equal(goal, self._cached_goal):
            return self._cached_q_goal.copy()
        for q_seed in (Q_READY, q_ref):
            try:
                solution = self.kinematics.inverse_kinematics(pose.translation, pose.rotation, q_seed)
                if self.is_collision_free(solution):
                    self._cached_goal, self._cached_q_goal = np.asarray(goal, dtype=float).copy(), solution.copy()
                    return solution
            except RuntimeError:
                pass
        rng = np.random.default_rng(0)
        found: list[np.ndarray] = []
        for _ in range(IK_ATTEMPTS):
            try:
                q_start = rng.uniform(self.model.lowerPositionLimit, self.model.upperPositionLimit)
                candidate = self.kinematics.inverse_kinematics(pose.translation, pose.rotation, q_start)
                if self.is_collision_free(candidate):
                    found.append(candidate)
            except RuntimeError:
                continue
            if len(found) == IK_SOLUTIONS:
                break
        if not found:
            raise RuntimeError("Inverse kinematics did not converge from any start; the pose may be unreachable.")
        solution = min(found, key=lambda q: float(np.linalg.norm(q - Q_READY)))
        self._cached_goal, self._cached_q_goal = np.asarray(goal, dtype=float).copy(), solution.copy()
        return solution

    # --- dataset and sampling ------------------------------------------------

    def get_dataset_features(self, prediction_horizon: int) -> dict[str, Any]:
        features = super().get_dataset_features(prediction_horizon)
        features["observation.environment_state"] = {
            "dtype": "float32",
            "shape": (self.environment_dim,),
            "names": [f"goal_in_tcp.{component}" for component in POSE_COMPONENTS]
            + [f"goal_rel.{name}" for name in self._coordinate_names()],
        }
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (self.nx,),
            "names": [f"{name}.{kind}" for kind in ("pos", "vel", "acc") for name in self._coordinate_names()],
        }
        return features

    def _coordinate_names(self) -> list[str]:
        return list(self.model.names[1:])

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Random arm configuration near ``Q_READY`` inside the joint limits, in front of the base and above the floor."""
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        margin = JOINT_SAMPLING_MARGIN * (upper - lower)
        sample_low = np.maximum(lower + margin, Q_READY - SAMPLING_RADIUS)
        sample_high = np.minimum(upper - margin, Q_READY + SAMPLING_RADIUS)
        (x_min, x_max), (y_min, y_max), (z_min, z_max) = WORKSPACE_BOUNDS
        for _ in range(SAMPLING_ATTEMPTS):
            q = rng.uniform(sample_low, sample_high)
            tcp = self.tcp_placement(q).translation
            if (
                x_min <= tcp[0] <= x_max
                and y_min <= tcp[1] <= y_max
                and z_min <= tcp[2] <= z_max
                and all(joint.translation[2] >= FLOOR_CLEARANCE for joint in self.data.oMi[1:])
                and self.is_collision_free(q)
            ):
                return q
        raise RuntimeError(f"Unable to sample an arm configuration in the workspace in {SAMPLING_ATTEMPTS} attempts.")

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """The TCP pose of a random configuration (so it is reachable)."""
        return self.pose_vector(self.tcp_placement(self._sample_configuration(rng)))
