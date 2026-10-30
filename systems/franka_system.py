"""Franka FR3 arm (7 joints) with the gripper as fixed geometry.

State  x = [q, v, a]  (joint positions, velocities, accelerations), nx = 21
Action u = j          (joint jerks), nu = 7
Goal   g = target TCP pose [position (3), rotation matrix row-major (9)], 12 numbers
Observation o = [g, x] (obs_dim = 33), what the planners read. The policy sees the goal in the TCP frame instead,
T_rel = T_tcp^-1 T_goal as [p_rel, R_rel] with p_rel = R^T (p_d - p) and R_rel = R^T R_d (``policy_observation``, the same
12-number encoding): the LeRobot frame stores it as ``observation.environment_state`` and x as ``observation.state``.
The policy sees no joint configuration of the goal: inverse kinematics is the experts' privilege (the planners reach it
through ``goal_configuration``), so it has to learn to reach a pose from the pose itself. Next to the goal it sees where
its own body is: the centers of its bounding spheres in the base frame (forward kinematics of the state, no obstacle
information), which lets it relate its body to fixed obstacles without perceiving them.

Kinematic model: inherits the triple-integrator plant from ``TripleIntegratorBase`` with SafeFlowMPC's FR3 limits (the
joint position limits narrowed by ``JOINT_LIMIT_MARGIN``, the velocity limits scaled by ``VELOCITY_LIMIT_SCALE``,
acceleration and jerk limits from the config). No masses or torques: a joint tracking controller is assumed to
follow the motion.

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. Start configurations
are sampled well inside the joint limits and free of spherical obstacles, goals as the TCP pose of such a configuration.
The goal is reached when the TCP is within ``pos_tol`` of the goal position and the arm is at rest: the orientation is
not tested (``goal_errors`` reports it).
"""

from typing import Any, Mapping

import coal
import numpy as np
import pinocchio as pin

from .franka_kinematics import franka_kinematics
from .franka_model import Q_READY, TCP_FRAME, build_model, build_robot
from .triple_integrator import SAMPLING_ATTEMPTS, TripleIntegratorBase

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7

# Random start/goal configurations stay inside the joint margins, within SAMPLING_RADIUS of Q_READY, with the TCP in
# WORKSPACE_BOUNDS in front of the base and every joint frame at least FLOOR_CLEARANCE above the floor (z = 0)
JOINT_SAMPLING_MARGIN = 0.1
SAMPLING_RADIUS = 1.0
WORKSPACE_BOUNDS = ((0.2, 0.8), (-0.6, 0.6), (0.1, 0.8))
FLOOR_CLEARANCE = 0.1

POSE_COMPONENTS = ["x", "y", "z"] + [f"R{i}{j}" for i in range(3) for j in range(3)]
POSE_DIM = len(POSE_COMPONENTS)


class FrankaSystem(TripleIntegratorBase):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a TCP pose."""

    task_description = "reach target pose"

    def __init__(self, config: Mapping[str, Any]):
        model = build_model()
        n = model.nv
        if n != FRANKA_NUM_JOINTS:
            raise ValueError(f"Expected {FRANKA_NUM_JOINTS} joints, the reduced franka_description has {n}.")
        kinematics = franka_kinematics()
        ready_position, ready_rotation = kinematics.fk(Q_READY)
        per_joint = lambda key, default: np.broadcast_to(np.asarray(config.get(key, default), dtype=float), (n,)).copy()
        super().__init__(
            config,
            n=n,
            lower_position=model.lowerPositionLimit,
            upper_position=model.upperPositionLimit,
            max_velocity=model.velocityLimit,
            max_acceleration=per_joint("max_acceleration", FRANKA_DEFAULT_MAX_ACCELERATION),
            max_jerk=per_joint("max_jerk", FRANKA_DEFAULT_MAX_JERK),
            default_goal=np.concatenate([np.array(ready_position).ravel(), np.array(ready_rotation).ravel()]),
            default_min_distance=0.25,
            default_pos_tol=0.03,
            default_vel_tol=0.05,
        )
        self.model = model
        self.data = model.createData()
        self.kinematics = kinematics
        self.tcp_id = model.getFrameId(TCP_FRAME)
        self._mesh_collision: tuple | None = None  # (obstacles, geometry model, geometry data), built on first use
        self._cached_goal: np.ndarray | None = None  # the goal whose joint configuration is cached, and that configuration
        self._cached_q_goal: np.ndarray | None = None

    @property
    def environment_dim(self) -> int:
        return POSE_DIM + 3 * len(self.kinematics.sphere_radii)

    def reset(self, initial_state: np.ndarray) -> np.ndarray:
        """Reset for a new rollout. The cached joint configuration of the goal is dropped, so one rollout never reuses
        what was solved for another."""
        state = super().reset(initial_state)
        self._cached_goal = None
        self._cached_q_goal = None
        return state

    def collides(self, state: np.ndarray) -> bool:
        """True if a collision mesh of the arm (the real geometry, hand and fingers included) touches an obstacle. The
        bounding spheres of the planners cover the meshes, so this is the ground truth they approximate."""
        key = tuple((tuple(position), radius) for position, radius in self.collision.obstacles)
        if self._mesh_collision is None or self._mesh_collision[0] != key:
            geometry_model = build_robot().collision_model
            robot_objects = len(geometry_model.geometryObjects)
            for index, (position, radius) in enumerate(self.collision.obstacles):
                geometry_model.addGeometryObject(pin.GeometryObject(f"obstacle{index}", 0, pin.SE3(np.eye(3), position), coal.Sphere(radius)))
                for robot_object in range(robot_objects):
                    geometry_model.addCollisionPair(pin.CollisionPair(robot_object, robot_objects + index))
            self._mesh_collision = (key, geometry_model, pin.GeometryData(geometry_model))
        _, geometry_model, geometry_data = self._mesh_collision
        return bool(
            self.collision.obstacles
            and pin.computeCollisions(self.model, self.data, geometry_model, geometry_data, np.asarray(state, dtype=float)[: self.n], True)
        )

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """The goal pose in the TCP frame (``T_tcp^-1 T_goal``), the centers of the arm's bounding spheres in the base
        frame, and the state."""
        obs_array = np.asarray(obs, dtype=float)
        goal, state = obs_array[:POSE_DIM], obs_array[POSE_DIM:]
        relative = self.tcp_placement(state[: self.n]).actInv(self.vector_pose(goal))
        sphere_centers = self.collision_spheres(state[: self.n])[0]
        return {
            "observation.environment_state": np.concatenate([self.pose_vector(relative), sphere_centers.ravel()]).astype(
                np.float32
            ),
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

    def goal_task_error(self, q: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """The TCP position error and the rotation vector of the orientation error (world frame), and the 6 x n
        Jacobian of the TCP in the world-aligned frame, in ``q``."""
        q = np.asarray(q, dtype=float)[: self.n]
        placement, target = self.tcp_placement(q), self.vector_pose(goal)
        position_error = target.translation - placement.translation
        error = np.concatenate([position_error, pin.log3(target.rotation @ placement.rotation.T)])
        jacobian = pin.computeFrameJacobian(self.model, self.data, q, self.tcp_id, pin.LOCAL_WORLD_ALIGNED)
        return error, np.array(jacobian), float(np.linalg.norm(position_error))

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
        """The collision-free joint configuration of the ``goal`` pose (cached per goal)."""
        pose = self.vector_pose(goal)
        q_ref = np.clip(q_reference, self.kinematics.lower, self.kinematics.upper)
        if self.kinematics.reaches_pose(q_ref, pose.translation, pose.rotation) and self.collision.is_collision_free(q_ref):
            return q_ref
        if self._cached_goal is not None and np.array_equal(goal, self._cached_goal):
            return self._cached_q_goal.copy()
        solution = self.kinematics.solve_ik(pose.translation, pose.rotation, q_ref, self.collision.is_collision_free)
        self._cached_goal, self._cached_q_goal = np.asarray(goal, dtype=float).copy(), solution.copy()
        return solution

    # --- dataset metadata and sampling ---------------------------------------

    @property
    def coordinate_names(self) -> list[str]:
        return list(self.model.names[1:])

    @property
    def environment_state_names(self) -> list[str]:
        return [f"goal_in_tcp.{component}" for component in POSE_COMPONENTS] + [
            f"collision_sphere_{sphere}.{axis}" for sphere in range(len(self.kinematics.sphere_radii)) for axis in "xyz"
        ]

    @property
    def state_names(self) -> list[str]:
        return [f"{name}.{kind}" for kind in ("pos", "vel", "acc") for name in self.coordinate_names]

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
                and self.collision.is_collision_free(q)
            ):
                return q
        raise RuntimeError(f"Unable to sample an arm configuration in the workspace in {SAMPLING_ATTEMPTS} attempts.")

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        """The TCP pose of a random configuration (so it is reachable)."""
        return self.pose_vector(self.tcp_placement(self._sample_configuration(rng)))
