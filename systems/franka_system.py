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

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. A start (at rest) and
a goal are drawn in the workspace in front of the arm (``franka_sampling``): the start is a configuration, the goal a pose,
both of TCP positions and hand directions that inverse kinematics reaches well inside the joint limits and free of
spherical obstacles. The goal is reached when the TCP is within ``pos_tol`` of the goal position and the arm is at rest: the orientation is
not tested (``goal_errors`` reports it).
"""

from typing import Any, Mapping

import coal
import numpy as np
import pinocchio as pin

from .config_fields import SYSTEM_CONFIG_KEYS
from .franka_config import (
    FRANKA_NUM_JOINTS,
    POSE_COMPONENTS,
    POSE_DIM,
    validate_franka_config,
)
from .franka_kinematics import franka_kinematics
from .franka_sampling import WorkspaceSampler
from .franka_model import Q_READY, TCP_FRAME, build_model, build_robot
from .triple_integrator import PlantLimits, TripleIntegratorBase


# Sampled configurations stay JOINT_SAMPLING_MARGIN (a fraction of the range) inside the joint limits, with every joint
# frame at least FLOOR_CLEARANCE above the floor (z = 0)
JOINT_SAMPLING_MARGIN = 0.1
FLOOR_CLEARANCE = 0.1


class FrankaSystem(TripleIntegratorBase):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a TCP pose."""

    task_description = "reach target pose"

    @classmethod
    def config_keys(cls) -> set[str]:
        return SYSTEM_CONFIG_KEYS

    @classmethod
    def validate_config(cls, raw_config: Mapping[str, Any]) -> dict[str, Any]:
        return validate_franka_config(raw_config)

    def __init__(self, config: Mapping[str, Any]):
        model = build_model()
        n = model.nv
        if n != FRANKA_NUM_JOINTS:
            raise ValueError(f"Expected {FRANKA_NUM_JOINTS} joints, the reduced franka_description has {n}.")
        kinematics = franka_kinematics()
        ready_position, ready_rotation = kinematics.fk(Q_READY)
        fields = self.validate_config(config)
        super().__init__(
            config,
            fields,
            limits=PlantLimits(
                n=n,
                lower_position=model.lowerPositionLimit,
                upper_position=model.upperPositionLimit,
                max_velocity=model.velocityLimit,
                max_acceleration=np.asarray(fields["max_acceleration"], dtype=float),
                max_jerk=np.asarray(fields["max_jerk"], dtype=float),
            ),
            default_goal=np.concatenate([np.array(ready_position).ravel(), np.array(ready_rotation).ravel()]),
        )
        self.model = model
        self.data = model.createData()
        self.kinematics = kinematics
        self.tcp_id = model.getFrameId(TCP_FRAME)
        self.sampler = WorkspaceSampler(
            self.kinematics.inverse_kinematics,
            self.kinematics.lower,
            self.kinematics.upper,
            self._is_valid,
        )
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

    def policy_observation(self, obs: np.ndarray, history: np.ndarray) -> dict[str, np.ndarray]:
        """The goal pose in the TCP frame (``T_tcp^-1 T_goal``) and the centers of the arm's bounding spheres in the base
        frame, the state, and the history of positions."""
        obs_array = np.asarray(obs, dtype=float)
        goal, state = obs_array[:POSE_DIM], obs_array[POSE_DIM:]
        relative = self.tcp_placement(state[: self.n]).actInv(self.vector_pose(goal))
        sphere_centers = self.collision_spheres(state[: self.n])[0]
        return {
            "observation.environment_state": np.concatenate([self.pose_vector(relative), sphere_centers.ravel()]).astype(
                np.float32
            ),
            "observation.state": state.astype(np.float32),
            "observation.history": np.asarray(history, dtype=np.float32),
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
        """The pose error of the TCP to the goal as the logarithm ``xi = log(T^-1 T_goal)`` in se(3) (translation part
        ``rho``, rotation vector ``phi``, in the TCP frame, like the policy observes the goal), its 6 x n Jacobian in ``q``
        with ``xi(q + dq) = xi(q) - J dq`` to first order, and the distance [m] of the TCP to the goal position.

        With the twist ``nu = J_b dq`` of the TCP in its own frame (the body Jacobian), ``T`` moves as ``T exp(nu)``, so
        ``T^-1 T_goal`` moves as ``exp(-nu) T^-1 T_goal``, a left perturbation: ``d xi = -J_l(xi)^-1 nu`` (``J_l`` the left
        Jacobian of SE(3)). ``rho`` is the translation of the screw that carries the TCP onto the goal, which couples to
        the rotation; it equals the position error in the TCP frame when there is no rotation error.
        """
        q = np.asarray(q, dtype=float)[: self.n]
        placement, target = self.tcp_placement(q), self.vector_pose(goal)
        relative = placement.actInv(target)
        error = np.array(pin.log6(relative).vector)
        body_jacobian = np.array(pin.computeFrameJacobian(self.model, self.data, q, self.tcp_id, pin.LOCAL))
        jacobian = np.array(pin.Jlog6(relative.inverse())) @ body_jacobian
        return error, jacobian, float(np.linalg.norm(relative.translation))

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position of the TCP in configuration ``q`` (what the viewers draw)."""
        return self.tcp_placement(q).translation

    def goal_transform(self, goal: np.ndarray) -> np.ndarray:
        """The 4x4 homogeneous transform of the goal TCP pose."""
        return self.vector_pose(goal).homogeneous

    def collision_spheres(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (S, 3), position Jacobians (S, 3, n), and radii (S,) at configuration ``q``."""
        return self.kinematics.collision_spheres(np.asarray(q, dtype=float)[: self.n])

    def collision_spheres_along(self, q_trajectory: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bounding sphere centers (H, S, 3), position Jacobians (H, S, 3, n), and radii (S,) at every row of ``q_trajectory``."""
        return self.kinematics.collision_spheres_along(np.asarray(q_trajectory, dtype=float)[:, : self.n])

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
        return self.sampler.configuration(rng)

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        return self.sampler.goal(rng)

    def _is_valid(self, q: np.ndarray) -> bool:
        """Inside the joint margins, every joint frame at least ``FLOOR_CLEARANCE`` above the floor and no obstacle touched."""
        margin = JOINT_SAMPLING_MARGIN * (self.model.upperPositionLimit - self.model.lowerPositionLimit)
        if np.any(q < self.model.lowerPositionLimit + margin) or np.any(q > self.model.upperPositionLimit - margin):
            return False
        self.tcp_placement(q)
        return all(joint.translation[2] >= FLOOR_CLEARANCE for joint in self.data.oMi[1:]) and self.collision.is_collision_free(q)
