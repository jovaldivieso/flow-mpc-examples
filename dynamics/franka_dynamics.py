"""Franka FR3 arm (7 joints) with the gripper as fixed geometry.

State  x = [q, v, a]  (joint positions, velocities, accelerations), nx = 21
Action u = j          (joint jerks), nu = 7
Goal   g = target joint configuration q_goal
Observation o = [g, x] (obs_dim = 28); the LeRobot frame stores g as ``observation.environment_state`` and x as
``observation.state``.

Kinematic model: inherits the triple-integrator plant from ``TripleIntegrator`` with SafeFlowMPC's FR3 limits (the
joint position limits narrowed by ``JOINT_LIMIT_MARGIN``, the velocity limits scaled by ``VELOCITY_LIMIT_SCALE``,
acceleration and jerk limits from the config). No masses or torques: a joint tracking controller is assumed to
follow the motion.

The model comes from the franka_description xacro via Pinocchio, the finger joints locked open. Start and goal
configurations are sampled well inside the joint limits, and there is no collision checking.
"""

import functools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

from .triple_integrator import TripleIntegrator

FRANKA_NUM_JOINTS = 7
FRANKA_DEFAULT_MAX_ACCELERATION = (5.0,) * 7
FRANKA_DEFAULT_MAX_JERK = (15.0,) * 7

# SafeFlowMPC's joint limits: positions 3 degrees inside the description's, velocities at 92 %
JOINT_LIMIT_MARGIN = np.deg2rad(3.0)
VELOCITY_LIMIT_SCALE = 0.92
# Random start/goal configurations keep this fraction of every joint range away from the limits
JOINT_SAMPLING_MARGIN = 0.1

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


class FrankaDynamics(TripleIntegrator):
    """Triple-integrator 7-joint FR3 with jerk as the action; the goal is a joint configuration."""

    task_description = "reach target configuration"

    def __init__(self, config: Mapping[str, Any]):
        # Call DynamicsSimulator.__init__ first, then configure the 7-DoF FR3 bounds on TripleIntegrator
        super(TripleIntegrator, self).__init__(config)
        self.model = build_model()
        self.data = self.model.createData()
        self._robot: RobotWrapper | None = None
        self.tcp_id = self.model.getFrameId(TCP_FRAME)
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
            default_goal=Q_READY,
            default_min_distance=0.25,
            default_pos_tol=2e-3,
            default_vel_tol=2e-2,
        )

    @property
    def robot(self) -> RobotWrapper:
        """Model with collision and visual geometry, built on first use (for Meshcat)."""
        if self._robot is None:
            self._robot = build_robot()
        return self._robot

    def tcp_placement(self, q: np.ndarray) -> pin.SE3:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.tcp_id].copy()

    def tcp_position(self, q: np.ndarray) -> np.ndarray:
        """The 3-D position of the TCP in configuration ``q`` (what the viewers draw)."""
        return self.tcp_placement(q).translation

    def _coordinate_names(self) -> list[str]:
        return list(self.model.names[1:])

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        """Random arm configuration inside the joint limits (by ``JOINT_SAMPLING_MARGIN``)."""
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        margin = JOINT_SAMPLING_MARGIN * (upper - lower)
        return rng.uniform(lower + margin, upper - margin)
