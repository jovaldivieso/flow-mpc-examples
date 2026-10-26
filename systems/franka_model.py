"""The FR3 arm as a Pinocchio model: the franka_description xacro expanded to a URDF, the finger joints locked open, and
SafeFlowMPC's joint limits. This is the robot description and its constants; the kinematics built on it (CasADi forward
kinematics, bounding spheres, inverse kinematics) are in ``franka_kinematics``.
"""

import contextlib
import functools
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pinocchio as pin
import xacro
from pinocchio.robot_wrapper import RobotWrapper
from xacro import substitution_args

# SafeFlowMPC's joint limits: positions 3 degrees inside the description's, velocities at 92 %
JOINT_LIMIT_MARGIN = np.deg2rad(3.0)
VELOCITY_LIMIT_SCALE = 0.92

# Official franka_description, pinned as a git submodule
FRANKA_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"
XACRO_PATH = FRANKA_DESCRIPTION / "robots" / "fr3" / "fr3.urdf.xacro"

# The finger joints are locked at this opening, the hand and fingers are fixed geometry
FINGER_OPEN = 0.04
FINGER_JOINTS = ("fr3_finger_joint1", "fr3_finger_joint2")
Q_READY = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
TCP_FRAME = "fr3_hand_tcp"


@contextlib.contextmanager
def _xacro_finds_the_description() -> Generator[None]:
    """While active, xacro resolves ``$(find pkg)`` to the franka_description submodule. Xacro does it through
    ament_index_python (ROS-only, not on PyPI), so it is overridden here and restored on exit instead of patching xacro
    for the whole process."""
    original = substitution_args._eval_find
    substitution_args._eval_find = lambda pkg: str(FRANKA_DESCRIPTION)
    try:
        yield
    finally:
        substitution_args._eval_find = original


@functools.lru_cache(maxsize=None)
def _urdf() -> str:
    with _xacro_finds_the_description():
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
