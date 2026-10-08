"""Load the FR3 URDF with Pinocchio and compute basic rigid body dynamics."""

from pathlib import Path

import xacro
from xacro import substitution_args

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Official franka_description, pinned as a git submodule
FRANKA_DESCRIPTION = Path(__file__).resolve().parents[1] / "third_party" / "franka_description"
XACRO_PATH = FRANKA_DESCRIPTION / "robots" / "fr3" / "fr3.urdf.xacro"

# xacro resolves $(find pkg) via ament_index_python (ROS-only, not on PyPI)
substitution_args._eval_find = lambda pkg: str(FRANKA_DESCRIPTION)


def build_robot() -> RobotWrapper:
    urdf = xacro.process_file(XACRO_PATH, mappings={"hand": "true"}).toxml()
    # meshes are package://franka_description/..., resolved from the clone's parent
    pkg = [str(FRANKA_DESCRIPTION.parent)]
    model = pin.buildModelFromXML(urdf)
    visual = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.VISUAL, package_dirs=pkg)
    collision = pin.buildGeomFromUrdfString(model, urdf, pin.GeometryType.COLLISION, package_dirs=pkg)
    return RobotWrapper(model, collision, visual)


def main() -> None:
    robot = build_robot()
    model, data = robot.model, robot.data
    print(
        f"Geometry: {len(robot.visual_model.geometryObjects)} visual, "
        f"{len(robot.collision_model.geometryObjects)} collision"
    )
    print(f"Model '{model.name}': nq={model.nq}, nv={model.nv}")
    for name in model.names[1:]:
        print(f"  joint: {name}")

    q = pin.neutral(model)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)

    # Inverse dynamics (RNEA): torque for given q, v, a. At rest this is gravity.
    tau = pin.rnea(model, data, q, v, a)
    print("\nGravity torques at neutral [Nm]:", tau)

    # Joint-space mass matrix (CRBA); Pinocchio fills only the upper triangle.
    M = pin.crba(model, data, q)
    M = np.triu(M) + np.triu(M, 1).T
    print("\nMass matrix diagonal:", M.diagonal())

    # Forward dynamics (ABA): acceleration under zero torque.
    ddq = pin.aba(model, data, q, v, np.zeros(model.nv))
    print("\nFree-fall acceleration [rad/s^2]:", ddq)

    # Forward kinematics of the last frame.
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    ee = "fr3_hand_tcp"
    print(f"\nFrame '{ee}' placement:\n{data.oMf[model.getFrameId(ee)]}")


if __name__ == "__main__":
    main()
