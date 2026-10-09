"""End-to-end check: CasADi/IPOPT MPC on the FR3 dynamics, played back in Meshcat."""

import time

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

from dynamics.franka_dynamics import Q_READY, FrankaDynamics
from planning.casadi_planner import CasadiPlanner

NUM_STEPS = 60


def tcp_pose(xyz: list[float], yaw: float, tilt: float = 0.0) -> pin.SE3:
    """Gripper pointing down (rotated pi about x), then tilted about y and yawed about z."""
    R = (pin.utils.rpyToMatrix(0.0, 0.0, yaw) @ pin.utils.rpyToMatrix(0.0, tilt, 0.0)
         @ pin.utils.rpyToMatrix(np.pi, 0.0, 0.0))
    return pin.SE3(R, np.array(xyz))


def main() -> None:
    start_pose = tcp_pose([0.45, -0.35, 0.50], yaw=-0.6)
    goal_pose = tcp_pose([0.35, 0.40, 0.25], yaw=0.9, tilt=0.4)

    config = {
        "dt": 0.05,
        "horizon": 20,
        # Arm joint positions are free: the TCP pose cost drives them. Fingers are only
        # softly kept at their starting width, and velocities are damped.
        "Q_diag": [0.0] * 7 + [500.0] * 2 + [2.0] * 9,
        "R_diag": [1e-3] * 7 + [1e-2] * 2,
    }
    dyn = FrankaDynamics(config)
    dyn.set_task_goal(goal_pose)
    planner = CasadiPlanner(dyn, config)

    # MPC needs a joint-space start state: one IK solve for the start TCP pose
    q0 = dyn.inverse_kinematics(start_pose, Q_READY)
    x0 = np.concatenate([q0, np.zeros(dyn.n)])

    print("Planning (closed-loop MPC)...")
    t0 = time.time()
    traj = dyn.simulate(x0, planner, NUM_STEPS)
    states = np.vstack([x0, traj["states"]])
    final = dyn.tcp_placement(states[-1, : dyn.n])
    print(f"{len(traj['states'])} steps in {time.time() - t0:.1f}s, "
          f"final TCP error: {np.linalg.norm(final.translation - goal_pose.translation):.4f} m, "
          f"{np.linalg.norm(pin.log3(final.rotation.T @ goal_pose.rotation)):.4f} rad")

    robot = dyn.robot
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    goal_tcp = goal_pose.translation
    viz.viewer["goal"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xFF0000, opacity=0.6))
    viz.viewer["goal"].set_transform(tf.translation_matrix(goal_tcp))

    print("Playing back in Meshcat (Ctrl+C to stop)")
    try:
        while True:
            for x in states:
                viz.display(x[: dyn.n])
                time.sleep(dyn.dt)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
