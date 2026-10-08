"""Show the FR3 (visual meshes used by franka_dynamics.py) in a Meshcat browser viewer."""

import sys
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rigid_body_dynamics.franka_dynamics import build_robot  # noqa: E402


def main() -> None:
    robot = build_robot()
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()

    q = pin.neutral(robot.model)
    viz.display(q)
    print(f"Viewer: {viz.viewer.url()}  (Ctrl+C to quit)")

    # Sweep the arm joints slowly to check that every mesh follows its link.
    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            q_t = q.copy()
            q_t[:7] += 0.5 * np.sin(0.5 * t + np.arange(7))
            viz.display(q_t)
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
