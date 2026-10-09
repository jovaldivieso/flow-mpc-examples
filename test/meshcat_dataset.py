"""Play back episodes of a collected LeRobot dataset in Meshcat.

    venv/bin/python -m test.meshcat_dataset --root data/franka_expert --episodes 10
"""

import argparse
import time
from pathlib import Path

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pinocchio.visualize import MeshcatVisualizer

from dynamics.franka_dynamics import FrankaDynamics
from learning.collect import CONFIG


def load_episodes(root: Path, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Random episodes as (joint positions [T, nq], goal TCP position [3])."""
    dataset = LeRobotDataset("local/franka_expert", root=root)
    total = dataset.meta.total_episodes
    chosen = np.random.default_rng(seed).choice(total, size=min(count, total), replace=False)
    episodes = []
    for index in sorted(chosen):
        info = dataset.meta.episodes[int(index)]
        rows = dataset.hf_dataset[int(info["dataset_from_index"]) : int(info["dataset_to_index"])]
        state = np.asarray(rows["observation.state"])
        goal = np.asarray(rows["observation.environment_state"])[0, :3]
        episodes.append((state[:, : state.shape[1] // 2], goal))
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dyn = FrankaDynamics(CONFIG)
    episodes = load_episodes(args.root, args.episodes, args.seed)

    robot = dyn.robot
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    viz.viewer["goal"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xFF0000, opacity=0.6))

    viz.viewer["start"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x2060FF, opacity=0.9))

    print(f"Playing {len(episodes)} episodes in a loop (Ctrl+C to stop): blue dot = start TCP, red ball = goal TCP")
    try:
        while True:
            for number, (positions, goal) in enumerate(episodes, 1):
                print(f"episode {number}/{len(episodes)}: {len(positions)} frames")
                viz.viewer["goal"].set_transform(tf.translation_matrix(goal))
                start = dyn.tcp_placement(positions[0]).translation
                viz.viewer["start"].set_transform(tf.translation_matrix(start))
                for q in positions:
                    viz.display(q)
                    time.sleep(dyn.dt)
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
