"""Expert data collection: closed-loop MPC from random start/goal poses into a LeRobot v3.0 dataset.

    venv/bin/python -m learning.collect --episodes 20 --root data/franka_expert
"""

import argparse
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from dynamics.franka_dynamics import FrankaDynamics
from planning.casadi_planner import CasadiPlanner
from planning.planner import PlannerSolveError

CONFIG = {
    "dt": 0.05,
    "horizon": 20,
    "Q_diag": [0.0] * 7 + [500.0] * 2 + [2.0] * 9,
    "R_diag": [1e-3] * 7 + [1e-2] * 2,
}
TASK = "reach target pose"


def collect_episode(dyn: FrankaDynamics, planner: CasadiPlanner, seed: int, steps: int, noise_std: float):
    """Returns (frames, reached_goal); frames is None if the planner failed."""
    rng = np.random.default_rng(seed)
    x0, goal_pose = dyn.sample_start_goal(rng)
    dyn.set_task_goal(goal_pose)
    planner.reset()
    labels: list[np.ndarray] = []

    def policy(state: np.ndarray) -> np.ndarray:
        # Execution noise perturbs the visited states; the stored label stays the expert action.
        expert_action = planner(state)
        labels.append(expert_action)
        noise = noise_std * rng.standard_normal(expert_action.shape)
        return np.clip(expert_action + noise, -dyn.max_action, dyn.max_action)

    try:
        traj = dyn.simulate(x0, policy, steps)
    except PlannerSolveError:
        return None, False
    frames = [dyn.format_dataset_frame(obs, label) for obs, label in zip(traj["observations"], labels)]
    return frames, dyn.is_done(traj["states"][-1])


def collect(root: Path, episodes: int, steps: int, seed: int, noise_std: float) -> None:
    dyn = FrankaDynamics(CONFIG)
    planner = CasadiPlanner(dyn, CONFIG)
    dataset = LeRobotDataset.create(
        repo_id="local/franka_expert",
        fps=round(1 / dyn.dt),
        features=dyn.get_dataset_features(),
        root=root,
        robot_type="franka_fr3",
        use_videos=False,
    )

    saved = attempts = 0
    max_attempts = 3 * episodes
    start = time.time()
    while saved < episodes and attempts < max_attempts:
        frames, reached = collect_episode(dyn, planner, seed + attempts, steps, noise_std)
        attempts += 1
        if frames is None or not reached:
            print(f"attempt {attempts}: discarded ({'solver failure' if frames is None else 'goal not reached'})")
            continue
        for frame in frames:
            dataset.add_frame({**frame, "task": TASK})
        dataset.save_episode()
        saved += 1
        print(f"attempt {attempts}: saved episode {saved}/{episodes} ({len(frames)} frames)")
    dataset.finalize()
    print(f"{saved} episodes, {attempts} attempts, {time.time() - start:.1f}s -> {root}")
    if saved < episodes:
        raise RuntimeError(f"Only {saved}/{episodes} episodes succeeded in {attempts} attempts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200, help="maximum steps per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0, help="execution noise on accelerations [rad/s^2]")
    args = parser.parse_args()
    collect(args.root, args.episodes, args.steps, args.seed, args.noise_std)
