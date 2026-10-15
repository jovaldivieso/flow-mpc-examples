"""Play back episodes of a collected LeRobot dataset in Meshcat.

    venv/bin/python -m scripts.meshcat_dataset --root data/lerobot_dataset_franka_dagger_<timestamp> --first --episodes 10 --show-plan

The episodes are the first --episodes ones (--first), the explicit --indices, or random ones (--seed). With --show-plan
the expert's plan that labels each frame (the action chunk the flow network is trained to predict from that frame's
observation) is drawn as the TCP path of the motion it leads to, in cyan, while the arm moves. --no-viewer prints the
per-episode table only. With --overview the TCP paths of every episode are also drawn at once (green if the episode
ended within --success-tol of the goal, orange otherwise, blue dots at the starts, red dots at the goals).

The parquet files are read directly, so a dataset a DAgger run is still writing to can be viewed: files that are not
finished yet are skipped (their episodes are missing).
"""

import argparse
import ast
import glob
import time
from pathlib import Path
from typing import NamedTuple

import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import pandas as pd
from pinocchio.visualize import MeshcatVisualizer

from planning.config import load_and_validate_system_config
from dynamics.franka_dynamics import FrankaDynamics

from .meshcat_utils import draw_plan, draw_tcp_paths, plan_tcp_points

PLAN_COLOR = 0x00D0FF
# A frame whose labelled plan stays within this many rad of the current configuration is a "stay" label
STAY_THRESHOLD = 0.05
AT_REST_SPEED = 1e-3


class Episode(NamedTuple):
    index: int
    states: np.ndarray  # (T, 3n) joint positions, velocities and accelerations
    goal: np.ndarray  # (12,) goal TCP pose [position, rotation matrix row-major]
    actions: np.ndarray  # (T, H, n) the expert's planned jerks that label each frame

    @property
    def positions(self) -> np.ndarray:
        return self.states[:, : self.actions.shape[2]]

    @property
    def velocities(self) -> np.ndarray:
        return self.states[:, self.actions.shape[2] : 2 * self.actions.shape[2]]


def plan_positions(dyn: FrankaDynamics, episode: Episode) -> np.ndarray:
    """The joint positions (T, H + 1, n) of the plan that labels each frame, row 0 the frame's own configuration."""
    return np.stack([dyn.rollout_positions(state, actions) for state, actions in zip(episode.states, episode.actions)])


def read_frames(root: Path) -> pd.DataFrame:
    """All frames of the dataset's finished parquet files; a file still being written (no footer yet) is skipped."""
    parts = []
    for path in sorted(glob.glob(str(root / "data" / "chunk-*" / "file-*.parquet"))):
        try:
            parts.append(pd.read_parquet(path))
        except Exception as error:  # an unfinished file raises pyarrow's ArrowInvalid, which has no common base to name
            print(f"skipping {Path(path).name} (not readable yet: {type(error).__name__})")
    if not parts:
        raise FileNotFoundError(f"No readable parquet file under {root / 'data'}.")
    return pd.concat(parts, ignore_index=True)


def load_episodes(
    root: Path,
    count: int,
    seed: int,
    first: bool = False,
    indices: list[int] | None = None,
) -> list[Episode]:
    """Episodes of the dataset at ``root``: ``indices`` if given, else the first ``count`` or ``count`` random ones."""
    frames = read_frames(root)
    available = sorted(frames["episode_index"].unique().tolist())
    if indices is not None:
        missing = [index for index in indices if index not in available]
        if missing:
            raise ValueError(f"Episodes {missing} are not in the readable part of the dataset ({len(available)} episodes).")
        chosen = list(indices)
    elif first:
        chosen = available[:count]
    else:
        chosen = sorted(np.random.default_rng(seed).choice(available, size=min(count, len(available)), replace=False).tolist())
    episodes = []
    for index in chosen:
        rows = frames[frames["episode_index"] == index].sort_values("frame_index")
        state = np.stack(rows["observation.state"].to_numpy()).astype(float)
        n = state.shape[1] // 3
        goal = np.stack(rows["observation.environment_state"].to_numpy()).astype(float)[0]
        action = np.stack(rows["action"].to_numpy()).astype(float)
        episodes.append(Episode(int(index), state, goal, action.reshape(len(rows), -1, n)))
    return episodes


def episode_summary(dyn: FrankaDynamics, episode: Episode) -> dict[str, float]:
    """What the episode ends like and what its labels say: how far from the goal, how much of it at rest or 'stay'."""
    dyn.set_goal(episode.goal)
    position_error, rotation_error = dyn.goal_errors(episode.positions[-1])
    plans = plan_positions(dyn, episode)
    excursion = np.abs(plans[:, 1:] - plans[:, :1]).max(axis=(1, 2))
    return {
        "frames": len(episode.positions),
        "final_position_error": position_error,
        "final_rotation_error": rotation_error,
        "clearance": dyn.clearance(episode.positions[-1]),
        "at_rest_fraction": float(np.mean(np.abs(episode.velocities).max(axis=1) < AT_REST_SPEED)),
        "stay_label_fraction": float(np.mean(excursion < STAY_THRESHOLD)),
        "plan_excursion": float(excursion.max()),
    }


def print_table(dyn: FrankaDynamics, episodes: list[Episode], cap: int) -> None:
    for episode in episodes:
        s = episode_summary(dyn, episode)
        note = "  <- ran into the step cap" if s["frames"] >= cap else ""
        print(
            f"episode {episode.index:5d}: {s['frames']:3d} frames | final TCP error {1000 * s['final_position_error']:6.1f} mm, "
            f"{s['final_rotation_error']:.3f} rad | final clearance {1000 * s['clearance']:5.1f} mm | at rest {s['at_rest_fraction']:4.0%} "
            f"| 'stay' labels {s['stay_label_fraction']:4.0%} | largest plan excursion {s['plan_excursion']:.2f} rad{note}"
        )


def show_overview(viz, dyn: FrankaDynamics, episodes: list[Episode], success_tol: float) -> None:
    """Draw the TCP path of every episode at once."""
    paths = [np.array([dyn.tcp_placement(q).translation for q in episode.positions]) for episode in episodes]
    goals = [episode.goal[:3] for episode in episodes]
    reached = [bool(np.linalg.norm(path[-1] - goal) < success_tol) for path, goal in zip(paths, goals)]
    draw_tcp_paths(viz, paths, goals, reached)
    print(
        f"Drawn {len(episodes)} episodes: {sum(reached)} ended within {success_tol} m of the goal (green), "
        f"{len(episodes) - sum(reached)} did not (orange); blue = starts, red = goals"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scripts/config/franka_casadi_config.yaml"),
        help="expert yaml config, only its dt and the robot model are used (default: %(default)s)",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--first", action="store_true", help="the first --episodes episodes instead of random ones")
    parser.add_argument("--indices", type=str, default=None, help="explicit episode indices, e.g. '[0, 5, 8, 9]'")
    parser.add_argument("--seed", type=int, default=0, help="seed of the random choice of episodes")
    parser.add_argument("--show-plan", action="store_true", help="draw the expert's labelled plan of every frame as a TCP path")
    parser.add_argument("--cap", type=int, default=250, help="episode length at which the table notes the step cap (default: %(default)s)")
    parser.add_argument("--no-viewer", action="store_true", help="print the per-episode table only")
    parser.add_argument("--overview", action="store_true", help="additionally draw the TCP paths of all episodes at once")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time (default: %(default)s)")
    parser.add_argument("--success-tol", type=float, default=0.01, help="goal distance [m] for a green path in --overview (default: %(default)s)")
    args = parser.parse_args()

    dyn = FrankaDynamics(load_and_validate_system_config(args.config))
    indices = [int(i) for i in ast.literal_eval(args.indices)] if args.indices else None
    episodes = load_episodes(args.root, 10**9 if args.overview else args.episodes, args.seed, args.first, indices)
    played = episodes[: args.episodes] if indices is None else episodes
    print_table(dyn, played, args.cap)
    if args.no_viewer:
        return

    robot = dyn.robot
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    if args.overview:
        show_overview(viz, dyn, episodes, args.success_tol)
    episodes = played
    viz.viewer["goal"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xFF0000, opacity=0.6))

    viz.viewer["start"].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x2060FF, opacity=0.9))

    print(f"Playing {len(episodes)} episodes in a loop (Ctrl+C to stop): blue dot = start TCP, red ball = goal TCP"
          + (", cyan line = the expert's labelled plan of the frame" if args.show_plan else ""))
    try:
        while True:
            for number, episode in enumerate(episodes, 1):
                print(f"episode {episode.index} ({number}/{len(episodes)}): {len(episode.positions)} frames")
                viz.viewer["goal"].set_transform(tf.translation_matrix(episode.goal[:3]))
                start = dyn.tcp_placement(episode.positions[0]).translation
                viz.viewer["start"].set_transform(tf.translation_matrix(start))
                for q, plan in zip(episode.positions, plan_positions(dyn, episode)):
                    viz.display(q)
                    if args.show_plan:
                        draw_plan(viz, "plan", plan_tcp_points(dyn, plan), PLAN_COLOR)
                    time.sleep(dyn.dt / args.speed)
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
