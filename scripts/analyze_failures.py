"""Replays the evaluation episodes of a DAgger checkpoint and tells how each one ends.

For every episode it records the outcome, the geometry of the task (how close the obstacles are to the start, the goal
and the straight line between them) and, for a collision, where and when it happened (the link, the obstacle, the
progress made, the speed). With ``--expert`` it also runs the expert from the episodes the policy fails, to tell the
episodes nobody can solve from those the policy gets wrong.

    python scripts/analyze_failures.py --checkpoint outputs/train_dagger/<run>/flow_dagger_iter_003.pt \
        --expert-config scripts/config/franka_ompl_casadi_config.yaml --policy-config learning/config/<policy>.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.config_loaders import load_dagger_training_config
from learning.dagger import apply_config_overrides, build_policy_action, evaluation_seed_specs
from learning.dagger.episode_starts import EpisodeStarts
from planning.config import load_and_validate_system_config, validate_system_config
from planning.ompl_planner import initialize_ompl
from planning.planner import PlannerSolveError
from planning.planner_factory import PlannerFactory
from scripts.evaluate_policy import load_policy
from systems.rollout_termination import RolloutTerminationTracker
from systems.seed_utils import default_action_noise_seed_for_config, torch_inference_seed_for_rollout
from systems.system_factory import SystemFactory

NEAR = 0.05  # m: an obstacle this close (surface to surface) to the start, the goal or the straight line is "in the way"


def clearance(simulator, state: np.ndarray) -> float:
    """Smallest gap [m] between a bounding sphere of the arm and an obstacle (negative when they overlap)."""
    centers, _, radii = simulator.collision_spheres(state[: simulator.n])
    return min(
        float(np.min(np.linalg.norm(centers - position, axis=1) - radii - radius))
        for position, radius in simulator.collision.obstacles
    )


def point_clearance(simulator, point: np.ndarray) -> float:
    return min(float(np.linalg.norm(point - position) - radius) for position, radius in simulator.collision.obstacles)


def segment_clearance(simulator, start: np.ndarray, end: np.ndarray) -> float:
    """Smallest gap [m] between the straight segment ``start`` -> ``end`` and an obstacle."""
    direction = end - start
    gaps = []
    for position, radius in simulator.collision.obstacles:
        fraction = np.clip(np.dot(position - start, direction) / max(np.dot(direction, direction), 1e-12), 0.0, 1.0)
        gaps.append(float(np.linalg.norm(start + fraction * direction - position) - radius))
    return min(gaps)


def contact(simulator) -> tuple[str, int]:
    """The link and the obstacle of the first contact found by the last ``collides`` call."""
    _, geometry_model, geometry_data = simulator._mesh_collision
    for index, result in enumerate(geometry_data.collisionResults):
        if result.isCollision():
            pair = geometry_model.collisionPairs[index]
            return geometry_model.geometryObjects[pair.first].name, int(geometry_model.geometryObjects[pair.second].name[len("obstacle"):])
    return "?", -1


def task_geometry(simulator, initial_state: np.ndarray) -> dict[str, float]:
    goal = simulator.vector_pose(simulator.goal).translation
    start = simulator.tcp_position(initial_state[: simulator.n])
    q_goal = simulator.goal_configuration(simulator.goal, initial_state[: simulator.n])
    return {
        "tcp_distance": float(np.linalg.norm(goal - start)),
        "joint_distance": float(np.linalg.norm(q_goal - initial_state[: simulator.n])),
        "start_clearance": clearance(simulator, initial_state),
        "goal_clearance": point_clearance(simulator, goal),
        "line_clearance": segment_clearance(simulator, start, goal),
    }


def run_policy(simulator, policy, initial_state: np.ndarray, num_steps: int, seed: int, noise_seed: int) -> dict[str, Any]:
    torch.manual_seed(torch_inference_seed_for_rollout(noise_seed, seed_spec=seed))
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    policy.reset()
    initial_error = simulator.goal_errors(state)[0]
    record: dict[str, Any] = {"outcome": "timeout", "steps": num_steps, "min_clearance": clearance(simulator, state)}
    if termination.update(state):
        return {**record, "outcome": "success", "steps": 0}
    for step in range(1, num_steps + 1):
        action = build_policy_action(simulator, policy, simulator.observe(state), torch.device("cpu"))
        state = simulator.step(state, action)
        record["min_clearance"] = min(record["min_clearance"], clearance(simulator, state))
        if simulator.collides(state):
            link, obstacle = contact(simulator)
            return {
                **record, "outcome": "collision", "steps": step, "link": link, "obstacle": obstacle,
                "progress": 1.0 - simulator.goal_errors(state)[0] / initial_error,
                "goal_distance": simulator.goal_errors(state)[0], "speed": float(np.linalg.norm(state[simulator.n: 2 * simulator.n])),
            }
        if termination.update(state):
            return {**record, "outcome": "success", "steps": step}
    return {**record, "goal_distance": simulator.goal_errors(state)[0], "speed": float(np.linalg.norm(state[simulator.n: 2 * simulator.n]))}


def run_expert(simulator, planner, initial_state: np.ndarray, num_steps: int) -> dict[str, Any]:
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    planner.reset()
    for step in range(1, num_steps + 1):
        try:
            state = simulator.step(state, planner.plan(simulator.observe(state)).actions[0])
        except PlannerSolveError:
            return {"expert": "planner_failure", "expert_steps": step - 1}
        if simulator.collides(state):
            return {"expert": "collision", "expert_steps": step}
        if termination.update(state):
            return {"expert": "success", "expert_steps": step}
    return {"expert": "timeout", "expert_steps": num_steps}


def analyze(
    checkpoint: str,
    config: Mapping[str, Any],
    num_episodes: int,
    seed_start: int,
    num_steps: int,
    eval_tolerances: Mapping[str, float],
    expert_tolerances: Mapping[str, float],
    with_expert: bool,
) -> list[dict[str, Any]]:
    eval_config = validate_system_config(apply_config_overrides(config, eval_tolerances))
    simulator = SystemFactory.create(eval_config["system"], eval_config)
    policy = load_policy("flow", checkpoint, simulator, eval_config, torch.device("cpu"))
    starts = EpisodeStarts.from_specs(simulator, None, None)
    noise_seed = default_action_noise_seed_for_config(eval_config)
    expert_config = validate_system_config(apply_config_overrides(config, expert_tolerances))
    expert_simulator = SystemFactory.create(expert_config["system"], expert_config)
    planner = PlannerFactory.create(expert_config["expert"], expert_simulator, expert_config) if with_expert else None

    episodes = []
    for index, seed in enumerate(evaluation_seed_specs(num_episodes, seed_start)):
        initial_state = starts.begin(simulator, index, seed)
        episode = {"seed": seed, **task_geometry(simulator, initial_state)}
        episode.update(run_policy(simulator, policy, initial_state, num_steps, seed, noise_seed))
        if planner is not None and episode["outcome"] != "success":
            expert_simulator.set_goal(simulator.goal)
            episode.update(run_expert(expert_simulator, planner, initial_state, num_steps * 2))
        episodes.append(episode)
    return episodes


def report(episodes: list[dict[str, Any]]) -> None:
    outcomes = Counter(e["outcome"] for e in episodes)
    print(f"\n{len(episodes)} episodes: " + ", ".join(f"{name} {count}" for name, count in outcomes.most_common()))

    print("\nTask geometry by outcome (mean; fraction with an obstacle within "
          f"{NEAR * 100:.0f} cm of the start / goal / straight line):")
    for outcome in outcomes:
        group = [e for e in episodes if e["outcome"] == outcome]
        mean = lambda key: np.mean([e[key] for e in group])
        fraction = lambda key: np.mean([e[key] < NEAR for e in group])
        print(f"  {outcome:<10} n={len(group):<3} tcp_dist {mean('tcp_distance'):.2f} m  joint_dist {mean('joint_distance'):.2f} rad  "
              f"start {fraction('start_clearance'):.0%}  goal {fraction('goal_clearance'):.0%}  line {fraction('line_clearance'):.0%}")

    collisions = [e for e in episodes if e["outcome"] == "collision"]
    if collisions:
        print("\nCollisions by link:     " + ", ".join(f"{k} {v}" for k, v in Counter(e["link"] for e in collisions).most_common()))
        print("Collisions by obstacle: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(e["obstacle"] for e in collisions).items())))
        steps = np.array([e["steps"] for e in collisions])
        print(f"At impact: step {np.median(steps):.0f} (median, range {steps.min()}-{steps.max()}), "
              f"progress {np.median([e['progress'] for e in collisions]):.0%}, "
              f"goal distance {np.median([e['goal_distance'] for e in collisions]):.2f} m, "
              f"joint speed {np.median([e['speed'] for e in collisions]):.2f} rad/s")
    failures = [e for e in episodes if "expert" in e]
    if failures:
        print("\nExpert from the failed episodes: " + ", ".join(f"{k} {v}" for k, v in Counter(e["expert"] for e in failures).most_common()))
        print("\nFailures (seed, outcome, step, link/obstacle, progress, line gap, expert):")
        for e in sorted(failures, key=lambda e: (e["outcome"], e["steps"])):
            where = f"{e['link']}/{e['obstacle']}" if e["outcome"] == "collision" else f"{e['goal_distance']:.2f} m from goal"
            progress = f"{e['progress']:+.0%}" if e["outcome"] == "collision" else "-"
            print(f"  {e['seed']}  {e['outcome']:<9} {e['steps']:>3}  {where:<26} {progress:>5}  {e['line_clearance']:+.2f} m  {e['expert']} ({e['expert_steps']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expert-config", required=True)
    parser.add_argument("--policy-config", type=Path, required=True, help="the training config of the run (evaluation settings)")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--expert", action="store_true", help="run the expert from the failed episodes")
    parser.add_argument("--output", type=Path, help="write the per-episode records as YAML")
    args = parser.parse_args()

    training = load_dagger_training_config(args.policy_config)
    config = load_and_validate_system_config(args.expert_config)
    initialize_ompl(config["ompl_seed"])
    episodes = analyze(
        args.checkpoint,
        config,
        args.episodes or int(training.get("eval_episodes", 100)),
        args.seed_start or int(training.get("eval_seed_start", 10000)),
        args.steps or int(training.get("steps_per_trajectory", 100)),
        training.get("eval_tolerance_overrides") or training.get("tolerance_overrides") or {},
        training.get("tolerance_overrides") or {},
        args.expert,
    )
    report(episodes)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml.safe_dump(episodes, sort_keys=False))


if __name__ == "__main__":
    main()
