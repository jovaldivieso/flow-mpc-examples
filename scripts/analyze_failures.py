"""Replays the evaluation episodes of a DAgger checkpoint and tells how each one ends.

For every episode it records the outcome, the geometry of the task (how close the obstacles are to the start, the goal
and the straight line between them) and, for a collision, where and when it happened (the link, the obstacle, the
progress made, the speed). An episode that times out is sorted into one of four failure modes by how its last
``STALL_WINDOW`` steps look: creep near the goal, sliding along an obstacle, wandering in free space, or at rest far from
the goal. With ``--expert`` it also runs the expert from the episodes the policy fails, to tell the episodes nobody can
solve from those the policy gets wrong, and compares, at the state a timeout ends in, the expert's plan with the policy's
chunk (without projection, with guidance, and the plan the policy settles on).

    python scripts/analyze_failures.py --checkpoint outputs/train_dagger/<run>/flow_dagger_iter_003.pt \
        --expert-config scripts/config/franka_obstacles_config.yaml --policy-config learning/config/<policy>.yaml
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
from learning.dagger import apply_config_overrides, build_policy_actions, evaluation_seed_specs
from learning.dagger.episode_starts import EpisodeStarts
from learning.models.policy import observation_features
from planning.config import load_and_validate_system_config, validate_system_config
from planning.ompl_planner import initialize_ompl
from planning.planner import PlannerSolveError
from planning.planner_factory import PlannerFactory
from scripts.evaluate_policy import checkpoint_policy_type, load_policy
from systems.position_history import PositionHistory
from systems.rollout_termination import RolloutTerminationTracker
from systems.seed_utils import default_action_noise_seed_for_config, torch_inference_seed_for_rollout
from systems.system_factory import SystemFactory

NEAR = 0.05  # m: an obstacle this close (surface to surface) to the start, the goal or the straight line is "in the way"

STALL_WINDOW = 25  # steps the end of a timed-out episode is judged over
NEAR_GOAL = 0.10  # m from the goal at the end: closer is creep (near the goal, not settled)
AT_REST = 0.02  # rad/s: no joint moves faster than this (on average over the window)
HUGGING = 0.02  # m: a mean clearance over the window below this is "along an obstacle"
APPROACHING = 0.05  # m the goal distance fell over the window: still making progress
CREEP, SLIDING, WANDERING, RESTING = (
    "creep near the goal",
    "sliding along an obstacle",
    "wandering in free space",
    "at rest far from the goal",
)
FAILURE_MODES = (CREEP, SLIDING, WANDERING, RESTING)


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


def joint_speed(simulator, state: np.ndarray) -> float:
    """The largest joint speed [rad/s] of ``state``."""
    return float(np.abs(state[simulator.n : 2 * simulator.n]).max())


def contact(simulator) -> tuple[str, int]:
    """The link and the obstacle of the first contact found by the last ``collides`` call."""
    _, geometry_model, geometry_data = simulator._mesh_collision
    for index, result in enumerate(geometry_data.collisionResults):
        if result.isCollision():
            pair = geometry_model.collisionPairs[index]
            return geometry_model.geometryObjects[pair.first].name, int(geometry_model.geometryObjects[pair.second].name[len("obstacle"):])
    return "?", -1


def task_geometry(simulator, initial_state: np.ndarray) -> dict[str, Any]:
    goal = simulator.vector_pose(simulator.goal).translation
    start = simulator.tcp_position(initial_state[: simulator.n])
    try:
        q_goal = simulator.goal_configuration(simulator.goal, initial_state[: simulator.n])
        joint_distance = float(np.linalg.norm(q_goal - initial_state[: simulator.n]))
    except RuntimeError:  # no configuration reaches the goal from this start (inverse kinematics does not converge)
        joint_distance = float("nan")
    return {
        "tcp_distance": float(np.linalg.norm(goal - start)),
        "joint_distance": joint_distance,
        "start_clearance": clearance(simulator, initial_state),
        "goal_clearance": point_clearance(simulator, goal),
        "line_clearance": segment_clearance(simulator, start, goal),
        "opposite_sides": bool(start[1] * goal[1] < 0.0),  # start and goal on the two sides of the plane y = 0
    }


def classify_timeout(goal_distance: float, speed: float, mean_clearance: float) -> str:
    """How an episode that timed out ends: near the goal it is creep; far from it, at rest, along an obstacle (sliding) or
    neither (wandering). ``speed`` and ``mean_clearance`` are averages over the last ``STALL_WINDOW`` steps."""
    if goal_distance < NEAR_GOAL:
        return CREEP
    if speed < AT_REST:
        return RESTING
    if mean_clearance < HUGGING:
        return SLIDING
    return WANDERING


def stall_features(goal_distance: list[float], clearances: list[float], speeds: list[float], tcp_y: list[float]) -> dict[str, Any]:
    """The end of a timed-out episode from its per-step traces: the failure mode and what it was judged by."""
    window = slice(-(STALL_WINDOW + 1), None)
    end, mean_clearance, speed = goal_distance[-1], float(np.mean(clearances[window])), float(np.mean(speeds[window]))
    return {
        "mode": classify_timeout(end, speed, mean_clearance),
        "goal_distance": end,
        "window_speed": speed,
        "window_clearance": mean_clearance,
        "window_progress": float(goal_distance[window][0] - end),
        "crossed_y0": bool(min(tcp_y) < 0.0 < max(tcp_y)),
    }


def run_policy(
    simulator, policy, initial_state: np.ndarray, num_steps: int, seed: int, noise_seed: int
) -> tuple[dict[str, Any], np.ndarray, PositionHistory]:
    """One evaluation episode: its record, the state it ends in and the history of positions there."""
    torch.manual_seed(torch_inference_seed_for_rollout(noise_seed, seed_spec=seed))
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    policy.reset()
    history = PositionHistory(simulator.n, policy.history_length)
    history.reset(state)
    initial_error = simulator.goal_errors(state)[0]
    record: dict[str, Any] = {"outcome": "timeout", "steps": num_steps, "min_clearance": clearance(simulator, state)}
    if termination.update(state):
        return {**record, "outcome": "success", "steps": 0}, state, history
    goal_distance, clearances, speeds = [initial_error], [record["min_clearance"]], [joint_speed(simulator, state)]
    tcp_y = [float(simulator.tcp_position(state[: simulator.n])[1])]
    step = 0
    while step < num_steps:
        try:
            actions = build_policy_actions(simulator, policy, simulator.observe(state), history.flat, torch.device("cpu"))
        except PlannerSolveError:
            return {**record, "outcome": "solve_failure", "steps": step, "goal_distance": goal_distance[-1]}, state, history
        for action in actions:  # the policy's jerks are executed one after the other until it is asked again
            if step >= num_steps:
                break
            step += 1
            state = simulator.step(state, action)
            history.push(state)
            goal_distance.append(simulator.goal_errors(state)[0])
            clearances.append(clearance(simulator, state))
            speeds.append(joint_speed(simulator, state))
            tcp_y.append(float(simulator.tcp_position(state[: simulator.n])[1]))
            record["min_clearance"] = min(record["min_clearance"], clearances[-1])
            if simulator.collides(state):
                link, obstacle = contact(simulator)
                return {
                    **record, "outcome": "collision", "steps": step, "link": link, "obstacle": obstacle,
                    "progress": 1.0 - goal_distance[-1] / initial_error,
                    "goal_distance": goal_distance[-1], "speed": speeds[-1],
                }, state, history
            if termination.update(state):
                return {**record, "outcome": "success", "steps": step}, state, history
    return {**record, **stall_features(goal_distance, clearances, speeds, tcp_y)}, state, history


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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def compare_proposals(
    simulator, policy, expert_positions: np.ndarray, state: np.ndarray, history: PositionHistory, initial_state: np.ndarray
) -> dict[str, float]:
    """The move of the chunk the policy proposes at ``state`` and the expert's plan, both at the last row it executes.

    For each chunk: its move from the state [rad] and the cosine of that move with the expert's. The chunks are the flow's
    proposal without projection, with guidance, and, for a safe flow policy, the plan its projection settles on (a tick
    from the coasting chunk: the plans of earlier ticks are not reused).
    """
    inner = getattr(policy, "inner_policy", policy)
    row = policy.execution_horizon - 1
    q = state[: simulator.n]
    expert_move = expert_positions[row] - q

    def tensors(observation_state: np.ndarray, observation_history: np.ndarray) -> dict[str, torch.Tensor]:
        observation = simulator.policy_observation(simulator.observe(observation_state), observation_history)
        return {name: torch.as_tensor(np.expand_dims(value, axis=0), dtype=torch.float32) for name, value in observation.items()}

    start_history = PositionHistory(simulator.n, policy.history_length)
    start_history.reset(initial_state)
    start_observation = tensors(initial_state, start_history.flat)
    policy.reset()
    # The guidance takes the start distance of the episode from its first call
    inner.guide(start_observation, np.tile(initial_state[: simulator.n], (inner.guidance.horizon, 1)))
    observation = tensors(state, history.flat)
    features = observation_features(observation)
    projector = getattr(policy, "projector", None)
    start_chunk = None
    if projector is not None:
        projector.begin_tick()
        coasting = simulator.rollout_positions(state, np.zeros((projector.tracked_horizon, simulator.nu)))[1:]
        start_chunk = projector.project(state, coasting).positions

    def project(proposal: np.ndarray) -> np.ndarray:
        return projector.project(state, inner.guide(observation, proposal)).positions

    variants = {"flow": lambda proposal: proposal, "guided": lambda proposal: inner.guide(observation, proposal)}
    if projector is not None:
        variants["plan"] = project
    result = {"expert_move": float(np.linalg.norm(expert_move))}
    with torch.inference_mode():
        for name, correct in variants.items():
            samples = 1 if start_chunk is not None else 5  # from Gaussian noise the proposal varies
            chunks = []
            for _ in range(samples):
                if projector is not None:
                    projector.begin_tick()
                chunks.append(inner.velocity_net.integrate(features, correct, initial=start_chunk).numpy()[0].astype(float))
            moves = [chunk[row] - q for chunk in chunks]
            median = sorted(moves, key=np.linalg.norm)[len(moves) // 2]
            result[f"{name}_move"], result[f"{name}_cos"] = float(np.linalg.norm(median)), cosine(median, expert_move)
    return result


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
    policy = load_policy(checkpoint_policy_type(checkpoint), checkpoint, simulator, eval_config, torch.device("cpu"))
    starts = EpisodeStarts.from_specs(simulator, None, None)
    noise_seed = default_action_noise_seed_for_config(eval_config)
    expert_config = validate_system_config(apply_config_overrides(config, expert_tolerances))
    expert_simulator = SystemFactory.create(expert_config["system"], expert_config)
    planner = PlannerFactory.create(expert_config["expert"], expert_simulator, expert_config) if with_expert else None

    episodes = []
    for index, seed in enumerate(evaluation_seed_specs(num_episodes, seed_start)):
        initial_state = starts.begin(simulator, index, seed)
        episode = {"seed": seed, **task_geometry(simulator, initial_state)}
        record, state, history = run_policy(simulator, policy, initial_state, num_steps, seed, noise_seed)
        episode.update(record)
        episode["rotation_error"] = float(simulator.goal_errors(state)[1])
        if planner is not None and episode["outcome"] != "success":
            expert_simulator.set_goal(simulator.goal)
            episode.update(run_expert(expert_simulator, planner, initial_state, num_steps * 2))
            if episode["outcome"] == "timeout":
                planner.reset()
                try:
                    plan = planner.plan(expert_simulator.observe(state)).positions
                except PlannerSolveError as error:
                    episode["expert_plan"] = str((error.__cause__ or error))[:60]
                else:
                    try:
                        episode.update(compare_proposals(simulator, policy, plan, state, history, initial_state))
                    except PlannerSolveError:
                        episode["comparison"] = "no safe stop from the state"
        episodes.append(episode)
    return episodes


def report(episodes: list[dict[str, Any]]) -> None:
    outcomes = Counter(e["outcome"] for e in episodes)
    print(f"\n{len(episodes)} episodes: " + ", ".join(f"{name} {count}" for name, count in outcomes.most_common()))

    print("\nTask geometry by outcome (mean; fraction with an obstacle within "
          f"{NEAR * 100:.0f} cm of the start / goal / straight line, start and goal on opposite sides of y = 0):")
    for outcome in outcomes:
        group = [e for e in episodes if e["outcome"] == outcome]
        mean = lambda key: np.nanmean([e[key] for e in group])
        fraction = lambda key: np.mean([e[key] < NEAR for e in group])
        print(f"  {outcome:<13} n={len(group):<3} tcp_dist {mean('tcp_distance'):.2f} m  joint_dist {mean('joint_distance'):.2f} rad  "
              f"start {fraction('start_clearance'):.0%}  goal {fraction('goal_clearance'):.0%}  line {fraction('line_clearance'):.0%}  "
              f"opposite {np.mean([e['opposite_sides'] for e in group]):.0%}")

    timeouts = [e for e in episodes if e["outcome"] == "timeout"]
    if timeouts:
        print(f"\nTimeouts by failure mode (over the last {STALL_WINDOW} steps; progress = goal distance fallen over the window):")
        for mode in FAILURE_MODES:
            group = [e for e in timeouts if e["mode"] == mode]
            if not group:
                continue
            mean = lambda key: np.mean([e[key] for e in group])
            print(f"  {mode:<27} n={len(group):<3} opposite sides {np.mean([e['opposite_sides'] for e in group]):.0%}  "
                  f"goal {mean('goal_distance'):.2f} m  rot {mean('rotation_error'):.2f} rad  clearance {mean('window_clearance') * 100:.1f} cm  speed {mean('window_speed'):.3f} rad/s  "
                  f"progress {mean('window_progress'):+.2f} m ({sum(e['window_progress'] > APPROACHING for e in group)} still approaching)  "
                  f"crossed y=0 {np.mean([e['crossed_y0'] for e in group]):.0%}")
        compared = [e for e in timeouts if "expert_move" in e]
        if compared:
            print("\nMove of the chunk at the last executed row [rad] / cosine with the expert's, median by failure mode:")
            keys = [key[: -len("_move")] for key in compared[0] if key.endswith("_move")]
            for mode in FAILURE_MODES:
                group = [e for e in compared if e["mode"] == mode]
                if group:
                    cells = [f"{key} {np.median([e[key + '_move'] for e in group]):.2f}" + (
                        f" ({np.median([e[key + '_cos'] for e in group]):+.2f})" if key != "expert" else "") for key in keys]
                    print(f"  {mode:<27} n={len(group):<3} " + "  ".join(cells))

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
        print("\nFailures (seed, outcome, step, where, progress, line gap, expert):")
        for e in sorted(failures, key=lambda e: (e["outcome"], e["steps"])):
            if e["outcome"] == "collision":
                where, progress = f"{e['link']}/{e['obstacle']}", f"{e['progress']:+.0%}"
            else:
                where, progress = f"{e['goal_distance']:.2f} m from goal" + (f", {e['mode']}" if "mode" in e else ""), "-"
            print(f"  {e['seed']}  {e['outcome']:<13} {e['steps']:>3}  {where:<52} {progress:>5}  {e['line_clearance']:+.2f} m  "
                  f"{e['expert']} ({e['expert_steps']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expert-config", required=True)
    parser.add_argument("--policy-config", type=Path, required=True, help="the training config of the run (evaluation settings)")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--expert", action="store_true", help="run the expert from the failed episodes and compare its plan with the policy's chunk")
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
        args.steps or int(training.get("eval_steps") or training.get("steps_per_trajectory", 100)),
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
