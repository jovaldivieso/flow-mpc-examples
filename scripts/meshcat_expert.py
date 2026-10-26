"""Check the CasADi MPC expert before running DAgger: random start/goal rollouts, statistics, Meshcat playback.

    venv/bin/python -m scripts.meshcat_expert --num-traj 10
    venv/bin/python -m scripts.meshcat_expert --seeds "[0, 3, 7]" --tolerance-overrides '{"pos_tol": 0.01, "vel_tol": 0.05}'

Every rollout samples its start and goal from the expert config exactly like the DAgger collection and the evaluation do
(seeded), runs the expert for --num-steps and prints per trajectory: reached the goal, final error,
and how smooth the commanded jerks are and how much the arm detours (joint and TCP path length relative to the straight line). Then all TCP paths are drawn (green = reached the goal,
orange = did not) and the trajectories are played back with the moving arm. --no-viewer prints the statistics only.
"""

import argparse
import ast
import time
from pathlib import Path

import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from systems.franka_system import FrankaSystem
from systems.rollout_termination import RolloutTerminationTracker
from systems.seed_utils import action_noise_rng_for_rollout, default_action_noise_seed_for_config
from learning.dagger.rollouts import apply_execution_noise
from learning.dagger.utils import sample_initial_state
from planning.config import load_and_validate_system_config, validate_system_config
from planning.ompl_planner import initialize_ompl
from planning.planner import Planner, PlannerSolveError
from planning.planner_factory import PlannerFactory

from .meshcat_utils import draw_frame, draw_obstacles, draw_tcp_paths, load_robot


def rollout_expert(dyn: FrankaSystem, planner: Planner, initial_state: np.ndarray, num_steps: int, noise_std: float, noise_rng) -> dict:
    """One closed-loop expert rollout; stops at the goal or when the planner fails."""
    state = dyn.reset(initial_state)
    termination = RolloutTerminationTracker(dyn)
    planner.reset()
    states, actions, solve_times = [state.copy()], [], []
    outcome = "timeout"
    if termination.update(state):
        outcome = "reached"
    for _ in range(0 if outcome == "reached" else num_steps):
        t0 = time.time()
        try:
            action = planner.plan(dyn.observe(state)).actions[0]
        except PlannerSolveError as exc:
            outcome = f"planner failure ({str(exc)[:60]})"
            break
        solve_times.append(time.time() - t0)
        executed = apply_execution_noise(dyn, action, noise_std, noise_rng)
        state = dyn.step(state, executed)
        states.append(state.copy())
        actions.append(executed)
        if dyn.collides(state):
            outcome = "collision"
            break
        if termination.update(state):
            outcome = "reached"
            break
    return {"states": np.array(states), "actions": np.array(actions).reshape(-1, dyn.nu), "solve_times": solve_times, "outcome": outcome}


def summarize(dyn: FrankaSystem, rollout: dict) -> dict:
    states, actions = rollout["states"], rollout["actions"]
    position_error, rotation_error = dyn.goal_errors(states[-1])
    stats = {
        "steps": len(actions),
        "position_error": position_error,
        "rotation_error": rotation_error,
        "max_joint_speed": float(np.abs(states[:, dyn.n : 2 * dyn.n]).max()),
        "ms_per_step": 1000.0 * float(np.mean(rollout["solve_times"])) if rollout["solve_times"] else 0.0,
    }
    arm = states[:, :7]
    tcp = np.array([dyn.tcp_placement(s[: dyn.n]).translation for s in states])
    # Path length relative to the straight line start -> end: 1.0 = no detour, large = the arm wanders
    stats["joint_path"] = float(np.linalg.norm(np.diff(arm, axis=0), axis=1).sum())
    stats["joint_ratio"] = stats["joint_path"] / max(float(np.linalg.norm(arm[-1] - arm[0])), 1e-2)
    stats["tcp_ratio"] = float(np.linalg.norm(np.diff(tcp, axis=0), axis=1).sum()) / max(float(np.linalg.norm(tcp[-1] - tcp[0])), 1e-2)
    if len(actions) > 1:
        normalized = actions / dyn.max_action
        stats["at_jerk_limit"] = float(np.mean(np.abs(normalized) > 0.99))
        stats["jerkiness"] = float(np.mean(np.abs(np.diff(normalized, axis=0))))  # mean change of the normalized action per step
    else:
        stats.update(at_jerk_limit=0.0, jerkiness=0.0)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("scripts/config/franka_casadi_config.yaml"), help="expert yaml config (default: %(default)s)")
    parser.add_argument("--num-traj", type=int, default=10, help="number of random rollouts, seeds 0..N-1 (default: %(default)s)")
    parser.add_argument("--seeds", type=str, default=None, help="explicit seeds instead of 0..N-1, e.g. '[0, 3, 7]'")
    parser.add_argument("--num-steps", type=int, default=250, help="maximum steps per rollout (default: %(default)s)")
    parser.add_argument("--tolerance-overrides", type=str, default=None, help="dict of pos_tol / vel_tol overriding the config's")
    parser.add_argument("--action-noise-std", type=float, default=0.0, help="execution noise on the expert's action [rad/s^3], like DAgger collection (default: %(default)s)")
    parser.add_argument("--no-viewer", action="store_true", help="print the statistics only")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time (default: %(default)s)")
    args = parser.parse_args()

    config = load_and_validate_system_config(args.config)
    initialize_ompl(config["ompl_seed"])
    if args.tolerance_overrides:
        config = validate_system_config({**config, **ast.literal_eval(args.tolerance_overrides)})
    seeds = [int(s) for s in ast.literal_eval(args.seeds)] if args.seeds else list(range(args.num_traj))
    dyn = FrankaSystem(config)
    planner = PlannerFactory.create(config["expert"], dyn, config)
    noise_seed = default_action_noise_seed_for_config(config)
    print(f"expert: horizon={config['horizon']} dt={config['dt']} | tolerances pos={config['pos_tol']} vel={config['vel_tol']}")

    baseline_goal = dyn.goal.copy()
    results = []
    for number, seed in enumerate(seeds, 1):
        dyn.set_goal(baseline_goal)
        initial_state = sample_initial_state(dyn, seed)
        rollout = rollout_expert(dyn, planner, initial_state, args.num_steps, args.action_noise_std, action_noise_rng_for_rollout(noise_seed, seed_spec=seed))
        stats = summarize(dyn, rollout)
        results.append((seed, dyn.goal.copy(), rollout, stats))
        print(
            f"[{number}/{len(seeds)}] seed {seed:3d}: {rollout['outcome']:10s} steps {stats['steps']:3d} | final error {stats['position_error']:.4f} m"
            f" | max |v| {stats['max_joint_speed']:.2f} rad/s"
            f" | at jerk limit {100 * stats['at_jerk_limit']:4.1f}% | jerkiness {stats['jerkiness']:.3f}"
            f" | joint path {stats['joint_path']:.1f} rad (x{stats['joint_ratio']:.1f} straight) | TCP path x{stats['tcp_ratio']:.1f} straight | {stats['ms_per_step']:.0f} ms/step",
            flush=True,
        )

    reached = [r[2]["outcome"] == "reached" for r in results]
    print(f"\ngoal reached in {sum(reached)}/{len(results)} rollouts")
    for kind in ("collision", "timeout", "planner failure"):
        count = sum(r[2]["outcome"].startswith(kind) for r in results)
        if count:
            print(f"  {kind}: {count}")
    print(f"  mean steps {np.mean([r[3]['steps'] for r in results]):.0f}, "
          f"mean jerkiness {np.mean([r[3]['jerkiness'] for r in results]):.3f}, mean at-limit fraction {100 * np.mean([r[3]['at_jerk_limit'] for r in results]):.1f}%,"
          f" mean joint path x{np.mean([r[3]['joint_ratio'] for r in results]):.1f} / TCP path x{np.mean([r[3]['tcp_ratio'] for r in results]):.1f} the straight line")
    if args.no_viewer:
        return

    robot = load_robot()
    viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer(open=True)
    viz.loadViewerModel()
    draw_obstacles(viz, dyn.obstacles)
    tcp_paths = [np.array([dyn.tcp_placement(s[: dyn.n]).translation for s in r[2]["states"]]) for r in results]
    draw_tcp_paths(viz, tcp_paths, [r[1][:3] for r in results], reached)
    draw_frame(viz, "goal")
    print("\nPlaying the rollouts in a loop (Ctrl+C to stop): green path = reached the goal, orange = did not")
    try:
        while True:
            for (seed, goal, rollout, stats), path in zip(results, tcp_paths):
                print(f"seed {seed}: {rollout['outcome']}")
                viz.viewer["goal"].set_transform(dyn.goal_transform(goal))
                for state in rollout["states"]:
                    viz.display(state[: dyn.n])
                    time.sleep(dyn.dt / args.speed)
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
