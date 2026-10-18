"""Roll a trained policy out in the simulator and play it in Meshcat, next to what the expert would do.

    venv/bin/python -m scripts.meshcat_policy --model-dir outputs/train_dagger/franka_flow/flow_dagger_checkpoint.pt \
        --seeds "[10000, 10001]" --show-expert

Starts and goals are sampled from the seeds like the DAgger evaluation does (--seeds). At every step the policy's
predicted action chunk is drawn as the TCP path of the motion it leads to (cyan) and, with --show-expert, the plan the
expert would make from the very same state (green): the two are the network's output and the label it is trained on.
The executed action is the first row of the policy's chunk. --no-viewer prints the outcome of every rollout only
(reached, collision, timeout, solve failure).
"""

import argparse
import ast
import time
from pathlib import Path
from typing import NamedTuple

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
import torch
from pinocchio.visualize import MeshcatVisualizer

from dynamics.franka_dynamics import FrankaDynamics
from learning.dagger.utils import sample_initial_state
from planning.config import load_and_validate_system_config, validate_system_config
from planning.factory import DynamicsFactory, PlannerFactory
from planning.planner import Planner, PlannerSolveError

from .evaluate_policy import load_policy
from .meshcat_utils import action_tcp_points, draw_plan

POLICY_COLOR = 0x00D0FF
EXPERT_COLOR = 0x20D040


class PolicyRollout(NamedTuple):
    states: np.ndarray  # (T + 1, 2n) visited states
    chunks: np.ndarray  # (T, H, nu) the policy's predicted action chunk at states[:T]
    expert_plans: np.ndarray | None  # (T, H, nu) the expert's planned actions from the same states, None without an expert
    outcome: str
    detail: str


def rollout_policy(
    dyn: FrankaDynamics,
    policy,
    initial_state: np.ndarray,
    num_steps: int,
    device: torch.device,
    planner: Planner | None = None,
) -> PolicyRollout:
    """Closed loop: the first row of the policy's chunk is executed; the expert (if any) only plans from each visited state."""
    state = dyn.reset(initial_state)
    policy.reset()
    if planner is not None:
        planner.reset()
    states, chunks, expert_plans = [state.copy()], [], []
    outcome, detail = "timeout", ""
    if dyn.should_terminate_rollout(state):
        outcome = "reached"
    for _ in range(0 if outcome == "reached" else num_steps):
        observation = dyn.observe(state)
        policy_input = {name: torch.as_tensor(value)[None].to(device) for name, value in dyn.policy_observation(observation).items()}
        try:
            with torch.inference_mode():
                chunk = policy.select_action(policy_input)[0].detach().cpu().numpy().astype(float)
        except PlannerSolveError as error:
            outcome, detail = "solve failure", str(error)[:80]
            break
        if planner is not None:
            try:
                expert_plans.append(planner.plan(observation).first_actions(len(chunk)))
            except PlannerSolveError:
                expert_plans.append(np.full((len(chunk), dyn.nu), np.nan))
        chunks.append(chunk)
        state = dyn.step(state, chunk[0])
        states.append(state.copy())
        if dyn.is_collision(state):
            outcome = "collision"
            break
        if dyn.should_terminate_rollout(state):
            outcome = "reached"
            break
    shape = (0, 0, dyn.nu)
    return PolicyRollout(
        np.array(states),
        np.array(chunks) if chunks else np.empty(shape),
        (np.array(expert_plans) if expert_plans else np.empty(shape)) if planner is not None else None,
        outcome,
        detail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, help="DAgger checkpoint, e.g. outputs/train_dagger/<name>/flow_dagger_checkpoint.pt")
    parser.add_argument("--policy-type", choices=["flow", "safeflow"], default=None, help="default: the checkpoint's own policy type")
    parser.add_argument("--config", type=Path, default=Path("scripts/config/franka_casadi_config.yaml"), help="expert yaml config (default: %(default)s)")
    parser.add_argument("--seeds", type=str, default="[10000]", help="seeds of the starts and goals, e.g. '[10000, 10001]' (default: %(default)s)")
    parser.add_argument("--num-steps", type=int, default=125, help="maximum steps per rollout (default: %(default)s)")
    parser.add_argument("--tolerance-overrides", type=str, default=None, help="dict of pos_tol / vel_tol overriding the config's")
    parser.add_argument("--show-expert", action="store_true", help="also plan with the expert from every visited state and draw its plan")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu", help="device of the policy (default: %(default)s)")
    parser.add_argument("--no-viewer", action="store_true", help="print the outcomes only")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time (default: %(default)s)")
    args = parser.parse_args()

    config = load_and_validate_system_config(args.config)
    if args.tolerance_overrides:
        config = validate_system_config({**config, **ast.literal_eval(args.tolerance_overrides)})
    device = torch.device(args.device)
    dyn = DynamicsFactory.create(config["system"], config)
    policy_type = args.policy_type or torch.load(args.model_dir, map_location="cpu", weights_only=True)["policy_type"]
    policy = load_policy(policy_type, args.model_dir, dyn, config, device)
    planner = PlannerFactory.create(config["expert"], dyn, config) if args.show_expert else None
    seeds = [int(s) for s in ast.literal_eval(args.seeds)]

    baseline_goal = dyn.goal.copy()
    rollouts = []
    for seed in seeds:
        dyn.set_goal(baseline_goal)
        initial_state = sample_initial_state(dyn, seed)
        goal = dyn.goal.copy()
        rollout = rollout_policy(dyn, policy, initial_state, args.num_steps, device, planner)
        position_error, rotation_error = dyn.goal_errors(rollout.states[-1])
        print(
            f"seed {seed}: {rollout.outcome}{' (' + rollout.detail + ')' if rollout.detail else ''} after {len(rollout.chunks)} steps | "
            f"final error {position_error:.4f} | "
            f"largest planned excursion of the policy {max([np.abs(np.diff(dyn.rollout_positions(s, c), axis=0)).sum(axis=0).max() for s, c in zip(rollout.states, rollout.chunks)], default=0.0):.2f}"
        )
        rollouts.append((seed, goal, rollout))
    if args.no_viewer:
        return

    if config["system"] == "franka":
        robot = dyn.robot
        viz = MeshcatVisualizer(robot.model, robot.collision_model, robot.visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        show = lambda state: viz.display(state[: dyn.n])
    else:  # a point mass has no model: a sphere at its position (padded to three axes)
        viz = type("Viewer", (), {"viewer": meshcat.Visualizer()})()
        viz.viewer.open()
        viz.viewer["point"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0x2060FF))
        show = lambda state: viz.viewer["point"].set_transform(tf.translation_matrix(dyn.tcp_position(state[: dyn.n])))
    viz.viewer["goal"].set_object(g.Sphere(0.03), g.MeshLambertMaterial(color=0xFF0000, opacity=0.6))
    print("Playing the rollouts in a loop (Ctrl+C to stop): red ball = goal TCP, cyan line = the policy's chunk"
          + (", green line = the expert's plan from the same state" if args.show_expert else ""))
    try:
        while True:
            for seed, goal, rollout in rollouts:
                print(f"seed {seed}: {rollout.outcome} after {len(rollout.chunks)} steps")
                viz.viewer["goal"].set_transform(tf.translation_matrix(dyn.tcp_position(goal)))
                for index, state in enumerate(rollout.states):
                    show(state)
                    if index < len(rollout.chunks):
                        draw_plan(viz, "policy_plan", action_tcp_points(dyn, state, rollout.chunks[index]), POLICY_COLOR)
                        if rollout.expert_plans is not None and not np.isnan(rollout.expert_plans[index]).any():
                            draw_plan(viz, "expert_plan", action_tcp_points(dyn, state, rollout.expert_plans[index]), EXPERT_COLOR)
                    time.sleep(dyn.dt / args.speed)
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
