from __future__ import annotations

import argparse
import ast
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.initial_state_utils import parse_goal_states_argument, parse_initial_states_argument
from learning.config_loaders import (
    default_checkpoint_dir,
    default_dataset_root,
    default_repo_id,
    load_dagger_training_config,
    load_flow_config,
    load_policy_type,
    load_prediction_horizon,
    load_unet_config,
)
from learning.dagger import DaggerConfig, DaggerTrainer
from planning.config import load_and_validate_system_config

"""High-level CLI plumbing and orchestration for DAgger training.

The reusable DAgger pipeline components live under learning/dagger/:
DaggerConfig (learning/dagger/dagger_config.py) and DaggerTrainer
(learning/dagger/dagger_trainer.py) hold the actual training logic; this
file only parses CLI args/policy YAML, resolves them into a DaggerConfig,
and hands off to DaggerTrainer.
"""


def _yaml_safe(value: Any) -> Any:
    """Recursively coerce a value into plain types yaml.safe_dump can render."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


def _print_yaml_block(title: str, data: dict[str, Any]) -> None:
    print(f"{title}:")
    for line in yaml.safe_dump(_yaml_safe(data), sort_keys=False).splitlines():
        print(f"  {line}")


def _parse_dict_override(raw_value: Any, flag_name: str, example: str) -> dict[str, float] | None:
    if not isinstance(raw_value, str):
        return raw_value
    try:
        parsed = ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse {flag_name}. Use Python-literal dict syntax like '{example}'."
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must evaluate to a dict.")
    return parsed


def _summarize_cli_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment_name": args.experiment_name,
        "expert_config": args.expert_config,
        "dagger_iterations": args.dagger_iterations,
        "trajectories_per_iteration": args.trajectories_per_iteration,
        "steps_per_trajectory": args.steps_per_trajectory,
        "action_noise_std": args.action_noise_std,
        "expert_beta_per_iteration": args.expert_beta_per_iteration,
        "training_curriculum": args.training_curriculum,
        "round_seeds": args.round_seeds,
        "restart_round_seed": args.restart_round_seed,
        "initial_states_count": len(parse_initial_states_argument(args.initial_states)) if isinstance(args.initial_states, str) and args.initial_states else (len(args.initial_states) if args.initial_states is not None else 0),
        "goal_states_count": len(parse_goal_states_argument(args.goal_states)) if isinstance(args.goal_states, str) and args.goal_states else (len(args.goal_states) if args.goal_states is not None else 0),
        "eval_episodes": args.eval_episodes,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "policy_config": str(args.policy_config),
        "checkpoint_dir": str(args.checkpoint_dir) if args.checkpoint_dir is not None else None,
        "seed": args.seed,
        "num_workers": args.num_workers,
    }


def _summarize_dagger_config(cfg: Any) -> dict[str, Any]:
    return {
        "dagger_iterations": cfg.dagger_iterations,
        "trajectories_per_iteration": cfg.trajectories_per_iteration,
        "steps_per_trajectory": cfg.steps_per_trajectory,
        "action_noise_std": cfg.action_noise_std,
        "expert_beta_per_iteration": cfg.expert_beta_per_iteration,
        "training_curriculum": cfg.training_curriculum,
        "initial_states_count": len(cfg.initial_states) if cfg.initial_states is not None else 0,
        "goal_states_count": len(cfg.goal_states) if cfg.goal_states is not None else 0,
        "eval_episodes": cfg.eval_episodes,
        "eval_seed_start": cfg.eval_seed_start,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "policy_type": cfg.policy_type,
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "seed": cfg.seed,
        "num_workers": cfg.num_workers,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a policy with DAgger")
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--expert-config", required=True)
    p.add_argument("--repo-id")
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--dagger-iterations", type=int)
    p.add_argument("--trajectories-per-iteration", nargs="+", type=int)
    p.add_argument("--steps-per-trajectory", type=int)
    p.add_argument("--action-noise-std", type=float)
    p.add_argument(
        "--training-curriculum",
        nargs="+",
        type=str,
        default=None,
        help=(
            "one 'random' or 'config' entry per DAgger round, selecting whether that round's "
            "rollouts use random goals/states or the explicit --initial-states/--goal-states lists."
        ),
    )
    p.add_argument(
        "--round-seeds",
        nargs="+",
        type=int,
        default=None,
        help=(
            "one base seed per DAgger round, controlling which seed drives that round's "
            "random initial-state/goal sampling instead of the single global --seed."
        ),
    )
    p.add_argument(
        "--restart-round-seed",
        action=argparse.BooleanOptionalAction,
        help=(
            "when set, rounds sharing the same --round-seeds value replay identical random "
            "initial-state/goal draws (a true stream restart); when unset, the round index "
            "still varies the draws even if the nominal seed repeats (today's behavior)."
        ),
    )
    p.add_argument(
        "--initial-states",
        type=str,
        default=None,
        help=(
            "explicit initial state specs. Examples: '[q1, ..., q7, v1, ..., v7]' for one rollout "
            "or '[[...], [...]]' for multiple rollouts. "
            "When exhausted, collection falls back to simulator RNG sampling."
        ),
    )
    p.add_argument(
        "--goal-states",
        type=str,
        default=None,
        help=(
            "explicit goal state specs, paired index-for-index with --initial-states. "
            "Examples: '[q1, ..., q7]' for one rollout or '[[...], [...]]' for multiple rollouts. "
            "When either list is exhausted, remaining rollouts in that round fall back to random."
        ),
    )
    p.add_argument(
        "--tolerance-overrides",
        type=str,
        default=None,
        help=(
            "per-experiment override for the expert config's convergence tolerances, as a Python-literal "
            "dict of the tolerance keys pos_tol and vel_tol, e.g. "
            "'{\"pos_tol\": 0.01, \"vel_tol\": 0.05}'."
        ),
    )
    p.add_argument(
        "--eval-tolerance-overrides",
        type=str,
        default=None,
        help=(
            "same format as --tolerance-overrides, but applied only to evaluation rollouts, not "
            "data collection/training. Falls back to --tolerance-overrides when omitted."
        ),
    )
    p.add_argument("--expert-beta-per-iteration", nargs="+", type=float)
    p.add_argument("--target-epochs-per-round", nargs="+", type=float)
    p.add_argument("--eval-episodes", type=int)
    p.add_argument("--eval-steps", type=int)
    p.add_argument("--eval-seed-start", type=int)
    p.add_argument("--eval-action-noise-std", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument(
        "--policy-config",
        type=Path,
        default=Path(PROJECT_ROOT) / "learning/config/franka_casadi_flow_config.yaml",
    )
    p.add_argument("--checkpoint-dir", type=Path)
    p.add_argument("--seed", type=int)
    p.add_argument("--max-train-steps", type=int)
    p.add_argument("--num-workers", type=int, help="parallel worker processes for rollout collection and evaluation")
    return p.parse_args()


def save_experiment_configs(args: argparse.Namespace, experiment_dir: Path, repo_id: str, dataset_root: Path) -> None:
    """Save experiment configs and run arguments to experiment directory."""
    shutil.copy2(args.expert_config, experiment_dir / "expert_config.yaml")
    shutil.copy2(args.policy_config, experiment_dir / "policy_config.yaml")

    run_args = vars(args).copy()
    run_args["repo_id"] = repo_id
    run_args["dataset_root"] = dataset_root
    run_args["checkpoint_dir"] = experiment_dir
    run_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_args.items()
    }

    with (experiment_dir / "run_args.yaml").open("w") as f:
        yaml.safe_dump(run_args, f, sort_keys=False)


def main() -> None:
    args = parse_args()
    _print_yaml_block(
        "CLI arguments (summary; full initial_states/goal_states omitted)",
        _summarize_cli_args(args),
    )
    validated = load_and_validate_system_config(args.expert_config)
    training_config = load_dagger_training_config(args.policy_config)

    def option(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        return cli_value if cli_value is not None else training_config.get(name, default)

    dagger_iterations = int(option("dagger_iterations", 4))
    trajectories_per_iteration = [
        int(value) for value in option("trajectories_per_iteration", [20])
    ]
    target_epochs_per_round = [
        float(value) for value in option("target_epochs_per_round", [30.0])
    ]
    expert_beta_per_iteration = [
        float(value) for value in option("expert_beta_per_iteration", [0.8])
    ]
    initial_states_config = option("initial_states", None)
    initial_states = (
        parse_initial_states_argument(initial_states_config)
        if isinstance(initial_states_config, str)
        else initial_states_config
    )
    goal_states_config = option("goal_states", None)
    goal_states = (
        parse_goal_states_argument(goal_states_config)
        if isinstance(goal_states_config, str)
        else goal_states_config
    )
    training_curriculum_config = option("training_curriculum", None)
    round_seeds_config = option("round_seeds", None)
    restart_round_seed = bool(option("restart_round_seed", False))
    tolerance_overrides = _parse_dict_override(
        option("tolerance_overrides", None),
        "--tolerance-overrides",
        '{"pos_tol": 0.01, "vel_tol": 0.05}',
    )
    eval_tolerance_overrides = _parse_dict_override(
        option("eval_tolerance_overrides", None),
        "--eval-tolerance-overrides",
        '{"pos_tol": 0.02, "vel_tol": 0.05}',
    )
    if (args.repo_id is None) != (args.dataset_root is None):
        raise ValueError("Provide both --repo-id and --dataset-root together, or omit both.")
    fresh = args.repo_id is None
    timestamp = time.time_ns()
    repo_id = default_repo_id(timestamp) if fresh else str(args.repo_id)
    dataset_root = default_dataset_root(timestamp) if fresh else Path(args.dataset_root)
    trajectories, epochs, betas, training_curriculum, round_seeds = DaggerTrainer.schedules(
        trajectories_per_iteration,
        target_epochs_per_round,
        expert_beta_per_iteration,
        dagger_iterations,
        training_curriculum_config,
        round_seeds_config,
    )

    experiment_dir = (args.checkpoint_dir or default_checkpoint_dir()) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_configs(
        args=args,
        experiment_dir=experiment_dir,
        repo_id=repo_id,
        dataset_root=dataset_root,
    )

    cfg = DaggerConfig(
        experiment_config=validated,
        repo_id=repo_id,
        dataset_root=dataset_root,
        start_with_aggregation=fresh,
        dagger_iterations=dagger_iterations,
        trajectories_per_iteration=trajectories,
        steps_per_trajectory=int(option("steps_per_trajectory", 150)),
        action_noise_std=float(option("action_noise_std", 0.0)),
        initial_states=initial_states,
        goal_states=goal_states,
        training_curriculum=training_curriculum,
        round_seeds=round_seeds,
        restart_round_seed=restart_round_seed,
        tolerance_overrides=tolerance_overrides,
        eval_tolerance_overrides=eval_tolerance_overrides,
        expert_beta_per_iteration=betas,
        target_epochs_per_round=epochs,
        eval_episodes=int(option("eval_episodes", 10)),
        eval_steps=option("eval_steps", None),
        eval_seed_start=int(option("eval_seed_start", 10000)),
        eval_action_noise_std=float(option("eval_action_noise_std", 0.0)),
        batch_size=int(option("batch_size", 1024)),
        learning_rate=float(option("learning_rate", 1e-4)),
        unet=load_unet_config(args.policy_config),
        prediction_horizon=load_prediction_horizon(args.policy_config),
        policy_type=load_policy_type(args.policy_config),
        flow_config=load_flow_config(args.policy_config),
        checkpoint_dir=experiment_dir,
        seed=int(option("seed", 99)),
        max_train_steps=option("max_train_steps", None),
        num_workers=int(option("num_workers", 1)),
    )
    _print_yaml_block(
        "Resolved DaggerConfig (summary; full explicit states omitted)",
        _summarize_dagger_config(cfg),
    )
    DaggerTrainer(cfg).run()


if __name__ == "__main__":
    main()
