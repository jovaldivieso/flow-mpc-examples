import os
import sys
import time
import torch
import ast
import argparse
import numpy as np
from typing import Any, Mapping

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from planning.config import load_and_validate_system_config, validate_system_config
from systems.initial_state_utils import (
    normalize_goal_state_specs,
    normalize_initial_state_specs,
    parse_goal_states_argument,
    parse_initial_states_argument,
)
from systems.seed_utils import (
    DEFAULT_SINGLE_ROBOT_SEED,
    action_noise_seed_for_rollout,
    default_action_noise_seed_for_config,
)
from systems.rollout_termination import RolloutTerminationTracker
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from learning.config_loaders import UnetConfig
from learning.data_utils import build_dataset_features
from learning.dagger import apply_config_overrides, build_policy_action
from learning.dagger.policy_action import apply_execution_noise
from learning.dagger.utils import rng_for_seed_spec as _rng_for_seed_spec, sample_initial_state
from learning.models.policy import ActionPolicy
from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.ompl_planner import initialize_ompl
from planning.planner import Planner, PlannerSolveError
from planning.planner_factory import PlannerFactory


def resolve_checkpoint_observation_dimensions(
    checkpoint: Mapping[str, Any],
    simulator: SystemSimulator,
    requested_policy_type: str,
) -> tuple[int, int]:
    """Resolve and validate checkpoint dimensions against the evaluation simulator."""
    features = build_dataset_features(simulator, int(checkpoint.get("prediction_horizon", 1)))

    # 'flow' and 'safeflow' share the exact same underlying network (safeflow
    # only wraps it with a CasADi projection at inference time), so a checkpoint
    # trained under either is interchangeable with the other.
    _POLICY_TYPE_EQUIVALENCE = {"flow": {"flow", "safeflow"}, "safeflow": {"flow", "safeflow"}}
    checkpoint_policy_type = checkpoint.get("policy_type")
    if checkpoint_policy_type is not None:
        checkpoint_type_normalized = str(checkpoint_policy_type).lower()
        compatible_types = _POLICY_TYPE_EQUIVALENCE.get(checkpoint_type_normalized, {checkpoint_type_normalized})
        if requested_policy_type not in compatible_types:
            raise ValueError(
                f"Checkpoint was trained as '{checkpoint_policy_type}', but evaluation requested "
                f"'{requested_policy_type}'."
            )

    environment_state_dim = int(features["observation.environment_state"]["shape"][0])
    proprioception_dim = int(features["observation.state"]["shape"][0])
    expected_state_dim = environment_state_dim + proprioception_dim
    state_dim = int(checkpoint.get("state_dim", expected_state_dim))
    if state_dim != expected_state_dim:
        raise ValueError(
            "Checkpoint ego observation schema is incompatible with the evaluation simulator: "
            f"checkpoint=(state_dim={state_dim}), "
            f"expected_state_dim={expected_state_dim}. "
            "Use a checkpoint trained with the same ego observation schema."
        )
    # The dataset's action is the flattened chunk (prediction_horizon rows), the policy's action_dim one row: one action
    expected_action_dim = int(simulator.nu)
    action_dim = int(checkpoint.get("action_dim", expected_action_dim))
    if action_dim != expected_action_dim:
        raise ValueError(
            "Checkpoint action schema is incompatible with the evaluation simulator: "
            f"checkpoint=(action_dim={action_dim}), "
            f"expected_action_dim={expected_action_dim}. "
            "Use a checkpoint trained with the same action schema."
        )
    return state_dim, action_dim


def parse_seed_argument(raw_seeds: str | None) -> list[int]:
    if raw_seeds is None:
        return []

    text = raw_seeds.strip()
    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parts = [chunk.strip() for chunk in text.split(",") if chunk.strip()]
        if len(parts) > 1:
            return [int(part) for part in parts]
        return [int(text)]

    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, list):
        if len(parsed) == 0:
            return []
        if all(isinstance(x, int) for x in parsed):
            return [int(x) for x in parsed]

    raise ValueError(
        "Unable to parse seeds. Use an int (e.g. '42') or a list (e.g. '[42, 7]')."
    )


def normalize_seed_specs(
    seeds: list[int] | None,
) -> list[int]:
    if seeds is None or len(seeds) == 0:
        seeds = [DEFAULT_SINGLE_ROBOT_SEED]

    return [int(seed) for seed in seeds]


def get_inference_device():
    """Return the best available device for policy inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _synchronize_device(device: torch.device) -> None:
    """Block until pending async accelerator work completes, for fair wall-clock timing.

    Cheap on CPU (no-op). Needed on cuda/mps because kernel launches are
    asynchronous; without this a timer around a GPU call would measure launch
    overhead, not actual compute.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def rollout_planner(
    simulator: SystemSimulator,
    planner: Planner,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    rollout_id: int | None = None,
    seed_value: Any | None = None,
    initial_state_source: str | None = None,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[float]]:
    """
    Rolls out the expert planner from a given initial state.

    Returns (trajectory, solve_times): solve_times holds one wall-clock
    duration (seconds) per successful planner solve, for benchmarking the
    expert against the policy.
    """
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    planner.reset()
    trajectory = [state.copy()]
    solve_times: list[float] = []

    if termination.update(state):
        return np.asarray(trajectory), solve_times

    for _ in range(num_steps):
        obs = simulator.observe(state)

        solve_start = time.perf_counter()
        try:
            action = planner.plan(obs).actions[0]
        except PlannerSolveError as exc:
            rollout_label = "?" if rollout_id is None else str(rollout_id)
            print(
                "Expert planner failed during evaluation "
                f"(rollout={rollout_label}, source={initial_state_source}, seed={seed_value}, "
                f"action_noise_std={action_noise_std:.6f}): {exc} "
                f"initial_state={np.array2string(np.asarray(initial_state), precision=6)}, "
                f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                f"goal_state={np.array2string(np.asarray(simulator.goal), precision=6)}"
            )
            break
        solve_times.append(time.perf_counter() - solve_start)

        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )
        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

        if termination.update(state):
            break

    return np.asarray(trajectory), solve_times


def rollout_policy(
    simulator: SystemSimulator,
    policy: ActionPolicy,
    device: torch.device,
    initial_state: np.ndarray,
    num_steps: int,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool, int, bool, list[float]]:
    """
    Rolls out the neural policy from a given initial state.

    returns:
        trajectory: array containing visited simulator states
        reached_goal: whether simulator reached goal state
        steps_taken: number of executed simulation steps
        solve_failed: whether the policy's own forward pass raised
            (e.g. a SafeFlow projector solve failure) instead of running
            out of steps -- distinct from timeout so callers can report it
            as its own failure mode
        solve_times: one wall-clock duration (seconds) per step, for the
            single call that produces the action
    """
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    trajectory = [state.copy()]
    policy.reset()
    solve_times: list[float] = []
    if termination.update(state):
        return np.asarray(trajectory), True, 0, False, solve_times

    for step in range(1, num_steps + 1):
        observation = simulator.observe(state)

        _synchronize_device(device)
        solve_start = time.perf_counter()
        try:
            action = build_policy_action(
                simulator=simulator,
                policy=policy,
                observation=observation,
                device=device,
            )
        except PlannerSolveError as exc:
            print(f"Policy rollout solve failed (step={step}): {exc}")
            return np.asarray(trajectory), False, len(trajectory) - 1, True, solve_times
        _synchronize_device(device)
        solve_times.append(time.perf_counter() - solve_start)

        executed_action = apply_execution_noise(
            simulator=simulator,
            action=action,
            action_noise_std=action_noise_std,
            rng=action_noise_rng,
        )

        state = simulator.step(state, executed_action)
        trajectory.append(state.copy())

        if termination.update(state):
            return np.asarray(trajectory), True, step, False, solve_times

    return np.asarray(trajectory), False, len(trajectory) - 1, False, solve_times


def _load_checkpoint_policy_components(
    model_dir: str,
    simulator: SystemSimulator,
    policy_type: str,
    device: torch.device,
) -> tuple[
    Mapping[str, Any],
    Mapping[str, torch.Tensor],
    int,
    int,
    UnetConfig,
    int,
]:
    """Load a metadata checkpoint and resolve its state/action/network-size schema.

    Returns (checkpoint, state_dict, state_dim, action_dim, unet,
    prediction_horizon).
    """
    checkpoint = torch.load(model_dir, map_location=device, weights_only=True)
    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise ValueError(
            f"'{policy_type}' models require a metadata checkpoint to infer the prediction_horizon "
            "and cannot be loaded from raw state dictionaries."
        )
    # Inference runs with the exponential moving average of the weights when the checkpoint has one
    state_dict = checkpoint.get("ema_model_state_dict", checkpoint["model_state_dict"])
    if not isinstance(state_dict, Mapping):
        raise ValueError(f"'{policy_type}' checkpoint must contain a state dictionary.")

    prediction_horizon = int(checkpoint.get("prediction_horizon", 1))

    state_dim, action_dim = resolve_checkpoint_observation_dimensions(
        checkpoint, simulator, policy_type
    )
    unet_raw = checkpoint.get("unet")
    if not isinstance(unet_raw, Mapping):
        raise ValueError(
            f"'{policy_type}' checkpoint has no 'unet' entry (the size of its temporal U-Net)."
        )
    unet = UnetConfig(dim=int(unet_raw["dim"]), dim_mults=tuple(int(m) for m in unet_raw["dim_mults"]))

    return (
        checkpoint,
        state_dict,
        state_dim,
        action_dim,
        unet,
        prediction_horizon,
    )


def load_policy(
    policy_type: str,
    model_dir: str,
    simulator: SystemSimulator,
    planner_config: Mapping[str, Any],
    device: torch.device,
):
    """The policy of a DAgger checkpoint (the EMA weights when it has them), in eval mode on ``device``.

    ``planner_config`` is the validated expert config a ``safeflow`` policy builds its projector from.
    """
    checkpoint, state_dict, state_dim, action_dim, unet, prediction_horizon = (
        _load_checkpoint_policy_components(model_dir, simulator, policy_type, device)
    )
    flow_config = checkpoint["flow_config"]
    spec = PolicySpec(
        policy_type=policy_type,
        action_dim=action_dim,
        state_dim=state_dim,
        prediction_horizon=prediction_horizon,
        unet_dim=unet.dim,
        unet_dim_mults=unet.dim_mults,
        num_inference_steps=int(flow_config["num_inference_steps"]),
        action_scale=tuple(flow_config["action_scale"]),
        flow_init=flow_config["safeflow_init"],
    )
    policy = PolicyFactory.create(spec, simulator, planner_config)
    policy.load_network_state_dict(state_dict)

    policy.eval()
    policy.to(device)

    policy.reset()
    return policy


def checkpoint_policy_type(model_dir: str) -> str:
    """The policy type a DAgger checkpoint was trained as."""
    return torch.load(model_dir, map_location="cpu", weights_only=True)["policy_type"]


def run_evaluation(
    policy_type: str,
    config: Mapping[str, Any],
    model_dir: str,
    num_steps: int = 125,
    seeds: list[int] | None = None,
    initial_states: Any | None = None,
    goal_states: Any | None = None,
    tolerance_overrides: Mapping[str, float] | None = None,
    action_noise_std: float = 0.0,
    output_path: str | None = None,
    device_override: str | None = None,
):
    if tolerance_overrides:
        config = apply_config_overrides(config, tolerance_overrides)
    validated_config = validate_system_config(raw_config=config)
    action_noise_seed = default_action_noise_seed_for_config(validated_config)

    simulator = SystemFactory.create(validated_config["system"], validated_config)
    seed_specs = normalize_seed_specs(seeds=seeds)
    initial_state_specs = normalize_initial_state_specs(
        simulator=simulator,
        initial_states=initial_states,
    )
    goal_state_specs = normalize_goal_state_specs(
        simulator=simulator,
        goal_states=goal_states,
    )

    device = torch.device(device_override) if device_override else get_inference_device()
    print(f"running inference on {device}")
    print(f"action noise seed: {action_noise_seed}")

    policy = load_policy(policy_type, model_dir, simulator, validated_config, device)

    # Instantiate expert planner once and reset it for each rollout.
    expert_planner = PlannerFactory.create(validated_config["expert"], simulator, validated_config)
    expert_trajectories: list[np.ndarray] = []
    policy_trajectories: list[np.ndarray] = []
    rollout_goal_states: list[np.ndarray] = []
    per_seed_metrics: list[dict[str, Any]] = []

    if len(initial_state_specs) > 0:
        total_rollouts = max(len(seed_specs), len(initial_state_specs), len(goal_state_specs))
        print(
            "evaluating "
            f"{total_rollouts} trajectories "
            f"({len(initial_state_specs)} explicit initial states + seeded/RNG fallback)"
        )
        rollout_plan: list[tuple[Any, str, int, Any, int | None]] = []
        for rollout_idx in range(total_rollouts):
            if rollout_idx < len(initial_state_specs):
                initial_state_spec: Any = np.asarray(initial_state_specs[rollout_idx], dtype=float).copy()
                initial_state_source = "provided"
                seed_value: Any = None
                noise_seed_spec: int | None = None
                if rollout_idx < len(seed_specs):
                    seed_spec = seed_specs[rollout_idx]
                    torch_seed = int(seed_spec)
                    noise_seed_spec = seed_spec
                else:
                    torch_seed = rollout_idx + 1
            elif rollout_idx < len(seed_specs):
                initial_state_spec = seed_specs[rollout_idx]
                initial_state_source = "seeded"
                seed_spec = seed_specs[rollout_idx]
                torch_seed = int(seed_spec)
                seed_value = seed_spec
                noise_seed_spec = seed_spec
            else:
                initial_state_spec = None
                initial_state_source = "rng_fallback"
                torch_seed = rollout_idx + 1
                seed_value = None
                noise_seed_spec = None

            rollout_plan.append((initial_state_spec, initial_state_source, torch_seed, seed_value, noise_seed_spec))
    else:
        total_rollouts = max(len(seed_specs), len(goal_state_specs))
        print(f"evaluating {total_rollouts} trajectories ({len(seed_specs)} seeded + goal/RNG fallback)")
        rollout_plan = []
        for rollout_idx in range(total_rollouts):
            if rollout_idx < len(seed_specs):
                seed_spec = seed_specs[rollout_idx]
                torch_seed = int(seed_spec)
                rollout_plan.append((seed_spec, "seeded", torch_seed, seed_spec, seed_spec))
            else:
                rollout_plan.append((None, "rng_fallback", rollout_idx + 1, None, None))

    baseline_goal = simulator.goal.copy()
    for rollout_idx, (rollout_spec, initial_state_source, torch_seed, seed_value, noise_seed_spec) in enumerate(rollout_plan, start=1):
        torch.manual_seed(torch_seed)

        explicit_goal = rollout_idx - 1 < len(goal_state_specs)
        if explicit_goal:
            simulator.set_goal(goal_state_specs[rollout_idx - 1])
        else:
            simulator.set_goal(baseline_goal)

        if initial_state_source == "seeded":
            seed_spec = rollout_spec
            if explicit_goal:
                initial_state = simulator.random_initial_state(_rng_for_seed_spec(seed_spec))
            else:
                initial_state = sample_initial_state(simulator=simulator, seed_spec=seed_spec)
        elif initial_state_source == "provided":
            if not explicit_goal:
                simulator.randomize_goal_for_reset(
                    _rng_for_seed_spec(noise_seed_spec)
                    if noise_seed_spec is not None
                    else np.random.default_rng(torch_seed)
                )
            initial_state = np.asarray(rollout_spec, dtype=float).copy()
        else:
            if explicit_goal:
                initial_state = simulator.random_initial_state(_rng_for_seed_spec(torch_seed))
            else:
                initial_state = sample_initial_state(simulator=simulator, seed_spec=torch_seed)

        if noise_seed_spec is not None:
            rollout_noise_seed = action_noise_seed_for_rollout(
                action_noise_seed,
                seed_spec=noise_seed_spec,
            )
        else:
            rollout_noise_seed = action_noise_seed_for_rollout(
                action_noise_seed,
                rollout_index=rollout_idx,
            )

        expert_action_noise_rng = np.random.default_rng(rollout_noise_seed)
        policy_action_noise_rng = np.random.default_rng(rollout_noise_seed)

        goal_state = simulator.goal.copy()

        expert_trajectory, expert_solve_times = rollout_planner(
            simulator=simulator,
            planner=expert_planner,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
            rollout_id=rollout_idx,
            seed_value=seed_value,
            initial_state_source=initial_state_source,
            action_noise_rng=expert_action_noise_rng,
        )

        policy_trajectory, reached_goal, steps_taken, policy_solve_failed, policy_solve_times = rollout_policy(
            simulator=simulator,
            policy=policy,
            device=device,
            initial_state=initial_state,
            num_steps=num_steps,
            action_noise_std=action_noise_std,
            action_noise_rng=policy_action_noise_rng,
        )

        expert_solve_time_mean = float(np.mean(expert_solve_times)) if expert_solve_times else None
        policy_solve_time_mean = float(np.mean(policy_solve_times)) if policy_solve_times else None

        policy_collided = any(simulator.collides(state) for state in policy_trajectory)
        policy_final_state = policy_trajectory[-1]
        expert_final_state = expert_trajectory[-1]
        policy_goal_position_error, policy_goal_rotation_error = simulator.goal_errors(policy_final_state)
        expert_goal_position_error, expert_goal_rotation_error = simulator.goal_errors(expert_final_state)

        expert_trajectories.append(expert_trajectory)
        policy_trajectories.append(policy_trajectory)
        rollout_goal_states.append(goal_state)
        per_seed_metrics.append(
            {
                "seed": seed_value,
                "initial_state_source": initial_state_source,
                "goal_state": goal_state,
                "initial_state": initial_state,
                "policy_reached_goal": reached_goal,
                "policy_solve_failed": policy_solve_failed,
                "policy_collided": policy_collided,
                "policy_steps": max(len(policy_trajectory) - 1, 0),
                "expert_steps": max(len(expert_trajectory) - 1, 0),
                "policy_goal_position_error": policy_goal_position_error,
                "policy_goal_rotation_error": policy_goal_rotation_error,
                "expert_goal_position_error": expert_goal_position_error,
                "expert_goal_rotation_error": expert_goal_rotation_error,
                "expert_solve_time_mean_s": expert_solve_time_mean,
                "policy_solve_time_mean_s": policy_solve_time_mean,
                "expert_solve_times": expert_solve_times,
                "policy_solve_times": policy_solve_times,
            }
        )

    total_runs = len(per_seed_metrics)
    total_successes = sum(
        1
        for metric in per_seed_metrics
        if metric["policy_reached_goal"] and not metric["policy_collided"]
    )
    success_rate = (total_successes / total_runs) if total_runs > 0 else 0.0
    mean_policy_position_error = float(
        np.mean([metric["policy_goal_position_error"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_policy_rotation_error = float(
        np.mean([metric["policy_goal_rotation_error"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_position_error = float(
        np.mean([metric["expert_goal_position_error"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_rotation_error = float(
        np.mean([metric["expert_goal_rotation_error"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_policy_steps = float(
        np.mean([metric["policy_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0
    mean_expert_steps = float(
        np.mean([metric["expert_steps"] for metric in per_seed_metrics])
    ) if total_runs > 0 else 0.0

    def _pooled_mean_or_none(key: str) -> float | None:
        pooled = [t for metric in per_seed_metrics for t in metric[key]]
        return float(np.mean(pooled)) if pooled else None

    mean_expert_solve_time = _pooled_mean_or_none("expert_solve_times")
    mean_policy_solve_time = _pooled_mean_or_none("policy_solve_times")

    print("\n--- Evaluation Summary ---")
    print(f"policy_type: {policy_type}")
    print(f"seeds: {seed_specs}")
    print(f"device: {device}")
    print(f"action_noise_std: {action_noise_std:.6f}")
    unique_goal_states = {
        tuple(np.asarray(metric["goal_state"], dtype=float).tolist())
        for metric in per_seed_metrics
    }
    summary_goal_state: list[float] | None = None
    if len(unique_goal_states) == 1:
        only_goal_state = np.asarray(next(iter(unique_goal_states)), dtype=float)
        summary_goal_state = only_goal_state.tolist()
        print(f"goal_state: {np.array2string(only_goal_state, precision=4)}")
    else:
        print("goal_state: varies per seed (goal randomization enabled)")
    print(f"num_trajectories: {total_runs}")
    print(f"policy_successes: {total_successes}/{total_runs}")
    print(f"success_rate: {success_rate:.4f}")
    policy_solve_failure_rate = (
        np.mean([metric["policy_solve_failed"] for metric in per_seed_metrics])
        if total_runs > 0 else 0.0
    )
    policy_collision_rate = (
        np.mean([metric["policy_collided"] for metric in per_seed_metrics]) if total_runs > 0 else 0.0
    )
    policy_timeout_rate = (
        np.mean(
            [
                (not metric["policy_reached_goal"])
                and (not metric["policy_solve_failed"])
                and (not metric["policy_collided"])
                for metric in per_seed_metrics
            ]
        )
        if total_runs > 0 else 0.0
    )
    print(f"policy_timeout_rate: {policy_timeout_rate:.4f}")
    print(f"policy_solve_failure_rate: {policy_solve_failure_rate:.4f}")
    print(f"policy_collision_rate: {policy_collision_rate:.4f}")
    print(f"mean_policy_steps: {mean_policy_steps:.3f}")
    print(f"mean_expert_steps: {mean_expert_steps:.3f}")
    print(f"mean_policy_goal_position_error: {mean_policy_position_error:.6f} m")
    print(f"mean_policy_goal_rotation_error: {mean_policy_rotation_error:.6f} rad")
    print(f"mean_expert_goal_position_error: {mean_expert_position_error:.6f} m")
    print(f"mean_expert_goal_rotation_error: {mean_expert_rotation_error:.6f} rad")

    def _fmt_solve_time(value: float | None) -> str:
        return f"{value * 1000.0:.3f} ms" if value is not None else "n/a"

    print(
        "mean_expert_solve_time (CasADi MPC solve): "
        f"{_fmt_solve_time(mean_expert_solve_time)} / step"
    )
    print(
        "mean_policy_solve_time (single policy call): "
        f"{_fmt_solve_time(mean_policy_solve_time)} / step"
    )

    if output_path is not None:
        if not output_path.endswith(".npz"):
            output_path += ".npz"
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # Rollouts end at different steps, so each rollout's trajectories get
        # their own array; goals[i] is the goal TCP pose that
        # expert_trajectory_i and policy_trajectory_i (steps + 1, nx) share.
        trajectory_arrays: dict[str, np.ndarray] = {}
        for rollout_idx, (expert_trajectory, policy_trajectory) in enumerate(
            zip(expert_trajectories, policy_trajectories)
        ):
            trajectory_arrays[f"expert_trajectory_{rollout_idx}"] = expert_trajectory
            trajectory_arrays[f"policy_trajectory_{rollout_idx}"] = policy_trajectory
        np.savez(output_path, goals=np.asarray(rollout_goal_states), **trajectory_arrays)
        print(f"Trajectories saved to {output_path}")

    metrics = {
        "policy_type": policy_type,
        "seeds": seed_specs,
        "device": str(device),
        "action_noise_std": action_noise_std,
        "goal_state": summary_goal_state,
        "goal_state_varies_by_seed": len(unique_goal_states) > 1,
        "num_trajectories": total_runs,
        "policy_successes": total_successes,
        "success_rate": success_rate,
        "policy_timeout_rate": float(policy_timeout_rate),
        "policy_solve_failure_rate": float(policy_solve_failure_rate),
        "policy_collision_rate": float(policy_collision_rate),
        "mean_policy_steps": mean_policy_steps,
        "mean_expert_steps": mean_expert_steps,
        "mean_policy_goal_position_error": mean_policy_position_error,
        "mean_policy_goal_rotation_error": mean_policy_rotation_error,
        "mean_expert_goal_position_error": mean_expert_position_error,
        "mean_expert_goal_rotation_error": mean_expert_rotation_error,
        "mean_expert_solve_time_s": mean_expert_solve_time,
        "mean_policy_solve_time_s": mean_policy_solve_time,
        "per_seed": per_seed_metrics,
        "output_path": output_path,
    }

    return metrics


def main():
    """
    evaluates a trained policy on the Franka arm
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="DAgger checkpoint, e.g. outputs/train_dagger/<name>/flow_dagger_checkpoint.pt",
    )
    parser.add_argument(
        "--policy-type",
        type=str.lower,
        choices=["flow", "safeflow"],
        default=None,
        help="default: the checkpoint's own policy type",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml"),
        help="expert yaml config (default: %(default)s)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="[10000]",
        help="seeds of the starts and goals, e.g. '[10000, 10001]' (default: %(default)s)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=125,
        help="maximum steps per rollout (default: %(default)s)",
    )
    parser.add_argument(
        "--initial-states",
        type=str,
        default=None,
        help=(
            "explicit initial state specs. Examples: '[q1, ..., q7, v1, ..., v7]' for one rollout "
            "or '[[...], [...]]' for multiple rollouts. "
            "When exhausted, evaluation falls back to simulator RNG sampling."
        ),
    )
    parser.add_argument(
        "--goal-states",
        type=str,
        default=None,
        help=(
            "explicit goal state specs, independently indexed from --initial-states. "
            "Examples: '[q1, ..., q7]' for one rollout or "
            "'[[...], [...]]' for multiple rollouts. "
            "When exhausted, evaluation falls back to simulator RNG sampling."
        ),
    )
    parser.add_argument(
        "--tolerance-overrides",
        type=str,
        default=None,
        help="dict of pos_tol / vel_tol overriding the config's, e.g. '{\"pos_tol\": 0.02, \"vel_tol\": 0.1}'",
    )
    parser.add_argument(
        "--action-noise-std",
        type=float,
        default=0.0,
        help=(
            "std-dev of Gaussian action noise applied during expert and policy rollout execution; "
            "default 0.0 keeps evaluation deterministic and clean"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "path to a .npz file the expert and policy state trajectories and the goals "
            "of every rollout are saved to (nothing is saved when omitted)"
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="device of the policy (default: %(default)s)",
    )

    args = parser.parse_args()
    config = load_and_validate_system_config(config_path=args.config)
    initialize_ompl(config["ompl_seed"])

    tolerance_overrides = None
    if args.tolerance_overrides:
        try:
            tolerance_overrides = ast.literal_eval(args.tolerance_overrides)
        except (SyntaxError, ValueError) as exc:
            parser.error(f"Unable to parse --tolerance-overrides: {exc}")
        if not isinstance(tolerance_overrides, dict):
            parser.error("--tolerance-overrides must evaluate to a dict.")

    run_evaluation(
        policy_type=args.policy_type or checkpoint_policy_type(args.model_dir),
        config=config,
        model_dir=args.model_dir,
        num_steps=args.num_steps,
        seeds=parse_seed_argument(args.seeds),
        initial_states=parse_initial_states_argument(args.initial_states),
        goal_states=parse_goal_states_argument(args.goal_states),
        tolerance_overrides=tolerance_overrides,
        action_noise_std=args.action_noise_std,
        output_path=args.output_path,
        device_override=args.device,
    )


if __name__ == "__main__":
    main()