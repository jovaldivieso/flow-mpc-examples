from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.data_utils import format_sample_for_policy
from learning.models.ema import ExponentialMovingAverage
from learning.models.flow_policy import FlowPolicy
from learning.models.flow_velocity_net import TIME_BETA_ALPHA, TIME_BETA_BETA
from planning.casadi_projector import CasadiTrajectoryProjector
from planning.config import load_and_validate_system_config
from planning.factory import DynamicsFactory
from planning.planner import PlannerSolveError

"""Safety fine-tuning of a flow policy on projected paths (second training stage of SafeFlowMPC, Oelerich et al., 2026).

The first stage (DAgger, learning/train_dagger.py) trains the flow network on straight lines between Gaussian noise and
the expert's plan. The safe policy (``policy_type: safeflow``) instead integrates the flow with the projection onto the
safe set after every Euler step, so its states are *projected* chunks. This stage adapts the network to them: for
frames of the dataset it computes a path from the projected noise to the projected expert plan,

    path_l = project(plan + (project(noise) - plan) * (1 - l / n)),   l = 0 .. n   (noise at l = 0, plan at l = n)

and regresses the network's velocity at the grid times l / n onto the path's finite-difference velocity. As in the
original, a batch is a safe-path batch with probability 1/2 and an ordinary Gaussian CondOT batch otherwise, the
learning rate is tiny (1e-6) and the exponential moving average of the weights is kept.
"""

DEFAULT_EXPERT_CONFIG = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

# Per-process state of the path generation workers (built once by ``_init_worker``)
_worker_projector: CasadiTrajectoryProjector | None = None
_worker_simulator = None


def projected_path(
    projector: CasadiTrajectoryProjector,
    observation: np.ndarray,
    plan: np.ndarray,
    noise: np.ndarray,
    num_steps: int,
) -> np.ndarray:
    """The projected path from the noise chunk (index 0) to the expert plan (index ``num_steps``), (num_steps + 1, H, n).

    Raises ``PlannerSolveError`` when the projector cannot make a point of the path safe.
    """
    projector.reset()
    safe_noise = projector.project(observation, noise)
    path = np.empty((num_steps + 1, *plan.shape), dtype=float)
    for index in range(num_steps + 1):
        # Flow time index / num_steps: the noise end first, the (projected) plan last
        mix = 1.0 - index / num_steps
        path[index] = projector.project(observation, plan + (safe_noise - plan) * mix)
    return path


def _init_worker(config: Mapping[str, Any], horizon: int) -> None:
    global _worker_projector, _worker_simulator
    _worker_simulator = DynamicsFactory.create(config["system"], config)
    _worker_projector = CasadiTrajectoryProjector(_worker_simulator, config, horizon=horizon)


def _generate_path(task: tuple[np.ndarray, np.ndarray, int, int]) -> np.ndarray | None:
    """One worker task (observation, plan, noise seed, number of path steps); None when the projection failed."""
    observation, plan, seed, num_steps = task
    assert _worker_projector is not None
    noise = np.random.default_rng(seed).normal(size=plan.shape) * _worker_simulator.max_action  # unit noise of the scaled jerks
    try:
        return projected_path(_worker_projector, observation, plan, noise, num_steps)
    except PlannerSolveError:
        return None


def generate_paths(
    config: Mapping[str, Any],
    observations: np.ndarray,
    plans: np.ndarray,
    noise_samples: int,
    num_steps: int,
    seed: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Projected paths for every (frame, noise draw): returns (frame indices, paths, observation rows, failures).

    ``paths`` is (count, num_steps + 1, H, n) and ``observations`` the matching (count, obs_dim) conditioning rows;
    pairs whose projection failed are left out and counted.
    """
    tasks = [
        (observations[frame], plans[frame], seed * 1_000_003 + frame * noise_samples + draw, num_steps)
        for frame in range(len(plans))
        for draw in range(noise_samples)
    ]
    frame_of_task = np.repeat(np.arange(len(plans)), noise_samples)
    if num_workers > 1:
        with multiprocessing.get_context("spawn").Pool(
            num_workers, initializer=_init_worker, initargs=(dict(config), plans.shape[1])
        ) as pool:
            results = pool.map(_generate_path, tasks, chunksize=max(1, len(tasks) // (4 * num_workers)))
    else:
        _init_worker(config, plans.shape[1])
        results = [_generate_path(task) for task in tasks]
    kept = [index for index, path in enumerate(results) if path is not None]
    failures = len(results) - len(kept)
    if not kept:
        raise RuntimeError("Every projected path failed; nothing to fine-tune on.")
    paths = np.stack([results[index] for index in kept])
    return frame_of_task[kept], paths, observations[frame_of_task[kept]], failures


def path_batch(
    paths: torch.Tensor,
    observation_rows: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A batch of (conditioning, x_t, dx_t, t) from the projected paths at Beta-distributed grid times.

    Like the flow time of the ordinary loss, the grid index is drawn as ``t = 1 - Beta(1.5, 1)``, so the early part of
    the paths gets more samples.
    """
    count, grid_points = paths.shape[0], paths.shape[1]
    num_steps = grid_points - 1
    picked = torch.randint(0, count, (batch_size,), generator=generator)
    beta = torch.distributions.Beta(torch.tensor(TIME_BETA_ALPHA), torch.tensor(TIME_BETA_BETA))
    t_continuous = 1.0 - beta.sample((batch_size,))  # drawn from torch's global generator, seeded by ``finetune``
    index = (t_continuous * num_steps).to(torch.int64).clamp(0, num_steps - 1)
    x_t = paths[picked, index]
    dx_t = (paths[picked, index + 1] - x_t) * num_steps
    return observation_rows[picked], x_t, dx_t, index.to(paths.dtype) / num_steps


def _observation_dict(rows: torch.Tensor, goal_dim: int) -> dict[str, torch.Tensor]:
    return {"observation.environment_state": rows[:, :goal_dim], "observation.state": rows[:, goal_dim:]}


def finetune(
    checkpoint_path: Path,
    dataset_repo_id: str,
    dataset_root: Path,
    expert_config: Mapping[str, Any],
    output_path: Path,
    *,
    num_frames: int,
    noise_samples: int,
    path_steps: int,
    iterations: int,
    batch_size: int,
    learning_rate: float,
    ema_decay: float,
    num_workers: int,
    seed: int,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    simulator = DynamicsFactory.create(expert_config["system"], expert_config)
    horizon = int(checkpoint["prediction_horizon"])
    flow_config = checkpoint["flow_config"]
    policy = FlowPolicy(
        action_dim=int(checkpoint["action_dim"]),
        state_dim=int(checkpoint["state_dim"]),
        prediction_horizon=horizon,
        unet_dim=int(checkpoint["unet"]["dim"]),
        unet_dim_mults=tuple(checkpoint["unet"]["dim_mults"]),
        num_inference_steps=int(flow_config["num_inference_steps"]),
        action_scale=flow_config["action_scale"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    ema = ExponentialMovingAverage(policy, decay=ema_decay)
    if "ema_model_state_dict" in checkpoint:
        averaged = FlowPolicy(
            action_dim=policy.velocity_net.action_dim,
            state_dim=policy.velocity_net.obs_cond_dim,
            prediction_horizon=horizon,
            unet_dim=int(checkpoint["unet"]["dim"]),
            unet_dim_mults=tuple(checkpoint["unet"]["dim_mults"]),
            action_scale=flow_config["action_scale"],
        )
        averaged.load_state_dict(checkpoint["ema_model_state_dict"])
        ema.load_state_dict(averaged.state_dict())

    dataset = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root)
    frames = rng.choice(len(dataset), size=min(num_frames, len(dataset)), replace=False)
    observations, plans = [], []
    for frame in frames:
        observation, plan = format_sample_for_policy(dataset[int(frame)], horizon)
        observations.append(torch.cat([observation["observation.environment_state"], observation["observation.state"]]).numpy())
        plans.append(plan.numpy().astype(float))
    observations_array, plans_array = np.stack(observations), np.stack(plans)

    started = time.time()
    _, paths, observation_rows, failures = generate_paths(
        expert_config, observations_array, plans_array, noise_samples, path_steps, seed, num_workers
    )
    print(
        f"Projected {len(paths)} paths of {path_steps + 1} points ({failures} failed) in {time.time() - started:.1f} s"
    )
    paths_tensor = torch.as_tensor(paths, dtype=torch.float32)
    rows_tensor = torch.as_tensor(observation_rows, dtype=torch.float32)
    plans_tensor = torch.as_tensor(plans_array, dtype=torch.float32)
    frame_rows_tensor = torch.as_tensor(observations_array, dtype=torch.float32)
    goal_dim = simulator.goal_dim

    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=1e-7)
    policy.train()
    losses: list[float] = []
    safe_batches = 0
    for iteration in range(iterations):
        optimizer.zero_grad()
        if torch.rand(1, generator=generator).item() <= 0.5:
            rows, x_t, dx_t, t = path_batch(paths_tensor, rows_tensor, batch_size, generator)
            loss = policy.compute_path_loss(_observation_dict(rows, goal_dim), x_t, dx_t, t)
            safe_batches += 1
        else:
            picked = torch.randint(0, len(plans_tensor), (batch_size,), generator=generator)
            loss = policy.compute_loss(_observation_dict(frame_rows_tensor[picked], goal_dim), plans_tensor[picked])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        ema.update(policy)
        losses.append(float(loss.item()))
        if iteration % max(1, iterations // 10) == 0 or iteration == iterations - 1:
            print(f"  iteration {iteration + 1}/{iterations} loss={losses[-1]:.6f}")
    policy.eval()

    result = {key: value for key, value in checkpoint.items() if key != "optimizer_state_dict"}
    result["model_state_dict"] = policy.state_dict()
    result["ema_model_state_dict"] = ema.state_dict()
    result["safety_finetune"] = {
        "iterations": iterations,
        "safe_path_batches": safe_batches,
        "num_paths": int(len(paths)),
        "failed_paths": int(failures),
        "path_steps": path_steps,
        "learning_rate": learning_rate,
        "base_checkpoint": str(checkpoint_path),
        "mean_loss": float(np.mean(losses)) if losses else float("nan"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    print(f"Saved {output_path}")
    return result["safety_finetune"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safety fine-tuning of a flow policy on projected paths.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="DAgger checkpoint (flow_dagger_checkpoint.pt)")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset id the DAgger run collected")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expert-config", default=DEFAULT_EXPERT_CONFIG)
    parser.add_argument("--output", type=Path, help="default: <checkpoint stem>_safe_finetuned.pt next to the checkpoint")
    parser.add_argument("--num-frames", type=int, default=200, help="dataset frames to build projected paths for")
    parser.add_argument("--noise-samples", type=int, default=2, help="noise draws (paths) per frame")
    parser.add_argument("--path-steps", type=int, default=10, help="grid steps n of each path (n + 1 projections)")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--num-workers", type=int, default=1, help="processes projecting the paths")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for name in ("num_frames", "noise_samples", "path_steps", "iterations", "batch_size", "num_workers"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive.")
    output = args.output or args.checkpoint.with_name(f"{args.checkpoint.stem}_safe_finetuned.pt")
    finetune(
        args.checkpoint,
        args.repo_id,
        args.dataset_root,
        load_and_validate_system_config(args.expert_config),
        output,
        num_frames=args.num_frames,
        noise_samples=args.noise_samples,
        path_steps=args.path_steps,
        iterations=args.iterations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        ema_decay=args.ema_decay,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
