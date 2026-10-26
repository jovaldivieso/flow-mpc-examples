from __future__ import annotations

import math
import random
from typing import Mapping

import numpy as np
import torch

from .metrics import DaggerEvalMetrics
from systems.system import SystemSimulator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def with_seeded_initial_state_config(
    config: Mapping[str, object],
    base_seed: int,
) -> dict[str, object]:
    """Ensure all simulator RNG entrypoints get deterministic initial-state seeds."""
    seeded_config = dict(config)
    seeded_config.setdefault("initial_state_seed", int(base_seed))
    return seeded_config


def apply_config_overrides(
    config: Mapping[str, object],
    overrides: Mapping[str, object],
) -> dict[str, object]:
    """Merge flat key/value overrides into a config dict."""
    merged_config = dict(config)
    if not overrides:
        return merged_config
    merged_config.update(overrides)
    return merged_config


def resolve_initial_state_seed(config: Mapping[str, object], fallback_seed: int) -> int:
    return int(config.get("initial_state_seed", fallback_seed))


def resolve_round_steps(
    num_frames: int,
    batch_size: int,
    target_epochs_per_round: float,
    max_train_steps: int | None,
) -> tuple[int, float]:
    if num_frames <= 0:
        raise ValueError("Training dataset must contain at least one frame.")
    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive.")
    if target_epochs_per_round <= 0:
        raise ValueError("'target_epochs_per_round' must be positive.")
    steps = math.ceil(float(target_epochs_per_round) * float(num_frames) / float(batch_size))
    if max_train_steps is not None:
        steps = min(steps, int(max_train_steps))
    if steps <= 0:
        raise ValueError("Per-round training steps must remain positive.")
    return steps, float(steps) * float(batch_size) / float(num_frames)


def print_rollout_metrics(label: str, prefix: str, metrics: DaggerEvalMetrics) -> None:
    print(
        f"{label}: {prefix}_success_rate={100.0 * metrics.success_rate:.1f}% "
        f"{prefix}_mean_steps={metrics.mean_steps:.2f} "
        f"{prefix}_min_steps={metrics.min_steps} {prefix}_max_steps={metrics.max_steps} "
        f"episodes={metrics.num_episodes}"
    )
    # Only evaluate_policy_rollouts populates this split (curriculum
    # initial_states/goal_states vs simulator RNG fallback); other
    # DaggerEvalMetrics producers (e.g. aggregation's own success
    # accounting) leave both counts at 0, so there's nothing worth a second
    # line for them.
    if metrics.config_num_episodes > 0 or metrics.random_num_episodes > 0:
        config_rate = (
            100.0 * metrics.config_successes / metrics.config_num_episodes
            if metrics.config_num_episodes > 0
            else None
        )
        random_rate = (
            100.0 * metrics.random_successes / metrics.random_num_episodes
            if metrics.random_num_episodes > 0
            else None
        )
        config_display = (
            f"{metrics.config_successes}/{metrics.config_num_episodes} ({config_rate:.1f}%)"
            if config_rate is not None
            else "n/a (0 episodes)"
        )
        random_display = (
            f"{metrics.random_successes}/{metrics.random_num_episodes} ({random_rate:.1f}%)"
            if random_rate is not None
            else "n/a (0 episodes)"
        )
        print(
            f"  {prefix}_success_by_source: config={config_display} random={random_display}"
        )
    total_failures = metrics.timeout_failures + metrics.solve_failures
    if total_failures > 0:
        print(
            f"  {prefix}_failure_breakdown: timeout={metrics.timeout_failures} "
            f"solve_failure={metrics.solve_failures} (of {total_failures} failures)"
        )


def evaluation_seed_specs(
    num_episodes: int,
    seed_start: int,
) -> list[int]:
    if num_episodes < 0:
        raise ValueError("'num_episodes' must be non-negative.")
    return [int(seed_start) + idx for idx in range(num_episodes)]


def rng_for_seed_spec(seed_spec: int) -> np.random.Generator:
    """Build the RNG for a seed spec, like sample_initial_state does."""
    return np.random.default_rng(int(seed_spec))


def sample_initial_state(simulator: SystemSimulator, seed_spec: int) -> np.ndarray:
    rng = rng_for_seed_spec(seed_spec)
    simulator.randomize_goal_for_reset(rng)
    return simulator.random_initial_state(rng)
