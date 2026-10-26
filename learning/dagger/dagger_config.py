from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from learning.config_loaders import FlowConfig, UnetConfig


def _per_round(values: list[Any], rounds: int, name: str) -> list[Any]:
    """One value per round: a single value is repeated for every round, else there must be exactly one per round."""
    if len(values) == 1:
        return list(values) * rounds if rounds else list(values)
    if len(values) != rounds:
        raise ValueError(f"'{name}' must contain one value or exactly one per round ({rounds}), got {len(values)}.")
    return list(values)


@dataclass(frozen=True)
class DaggerConfig:
    """The resolved settings of a DAgger run. The per-round lists (``trajectories_per_iteration``,
    ``expert_beta_per_iteration``, ``target_epochs_per_round``, and ``training_curriculum`` and ``round_seeds`` when
    given) are taken as they come, with one value for all rounds or one per round, and are expanded here to one value
    per round."""

    experiment_config: dict[str, Any]
    repo_id: str
    dataset_root: Path
    start_with_aggregation: bool
    dagger_iterations: int
    trajectories_per_iteration: list[int]
    steps_per_trajectory: int
    action_noise_std: float
    expert_beta_per_iteration: list[float]
    target_epochs_per_round: list[float]
    eval_episodes: int
    eval_steps: int | None
    eval_seed_start: int
    eval_action_noise_std: float
    batch_size: int
    learning_rate: float
    unet: UnetConfig
    prediction_horizon: int
    policy_type: str
    flow_config: FlowConfig
    checkpoint_dir: Path
    seed: int
    max_train_steps: int | None
    initial_states: list[np.ndarray] | None = None
    goal_states: list[np.ndarray] | None = None
    training_curriculum: list[str] | None = None
    round_seeds: list[int] | None = None
    restart_round_seed: bool = False
    tolerance_overrides: dict[str, float] | None = None
    # Independent of tolerance_overrides: that one shapes what gets
    # demonstrated/trained on (tight tolerances during collection encourage
    # precise demonstrations), while this one only affects whether an eval
    # rollout counts as a success, so it's reasonable for this to be more
    # relaxed without touching training dynamics or demonstration length.
    # Falls back to tolerance_overrides when unset.
    eval_tolerance_overrides: dict[str, float] | None = None
    num_workers: int = 1

    def __post_init__(self) -> None:
        if self.dagger_iterations < 0:
            raise ValueError("'dagger_iterations' must be non-negative.")
        rounds = self.dagger_iterations
        for name in ("trajectories_per_iteration", "expert_beta_per_iteration", "target_epochs_per_round"):
            object.__setattr__(self, name, _per_round(getattr(self, name), rounds, name))
        for name in ("training_curriculum", "round_seeds"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, _per_round(getattr(self, name), rounds, name))
        if any(v <= 0 for v in self.trajectories_per_iteration):
            raise ValueError("Trajectory targets must be positive.")
        if self.steps_per_trajectory <= 0:
            raise ValueError("'steps_per_trajectory' must be positive.")
        if self.action_noise_std < 0 or self.eval_action_noise_std < 0:
            raise ValueError("Action noise must be non-negative.")
        if any(not 0.0 <= v <= 1.0 for v in self.expert_beta_per_iteration):
            raise ValueError("Expert beta values must be in [0, 1].")
        if any(v <= 0 for v in self.target_epochs_per_round):
            raise ValueError("Epoch targets must be positive.")
        if self.eval_episodes < 0 or (
            self.eval_steps is not None and self.eval_steps <= 0
        ):
            raise ValueError("Evaluation counts must be valid.")
        if self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("Batch size and learning rate must be positive.")
        if self.max_train_steps is not None and self.max_train_steps <= 0:
            raise ValueError("Max train steps must be positive.")
        if self.training_curriculum is not None and any(
            mode not in {"random", "config"} for mode in self.training_curriculum
        ):
            raise ValueError("'training_curriculum' entries must be 'random' or 'config'.")
        if self.tolerance_overrides is not None and any(
            value <= 0 for value in self.tolerance_overrides.values()
        ):
            raise ValueError("'tolerance_overrides' values must be positive.")
        if self.eval_tolerance_overrides is not None and any(
            value <= 0 for value in self.eval_tolerance_overrides.values()
        ):
            raise ValueError("'eval_tolerance_overrides' values must be positive.")
        if self.num_workers <= 0:
            raise ValueError("'num_workers' must be positive.")
        if not self.dataset_root.exists() and not self.start_with_aggregation:
            raise FileNotFoundError(self.dataset_root)
