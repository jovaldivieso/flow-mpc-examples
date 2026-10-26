from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from systems.system import SystemSimulator

StructuredObservation = dict[str, torch.Tensor]


def build_dataset_features(simulator: SystemSimulator, prediction_horizon: int) -> dict[str, Any]:
    """The LeRobot features of ``simulator``'s dataset: what its policy observes and the expert's next
    ``prediction_horizon`` planned actions flattened row by row."""
    environment_names, state_names = simulator.environment_state_names, simulator.state_names
    return {
        "observation.environment_state": {"dtype": "float32", "shape": (len(environment_names),), "names": environment_names},
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "action": {
            "dtype": "float32",
            "shape": (prediction_horizon * simulator.nu,),
            "names": [f"{name}[t+{k}]" for k in range(prediction_horizon) for name in simulator.action_names],
        },
    }


def format_dataset_frame(simulator: SystemSimulator, obs: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    """An observation and the expert's planned actions ``(prediction_horizon, nu)`` in the LeRobot format (without
    ``task``, which the caller adds)."""
    actions_array = np.asarray(actions, dtype=float)
    if actions_array.ndim != 2 or actions_array.shape[1] != simulator.nu:
        raise ValueError(f"actions must have shape (prediction_horizon, {simulator.nu}), got {actions_array.shape}.")
    return {**simulator.policy_observation(obs), "action": actions_array.reshape(-1).astype(np.float32)}


def _tensor_field(sample: Mapping[str, Any], name: str) -> torch.Tensor:
    value = sample.get(name)
    if value is None:
        raise ValueError(f"Dataset sample must contain '{name}'.")
    return torch.as_tensor(value, dtype=torch.float32).reshape(-1)


def format_sample_for_policy(
    sample: Mapping[str, Any],
    prediction_horizon: int,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Convert one native LeRobot frame into policy tensors.

    The ``action`` of a frame is the expert's planned joint positions from that state, ``prediction_horizon`` rows
    of one configuration each, flattened row by row; it becomes the (prediction_horizon, n_joints) target.
    """
    if prediction_horizon <= 0:
        raise ValueError("'prediction_horizon' must be positive.")

    observation: StructuredObservation = {
        "observation.environment_state": _tensor_field(sample, "observation.environment_state"),
        "observation.state": _tensor_field(sample, "observation.state"),
    }
    plan = _tensor_field(sample, "action")
    if plan.numel() % prediction_horizon != 0:
        raise ValueError(
            f"A plan of {plan.numel()} values cannot be split into prediction_horizon={prediction_horizon} rows."
        )
    return observation, plan.reshape(prediction_horizon, -1)


def collate_batch_for_policy(
    batch: Sequence[Mapping[str, Any]],
    prediction_horizon: int,
) -> tuple[StructuredObservation, torch.Tensor]:
    """Collate native LeRobot frames into batched policy observations and plans (B, prediction_horizon, n_joints)."""
    if not batch:
        raise ValueError("Cannot collate an empty dataset batch.")

    formatted = [format_sample_for_policy(sample, prediction_horizon) for sample in batch]
    observations, plans = zip(*formatted)
    batched_observations = {
        name: torch.stack([observation[name] for observation in observations]) for name in observations[0]
    }
    return batched_observations, torch.stack(plans)


def create_collate_fn(prediction_horizon: int):
    """Collate function for torch.utils.data.DataLoader over a LeRobotDataset."""

    def collate_fn(batch: Sequence[Mapping[str, Any]]):
        return collate_batch_for_policy(batch=batch, prediction_horizon=prediction_horizon)

    return collate_fn
