from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from systems.system import SystemSimulator

if TYPE_CHECKING:
    from learning.dagger.safe_pairs import SafePairs

StructuredObservation = dict[str, torch.Tensor]

OBSERVATION_KEYS = ("observation.environment_state", "observation.state", "observation.history")


def build_dataset_features(simulator: SystemSimulator, prediction_horizon: int, history_length: int) -> dict[str, Any]:
    """The LeRobot features of ``simulator``'s dataset: what its policy observes (the environment, the state and the last
    ``history_length`` positions) and the expert's planned positions of the next ``prediction_horizon`` control periods
    flattened row by row (the ``action`` of a frame)."""
    environment_names, state_names = simulator.environment_state_names, simulator.state_names
    history_names, plan_names = simulator.history_names(history_length), simulator.plan_names(prediction_horizon)
    return {
        "observation.environment_state": {"dtype": "float32", "shape": (len(environment_names),), "names": environment_names},
        "observation.state": {"dtype": "float32", "shape": (len(state_names),), "names": state_names},
        "observation.history": {"dtype": "float32", "shape": (len(history_names),), "names": history_names},
        "action": {"dtype": "float32", "shape": (len(plan_names),), "names": plan_names},
    }


def format_dataset_frame(simulator: SystemSimulator, obs: np.ndarray, history: np.ndarray, positions: np.ndarray) -> dict[str, np.ndarray]:
    """An observation, the history of positions of its rollout and the expert's planned positions
    ``(prediction_horizon, n)`` in the LeRobot format (without ``task``, which the caller adds)."""
    positions_array = np.asarray(positions, dtype=float)
    if positions_array.ndim != 2 or positions_array.shape[1] != simulator.n:
        raise ValueError(f"positions must have shape (prediction_horizon, {simulator.n}), got {positions_array.shape}.")
    return {**simulator.policy_observation(obs, history), "action": positions_array.reshape(-1).astype(np.float32)}


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

    observation: StructuredObservation = {key: _tensor_field(sample, key) for key in OBSERVATION_KEYS}
    plan = _tensor_field(sample, "action")
    if plan.numel() % prediction_horizon != 0:
        raise ValueError(
            f"A plan of {plan.numel()} values cannot be split into prediction_horizon={prediction_horizon} rows."
        )
    return observation, plan.reshape(prediction_horizon, -1)


def collate_batch_for_policy(
    batch: Sequence[Mapping[str, Any]],
    prediction_horizon: int,
    pairs: SafePairs | None = None,
) -> tuple[StructuredObservation, torch.Tensor, torch.Tensor | None]:
    """Collate native LeRobot frames into batched policy observations, plans (B, prediction_horizon, n_joints) and, with
    ``pairs``, the safe paths (B, segments + 1, prediction_horizon, n_joints) of the frames, from their safe source to their
    safe target, along which the flow is learned (SafeFlowMPC, Sec. II-B.2); else None."""
    if not batch:
        raise ValueError("Cannot collate an empty dataset batch.")

    formatted = [format_sample_for_policy(sample, prediction_horizon) for sample in batch]
    observations, plans = zip(*formatted)
    batched_observations = {
        name: torch.stack([observation[name] for observation in observations]) for name in observations[0]
    }
    plans = torch.stack(plans)
    if pairs is None:
        return batched_observations, plans, None
    indices = torch.stack([_tensor_field(sample, "index") for sample in batch]).long().flatten().numpy()
    return batched_observations, plans, torch.from_numpy(pairs.paths[indices])


def create_collate_fn(prediction_horizon: int, pairs: SafePairs | None = None):
    """Collate function for torch.utils.data.DataLoader over a LeRobotDataset."""

    def collate_fn(batch: Sequence[Mapping[str, Any]]):
        return collate_batch_for_policy(batch, prediction_horizon, pairs)

    return collate_fn
