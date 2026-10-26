from __future__ import annotations

from typing import Any, Mapping

import numpy as np


DEFAULT_SINGLE_ROBOT_SEED = 1
_ACTION_NOISE_STREAM_ID = 1
_INITIAL_STATE_STREAM_ID = 2
_EXPERT_MIXING_STREAM_ID = 3
_TORCH_INFERENCE_STREAM_ID = 4


def default_action_noise_seed_for_config(config: Mapping[str, Any]) -> int:
    action_noise_seed = config.get("action_noise_seed")
    if action_noise_seed is None:
        raise KeyError("Validated config is missing 'action_noise_seed'.")
    return int(action_noise_seed)


def action_noise_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
    round_index: int | None = None,
) -> int:
    return _derived_rollout_seed(
        base_seed,
        stream_id=_ACTION_NOISE_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
        round_index=round_index,
    )


def initial_state_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
    round_index: int | None = None,
) -> int:
    return _derived_rollout_seed(
        base_seed,
        stream_id=_INITIAL_STATE_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
        round_index=round_index,
    )


def expert_mixing_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
    round_index: int | None = None,
) -> int:
    return _derived_rollout_seed(
        base_seed,
        stream_id=_EXPERT_MIXING_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
        round_index=round_index,
    )


def torch_inference_seed_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
    round_index: int | None = None,
) -> int:
    """Seed for torch.manual_seed(...), pinning a policy's own stochastic
    inference (e.g. FlowPolicy's ODE initial noise) per rollout. Own stream
    ID keeps this independent of action_noise_seed_for_rollout/initial_state
    _seed_for_rollout even though all three commonly share the same
    base_seed and seed_spec -- see _derived_rollout_seed.
    """
    return _derived_rollout_seed(
        base_seed,
        stream_id=_TORCH_INFERENCE_STREAM_ID,
        seed_spec=seed_spec,
        rollout_index=rollout_index,
        round_index=round_index,
    )


def _derived_rollout_seed(
    base_seed: int,
    *,
    stream_id: int,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
    round_index: int | None = None,
) -> int:
    entropy = [int(base_seed), int(stream_id)]
    if seed_spec is not None:
        entropy.append(int(seed_spec))
    elif rollout_index is not None:
        entropy.append(int(rollout_index))
    if round_index is not None:
        entropy.append(int(round_index))

    seed_sequence = np.random.SeedSequence(entropy)
    return int(seed_sequence.generate_state(1, dtype=np.uint64)[0])


def action_noise_rng_for_rollout(
    base_seed: int,
    *,
    seed_spec: int | None = None,
    rollout_index: int | None = None,
) -> np.random.Generator:
    return np.random.default_rng(
        action_noise_seed_for_rollout(
            base_seed,
            seed_spec=seed_spec,
            rollout_index=rollout_index,
        )
    )