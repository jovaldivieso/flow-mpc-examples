from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from systems.system import SystemSimulator


def build_policy_action(
    simulator: SystemSimulator,
    policy,
    observation: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Query the policy once and return the action to execute: the first row of the predicted chunk.

    The policy predicts a chunk of actions (batch, horizon, nu), row k the action at step k.
    """
    policy_observation = simulator.policy_observation(observation)
    policy_input = {
        name: torch.as_tensor(
            np.expand_dims(policy_observation[name], axis=0),
            dtype=torch.float32,
        ).to(device)
        for name in policy_observation
    }

    with torch.inference_mode():
        chunk = policy.select_action(policy_input)

    if chunk.ndim != 3 or chunk.shape[1] < 1:
        raise ValueError("Policy must return an action chunk shaped (batch, horizon >= 1, nu).")
    return chunk[0, 0, :].detach().cpu().numpy().astype(float)


def apply_execution_noise(
    simulator: SystemSimulator,
    action: np.ndarray,
    action_noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if action_noise_std <= 0.0:
        return action
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, action_noise_std, size=action.shape).astype(action.dtype, copy=False)
    return np.clip(action + noise, -simulator.max_action, simulator.max_action)


def _policy_action_fn(simulator: SystemSimulator, policy) -> Callable[[np.ndarray], np.ndarray]:
    return lambda observation: build_policy_action(simulator, policy, observation, torch.device("cpu"))
