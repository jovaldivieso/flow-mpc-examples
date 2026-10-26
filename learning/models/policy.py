from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn


class ActionPolicy(nn.Module, ABC):
    """Abstract base class for all continuous-action imitation learning policies."""

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Do not call forward() directly on ActionPolicy. "
            "Use select_action() for inference or compute_loss() for training."
        )

    @abstractmethod
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


def observation_features(observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Flat conditioning vector ``[observation.environment_state, observation.state]``.

    This is what the flow policies are conditioned on directly (and what
    ``CasadiTrajectoryProjector``/``invert_obs`` take as their flat ego
    observation): the policy observation dict has exactly those two keys.
    """
    return torch.cat(
        [
            observation_dict["observation.environment_state"],
            observation_dict["observation.state"],
        ],
        dim=-1,
    )
