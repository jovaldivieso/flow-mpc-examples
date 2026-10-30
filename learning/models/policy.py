from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch
from torch import nn

INNER_POLICY_PREFIX = "inner_policy."  # of the keys of a safe flow policy's state dict
PROJECTOR_PREFIX = "projector."


class ActionPolicy(nn.Module, ABC):
    """Abstract base class for all continuous-action imitation learning policies."""

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Do not call forward() directly on ActionPolicy. "
            "Use select_action() for inference or compute_loss() for training."
        )

    @property
    @abstractmethod
    def network(self) -> nn.Module:
        """The module that holds the learned weights, the flow policy every policy type is made of."""

    def load_network_state_dict(
        self, state_dict: Mapping[str, torch.Tensor], strict: bool = True
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load the weights of a checkpoint of any policy type, with PyTorch's own ``load_state_dict``.

        A checkpoint can come from a flow policy or from a safe flow policy (which keeps its flow policy behind
        ``inner_policy.``, next to a projector that has no weights): the two share their weights. ``torch.compile``
        leaves ``_orig_mod`` in the names. The keys are mapped onto the ones of ``network`` first, so the loading, its
        hooks and its ``strict`` check are PyTorch's.
        """
        network_keys = {
            key.removeprefix(INNER_POLICY_PREFIX).replace("._orig_mod.", "."): value
            for key, value in state_dict.items()
            if not key.startswith(PROJECTOR_PREFIX)
        }
        return self.network.load_state_dict(network_keys, strict=strict)

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
