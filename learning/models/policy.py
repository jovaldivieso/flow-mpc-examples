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
    def execution_horizon(self) -> int:
        """How many control periods the first jerks of the selected chunk are executed before the policy is asked again."""

    @property
    @abstractmethod
    def history_length(self) -> int:
        """How many configurations of the arm's last positions (the current one last) the policy is conditioned on."""

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
        """The jerks (1, H, nu) that execute the policy's plan, the first row of which is executed."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def compute_loss(
        self, observation_dict: Mapping[str, torch.Tensor], positions: torch.Tensor, paths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """The training loss of the target chunks of positions ``positions`` (B, H, n); with ``paths`` (B, S + 1, H, n), the safe
        path of every row, the flow is learned along them (see ``FlowVelocityNet.compute_loss``), else from Gaussian noise
        towards the chunks."""
        raise NotImplementedError


def observation_features(observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Flat conditioning vector ``[observation.environment_state, observation.history]``: the policy is conditioned on what
    it sees of the world and the positions of the arm in its last configurations, not on its velocity and acceleration
    (``observation.state``, which the guidance and the projection start from)."""
    return torch.cat(
        [
            observation_dict["observation.environment_state"],
            observation_dict["observation.history"],
        ],
        dim=-1,
    )
