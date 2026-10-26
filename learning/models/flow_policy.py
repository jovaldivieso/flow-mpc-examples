from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from learning.models.flow_velocity_net import FlowVelocityNet
from learning.models.policy import ActionPolicy, observation_features


class FlowPolicy(ActionPolicy):
    """Conditional flow-matching policy with Euler ODE inference.

    It predicts a chunk of ``prediction_horizon`` actions (rows of ``action_dim`` jerks, row k the action k steps
    ahead, row 0 the one executed now), conditioned directly on the flat observation features
    ``[observation.environment_state, observation.state]`` (see ``observation_features``). The flow-matching
    network and math itself live in FlowVelocityNet.
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        prediction_horizon: int = 16,
        unet_dim: int = 32,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_inference_steps: int = 10,
        action_scale: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()

        self.velocity_net = FlowVelocityNet(
            action_dim=action_dim,
            obs_cond_dim=int(state_dim),
            prediction_horizon=prediction_horizon,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            num_inference_steps=num_inference_steps,
            action_scale=action_scale,
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load a checkpoint from either a bare FlowPolicy or a SafeFlowMPCPolicy wrapper."""
        normalized_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("inner_policy."):
                key = key[len("inner_policy.") :]
            elif key.startswith("projector."):
                continue
            key = key.replace("._orig_mod.", ".")
            normalized_state_dict[key] = value
        return super().load_state_dict(normalized_state_dict, strict=strict, assign=assign)

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = observation_features(observation_dict)
        return self.velocity_net.compute_loss(obs_cond, actions)

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs_cond = observation_features(observation_dict)
        return self.velocity_net.integrate(obs_cond)

    def reset(self) -> None:
        """Keeps parity with other policy APIs that expose a reset hook."""
        return None
