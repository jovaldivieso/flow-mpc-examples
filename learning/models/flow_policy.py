from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from learning.models.flow_velocity_net import FlowVelocityNet
from learning.models.policy import ActionPolicy, observation_features
from learning.models.pose_guidance import PoseGuidance


class FlowPolicy(ActionPolicy):
    """Conditional flow-matching policy with Euler ODE inference.

    It predicts a chunk of ``prediction_horizon`` actions (rows of ``action_dim`` jerks, row k the action k steps
    ahead, row 0 the one executed now), conditioned directly on the flat observation features
    ``[observation.environment_state, observation.state]`` (see ``observation_features``). The flow-matching
    network and math itself live in FlowVelocityNet.

    Every Euler step of the inference moves the proposal toward the goal pose (``guidance``, see ``guide``) and clips
    it to the jerk limits, the one constraint a flow policy knows and the network was trained within (Eq. 9 of
    SafeFlowMPC without its projection onto the safe set).
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        guidance: PoseGuidance,
        prediction_horizon: int = 16,
        unet_dim: int = 32,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_inference_steps: int = 10,
        action_scale: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        self.guidance = guidance

        self.velocity_net = FlowVelocityNet(
            action_dim=action_dim,
            obs_cond_dim=int(state_dim),
            prediction_horizon=prediction_horizon,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            num_inference_steps=num_inference_steps,
            action_scale=action_scale,
        )

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        obs_cond = observation_features(observation_dict)
        return self.velocity_net.compute_loss(obs_cond, actions)

    @property
    def network(self) -> nn.Module:
        return self

    def guide(self, observation_dict: Mapping[str, torch.Tensor], proposal: np.ndarray) -> np.ndarray:
        """The proposed chunk of jerks (H, action_dim) of one observation moved toward the goal."""
        return self.guidance.guide(
            observation_dict["observation.environment_state"][0].detach().cpu().numpy().astype(float),
            observation_dict["observation.state"][0].detach().cpu().numpy().astype(float),
            proposal,
        )

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        limit = self.guidance.max_action
        return self.velocity_net.integrate(
            observation_features(observation_dict),
            lambda proposal: np.clip(self.guide(observation_dict, proposal), -limit, limit),
        )

    def reset(self) -> None:
        """Start of a new episode: the network gets the same call, so a stateful layer would be cleared here."""
        self.velocity_net.reset()
        self.guidance.reset()
