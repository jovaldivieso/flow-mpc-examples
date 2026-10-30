from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch

from learning.models.flow_velocity_net import FlowVelocityNet
from learning.models.policy import ActionPolicy, observation_features
from learning.models.pose_guidance import PoseGuidance


class FlowPolicy(ActionPolicy):
    """Conditional flow-matching policy with Euler ODE inference.

    It predicts a chunk of ``prediction_horizon`` actions (rows of ``action_dim`` jerks, row k the action k steps
    ahead, row 0 the one executed now), conditioned directly on the flat observation features
    ``[observation.environment_state, observation.state]`` (see ``observation_features``). The flow-matching
    network and math itself live in FlowVelocityNet.

    With a ``guidance`` (``PoseGuidance``) every Euler step of the inference moves the proposal toward the goal pose
    (see ``guide``) and clips it to the jerk limits, the one constraint a flow policy knows and the network was trained
    within (Eq. 9 of SafeFlowMPC without its projection onto the safe set); without a guidance it is the plain
    integration of the network.
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
        guidance: PoseGuidance | None = None,
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

    def guide(self, observation_dict: Mapping[str, torch.Tensor], proposal: np.ndarray) -> np.ndarray:
        """The proposed chunk of jerks (H, action_dim) of one observation moved toward the goal, as it is without guidance."""
        if self.guidance is None:
            return proposal
        return self.guidance.guide(
            observation_dict["observation.environment_state"][0].detach().cpu().numpy().astype(float),
            observation_dict["observation.state"][0].detach().cpu().numpy().astype(float),
            proposal,
        )

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        obs_cond = observation_features(observation_dict)
        net = self.velocity_net
        if self.guidance is None:
            return net.integrate(obs_cond)
        if obs_cond.shape[0] != 1:
            raise ValueError(f"A guided FlowPolicy serves one system, so the batch must hold one row, got {obs_cond.shape[0]}.")
        scale = net.action_scale.detach().cpu().numpy()
        limit = self.guidance.sim.max_action
        x = torch.randn(1, net.prediction_horizon, net.action_dim, device=obs_cond.device, dtype=obs_cond.dtype)
        dt = 1.0 / float(net.num_inference_steps)
        for step in range(net.num_inference_steps):
            t_tensor = torch.full((1,), step * dt, device=obs_cond.device, dtype=obs_cond.dtype)
            velocity = net.predict_velocity(x.flatten(1), obs_cond, t_tensor).view_as(x)
            proposal = self.guide(observation_dict, (x + dt * velocity)[0].detach().cpu().numpy() * scale)
            proposal = np.clip(proposal, -limit, limit)
            x = torch.tensor(proposal / scale, device=x.device, dtype=x.dtype).unsqueeze(0)
        return x * net.action_scale

    def reset(self) -> None:
        """Start of a new episode: the network gets the same call, so a stateful layer would be cleared here."""
        self.velocity_net.reset()
        if self.guidance is not None:
            self.guidance.reset()
