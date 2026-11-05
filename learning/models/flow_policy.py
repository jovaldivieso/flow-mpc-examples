from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from learning.models.flow_velocity_net import FlowVelocityNet
from learning.models.policy import ActionPolicy, observation_features
from learning.models.pose_guidance import PoseGuidance
from planning.casadi_projector import CasadiTrajectoryProjector


class FlowPolicy(ActionPolicy):
    """Conditional flow-matching policy with Euler ODE inference over chunks of joint positions.

    It predicts a chunk of ``prediction_horizon`` configurations (rows of ``action_dim`` joint positions, row k the
    configuration k + 1 control periods ahead), conditioned on ``[observation.environment_state, observation.history]``
    (see ``observation_features``). The flow-matching network and math itself live in FlowVelocityNet.

    Every Euler step of the inference moves the proposal toward the goal pose (``guidance``, see ``guide``), and nothing else
    corrects the iterates (Eq. 9 of SafeFlowMPC without its projection onto the safe set): only the final chunk is looked at.
    The jerks that execute it are those of ``tracker``, the tracking of positions within the limits of the plant alone: the
    limits hold for what is executed, neither obstacles nor rest are enforced, which is what makes this policy the one without
    safety filter.
    The inner policy of a safe flow policy has no tracker (``None``): the projection of the safe flow policy executes it.
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        history_length: int,
        execution_horizon: int,
        guidance: PoseGuidance,
        tracker: CasadiTrajectoryProjector | None,
        prediction_horizon: int = 16,
        unet_dim: int = 32,
        unet_dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_inference_steps: int = 10,
        action_scale: Sequence[float] | torch.Tensor | None = None,
    ):
        super().__init__()
        self._history_length = int(history_length)
        self._execution_horizon = int(execution_horizon)
        self.guidance = guidance
        self.tracker = tracker

        self.velocity_net = FlowVelocityNet(
            action_dim=action_dim,
            obs_cond_dim=int(cond_dim),
            prediction_horizon=prediction_horizon,
            unet_dim=unet_dim,
            unet_dim_mults=unet_dim_mults,
            num_inference_steps=num_inference_steps,
            action_scale=action_scale,
        )

    def compute_loss(
        self, observation_dict: Mapping[str, torch.Tensor], positions: torch.Tensor, paths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.velocity_net.compute_loss(observation_features(observation_dict), positions, paths)

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def execution_horizon(self) -> int:
        return self._execution_horizon

    @property
    def network(self) -> nn.Module:
        return self

    def reset(self) -> None:
        """Start of a new episode: the network gets the same call, so a stateful layer would be cleared here."""
        self.velocity_net.reset()
        self.guidance.reset()
        if self.tracker is not None:
            self.tracker.reset()

    def state(self, observation_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        """The plant state of the one observation (the guidance and the projection start from it)."""
        return observation_dict["observation.state"][0].detach().cpu().numpy().astype(float)

    def guide(self, observation_dict: Mapping[str, torch.Tensor], proposal: np.ndarray) -> np.ndarray:
        """The proposed chunk of positions (H, n) of one observation moved toward the goal."""
        return self.guidance.guide(
            observation_dict["observation.environment_state"][0].detach().cpu().numpy().astype(float),
            self.state(observation_dict),
            proposal,
        )

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if self.tracker is None:
            raise RuntimeError("This flow policy has no tracker to execute its positions with; its safe flow policy selects the action.")
        positions = self.velocity_net.integrate(
            observation_features(observation_dict), lambda proposal: self.guide(observation_dict, proposal)
        )[0].numpy().astype(float)
        self.tracker.begin_tick()
        return torch.as_tensor(self.tracker.project(self.state(observation_dict), positions).actions, dtype=torch.float32).unsqueeze(0)
