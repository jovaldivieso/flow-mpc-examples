from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy, observation_features
from planning.casadi_projector import CasadiTrajectoryProjector

FLOW_INITIALIZATIONS = ("previous_safe", "gaussian")


class SafeFlowMPCPolicy(ActionPolicy):
    """Wraps a FlowPolicy with a CasADi safety projection at every inference step (SafeFlowMPC, Oelerich et al., 2026).

    Loss and reset delegate to the inner FlowPolicy. ``select_action`` integrates the flow over the chunk of actions:
    after every flow step the proposal is moved toward the goal by the inner policy's guidance and projected onto the
    safe set by ``CasadiTrajectoryProjector``, and the projected chunk is the flow's next state. The first action of
    the returned chunk is what the system executes.

    With ``flow_init="previous_safe"`` the integration does not start from Gaussian noise but from the previous
    tick's safe actions (shifted by one step), or, on the first tick of an episode, from zero actions.

    The projector takes the state the policy observes (``observation.state``) and nothing of the goal. The state dict is
    the inner policy's behind the ``inner_policy.`` prefix, which ``load_network_state_dict`` maps checkpoint keys onto.
    """

    def __init__(
        self,
        inner_policy: FlowPolicy,
        projector: CasadiTrajectoryProjector,
        flow_init: str = "previous_safe",
    ) -> None:
        super().__init__()
        if flow_init not in FLOW_INITIALIZATIONS:
            raise ValueError(f"'flow_init' must be one of {FLOW_INITIALIZATIONS}, got {flow_init!r}.")
        self.inner_policy = inner_policy
        self.projector = projector
        self.flow_init = flow_init

    def reset(self) -> None:
        self.inner_policy.reset()
        # The projector carries its own persistent primal/dual warm-start state across calls: without clearing it
        # here, a new episode's first solve would warm start from the previous episode's unrelated final trajectory.
        self.projector.reset()

    def compute_loss(
        self,
        observation_dict: Mapping[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner_policy.compute_loss(observation_dict, actions)

    def _initial_chunk(self) -> np.ndarray | None:
        """The safe actions the flow integration starts from, or None for Gaussian noise."""
        if self.flow_init == "gaussian":
            return None
        chunk = self.projector.next_chunk_guess()
        return np.zeros((self.projector.tracked_horizon, self.inner_policy.velocity_net.action_dim)) if chunk is None else chunk

    @property
    def network(self) -> nn.Module:
        return self.inner_policy

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy

        def project(proposal: np.ndarray) -> np.ndarray:
            # The projector serves one system and takes the state the policy observes (the integration checks the batch
            # holds just its one row).
            state = observation_dict["observation.state"][0].detach().cpu().numpy().astype(float)
            return self.projector.project(state, inner.guide(observation_dict, proposal))

        return inner.velocity_net.integrate(observation_features(observation_dict), project, initial=self._initial_chunk())
