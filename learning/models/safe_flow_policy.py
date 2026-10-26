from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy, observation_features
from planning.casadi_projector import CasadiTrajectoryProjector

FLOW_INITIALIZATIONS = ("previous_safe", "gaussian")


class SafeFlowMPCPolicy(ActionPolicy):
    """Wraps a FlowPolicy with a CasADi safety projection at every inference step (SafeFlowMPC, Oelerich et al., 2026).

    Loss and reset delegate to the inner FlowPolicy. ``select_action`` re-implements the Euler ODE integration loop
    over the chunk of actions: after every flow step the proposal is projected onto the safe set by
    ``CasadiTrajectoryProjector``, and the projected chunk is the flow's next state. The first action of the
    returned chunk is what the system executes.

    With ``flow_init="previous_safe"`` the integration does not start from Gaussian noise but from the previous
    tick's safe actions (shifted by one step), or, on the first tick of an episode, from zero actions.

    The projector takes the state the policy observes (``observation.state``) and nothing of the goal. The state dict is
    the inner policy's behind the ``inner_policy.`` prefix, which ``load_policy_state_dict`` maps checkpoint keys onto.
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

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy
        net = inner.velocity_net
        obs_cond = observation_features(observation_dict)
        batch_size = obs_cond.shape[0]
        # self.projector serves exactly one system, so the batch must hold exactly its one row (see
        # build_policy_action, the only caller, which always builds exactly one) -- not checking this would let a
        # batch of the wrong size silently leave every row past the first unprojected or crash with an opaque
        # IndexError instead of failing clearly.
        if batch_size != 1:
            raise ValueError(
                f"SafeFlowMPCPolicy.select_action received a batch of size {batch_size}, but "
                "this policy was constructed for a single system. The batch must contain exactly one row."
            )
        device = obs_cond.device
        state = observation_dict["observation.state"][0].detach().cpu().numpy().astype(float)

        # net.predict_velocity operates in FlowVelocityNet's normalized space (see FlowVelocityNet.compute_loss /
        # integrate), but the projector compares physical actions, so x stays normalized throughout (that's what the
        # network was trained to integrate) and only the values handed to / read back from the projector are rescaled.
        scale = net.action_scale.detach().cpu().numpy()
        initial = self._initial_chunk()
        if initial is None:
            x = torch.randn(batch_size, net.prediction_horizon, net.action_dim, device=device, dtype=obs_cond.dtype)
        else:
            x = torch.tensor(initial / scale, device=device, dtype=obs_cond.dtype).unsqueeze(0)

        dt = 1.0 / float(net.num_inference_steps)
        for step in range(net.num_inference_steps):
            t_tensor = torch.full((batch_size,), step * dt, device=device, dtype=obs_cond.dtype)
            pred_velocity = net.predict_velocity(x.flatten(1), obs_cond, t_tensor)
            pred_velocity = pred_velocity.view(batch_size, net.prediction_horizon, net.action_dim)
            proposal = (x + dt * pred_velocity)[0].detach().cpu().numpy() * scale
            safe = self.projector.project(state, proposal)
            x = torch.tensor(safe / scale, device=device, dtype=x.dtype).unsqueeze(0)

        # Back to physical units, matching FlowPolicy.select_action's own return convention
        return x * net.action_scale
