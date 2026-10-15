from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy, observation_features
from planning.casadi_projector import CasadiTrajectoryProjector
from dynamics.dynamics import DynamicsProtocol

FLOW_INITIALIZATIONS = ("previous_safe", "gaussian")


class SafeFlowMPCPolicy(ActionPolicy):
    """Wraps a FlowPolicy with a CasADi safety projection at every inference step (SafeFlowMPC, Oelerich et al., 2026).

    Loss and reset delegate to the inner FlowPolicy. ``select_action`` re-implements the Euler ODE integration loop
    over the chunk of actions: after every flow step the proposal is projected onto the safe set by
    ``CasadiTrajectoryProjector``, and the projected chunk is the flow's next state. The first action of the
    returned chunk is what the system executes.

    With ``flow_init="previous_safe"`` the integration does not start from Gaussian noise but from the previous
    tick's safe actions (shifted by one step), or, on the first tick of an episode, from zero actions. A network
    trained only from noise is not adapted to that start; ``learning/finetune_safe.py`` trains it on the projected
    paths from such starts.
    """

    def __init__(
        self,
        inner_policy: FlowPolicy,
        projector: CasadiTrajectoryProjector,
        local_sim: DynamicsProtocol,
        flow_init: str = "previous_safe",
    ) -> None:
        super().__init__()
        if flow_init not in FLOW_INITIALIZATIONS:
            raise ValueError(f"'flow_init' must be one of {FLOW_INITIALIZATIONS}, got {flow_init!r}.")
        self.inner_policy = inner_policy
        # This is PolicyFactory.create's own private, goal-zeroed copy: the projector reads no goal, so it is never a
        # live simulator reference and has nothing to keep in sync with whatever simulator drives a rollout.
        self.local_sim = local_sim
        self.projector = projector
        self.flow_init = flow_init

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """Load either a SafeFlowMPCPolicy checkpoint or a bare FlowPolicy checkpoint.

        Strips a leading ``inner_policy.`` prefix if present (a checkpoint saved from a SafeFlowMPCPolicy), otherwise
        takes keys as-is (a plain FlowPolicy checkpoint) -- ``projector`` never appears in a state dict since
        ``CasadiTrajectoryProjector`` isn't an nn.Module. Delegates to ``self.inner_policy.load_state_dict``
        directly rather than ``super().load_state_dict``: PyTorch's recursive module loading calls each submodule's
        private ``_load_from_state_dict`` hook, not its public ``load_state_dict`` override, so going through
        ``super()`` would silently skip FlowPolicy's own key remapping.
        """
        inner_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("inner_policy."):
                inner_state_dict[key[len("inner_policy.") :]] = value
            elif not key.startswith("projector."):
                inner_state_dict[key] = value
        return self.inner_policy.load_state_dict(inner_state_dict, strict=strict, assign=assign)

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

    def _extract_ego_observation(self, observation_dict: Mapping[str, torch.Tensor]) -> np.ndarray:
        """Reconstruct the flat ego observation ``local_sim.invert_obs`` expects."""
        parts = [
            observation_dict["observation.environment_state"],
            observation_dict["observation.state"],
        ]
        return torch.cat(parts, dim=-1).detach().cpu().numpy()

    def _initial_chunk(self) -> np.ndarray | None:
        """The safe actions the flow integration starts from, or None for Gaussian noise."""
        if self.flow_init == "gaussian":
            return None
        chunk = self.projector.next_chunk_guess()
        return np.zeros((self.projector.tracked_horizon, self.local_sim.nu)) if chunk is None else chunk

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy
        net = inner.velocity_net
        obs_cond = observation_features(observation_dict)
        batch_size = obs_cond.shape[0]
        # self.local_sim/self.projector serve exactly one system, so the batch must hold exactly its one row (see
        # build_policy_action, the only caller, which always builds exactly one) -- not checking this would let a
        # batch of the wrong size silently leave every row past the first unprojected or crash with an opaque
        # IndexError instead of failing clearly.
        if batch_size != 1:
            raise ValueError(
                f"SafeFlowMPCPolicy.select_action received a batch of size {batch_size}, but "
                "this policy was constructed for a single system. The batch must contain exactly one row."
            )
        device = obs_cond.device
        ego_obs_np = self._extract_ego_observation(observation_dict)[0]

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
            safe = self.projector.project(ego_obs_np, proposal)
            x = torch.tensor(safe / scale, device=device, dtype=x.dtype).unsqueeze(0)

        # Back to physical units, matching FlowPolicy.select_action's own return convention
        return x * net.action_scale
