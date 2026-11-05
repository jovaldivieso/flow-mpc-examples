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

    Loss and reset delegate to the inner FlowPolicy. ``select_action`` integrates the flow over the chunk of positions:
    after every flow step the proposal is moved toward the goal by the inner policy's guidance and projected onto the
    safe set by ``CasadiTrajectoryProjector``, and the projected positions are the flow's next state. The jerks of the
    last projection are what the system executes, the first of them now.

    With ``flow_init="previous_safe"`` the integration does not start from Gaussian noise but from the previous
    tick's safe positions (one control period later), or, on the first tick of an episode, from the safe positions nearest
    to the arm coasting from the state (zero jerk), which keep it within its limits and clear of the obstacles: SafeFlowMPC
    starts every tick on the safe set (Assumption 2 and Eq. 10).

    The jerks of the last projection are what the system executes for the next ``execution_horizon`` control periods; the next
    tick starts from the plan that many periods later.

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
        self, observation_dict: Mapping[str, torch.Tensor], positions: torch.Tensor, paths: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.inner_policy.compute_loss(observation_dict, positions, paths)

    def _initial_chunk(self, state: np.ndarray) -> np.ndarray | None:
        """The safe positions the flow integration starts from, or None for Gaussian noise."""
        if self.flow_init == "gaussian":
            return None
        chunk = self.projector.next_chunk_guess()
        if chunk is not None:
            return chunk
        sim = self.projector.sim
        coasting = sim.rollout_positions(state, np.zeros((self.projector.tracked_horizon, sim.nu)))[1:]
        return self.projector.project(state, coasting).positions

    @property
    def history_length(self) -> int:
        return self.inner_policy.history_length

    @property
    def execution_horizon(self) -> int:
        return self.inner_policy.execution_horizon

    @property
    def network(self) -> nn.Module:
        return self.inner_policy

    @torch.no_grad()
    def select_action(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        inner = self.inner_policy
        inner.velocity_net.require_one_system(observation_features(observation_dict))
        self.projector.begin_tick()
        state = inner.state(observation_dict)
        actions: list[np.ndarray] = []

        def project(proposal: np.ndarray) -> np.ndarray:
            projection = self.projector.project(state, inner.guide(observation_dict, proposal))
            actions[:] = [projection.actions]
            return projection.positions

        inner.velocity_net.integrate(observation_features(observation_dict), project, initial=self._initial_chunk(state))
        return torch.as_tensor(actions[0], dtype=torch.float32).unsqueeze(0)
