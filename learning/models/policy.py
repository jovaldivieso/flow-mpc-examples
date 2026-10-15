from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
import torch
from torch import nn

DEFAULT_POLICY_TYPE = "flow"


class ActionPolicy(nn.Module, ABC):
    """Abstract base class for all continuous-action imitation learning policies."""

    def forward(self, observation_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError(
            "Do not call forward() directly on ActionPolicy. "
            "Use select_action() for inference or compute_loss() for training."
        )

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


class PolicyFactory:
    @staticmethod
    def create(policy_type: str, **kwargs: object) -> ActionPolicy:
        normalized_type = policy_type.strip().lower()
        if normalized_type == "flow":
            from learning.models.flow_policy import FlowPolicy

            return FlowPolicy(**kwargs)
        if normalized_type == "safeflow":
            from learning.models.flow_policy import FlowPolicy
            from learning.models.safe_flow_policy import SafeFlowMPCPolicy
            from planning.casadi_projector import CasadiTrajectoryProjector

            flow_kwargs = dict(kwargs)
            simulator = flow_kwargs.pop("simulator", None)
            planner_config = flow_kwargs.pop("planner_config", None)
            safeflow_options = {name: flow_kwargs.pop(name) for name in ("flow_init",) if name in flow_kwargs}
            if simulator is None or planner_config is None:
                raise ValueError(
                    "policy_type 'safeflow' requires 'simulator' and 'planner_config' kwargs "
                    "to build its CasadiTrajectoryProjector."
                )
            inner_policy = FlowPolicy(**flow_kwargs)
            # This is deep-copied with the goal zeroed, once, here, rather
            # than kept as a live reference into `simulator`: every
            # constraint/cost this projector pipeline touches (dynamics,
            # action/state bounds, terminal
            # velocity) is built from the state invert_obs reconstructs
            # from the observation, not from the goal or any absolute
            # world position -- so the goal used here is never anything more
            # than a fixed, self-consistent local anchor, not the live
            # episode's actual goal. That means this policy never needs to
            # track -- or be resynced to -- whatever simulator instance
            # actually drives the live rollout (DAgger collection and
            # evaluation each construct their own, separate from this one,
            # every round/call). Deep-copying also protects a caller that
            # passes the same simulator object for both policy construction
            # and its own rollout from having that simulator's goal
            # overwritten to zero out from under it.
            #
            # This equivalence would break if this projector ever gained a
            # genuinely world-frame-referencing term (finite absolute
            # position bounds, static obstacles, a goal-tracking cost) --
            # anything added like that must take the real goal as an
            # explicit input rather than reading it off this local_sim.
            local_sim = copy.deepcopy(simulator)
            local_sim.set_goal(np.zeros_like(local_sim.goal))

            projector = CasadiTrajectoryProjector(
                local_sim,
                dict(planner_config),
                horizon=inner_policy.velocity_net.prediction_horizon,
            )
            return SafeFlowMPCPolicy(inner_policy, projector, local_sim, **safeflow_options)
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported policies: 'flow', 'safeflow'."
        )
