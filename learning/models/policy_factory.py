from __future__ import annotations

import copy

import numpy as np

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.casadi_projector import CasadiTrajectoryProjector


class PolicyFactory:
    @staticmethod
    def create(policy_type: str, **kwargs: object) -> ActionPolicy:
        normalized_type = policy_type.strip().lower()
        if normalized_type == "flow":
            return FlowPolicy(**kwargs)
        if normalized_type == "safeflow":
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
