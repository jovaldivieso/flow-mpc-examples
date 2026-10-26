from __future__ import annotations

from collections.abc import Mapping

import torch

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.casadi_projector import CasadiTrajectoryProjector

INNER_POLICY_PREFIX = "inner_policy."  # of the keys of a safe flow policy's state dict


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
            # The projector reads no goal of the simulator (its dynamics, limits and obstacles come from the system), so
            # the simulator is shared with it and not copied.
            projector = CasadiTrajectoryProjector(
                simulator,
                dict(planner_config),
                horizon=inner_policy.velocity_net.prediction_horizon,
            )
            return SafeFlowMPCPolicy(inner_policy, projector, **safeflow_options)
        raise ValueError(
            f"Unknown policy type '{policy_type}'. "
            f"Supported policies: 'flow', 'safeflow'."
        )


def load_policy_state_dict(
    policy: ActionPolicy, state_dict: Mapping[str, torch.Tensor], strict: bool = True
) -> torch.nn.modules.module._IncompatibleKeys:
    """Load the weights of a checkpoint of either policy type into ``policy``, with PyTorch's own ``load_state_dict``.

    A safe flow policy keeps its network behind ``inner_policy.``, a flow policy does not, and a checkpoint can come
    from either (the two share their weights); the projector has none, and ``torch.compile`` leaves ``_orig_mod`` in the
    names. The keys are mapped onto the ones ``policy`` itself uses first, so the loading, its hooks and its ``strict``
    check are PyTorch's.
    """
    prefix = INNER_POLICY_PREFIX if isinstance(policy, SafeFlowMPCPolicy) else ""
    network_keys = {
        key.removeprefix(INNER_POLICY_PREFIX).replace("._orig_mod.", "."): value
        for key, value in state_dict.items()
        if not key.startswith("projector.")
    }
    return policy.load_state_dict({prefix + key: value for key, value in network_keys.items()}, strict=strict)
