from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import ActionPolicy
from learning.models.pose_guidance import PoseGuidance
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.casadi_projector import CasadiTrajectoryProjector
from planning.planner_config import ProjectionConfig
from systems.triple_integrator import TripleIntegratorBase


@dataclass(frozen=True)
class PolicySpec:
    """Everything of a policy but the system it serves: what the factory builds it from, and what the rollout pool
    sends its workers."""

    policy_type: str
    action_dim: int  # the joint positions of a configuration of the chunk
    cond_dim: int  # of the conditioning vector: the environment and the history of positions
    history_length: int  # configurations of the history
    execution_horizon: int  # control periods the first jerks of the chunk are executed for before the next policy call
    prediction_horizon: int
    unet_dim: int
    unet_dim_mults: tuple[int, ...]
    num_inference_steps: int
    action_scale: tuple[float, ...]  # the positions are scaled by half the range of their joint, to the order of the unit-scale noise prior
    flow_init: str = "previous_safe"  # of a safe flow policy


PolicyBuilder = Callable[[PolicySpec, TripleIntegratorBase, Mapping[str, Any] | None], ActionPolicy]


def _projection_config(planner_config: Mapping[str, Any] | None, policy_type: str) -> ProjectionConfig:
    if planner_config is None:
        raise ValueError(f"policy_type '{policy_type}' requires a planner config to build its CasadiTrajectoryProjector.")
    return ProjectionConfig.from_mapping(planner_config)


def _flow_policy(spec: PolicySpec, simulator: TripleIntegratorBase, tracker: CasadiTrajectoryProjector | None) -> FlowPolicy:
    return FlowPolicy(
        action_dim=spec.action_dim,
        cond_dim=spec.cond_dim,
        history_length=spec.history_length,
        execution_horizon=spec.execution_horizon,
        guidance=PoseGuidance(simulator, spec.prediction_horizon),
        tracker=tracker,
        prediction_horizon=spec.prediction_horizon,
        unet_dim=spec.unet_dim,
        unet_dim_mults=spec.unet_dim_mults,
        num_inference_steps=spec.num_inference_steps,
        action_scale=spec.action_scale,
    )


def build_flow_policy(spec: PolicySpec, simulator: TripleIntegratorBase, planner_config: Mapping[str, Any] | None) -> FlowPolicy:
    """A flow policy whose flow is guided toward the goal pose of ``simulator``, executed by the tracking within the plant's
    limits."""
    tracker = CasadiTrajectoryProjector(
        simulator,
        _projection_config(planner_config, spec.policy_type),
        horizon=spec.prediction_horizon,
        safe=False,
        tick_steps=spec.execution_horizon,
    )
    return _flow_policy(spec, simulator, tracker)


def build_safe_flow_policy(
    spec: PolicySpec, simulator: TripleIntegratorBase, planner_config: Mapping[str, Any] | None
) -> SafeFlowMPCPolicy:
    """A flow policy with the projection of ``CasadiTrajectoryProjector``, built from the validated expert config
    ``planner_config``. The projector reads no goal of the simulator (its dynamics, limits and obstacles come from the
    system), so the simulator is shared with it and not copied."""
    projector = CasadiTrajectoryProjector(
        simulator, _projection_config(planner_config, spec.policy_type), horizon=spec.prediction_horizon, tick_steps=spec.execution_horizon
    )
    return SafeFlowMPCPolicy(_flow_policy(spec, simulator, None), projector, flow_init=spec.flow_init)


# The policies the config can choose (``model.policy_type``)
POLICY_REGISTRY: dict[str, PolicyBuilder] = {
    "flow": build_flow_policy,
    "safeflow": build_safe_flow_policy,
}


class PolicyFactory:
    """Factory for policy instances."""

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(POLICY_REGISTRY.keys())

    @staticmethod
    def create(
        spec: PolicySpec, simulator: TripleIntegratorBase, planner_config: Mapping[str, Any] | None = None
    ) -> ActionPolicy:
        try:
            builder = POLICY_REGISTRY[spec.policy_type]
        except KeyError as exc:
            raise ValueError(f"Unknown policy type '{spec.policy_type}'. Available: {sorted(POLICY_REGISTRY)}") from exc
        return builder(spec, simulator, planner_config)
