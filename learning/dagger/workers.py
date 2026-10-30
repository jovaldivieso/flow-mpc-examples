from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

import torch

from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.ompl_planner import initialize_ompl
from planning.planner import Planner
from planning.planner_factory import PlannerFactory
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory


class RunnerSettings(Protocol):
    """The settings of a pool of workers: they build the runner of one worker, a callable that runs one task."""

    def build_runner(
        self, simulator: SystemSimulator, expert_planner: Planner | None, policy
    ) -> Callable[[Any], Any]: ...


class RolloutWorkerSpec(NamedTuple):
    """Picklable specification for constructing a rollout worker's simulator, expert planner, and CPU policy."""

    config: dict[str, Any]
    policy_spec: PolicySpec | None = None
    policy_state_dict: dict[str, torch.Tensor] | None = None
    need_expert: bool = True


class RolloutWorker:
    """The state of one process of a rollout pool: the simulator, the expert planner and the CPU policy built from the
    spec, and the runner its settings build (a ``RolloutCollector`` or a ``RolloutEvaluator``).

    Called with a task: an attempt index when collecting, an ``(episode index, seed spec)`` pair when evaluating.
    """

    def __init__(self, spec: RolloutWorkerSpec, settings: RunnerSettings) -> None:
        torch.set_num_threads(1)
        initialize_ompl(spec.config["ompl_seed"])  # a spawned process has its own OMPL globals
        self.simulator = SystemFactory.create(spec.config["system"], spec.config)
        planner = PlannerFactory.create(spec.config["expert"], self.simulator, spec.config) if spec.need_expert else None
        self.policy = self._build_policy(spec)
        self._runner = settings.build_runner(self.simulator, planner, self.policy)

    def _build_policy(self, spec: RolloutWorkerSpec):
        if spec.policy_spec is None:
            return None
        policy = PolicyFactory.create(spec.policy_spec, self.simulator, spec.config).to(torch.device("cpu")).eval()
        if spec.policy_state_dict is not None:
            policy.load_network_state_dict(spec.policy_state_dict)
        return policy

    def __call__(self, task: int | tuple[int, int]) -> Any:
        return self._runner(task)


_worker: RolloutWorker | None = None  # the one of this process of a pool


def _init_rollout_worker(spec: RolloutWorkerSpec, settings: RunnerSettings) -> None:
    global _worker
    _worker = RolloutWorker(spec, settings)


def _run_worker_task(task: int | tuple[int, int]) -> Any:
    assert _worker is not None
    return _worker(task)
