from __future__ import annotations

import contextlib
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from planning.planner import Planner, PlannerSolveError
from systems.rollout_termination import RolloutTerminationTracker
from systems.seed_utils import action_noise_rng_for_rollout, torch_inference_seed_for_rollout
from systems.system import SystemSimulator

from .episode_starts import EpisodeStarts
from .metrics import DaggerEvalMetrics
from .policy_action import _policy_action_fn, apply_execution_noise
from .utils import evaluation_seed_specs
from .workers import RolloutWorkerSpec, _init_rollout_worker, _run_worker_task


class _EvalEpisodeResult(NamedTuple):
    reached_goal: bool
    rollout_steps: int
    failure_reason: str | None
    is_config_sourced: bool
    inference_seconds: float  # the wall-clock time of the policy calls of the episode that returned an action
    inference_calls: int


@dataclass(frozen=True)
class EvaluationSettings:
    num_steps: int
    action_noise_std: float
    action_noise_seed: int
    starts: EpisodeStarts

    def build_runner(self, simulator: SystemSimulator, expert_planner: Planner | None, policy) -> RolloutEvaluator:
        return RolloutEvaluator(simulator, self, _policy_action_fn(simulator, policy), policy.reset)


class RolloutEvaluator:
    """Rolls the policy out from the episodes' starts without an expert and tells how each episode ended."""

    def __init__(
        self,
        simulator: SystemSimulator,
        settings: EvaluationSettings,
        action_fn: Callable[[np.ndarray], np.ndarray],
        reset_fn: Callable[[], None] | None = None,
    ) -> None:
        self.simulator = simulator
        self.settings = settings
        self.action_fn = action_fn
        self.reset_fn = reset_fn

    def __call__(self, task: tuple[int, int]) -> _EvalEpisodeResult:
        """The task of a pool worker: the episode index and the seed spec of its start."""
        return self.run_episode(*task)

    def run_episode(self, episode_idx: int, seed_spec: int) -> _EvalEpisodeResult:
        settings = self.settings
        initial_state = settings.starts.begin(self.simulator, episode_idx, seed_spec)
        torch.manual_seed(torch_inference_seed_for_rollout(settings.action_noise_seed, seed_spec=seed_spec))
        call_times: list[float] = []
        reached_goal, rollout_steps, failure_reason = rollout_policy_with_action_fn(
            simulator=self.simulator,
            initial_state=initial_state,
            num_steps=settings.num_steps,
            action_fn=self.action_fn,
            reset_fn=self.reset_fn,
            action_noise_std=settings.action_noise_std,
            action_noise_rng=action_noise_rng_for_rollout(settings.action_noise_seed, seed_spec=seed_spec),
            call_times=call_times,
        )
        return _EvalEpisodeResult(
            reached_goal, int(rollout_steps), failure_reason, settings.starts.is_provided(episode_idx), sum(call_times), len(call_times)
        )


def rollout_policy_with_action_fn(
    simulator: SystemSimulator,
    initial_state: np.ndarray,
    num_steps: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
    call_times: list[float] | None = None,
) -> tuple[bool, int, str | None]:
    """Returns (reached_goal, steps_taken, failure_reason). The wall-clock duration of every ``action_fn`` call that
    returned an action is appended to ``call_times``.

    failure_reason is None on success, else one of "timeout" (ran out of steps
    without reaching tolerance), "solve_failure" (action_fn itself raised) or
    "collision" (the system touched an obstacle, which ends the episode) --
    see DaggerEvalMetrics for why these are tracked separately rather than
    folded into one failure count.
    """
    state = simulator.reset(initial_state)
    termination = RolloutTerminationTracker(simulator)
    if reset_fn is not None:
        reset_fn()
    if termination.update(state):
        return True, 0, None
    for step in range(1, num_steps + 1):
        call_start = time.perf_counter()
        try:
            action = action_fn(simulator.observe(state))
        except PlannerSolveError as exc:
            # No expert running alongside evaluation to fall back to (unlike
            # collect_dagger_rollouts) -- treat like any other episode
            # failure (a timeout) rather than crashing the
            # whole evaluation/training run over one solver hiccup.
            # simulator.step() never ran this iteration, so `step` (a
            # 1-indexed loop counter) overcounts the actually-executed steps
            # by one -- step - 1 matches every other return in this function,
            # each of which only reaches `step` after simulator.step() ran.
            print(f"Safeflow policy solve failed during evaluation at step={step}: {exc}")
            return False, step - 1, "solve_failure"
        if call_times is not None:
            call_times.append(time.perf_counter() - call_start)
        state = simulator.step(state, apply_execution_noise(simulator, action, action_noise_std, action_noise_rng))
        if simulator.collides(state):
            return False, step, "collision"
        if termination.update(state):
            return True, step, None
    return False, num_steps, "timeout"


@contextlib.contextmanager
def _isolated_torch_rng_state():
    """Snapshots/restores torch's global RNG state(s) around a block.

    torch.manual_seed(...) mutates the same process-wide generator(s) (CPU,
    every CUDA device, MPS) that a flow/safeflow policy's own training
    (FlowPolicy.compute_loss's torch.randn_like/torch.rand) and inference
    (select_action's ODE noise) both draw from -- there's no separate "eval"
    generator to scope a manual_seed call to. Without restoring the pre-eval
    state afterward, pinning eval's own seed for reproducibility (see the
    call site below) would leave the *next* DAgger round's training noise
    sequence depending on how many eval episodes ran and how each rollout
    played out -- a confound that has nothing to do with the policy actually
    changing.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mps_state = torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        if mps_state is not None:
            torch.mps.set_rng_state(mps_state)


def _summarize_evaluation(results: list[_EvalEpisodeResult]) -> DaggerEvalMetrics:
    """The metrics of the episodes, also split by whether an episode came from the provided starts (likely memorized or
    specialized) or was sampled (generalization), so the two are never averaged together without being visible on
    their own -- see DaggerEvalMetrics."""
    successes = config_successes = config_episodes = random_successes = random_episodes = 0
    timeout_failures = solve_failures = collision_failures = 0
    steps_taken: list[int] = []
    inference_seconds = sum(res.inference_seconds for res in results)
    inference_calls = sum(res.inference_calls for res in results)
    for res in results:
        successes += int(res.reached_goal)
        steps_taken.append(res.rollout_steps)
        if res.failure_reason == "timeout":
            timeout_failures += 1
        elif res.failure_reason == "solve_failure":
            solve_failures += 1
        elif res.failure_reason == "collision":
            collision_failures += 1
        if res.is_config_sourced:
            config_successes += int(res.reached_goal)
            config_episodes += 1
        else:
            random_successes += int(res.reached_goal)
            random_episodes += 1
    return DaggerEvalMetrics(
        success_rate=float(successes) / float(len(results)),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=len(results),
        config_successes=config_successes,
        config_num_episodes=config_episodes,
        random_successes=random_successes,
        random_num_episodes=random_episodes,
        timeout_failures=timeout_failures,
        solve_failures=solve_failures,
        collision_failures=collision_failures,
        mean_inference_ms=1000.0 * inference_seconds / inference_calls if inference_calls else None,
    )


def evaluate_policy_rollouts(
    simulator: SystemSimulator,
    num_episodes: int,
    num_steps: int,
    seed_start: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_seed: int = 0,
    initial_states: list[np.ndarray] | None = None,
    goal_states: list[np.ndarray] | None = None,
    num_workers: int = 1,
    worker_spec: RolloutWorkerSpec | None = None,
) -> DaggerEvalMetrics | None:
    if num_episodes == 0:
        return None
    if num_steps <= 0:
        raise ValueError("'num_steps' must be positive.")
    seed_specs = evaluation_seed_specs(num_episodes, seed_start)
    settings = EvaluationSettings(
        num_steps=num_steps,
        action_noise_std=action_noise_std,
        action_noise_seed=action_noise_seed,
        starts=EpisodeStarts.from_specs(simulator, initial_states, goal_states),
    )
    if num_workers > 1 and worker_spec is not None and num_episodes > 1:
        with multiprocessing.get_context("spawn").Pool(
            min(num_workers, num_episodes), initializer=_init_rollout_worker, initargs=(worker_spec, settings)
        ) as pool:
            results = pool.map(_run_worker_task, list(enumerate(seed_specs)))
    else:
        evaluator = RolloutEvaluator(simulator, settings, action_fn, reset_fn)
        results = []
        for episode_idx, seed_spec in enumerate(seed_specs):
            with _isolated_torch_rng_state():
                results.append(evaluator.run_episode(episode_idx, seed_spec))
    return _summarize_evaluation(results)
