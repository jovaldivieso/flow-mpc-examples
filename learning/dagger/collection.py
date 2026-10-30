from __future__ import annotations

import multiprocessing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from learning.data_utils import format_dataset_frame
from planning.planner import Planner, PlannerSolveError
from systems.rollout_termination import RolloutTerminationTracker
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    expert_mixing_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)
from systems.system import SystemSimulator

from .episode_starts import EpisodeStarts
from .metrics import DaggerEvalMetrics
from .policy_action import _policy_action_fn, apply_execution_noise
from .workers import RolloutWorkerSpec, _init_rollout_worker, _run_worker_task


class _CollectionAttemptResult(NamedTuple):
    frames: list[dict[str, object]] | None
    discard_reason: str | None
    reached_goal: bool
    rollout_steps: int
    expert_executed_steps: int
    total_executed_steps: int
    planner_failures: int
    policy_solve_fallbacks: int
    expert_statistics: dict[str, int]


@dataclass(frozen=True)
class CollectionSettings:
    steps_per_trajectory: int
    action_noise_std: float
    action_noise_seed: int
    initial_state_seed: int
    initial_state_round_index: int | None
    expert_mixing_beta: float
    prediction_horizon: int
    round_index: int
    starts: EpisodeStarts
    should_query_policy: bool

    def build_runner(self, simulator: SystemSimulator, expert_planner: Planner | None, policy) -> RolloutCollector:
        queries_policy = self.should_query_policy and policy is not None
        return RolloutCollector(
            simulator,
            expert_planner,
            self,
            policy_action_fn=_policy_action_fn(simulator, policy) if queries_policy else None,
            policy_reset_fn=policy.reset if queries_policy else None,
        )


@dataclass
class _CollectionEpisode:
    """What one collection attempt has so far."""

    attempt_index: int
    state: np.ndarray
    initial_state: np.ndarray
    noise_seed: int
    noise_rng: np.random.Generator
    mixing_rng: np.random.Generator
    termination: RolloutTerminationTracker
    rollout_steps: int
    frames: list[dict[str, object]]
    actor_is_expert: list[bool]
    planner_failures: int = 0
    policy_solve_fallbacks: int = 0
    discard_reason: str | None = None
    reached_goal: bool = False


class RolloutCollector:
    """Collects the episodes of a DAgger round: every visited state is labeled with the expert's plan from it, whoever
    executes the step (the expert or the policy, mixed by ``expert_mixing_beta``).

    An episode whose expert planner fails to solve is discarded (``discard_reason``); a failed policy solve falls back
    to the expert's action of that step.
    """

    def __init__(
        self,
        simulator: SystemSimulator,
        expert_planner: Planner,
        settings: CollectionSettings,
        *,
        policy_action_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        policy_reset_fn: Callable[[], None] | None = None,
        frame_builder: Callable[[np.ndarray, np.ndarray], Mapping[str, object]] | None = None,
    ) -> None:
        self.simulator = simulator
        self.expert_planner = expert_planner
        self.settings = settings
        self.policy_action_fn = policy_action_fn
        self.policy_reset_fn = policy_reset_fn
        self.frame_builder = frame_builder if frame_builder is not None else self._format_frame

    def __call__(self, attempt_index: int) -> _CollectionAttemptResult:
        """The task of a pool worker: one attempt, with the policy's sampling seeded by its index."""
        if self.policy_action_fn is not None:
            torch.manual_seed(
                torch_inference_seed_for_rollout(
                    self.settings.action_noise_seed,
                    rollout_index=attempt_index,
                    round_index=self.settings.round_index,
                )
            )
        return self.run_attempt(attempt_index)

    def run_attempt(self, attempt_index: int) -> _CollectionAttemptResult:
        """Roll out one episode (attempts are counted from 1)."""
        stats_before = dict(self.expert_planner.statistics)
        episode = self._begin_episode(attempt_index)
        for step in range(1, self.settings.steps_per_trajectory + 1):
            if not self._run_step(episode, step):
                break
        return self._result(episode, stats_before)

    def _format_frame(self, observation: np.ndarray, expert_plan: np.ndarray) -> Mapping[str, object]:
        return format_dataset_frame(self.simulator, observation, expert_plan)

    def _begin_episode(self, attempt_index: int) -> _CollectionEpisode:
        settings = self.settings
        sampled_initial_state = settings.starts.begin(
            self.simulator,
            attempt_index - 1,
            initial_state_seed_for_rollout(
                settings.initial_state_seed, rollout_index=attempt_index, round_index=settings.initial_state_round_index
            ),
        )
        state = self.simulator.reset(sampled_initial_state)
        noise_seed = action_noise_seed_for_rollout(
            settings.action_noise_seed, rollout_index=attempt_index, round_index=settings.round_index
        )
        mixing_seed = expert_mixing_seed_for_rollout(
            settings.action_noise_seed, rollout_index=attempt_index, round_index=settings.round_index
        )
        if self.policy_reset_fn is not None:
            self.policy_reset_fn()
        self.expert_planner.reset()
        return _CollectionEpisode(
            attempt_index=attempt_index,
            state=state,
            initial_state=state.copy(),
            noise_seed=noise_seed,
            noise_rng=np.random.default_rng(noise_seed),
            mixing_rng=np.random.default_rng(mixing_seed),
            termination=RolloutTerminationTracker(self.simulator),
            rollout_steps=settings.steps_per_trajectory,
            frames=[],
            actor_is_expert=[],
        )

    def _run_step(self, episode: _CollectionEpisode, step: int) -> bool:
        """Label the state of the episode and execute one step; whether the episode goes on."""
        settings, simulator = self.settings, self.simulator
        observation = simulator.observe(episode.state)
        try:
            expert_plan = self.expert_planner.plan(observation)
        except PlannerSolveError as exc:
            print(
                f"Planner failed to solve (attempt={episode.attempt_index}, step={step}, "
                f"action_noise_std={settings.action_noise_std:.6f}, noise_seed={episode.noise_seed}): {exc} "
                f"initial_state={np.array2string(np.asarray(episode.initial_state), precision=6)}, "
                f"current_state={np.array2string(np.asarray(episode.state), precision=6)}, "
                f"goal={np.array2string(np.asarray(simulator.goal), precision=6)}"
            )
            episode.planner_failures += 1
            episode.discard_reason = "the expert planner failed to solve"
            return False

        use_expert_action = bool(episode.mixing_rng.random() < settings.expert_mixing_beta)
        policy_action = None
        if settings.should_query_policy and self.policy_action_fn is not None:
            policy_action = self._query_policy(episode, step, observation)
            if policy_action is None:
                use_expert_action = True
        built_frame = self.frame_builder(observation, expert_plan.first_actions(settings.prediction_horizon))
        if not isinstance(built_frame, Mapping):
            raise TypeError("'frame_builder' must return a mapping, got " f"{type(built_frame).__name__}.")
        episode.frames.append({"task": simulator.task_description, **built_frame})
        episode.actor_is_expert.append(use_expert_action)
        base_action = expert_plan.actions[0] if use_expert_action or policy_action is None else policy_action
        episode.state = simulator.step(
            episode.state,
            apply_execution_noise(simulator, base_action, settings.action_noise_std, episode.noise_rng),
        )
        if episode.termination.update(episode.state):
            episode.reached_goal = True
            episode.rollout_steps = step
            return False
        return True

    def _query_policy(self, episode: _CollectionEpisode, step: int, observation: np.ndarray) -> np.ndarray | None:
        """The policy's action, or none if its solve failed. Unlike an expert-planner failure the episode survives
        this: the expert's action of the step is already computed, so the step falls back to it and the episode goes
        on instead of being discarded. It counts as an expert step, since that is what gets executed."""
        try:
            return self.policy_action_fn(observation)
        except PlannerSolveError as exc:
            print(
                "Policy action unavailable this step, falling back to the "
                f"expert action (attempt={episode.attempt_index}, step={step}): {exc}"
            )
            episode.policy_solve_fallbacks += 1
            return None

    def _result(self, episode: _CollectionEpisode, stats_before: dict[str, int]) -> _CollectionAttemptResult:
        stats_delta = {
            name: count - stats_before.get(name, 0) for name, count in self.expert_planner.statistics.items()
        }
        return _CollectionAttemptResult(
            frames=None if episode.discard_reason is not None else episode.frames,
            discard_reason=episode.discard_reason,
            reached_goal=episode.reached_goal,
            rollout_steps=episode.rollout_steps,
            expert_executed_steps=sum(episode.actor_is_expert),
            total_executed_steps=len(episode.actor_is_expert),
            planner_failures=episode.planner_failures,
            policy_solve_fallbacks=episode.policy_solve_fallbacks,
            expert_statistics=stats_delta,
        )


class _CollectionTally:
    """The episodes a collection has kept so far and what it took to get them."""

    def __init__(self, expert_planner: Planner, trajectories: int, dataset_writer) -> None:
        self.trajectories = trajectories
        self.dataset_writer = dataset_writer
        self.successful_episodes = self.attempted_episodes = self.reached_goal_count = 0
        self.steps_taken: list[int] = []
        self.expert_executed_steps = self.total_executed_steps = 0
        self.planner_failures = self.discarded_episodes = self.policy_solve_fallbacks = 0
        self.expert_statistics: dict[str, int] = {name: 0 for name in expert_planner.statistics}

    def record(self, result: _CollectionAttemptResult) -> None:
        """Count an attempt and, unless it was discarded, write its episode to the dataset."""
        self.attempted_episodes += 1
        self.planner_failures += result.planner_failures
        self.policy_solve_fallbacks += result.policy_solve_fallbacks
        for name, delta in result.expert_statistics.items():
            self.expert_statistics[name] = self.expert_statistics.get(name, 0) + delta
        if result.discard_reason is not None:
            self.discarded_episodes += 1
            print(
                f"Discarding DAgger episode ({result.discard_reason}); "
                "retrying with a new initial state and goal."
            )
            return
        self.total_executed_steps += result.total_executed_steps
        self.expert_executed_steps += result.expert_executed_steps
        if not result.frames:
            raise RuntimeError("DAgger produced an empty frame buffer for a non-discarded episode.")
        for frame_data in result.frames:
            self.dataset_writer.add_frame(frame_data)
        self.dataset_writer.save_episode()
        self.successful_episodes += 1
        self.reached_goal_count += int(result.reached_goal)
        self.steps_taken.append(int(result.rollout_steps))
        if self.successful_episodes % 10 == 0:
            print(f"Collected {self.successful_episodes}/{self.trajectories} trajectories")

    def metrics(self) -> DaggerEvalMetrics:
        return DaggerEvalMetrics(
            success_rate=float(self.reached_goal_count) / float(self.successful_episodes),
            mean_steps=float(np.mean(np.asarray(self.steps_taken, dtype=float))),
            min_steps=min(self.steps_taken),
            max_steps=max(self.steps_taken),
            num_episodes=self.successful_episodes,
            attempted_episodes=self.attempted_episodes,
            discarded_episodes=self.discarded_episodes,
            planner_failures=self.planner_failures,
            policy_solve_fallbacks=self.policy_solve_fallbacks,
            expert_statistics=self.expert_statistics,
        )


def collect_dagger_rollouts(
    simulator: SystemSimulator,
    expert_planner: Planner,
    dataset_writer,
    trajectories_per_iteration: int,
    steps_per_trajectory: int,
    action_noise_std: float,
    action_noise_seed: int,
    initial_state_seed: int,
    expert_mixing_beta: float,
    prediction_horizon: int,
    policy_action_fn: Callable[[np.ndarray], np.ndarray] | None,
    policy_reset_fn: Callable[[], None] | None = None,
    frame_builder: Callable[[np.ndarray, np.ndarray], Mapping[str, object]] | None = None,
    initial_states: list[np.ndarray] | None = None,
    goal_states: list[np.ndarray] | None = None,
    round_index: int = 0,
    restart_initial_state_round: bool = False,
    num_workers: int = 1,
    worker_spec: RolloutWorkerSpec | None = None,
) -> DaggerEvalMetrics:
    """Collect DAgger trajectories with expert relabeling and mixed execution.

    Every visited state is labeled with the expert's plan from that state (its next ``prediction_horizon`` planned
    actions, the length of the policy's chunk), whoever executes the step. An episode whose expert planner fails to
    solve is dropped as a whole and retried from a fresh initial state and goal.
    """
    # Scoped to initial-state/goal sampling only: action-noise and expert-mixing
    # randomness always keep varying by round_index, regardless of this flag.
    initial_state_round_index = None if restart_initial_state_round else round_index
    if policy_action_fn is None and expert_mixing_beta < 1.0:
        raise ValueError("'policy_action_fn' is required when 'expert_mixing_beta' is less than 1.0.")
    settings = CollectionSettings(
        steps_per_trajectory=steps_per_trajectory,
        action_noise_std=action_noise_std,
        action_noise_seed=action_noise_seed,
        initial_state_seed=initial_state_seed,
        initial_state_round_index=initial_state_round_index,
        expert_mixing_beta=expert_mixing_beta,
        prediction_horizon=prediction_horizon,
        round_index=round_index,
        starts=EpisodeStarts.from_specs(simulator, initial_states, goal_states),
        should_query_policy=policy_action_fn is not None and expert_mixing_beta < 1.0,
    )
    tally = _CollectionTally(expert_planner, trajectories_per_iteration, dataset_writer)
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)

    def too_many_failed_attempts() -> RuntimeError:
        return RuntimeError(
            "Too many failed DAgger rollout attempts. "
            f"Collected {tally.successful_episodes}/{trajectories_per_iteration} episodes."
        )

    if num_workers > 1 and worker_spec is not None and trajectories_per_iteration > 1:
        pool_size = min(num_workers, trajectories_per_iteration)
        with multiprocessing.get_context("spawn").Pool(
            pool_size, initializer=_init_rollout_worker, initargs=(worker_spec, settings)
        ) as pool:
            while tally.successful_episodes < trajectories_per_iteration:
                needed = trajectories_per_iteration - tally.successful_episodes
                batch = list(range(tally.attempted_episodes + 1, min(tally.attempted_episodes + needed, max_attempts) + 1))
                if not batch:
                    raise too_many_failed_attempts()
                for result in pool.map(_run_worker_task, batch):
                    tally.record(result)
    else:
        collector = RolloutCollector(
            simulator,
            expert_planner,
            settings,
            policy_action_fn=policy_action_fn,
            policy_reset_fn=policy_reset_fn,
            frame_builder=frame_builder,
        )
        while tally.successful_episodes < trajectories_per_iteration:
            if tally.attempted_episodes + 1 > max_attempts:
                raise too_many_failed_attempts()
            tally.record(collector.run_attempt(tally.attempted_episodes + 1))

    if tally.total_executed_steps > 0:
        print(
            "Execution mixing stats: "
            f"requested_beta={expert_mixing_beta:.3f}, "
            f"realized_expert_fraction={float(tally.expert_executed_steps) / float(tally.total_executed_steps):.3f}, "
            f"executed_steps={tally.total_executed_steps}"
        )
    return tally.metrics()
