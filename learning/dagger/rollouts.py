from __future__ import annotations

import contextlib
import multiprocessing
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import torch

from systems.rollout_termination import RolloutTerminationTracker
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from systems.initial_state_utils import normalize_goal_state_specs, normalize_initial_state_specs
from systems.seed_utils import (
    action_noise_rng_for_rollout,
    action_noise_seed_for_rollout,
    expert_mixing_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)
from learning.data_utils import format_dataset_frame
from learning.models.policy_factory import PolicyFactory, load_policy_state_dict
from planning.ompl_planner import initialize_ompl
from planning.planner import Planner, PlannerSolveError
from planning.planner_factory import PlannerFactory

from .metrics import DaggerEvalMetrics
from .utils import evaluation_seed_specs, rng_for_seed_spec, sample_initial_state


class RolloutWorkerSpec(NamedTuple):
    """Picklable specification for constructing a rollout worker's simulator, expert planner, and CPU policy."""

    config: dict[str, Any]
    policy_kwargs: dict[str, Any] | None = None
    policy_state_dict: dict[str, torch.Tensor] | None = None
    need_expert: bool = True


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


class _EvalEpisodeResult(NamedTuple):
    reached_goal: bool
    rollout_steps: int
    failure_reason: str | None
    is_config_sourced: bool
    inference_seconds: float  # the wall-clock time of the policy calls of the episode that returned an action
    inference_calls: int


def build_policy_action(
    simulator: SystemSimulator,
    policy,
    observation: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Query the policy once and return the action to execute: the first row of the predicted chunk.

    The policy predicts a chunk of actions (batch, horizon, nu), row k the action at step k.
    """
    policy_observation = simulator.policy_observation(observation)
    policy_input = {
        name: torch.as_tensor(
            np.expand_dims(policy_observation[name], axis=0),
            dtype=torch.float32,
        ).to(device)
        for name in policy_observation
    }

    with torch.inference_mode():
        chunk = policy.select_action(policy_input)

    if chunk.ndim != 3 or chunk.shape[1] < 1:
        raise ValueError("Policy must return an action chunk shaped (batch, horizon >= 1, nu).")
    return chunk[0, 0, :].detach().cpu().numpy().astype(float)


def apply_execution_noise(
    simulator: SystemSimulator,
    action: np.ndarray,
    action_noise_std: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if action_noise_std <= 0.0:
        return action
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, action_noise_std, size=action.shape).astype(action.dtype, copy=False)
    return np.clip(action + noise, -simulator.max_action, simulator.max_action)


def _setup_episode_start(
    simulator: SystemSimulator,
    index: int,
    usable_count: int,
    provided_initial_states: list[np.ndarray],
    provided_goal_states: list[np.ndarray],
    baseline_goal: np.ndarray,
    seed_spec: int,
) -> np.ndarray:
    """Set the goal of episode ``index`` (counted from 0) and return its initial state.

    The first ``usable_count`` episodes take their initial state from ``provided_initial_states`` and their goal from
    ``provided_goal_states``; with no goals provided the goal is drawn as ``randomize_goal_for_reset`` does, seeded with
    ``seed_spec``, so these episodes still vary per the config instead of freezing at the baseline goal. The others
    restore ``baseline_goal`` first (an earlier episode's explicit goal mutated the simulator, and
    ``randomize_goal_for_reset`` is a no-op under ``randomize_goal: false``, which would leak that goal into this
    episode) and sample a start (and goal) from ``seed_spec``.
    """
    if index < usable_count:
        if provided_goal_states:
            simulator.set_goal(provided_goal_states[index])
        else:
            simulator.randomize_goal_for_reset(rng_for_seed_spec(seed_spec))
        return provided_initial_states[index]
    simulator.set_goal(baseline_goal)
    return sample_initial_state(simulator, seed_spec)


@dataclass(frozen=True)
class EpisodeStarts:
    """Where the episodes of a collection or an evaluation start: the first ones from the initial states (and goals)
    that were provided, the others sampled (``_setup_episode_start``)."""

    initial_states: list[np.ndarray]
    goal_states: list[np.ndarray]
    baseline_goal: np.ndarray

    @classmethod
    def from_specs(
        cls,
        simulator: SystemSimulator,
        initial_states: list[np.ndarray] | None,
        goal_states: list[np.ndarray] | None,
    ) -> EpisodeStarts:
        return cls(
            normalize_initial_state_specs(simulator, initial_states),
            normalize_goal_state_specs(simulator, goal_states),
            simulator.goal.copy(),
        )

    @property
    def usable_count(self) -> int:
        return min(len(self.initial_states), len(self.goal_states)) if self.goal_states else len(self.initial_states)

    def is_provided(self, index: int) -> bool:
        return index < self.usable_count

    def begin(self, simulator: SystemSimulator, index: int, seed_spec: int) -> np.ndarray:
        """Set the goal of episode ``index`` (counted from 0) and return its initial state."""
        return _setup_episode_start(
            simulator,
            index,
            self.usable_count,
            self.initial_states,
            self.goal_states,
            self.baseline_goal,
            seed_spec,
        )


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


@dataclass(frozen=True)
class EvaluationSettings:
    num_steps: int
    action_noise_std: float
    action_noise_seed: int
    starts: EpisodeStarts


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
        episode.frames.append({"task": getattr(simulator, "task_description", "reach target"), **built_frame})
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


class RolloutWorker:
    """The state of one process of a rollout pool: the simulator, the expert planner and the CPU policy built from the
    spec, and the runner of its task (a ``RolloutCollector`` for ``CollectionSettings``, else a ``RolloutEvaluator``).

    Called with a task: an attempt index when collecting, an ``(episode index, seed spec)`` pair when evaluating.
    """

    def __init__(self, spec: RolloutWorkerSpec, settings: CollectionSettings | EvaluationSettings) -> None:
        torch.set_num_threads(1)
        initialize_ompl(spec.config["ompl_seed"])  # a spawned process has its own OMPL globals
        self.settings = settings
        self.simulator = SystemFactory.create(spec.config["system"], spec.config)
        planner = PlannerFactory.create(spec.config["expert"], self.simulator, spec.config) if spec.need_expert else None
        self.policy = self._build_policy(spec)
        policy_action_fn = None
        if self.policy is not None:
            policy_action_fn = lambda obs: build_policy_action(self.simulator, self.policy, obs, torch.device("cpu"))
        if isinstance(settings, CollectionSettings):
            queries_policy = settings.should_query_policy and self.policy is not None
            self._runner = RolloutCollector(
                self.simulator,
                planner,
                settings,
                policy_action_fn=policy_action_fn if queries_policy else None,
                policy_reset_fn=self.policy.reset if queries_policy else None,
            )
        else:
            self._runner = RolloutEvaluator(self.simulator, settings, policy_action_fn, self.policy.reset)

    def _build_policy(self, spec: RolloutWorkerSpec):
        if spec.policy_kwargs is None:
            return None
        kwargs = dict(spec.policy_kwargs)
        policy_type = kwargs.pop("policy_type")
        kwargs["simulator"] = self.simulator
        if policy_type == "safeflow":
            kwargs["planner_config"] = spec.config
        policy = PolicyFactory.create(policy_type, **kwargs).to(torch.device("cpu")).eval()
        if spec.policy_state_dict is not None:
            load_policy_state_dict(policy, spec.policy_state_dict)
        return policy

    def __call__(self, task: int | tuple[int, int]) -> _CollectionAttemptResult | _EvalEpisodeResult:
        if isinstance(self.settings, EvaluationSettings):
            return self._runner.run_episode(*task)
        if self.settings.should_query_policy and self.policy is not None:
            torch.manual_seed(
                torch_inference_seed_for_rollout(
                    self.settings.action_noise_seed,
                    rollout_index=task,
                    round_index=self.settings.round_index,
                )
            )
        return self._runner.run_attempt(task)


_worker: RolloutWorker | None = None  # the one of this process of a pool


def _init_rollout_worker(spec: RolloutWorkerSpec, settings: CollectionSettings | EvaluationSettings) -> None:
    global _worker
    _worker = RolloutWorker(spec, settings)


def _run_worker_task(task: int | tuple[int, int]) -> _CollectionAttemptResult | _EvalEpisodeResult:
    assert _worker is not None
    return _worker(task)


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
