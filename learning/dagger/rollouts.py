from __future__ import annotations

import contextlib
import multiprocessing
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch

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
from learning.models.policy_factory import PolicyFactory
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


_worker_simulator: SystemSimulator | None = None
_worker_planner: Planner | None = None
_worker_policy: Any = None
_worker_context: dict[str, Any] = {}


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


def _default_frame_builder(simulator: SystemSimulator, observation: np.ndarray, expert_plan: np.ndarray) -> dict[str, object]:
    frame = simulator.format_dataset_frame(observation, expert_plan)
    frame["task"] = getattr(simulator, "task_description", "reach target")
    return frame


def _ensure_task_field(simulator: SystemSimulator, frame: dict[str, object]) -> dict[str, object]:
    if "task" not in frame:
        frame = dict(frame)
        frame["task"] = getattr(simulator, "task_description", "reach target")
    return frame


def _init_rollout_worker(spec: RolloutWorkerSpec, context: dict[str, Any]) -> None:
    global _worker_simulator, _worker_planner, _worker_policy, _worker_context
    torch.set_num_threads(1)
    _worker_simulator = SystemFactory.create(spec.config["system"], spec.config)
    _worker_planner = PlannerFactory.create(spec.config["expert"], _worker_simulator, spec.config) if spec.need_expert else None
    if spec.policy_kwargs is not None:
        kwargs = dict(spec.policy_kwargs)
        policy_type = kwargs.pop("policy_type")
        if policy_type == "safeflow":
            kwargs["simulator"] = _worker_simulator
            kwargs["planner_config"] = spec.config
        _worker_policy = PolicyFactory.create(policy_type, **kwargs).to(torch.device("cpu")).eval()
        if spec.policy_state_dict is not None:
            _worker_policy.load_state_dict(spec.policy_state_dict)
    else:
        _worker_policy = None
    _worker_context = dict(context)


def _run_collection_attempt(
    attempt_index: int,
    *,
    simulator: SystemSimulator,
    expert_planner: Planner,
    steps_per_trajectory: int,
    action_noise_std: float,
    action_noise_seed: int,
    initial_state_seed: int,
    initial_state_round_index: int | None,
    expert_mixing_beta: float,
    prediction_horizon: int,
    round_index: int,
    provided_initial_states: list[np.ndarray],
    provided_goal_states: list[np.ndarray],
    have_goal_states: bool,
    usable_count: int,
    baseline_goal: np.ndarray,
    should_query_policy: bool,
    policy_action_fn: Callable[[np.ndarray], np.ndarray] | None,
    policy_reset_fn: Callable[[], None] | None,
    frame_builder: Callable[[np.ndarray, np.ndarray], Mapping[str, object]],
) -> _CollectionAttemptResult:
    stats_before = dict(expert_planner.statistics)
    if attempt_index <= usable_count:
        sampled_initial_state = provided_initial_states[attempt_index - 1]
        if have_goal_states:
            simulator.set_goal(provided_goal_states[attempt_index - 1])
        else:
            # An explicit initial state was provided without a paired goal:
            # still honor randomize_goal_for_reset so these episodes vary
            # per the collection config, instead of silently freezing at
            # the baseline goal for as many initial states as were given.
            episode_goal_seed = initial_state_seed_for_rollout(
                initial_state_seed,
                rollout_index=attempt_index,
                round_index=initial_state_round_index,
            )
            simulator.randomize_goal_for_reset(rng_for_seed_spec(episode_goal_seed))
    else:
        # A prior episode's explicit goal mutates the simulator; restore the
        # config's baseline goal before any fallback sampling, since
        # randomize_goal_for_reset() is a no-op under `randomize_goal: false`
        # and would otherwise silently leak that leftover explicit goal into
        # this episode instead of the configured/default one.
        simulator.set_goal(baseline_goal)
        episode_initial_state_seed = initial_state_seed_for_rollout(
            initial_state_seed,
            rollout_index=attempt_index,
            round_index=initial_state_round_index,
        )
        sampled_initial_state = sample_initial_state(simulator, episode_initial_state_seed)

    state = simulator.reset(sampled_initial_state)
    episode_initial_state = state.copy()
    episode_noise_seed = action_noise_seed_for_rollout(
        action_noise_seed,
        rollout_index=attempt_index,
        round_index=round_index,
    )
    episode_action_noise_rng = np.random.default_rng(episode_noise_seed)
    episode_expert_mixing_seed = expert_mixing_seed_for_rollout(
        action_noise_seed,
        rollout_index=attempt_index,
        round_index=round_index,
    )
    episode_expert_mixing_rng = np.random.default_rng(episode_expert_mixing_seed)
    if policy_reset_fn is not None:
        policy_reset_fn()
    expert_planner.reset()
    reached_goal = False
    rollout_steps = steps_per_trajectory
    episode_frame_buffer: list[dict[str, object]] = []
    episode_actor_is_expert: list[bool] = []
    planner_failures = policy_solve_fallbacks = 0
    discard_reason: str | None = None

    for step in range(1, steps_per_trajectory + 1):
        observation = simulator.observe(state)
        try:
            expert_plan = expert_planner.plan(observation)
        except PlannerSolveError as exc:
            print(
                f"Planner failed to solve (attempt={attempt_index}, step={step}, "
                f"action_noise_std={action_noise_std:.6f}, noise_seed={episode_noise_seed}): {exc} "
                f"initial_state={np.array2string(np.asarray(episode_initial_state), precision=6)}, "
                f"current_state={np.array2string(np.asarray(state), precision=6)}, "
                f"goal={np.array2string(np.asarray(simulator.goal), precision=6)}"
            )
            planner_failures += 1
            discard_reason = "the expert planner failed to solve"
            break

        expert_action = expert_plan.actions[0]
        use_expert_action = bool(episode_expert_mixing_rng.random() < expert_mixing_beta)
        policy_action = None
        if should_query_policy and policy_action_fn is not None:
            try:
                policy_action = policy_action_fn(observation)
            except PlannerSolveError as exc:
                # Unlike an expert-planner failure above, the episode survives this:
                # expert_action for this exact step is already computed,
                # so falling back to it (below, via policy_action=None) keeps the episode
                # going instead of discarding it. Recorded as an expert step since that's
                # what actually gets executed.
                print(
                    "Policy action unavailable this step, falling back to the "
                    f"expert action (attempt={attempt_index}, step={step}): {exc}"
                )
                policy_solve_fallbacks += 1
                use_expert_action = True
        built_frame = frame_builder(observation, expert_plan.first_actions(prediction_horizon))
        if not isinstance(built_frame, Mapping):
            raise TypeError("'frame_builder' must return a mapping, got " f"{type(built_frame).__name__}.")
        episode_frame_buffer.append(_ensure_task_field(simulator, dict(built_frame)))
        episode_actor_is_expert.append(use_expert_action)
        base_action = expert_action if use_expert_action or policy_action is None else policy_action
        state = simulator.step(
            state,
            apply_execution_noise(simulator, base_action, action_noise_std, episode_action_noise_rng),
        )
        if simulator.should_terminate_rollout(state):
            reached_goal = True
            rollout_steps = step
            break

    stats_delta = {name: count - stats_before.get(name, 0) for name, count in expert_planner.statistics.items()}
    return _CollectionAttemptResult(
        frames=None if discard_reason is not None else episode_frame_buffer,
        discard_reason=discard_reason,
        reached_goal=reached_goal,
        rollout_steps=rollout_steps,
        expert_executed_steps=sum(episode_actor_is_expert),
        total_executed_steps=len(episode_actor_is_expert),
        planner_failures=planner_failures,
        policy_solve_fallbacks=policy_solve_fallbacks,
        expert_statistics=stats_delta,
    )


def _worker_collect_attempt(attempt_index: int) -> _CollectionAttemptResult:
    assert _worker_simulator is not None and _worker_planner is not None
    ctx = _worker_context
    policy_action_fn = None
    policy_reset_fn = None
    if ctx["should_query_policy"] and _worker_policy is not None:
        torch.manual_seed(
            torch_inference_seed_for_rollout(
                ctx["action_noise_seed"],
                rollout_index=attempt_index,
                round_index=ctx["round_index"],
            )
        )
        policy_action_fn = lambda obs: build_policy_action(_worker_simulator, _worker_policy, obs, torch.device("cpu"))
        policy_reset_fn = _worker_policy.reset
    return _run_collection_attempt(
        attempt_index,
        simulator=_worker_simulator,
        expert_planner=_worker_planner,
        policy_action_fn=policy_action_fn,
        policy_reset_fn=policy_reset_fn,
        frame_builder=lambda obs, plan: _default_frame_builder(_worker_simulator, obs, plan),
        **ctx,
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
    should_query_policy = policy_action_fn is not None and expert_mixing_beta < 1.0
    successful_episodes = attempted_episodes = reached_goal_count = 0
    max_attempts = max(trajectories_per_iteration * 3, trajectories_per_iteration)
    steps_taken: list[int] = []
    expert_executed_steps = total_executed_steps = 0
    planner_failures = discarded_episodes = policy_solve_fallbacks = 0
    accumulated_expert_stats: dict[str, int] = {name: 0 for name in expert_planner.statistics}

    active_frame_builder = (
        frame_builder
        if frame_builder is not None
        else (lambda obs, plan: _default_frame_builder(simulator, obs, plan))
    )

    provided_initial_states = normalize_initial_state_specs(simulator, initial_states)
    provided_goal_states = normalize_goal_state_specs(simulator, goal_states)
    have_goal_states = bool(goal_states)
    usable_count = (
        min(len(provided_initial_states), len(provided_goal_states))
        if have_goal_states
        else len(provided_initial_states)
    )

    baseline_goal = simulator.goal.copy()
    attempt_context = {
        "steps_per_trajectory": steps_per_trajectory,
        "action_noise_std": action_noise_std,
        "action_noise_seed": action_noise_seed,
        "initial_state_seed": initial_state_seed,
        "initial_state_round_index": initial_state_round_index,
        "expert_mixing_beta": expert_mixing_beta,
        "prediction_horizon": prediction_horizon,
        "round_index": round_index,
        "provided_initial_states": provided_initial_states,
        "provided_goal_states": provided_goal_states,
        "have_goal_states": have_goal_states,
        "usable_count": usable_count,
        "baseline_goal": baseline_goal,
        "should_query_policy": should_query_policy,
    }

    def record_attempt(result: _CollectionAttemptResult) -> None:
        nonlocal successful_episodes, reached_goal_count, expert_executed_steps, total_executed_steps
        nonlocal planner_failures, discarded_episodes, policy_solve_fallbacks
        planner_failures += result.planner_failures
        policy_solve_fallbacks += result.policy_solve_fallbacks
        for name, delta in result.expert_statistics.items():
            accumulated_expert_stats[name] = accumulated_expert_stats.get(name, 0) + delta
        if result.discard_reason is not None:
            discarded_episodes += 1
            print(
                f"Discarding DAgger episode ({result.discard_reason}); "
                "retrying with a new initial state and goal."
            )
            return
        total_executed_steps += result.total_executed_steps
        expert_executed_steps += result.expert_executed_steps
        if not result.frames:
            raise RuntimeError("DAgger produced an empty frame buffer for a non-discarded episode.")
        for frame_data in result.frames:
            dataset_writer.add_frame(frame_data)
        dataset_writer.save_episode()
        successful_episodes += 1
        reached_goal_count += int(result.reached_goal)
        steps_taken.append(int(result.rollout_steps))
        if successful_episodes % 10 == 0:
            print(f"Collected {successful_episodes}/{trajectories_per_iteration} trajectories")

    if num_workers > 1 and worker_spec is not None and trajectories_per_iteration > 1:
        pool_size = min(num_workers, trajectories_per_iteration)
        with multiprocessing.get_context("spawn").Pool(
            pool_size, initializer=_init_rollout_worker, initargs=(worker_spec, attempt_context)
        ) as pool:
            while successful_episodes < trajectories_per_iteration:
                needed = trajectories_per_iteration - successful_episodes
                batch = list(range(attempted_episodes + 1, min(attempted_episodes + needed, max_attempts) + 1))
                if not batch:
                    raise RuntimeError(
                        "Too many failed DAgger rollout attempts. "
                        f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
                    )
                for result in pool.map(_worker_collect_attempt, batch):
                    attempted_episodes += 1
                    record_attempt(result)
    else:
        while successful_episodes < trajectories_per_iteration:
            attempted_episodes += 1
            if attempted_episodes > max_attempts:
                raise RuntimeError(
                    "Too many failed DAgger rollout attempts. "
                    f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
                )
            result = _run_collection_attempt(
                attempted_episodes,
                simulator=simulator,
                expert_planner=expert_planner,
                policy_action_fn=policy_action_fn,
                policy_reset_fn=policy_reset_fn,
                frame_builder=active_frame_builder,
                **attempt_context,
            )
            record_attempt(result)

    if total_executed_steps > 0:
        print(
            "Execution mixing stats: "
            f"requested_beta={expert_mixing_beta:.3f}, "
            f"realized_expert_fraction={float(expert_executed_steps) / float(total_executed_steps):.3f}, "
            f"executed_steps={total_executed_steps}"
        )
    return DaggerEvalMetrics(
        success_rate=float(reached_goal_count) / float(successful_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=successful_episodes,
        attempted_episodes=attempted_episodes,
        discarded_episodes=discarded_episodes,
        planner_failures=planner_failures,
        policy_solve_fallbacks=policy_solve_fallbacks,
        expert_statistics=accumulated_expert_stats,
    )


def rollout_policy_with_action_fn(
    simulator: SystemSimulator,
    initial_state: np.ndarray,
    num_steps: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_rng: np.random.Generator | None = None,
) -> tuple[bool, int, str | None]:
    """Returns (reached_goal, steps_taken, failure_reason).

    failure_reason is None on success, else one of "timeout" (ran out of steps
    without reaching tolerance) or "solve_failure" (action_fn itself raised)
    -- see DaggerEvalMetrics for why these are tracked separately rather than
    folded into one failure count.
    """
    state = simulator.reset(initial_state)
    if reset_fn is not None:
        reset_fn()
    if simulator.should_terminate_rollout(state):
        return True, 0, None
    for step in range(1, num_steps + 1):
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
            print(f"Policy solve failed during evaluation at step={step}: {exc}")
            return False, step - 1, "solve_failure"
        state = simulator.step(state, apply_execution_noise(simulator, action, action_noise_std, action_noise_rng))
        if simulator.should_terminate_rollout(state):
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


def _run_eval_episode(
    episode_idx: int,
    seed_spec: int,
    *,
    simulator: SystemSimulator,
    num_steps: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None,
    action_noise_std: float,
    action_noise_seed: int,
    provided_initial_states: list[np.ndarray],
    provided_goal_states: list[np.ndarray],
    have_goal_states: bool,
    usable_count: int,
    baseline_goal: np.ndarray,
) -> _EvalEpisodeResult:
    is_config_sourced = episode_idx < usable_count
    if is_config_sourced:
        initial_state = provided_initial_states[episode_idx]
        if have_goal_states:
            simulator.set_goal(provided_goal_states[episode_idx])
        else:
            # An explicit initial state was provided without a paired goal:
            # still honor randomize_goal_for_reset, matching the fallback
            # branch's own use of sample_initial_state below, instead of
            # silently freezing at the baseline goal.
            simulator.randomize_goal_for_reset(rng_for_seed_spec(seed_spec))
    else:
        simulator.set_goal(baseline_goal)
        initial_state = sample_initial_state(simulator, seed_spec)
    torch.manual_seed(torch_inference_seed_for_rollout(action_noise_seed, seed_spec=seed_spec))
    reached_goal, rollout_steps, failure_reason = rollout_policy_with_action_fn(
        simulator=simulator,
        initial_state=initial_state,
        num_steps=num_steps,
        action_fn=action_fn,
        reset_fn=reset_fn,
        action_noise_std=action_noise_std,
        action_noise_rng=action_noise_rng_for_rollout(action_noise_seed, seed_spec=seed_spec),
    )
    return _EvalEpisodeResult(reached_goal, int(rollout_steps), failure_reason, is_config_sourced)


def _worker_eval_episode(task: tuple[int, int]) -> _EvalEpisodeResult:
    episode_idx, seed_spec = task
    assert _worker_simulator is not None and _worker_policy is not None
    return _run_eval_episode(
        episode_idx,
        seed_spec,
        simulator=_worker_simulator,
        action_fn=lambda obs: build_policy_action(_worker_simulator, _worker_policy, obs, torch.device("cpu")),
        reset_fn=_worker_policy.reset,
        **_worker_context,
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
    provided_initial_states = normalize_initial_state_specs(simulator, initial_states)
    provided_goal_states = normalize_goal_state_specs(simulator, goal_states)
    have_goal_states = bool(goal_states)
    usable_count = (
        min(len(provided_initial_states), len(provided_goal_states))
        if have_goal_states
        else len(provided_initial_states)
    )
    eval_context = {
        "num_steps": num_steps,
        "action_noise_std": action_noise_std,
        "action_noise_seed": action_noise_seed,
        "provided_initial_states": provided_initial_states,
        "provided_goal_states": provided_goal_states,
        "have_goal_states": have_goal_states,
        "usable_count": usable_count,
        "baseline_goal": simulator.goal.copy(),
    }
    if num_workers > 1 and worker_spec is not None and num_episodes > 1:
        with multiprocessing.get_context("spawn").Pool(
            min(num_workers, num_episodes), initializer=_init_rollout_worker, initargs=(worker_spec, eval_context)
        ) as pool:
            results = pool.map(_worker_eval_episode, list(enumerate(seed_specs)))
    else:
        results = []
        for episode_idx, seed_spec in enumerate(seed_specs):
            with _isolated_torch_rng_state():
                results.append(
                    _run_eval_episode(
                        episode_idx,
                        seed_spec,
                        simulator=simulator,
                        action_fn=action_fn,
                        reset_fn=reset_fn,
                        **eval_context,
                    )
                )

    successes = config_successes = config_episodes = random_successes = random_episodes = 0
    timeout_failures = solve_failures = 0
    steps_taken: list[int] = []
    # Split the same running totals by source so a config-curriculum episode's
    # (likely memorized/specialized) outcome is never averaged together with
    # a random-sampled (generalization) one without also being visible on its
    # own -- see DaggerEvalMetrics for why this distinction is worth keeping.
    for res in results:
        successes += int(res.reached_goal)
        steps_taken.append(res.rollout_steps)
        if res.failure_reason == "timeout":
            timeout_failures += 1
        elif res.failure_reason == "solve_failure":
            solve_failures += 1
        if res.is_config_sourced:
            config_successes += int(res.reached_goal)
            config_episodes += 1
        else:
            random_successes += int(res.reached_goal)
            random_episodes += 1
    return DaggerEvalMetrics(
        success_rate=float(successes) / float(num_episodes),
        mean_steps=float(np.mean(np.asarray(steps_taken, dtype=float))),
        min_steps=min(steps_taken),
        max_steps=max(steps_taken),
        num_episodes=num_episodes,
        config_successes=config_successes,
        config_num_episodes=config_episodes,
        random_successes=random_successes,
        random_num_episodes=random_episodes,
        timeout_failures=timeout_failures,
        solve_failures=solve_failures,
    )
