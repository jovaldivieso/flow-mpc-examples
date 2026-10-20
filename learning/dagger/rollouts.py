from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping

import numpy as np
import torch

from planning.planner import Planner, PlannerSolveError
from dynamics.dynamics import DynamicsSimulator
from dynamics.seed_utils import (
    action_noise_rng_for_rollout,
    action_noise_seed_for_rollout,
    expert_mixing_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)

from .metrics import DaggerEvalMetrics
from .utils import evaluation_seed_specs, rng_for_seed_spec, sample_initial_state
from dynamics.initial_state_utils import normalize_goal_state_specs, normalize_initial_state_specs


def build_policy_action(
    simulator: DynamicsSimulator,
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
    simulator: DynamicsSimulator,
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


def collect_dagger_rollouts(
    simulator: DynamicsSimulator,
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
    statistics_before = expert_planner.statistics

    if frame_builder is None:
        def frame_builder(
            observation: np.ndarray,
            expert_plan: np.ndarray,
        ) -> dict[str, object]:
            frame = simulator.format_dataset_frame(observation, expert_plan)
            frame["task"] = getattr(simulator, "task_description", "reach target")
            return frame

    def ensure_task_field(frame: dict[str, object]) -> dict[str, object]:
        if "task" not in frame:
            frame = dict(frame)
            frame["task"] = getattr(simulator, "task_description", "reach target")
        return frame

    provided_initial_states = normalize_initial_state_specs(simulator, initial_states)
    provided_goal_states = normalize_goal_state_specs(simulator, goal_states)
    have_goal_states = bool(goal_states)
    usable_count = (
        min(len(provided_initial_states), len(provided_goal_states))
        if have_goal_states
        else len(provided_initial_states)
    )

    baseline_goal = simulator.goal.copy()
    while successful_episodes < trajectories_per_iteration:
        attempted_episodes += 1
        if attempted_episodes > max_attempts:
            raise RuntimeError(
                "Too many failed DAgger rollout attempts. "
                f"Collected {successful_episodes}/{trajectories_per_iteration} episodes."
            )
        if attempted_episodes <= usable_count:
            sampled_initial_state = provided_initial_states[attempted_episodes - 1]
            if have_goal_states:
                simulator.set_goal(provided_goal_states[attempted_episodes - 1])
            else:
                # An explicit initial state was provided without a paired goal:
                # still honor randomize_goal_for_reset so these episodes vary
                # per the collection config, instead of silently freezing at
                # the baseline goal for as many initial states as were given.
                episode_goal_seed = initial_state_seed_for_rollout(
                    initial_state_seed,
                    rollout_index=attempted_episodes,
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
                rollout_index=attempted_episodes,
                round_index=initial_state_round_index,
            )
            sampled_initial_state = sample_initial_state(simulator, episode_initial_state_seed)
        state = simulator.reset(sampled_initial_state)
        episode_initial_state = state.copy()
        episode_noise_seed = action_noise_seed_for_rollout(
            action_noise_seed,
            rollout_index=attempted_episodes,
            round_index=round_index,
        )
        episode_action_noise_rng = np.random.default_rng(episode_noise_seed)
        episode_expert_mixing_seed = expert_mixing_seed_for_rollout(
            action_noise_seed,
            rollout_index=attempted_episodes,
            round_index=round_index,
        )
        episode_expert_mixing_rng = np.random.default_rng(episode_expert_mixing_seed)
        discard_reason: str | None = None
        if policy_reset_fn is not None:
            policy_reset_fn()
        expert_planner.reset()
        reached_goal = False
        rollout_steps = steps_per_trajectory
        episode_frame_buffer: list[dict[str, object]] = []
        episode_actor_is_expert: list[bool] = []

        for step in range(1, steps_per_trajectory + 1):
            observation = simulator.observe(state)
            try:
                expert_plan = expert_planner.plan(observation)
            except PlannerSolveError as exc:
                print(
                    f"Planner failed to solve (attempt={attempted_episodes}, step={step}, "
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
            if should_query_policy:
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
                        f"expert action (attempt={attempted_episodes}, step={step}): {exc}"
                    )
                    policy_solve_fallbacks += 1
                    use_expert_action = True
            built_frame = frame_builder(observation, expert_plan.first_actions(prediction_horizon))
            if not isinstance(built_frame, Mapping):
                raise TypeError("'frame_builder' must return a mapping, got " f"{type(built_frame).__name__}.")
            episode_frame_buffer.append(ensure_task_field(dict(built_frame)))
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

        if discard_reason is not None:
            discarded_episodes += 1
            # The whole episode is dropped (nothing has been written to the dataset yet) and the
            # next attempt starts from a fresh initial state and goal.
            print(
                f"Discarding DAgger episode ({discard_reason}); "
                "retrying with a new initial state and goal."
            )
            continue
        total_executed_steps += len(episode_actor_is_expert)
        expert_executed_steps += sum(episode_actor_is_expert)
        if not episode_frame_buffer:
            raise RuntimeError(
                "DAgger produced an empty frame buffer for a non-discarded episode."
            )

        for frame_data in episode_frame_buffer:
            dataset_writer.add_frame(frame_data)

        dataset_writer.save_episode()

        successful_episodes += 1
        reached_goal_count += int(reached_goal)
        steps_taken.append(int(rollout_steps))
        if successful_episodes % 10 == 0:
            print(f"Collected {successful_episodes}/{trajectories_per_iteration} trajectories")

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
        expert_statistics={
            name: count - statistics_before.get(name, 0) for name, count in expert_planner.statistics.items()
        },
    )


def rollout_policy_with_action_fn(
    simulator: DynamicsSimulator,
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


def evaluate_policy_rollouts(
    simulator: DynamicsSimulator,
    num_episodes: int,
    num_steps: int,
    seed_start: int,
    action_fn: Callable[[np.ndarray], np.ndarray],
    reset_fn: Callable[[], None] | None = None,
    action_noise_std: float = 0.0,
    action_noise_seed: int = 0,
    initial_states: list[np.ndarray] | None = None,
    goal_states: list[np.ndarray] | None = None,
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
    successes = 0
    steps_taken: list[int] = []
    # Split the same running totals by source so a config-curriculum episode's
    # (likely memorized/specialized) outcome is never averaged together with
    # a random-sampled (generalization) one without also being visible on its
    # own -- see DaggerEvalMetrics for why this distinction is worth keeping.
    config_successes = 0
    config_episodes = 0
    random_successes = 0
    random_episodes = 0
    timeout_failures = 0
    solve_failures = 0
    baseline_goal = simulator.goal.copy()
    for episode_idx, seed_spec in enumerate(seed_specs):
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
        with _isolated_torch_rng_state():
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
        successes += int(reached_goal)
        steps_taken.append(int(rollout_steps))
        if failure_reason == "timeout":
            timeout_failures += 1
        elif failure_reason == "solve_failure":
            solve_failures += 1
        if is_config_sourced:
            config_successes += int(reached_goal)
            config_episodes += 1
        else:
            random_successes += int(reached_goal)
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
