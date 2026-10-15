from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from dynamics.franka_dynamics import Q_READY, FrankaDynamics
from learning.dagger.rollouts import (
    apply_execution_noise,
    build_policy_action,
    collect_dagger_rollouts,
    evaluate_policy_rollouts,
)
from learning.models.flow_policy import FlowPolicy
from planning.casadi_planner import CasadiPlanner
from planning.planner import Plan, Planner, PlannerSolveError
from dynamics.seed_utils import (
    action_noise_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")
PREDICTION_HORIZON = 2  # the chunk length of the fake system's labels (and of the arm's, in the arm tests)


class _FakeSimulator:
    """Minimal 1-D stand-in exercising only what collect_dagger_rollouts needs.

    ``predict_next_state`` (used for the pre-execution safety checks) and
    ``step`` (the actual transition) are allowed to diverge via
    ``first_step_disturbance``, so a test can force the real, post-execution
    collision path distinctly from the raw-expert-label collision path.
    """

    def __init__(self, collision_threshold: float, goal_threshold: float, done_hold_steps: int = 1) -> None:
        self.goal = np.array([0.0], dtype=float)
        self.nx = 1
        self.n = 1
        self.dt = 0.05
        self.max_action = np.array([1e9], dtype=float)
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.done_hold_steps = done_hold_steps
        self.state: np.ndarray | None = None
        self.step_call_count = 0
        self.first_step_disturbance = 0.0
        self._rollout_done_counter = 0

    @property
    def goal_dim(self) -> int:
        return 1

    def set_goal(self, goal: np.ndarray) -> None:
        self.goal = np.asarray(goal, dtype=float).copy()

    def randomize_goal_for_reset(self, rng: np.random.Generator) -> None:
        self.goal = np.array([rng.uniform(-1.0, 1.0)], dtype=float)

    def random_initial_state(self, rng: np.random.Generator) -> np.ndarray:
        return np.array([rng.uniform(-1.0, 1.0)], dtype=float)

    def reset(self, state: np.ndarray) -> np.ndarray:
        self.state = np.asarray(state, dtype=float).copy()
        self.reset_rollout_termination()
        return self.state.copy()

    def reset_rollout_termination(self) -> None:
        self._rollout_done_counter = 0

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).copy()

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=float).copy()

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float) + np.asarray(action, dtype=float)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.step_call_count += 1
        next_state = self.predict_next_state(state, action)
        if self.step_call_count == 1:
            next_state = next_state + self.first_step_disturbance
        return next_state

    def is_collision(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.collision_threshold)

    def is_done(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.goal_threshold)

    def should_terminate_rollout(self, state: np.ndarray) -> bool:
        """Mirrors DynamicsSimulator.should_terminate_rollout's consecutive-hold-steps contract."""
        if self.is_done(state):
            self._rollout_done_counter += 1
        else:
            self._rollout_done_counter = 0
        return self._rollout_done_counter >= self.done_hold_steps

    def format_dataset_frame(self, obs: np.ndarray, plan: np.ndarray) -> dict[str, np.ndarray]:
        """Mirrors FrankaDynamics.format_dataset_frame: ONE mapping (no ``task`` -- the caller adds it).

        The dataset action is the expert's planned actions (prediction_horizon, nu) flattened row by row.
        """
        return {
            "observation.environment_state": self.goal.astype(np.float32),
            "observation.state": np.asarray(obs, dtype=np.float32),
            "action": np.asarray(plan, dtype=np.float32).reshape(-1),
        }

    def policy_observation(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Mirrors DynamicsSimulator.policy_observation: the dataset's two observation keys, float32."""
        frame = self.format_dataset_frame(obs, np.zeros((1, self.n)))
        return {
            "observation.environment_state": frame["observation.environment_state"],
            "observation.state": frame["observation.state"],
        }


class _FakePlantPlanner(Planner):
    """A planner of the 1-D fake plant (state += action) that plans one step: the state and the state after the
    scripted action of this call. A subclass chooses the action (or fails)."""

    def __init__(self) -> None:
        self.call_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def plan(self, observation: np.ndarray, initial_plan: Plan | None = None) -> Plan:
        self.call_count += 1
        action = self._action_of_call()
        return Plan(positions=np.stack([observation, observation + action]), actions=action[None, :])

    def _action_of_call(self) -> np.ndarray:
        raise NotImplementedError


class _ScriptedPlanner(_FakePlantPlanner):
    """Plans an unsafe (colliding) action on its first call, then a safe one afterwards."""

    def __init__(self, unsafe_action: list[float], safe_action: list[float]) -> None:
        super().__init__()
        self.unsafe_action = np.asarray(unsafe_action, dtype=float)
        self.safe_action = np.asarray(safe_action, dtype=float)

    def _action_of_call(self) -> np.ndarray:
        return self.unsafe_action if self.call_count == 1 else self.safe_action


class _FailThenSucceedPlanner(_FakePlantPlanner):
    """Raises PlannerSolveError (a genuine solve failure) on its first call, then
    plans a constant safe action -- unlike _ScriptedPlanner, which always
    successfully solves but plans an action that's unsafe to execute."""

    def __init__(self, safe_action: list[float]) -> None:
        super().__init__()
        self.safe_action = np.asarray(safe_action, dtype=float)

    def _action_of_call(self) -> np.ndarray:
        if self.call_count == 1:
            raise PlannerSolveError("forced solver failure for test")
        return self.safe_action


class _ConstantPlanner(_FakePlantPlanner):
    def __init__(self, action: list[float]) -> None:
        super().__init__()
        self.action = np.asarray(action, dtype=float)

    def _action_of_call(self) -> np.ndarray:
        return self.action


class _FakeDatasetWriter:
    def __init__(self) -> None:
        self.frames: list[dict[str, object]] = []
        self.episode_boundaries: list[int] = []

    def add_frame(self, frame: dict[str, object]) -> None:
        self.frames.append(frame)

    def save_episode(self) -> None:
        self.episode_boundaries.append(len(self.frames))


def _frame_builder(observation: np.ndarray, action: np.ndarray) -> dict[str, object]:
    return {"observation": np.asarray(observation).tolist(), "action": np.asarray(action).tolist()}


def _assert_label_is_the_plan_of_one_step(test: unittest.TestCase, frame: dict[str, object], action: float) -> None:
    """The label of a frame is the expert's planned actions from its state: the planned action, then zeros."""
    test.assertEqual(frame["action"], [[action], [0.0]])


class _AlwaysFailingPlanner(_FakePlantPlanner):
    """Every solve raises PlannerSolveError (an expert that can never solve)."""

    def _action_of_call(self) -> np.ndarray:
        raise PlannerSolveError("forced solver failure for test")


class _FirstCallCollidesPolicy:
    """A learner whose first action drives the fake simulator into collision and whose later ones are safe."""

    def __init__(self, colliding_action: list[float], safe_action: list[float]) -> None:
        self.colliding_action = np.asarray(colliding_action, dtype=float)
        self.safe_action = np.asarray(safe_action, dtype=float)
        self.call_count = 0

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        self.call_count += 1
        return (self.colliding_action if self.call_count == 1 else self.safe_action).copy()


def _collect(simulator, planner, writer, *, trajectories=1, steps=3, beta=1.0, policy_action_fn=None, initial_states=None):
    """collect_dagger_rollouts with the boilerplate arguments fixed; returns (metrics, captured stdout)."""
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        metrics = collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=trajectories,
            steps_per_trajectory=steps,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=beta,
            prediction_horizon=PREDICTION_HORIZON,
            policy_action_fn=policy_action_fn,
            frame_builder=_frame_builder,
            initial_states=initial_states,
            goal_states=None,
        )
    return metrics, stdout.getvalue()


class InitialStateSeedTests(unittest.TestCase):
    def test_round_seeds_are_distinct_and_reproducible(self) -> None:
        first_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=0)
        second_round_seed = initial_state_seed_for_rollout(0, rollout_index=1, round_index=1)

        self.assertNotEqual(first_round_seed, second_round_seed)
        self.assertEqual(
            first_round_seed,
            initial_state_seed_for_rollout(0, rollout_index=1, round_index=0),
        )


class _CountingPlanner(_ConstantPlanner):
    """A planner with cumulative counters, like the expert with an OMPL path (``Planner.statistics``)."""

    EARLIER_CALLS = 3  # what it did before the collection

    @property
    def statistics(self) -> dict[str, int]:
        return {"calls": self.EARLIER_CALLS + self.call_count}


class CollectionStatisticsTests(unittest.TestCase):
    def test_the_experts_statistics_are_the_change_over_the_collection(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _CountingPlanner(action=[1.0])

        metrics, _ = _collect(simulator, planner, _FakeDatasetWriter(), trajectories=2, initial_states=[[0.0]])

        self.assertEqual(planner.statistics, {"calls": 3 + 2})
        self.assertEqual(metrics.expert_statistics, {"calls": 2})

    def test_a_planner_without_statistics_reports_none(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)

        metrics, _ = _collect(simulator, _ConstantPlanner(action=[1.0]), _FakeDatasetWriter(), initial_states=[[0.0]])

        self.assertEqual(metrics.expert_statistics, {})


class CollectDaggerRolloutsDropAndRetryTests(unittest.TestCase):
    """A failed episode is dropped as a whole and retried from a fresh initial state.

    The fake simulator's first attempt starts from the explicit initial state [0.0]; every later attempt
    starts from a sampled state in [-1, 1] and reaches the goal after one step (goal_threshold=-10), so a
    retry is recognisable by its start state and by the number of planner calls.
    """

    def test_unsafe_expert_label_drops_the_episode_and_the_retry_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _ScriptedPlanner(unsafe_action=[100.0], safe_action=[1.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, initial_states=[[0.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(planner.call_count, 2)  # the dropped attempt and the retry
        self.assertEqual(planner.reset_count, 2)
        # Nothing of the dropped attempt reaches the dataset, and the retry starts elsewhere than [0.0]
        self.assertEqual(len(writer.frames), 1)
        _assert_label_is_the_plan_of_one_step(self, writer.frames[0], 1.0)
        self.assertNotEqual(writer.frames[0]["observation"], [0.0])
        self.assertEqual(writer.episode_boundaries, [1])
        self.assertIn("Unsafe expert corrective label", output)
        self.assertIn("Discarding DAgger episode", output)

    def test_post_execution_collision_drops_the_episode_and_the_retry_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        simulator.first_step_disturbance = 20.0  # only the very first .step() call collides
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, initial_states=[[0.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(simulator.step_call_count, 2)
        self.assertEqual(planner.reset_count, 2)
        self.assertEqual(len(writer.frames), 1)
        self.assertNotEqual(writer.frames[0]["observation"], [0.0])
        self.assertEqual(writer.episode_boundaries, [1])
        self.assertIn("DAgger rollout collision", output)
        self.assertIn("Discarding DAgger episode", output)

    def test_planner_solve_failure_drops_the_episode_and_the_retry_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _FailThenSucceedPlanner(safe_action=[1.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, initial_states=[[0.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(len(writer.frames), 1)
        _assert_label_is_the_plan_of_one_step(self, writer.frames[0], 1.0)
        self.assertNotEqual(writer.frames[0]["observation"], [0.0])
        self.assertIn("Planner failed to solve", output)
        self.assertIn("Discarding DAgger episode", output)
        self.assertEqual(
            (metrics.attempted_episodes, metrics.discarded_episodes, metrics.planner_failures, metrics.policy_solve_fallbacks),
            (2, 1, 1, 0),
        )

    def test_frames_of_a_dropped_episode_are_never_written_even_after_several_good_steps(self) -> None:
        # The collision happens on step 3: the two collected frames before it must be dropped with the episode
        simulator = _FakeSimulator(collision_threshold=2.5, goal_threshold=100.0)
        planner = _ConstantPlanner(action=[1.0])  # state 0 -> 1 -> 2 -> 3 (collides) from the first start
        writer = _FakeDatasetWriter()

        # The retry starts from a sampled state in [-1, 1], which also reaches the collision threshold
        # eventually, so bound the attempts: every episode here collides and the run has to give up.
        with self.assertRaisesRegex(RuntimeError, "Too many failed DAgger rollout attempts"):
            _collect(simulator, planner, writer, steps=5, initial_states=[[0.0]])

        self.assertEqual(writer.frames, [])
        self.assertEqual(writer.episode_boundaries, [])

    def test_gives_up_after_three_times_the_requested_episodes(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _AlwaysFailingPlanner()
        writer = _FakeDatasetWriter()

        with self.assertRaisesRegex(RuntimeError, "Too many failed DAgger rollout attempts.*0/2"):
            _collect(simulator, planner, writer, trajectories=2)

        self.assertEqual(planner.call_count, 6)  # 3 * trajectories_per_iteration attempts, one solve each
        self.assertEqual(writer.frames, [])

    def test_dropped_policy_steps_are_not_counted_in_the_execution_mixing_stats(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _ConstantPlanner(action=[0.5])
        policy = _FirstCallCollidesPolicy(colliding_action=[100.0], safe_action=[0.4])
        writer = _FakeDatasetWriter()

        # beta = 0: every executed action comes from the policy; its first one collides, so that episode is dropped
        metrics, output = _collect(
            simulator, planner, writer, beta=0.0, policy_action_fn=policy, initial_states=[[0.0]]
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(policy.call_count, 2)
        self.assertIn("realized_expert_fraction=0.000", output)
        self.assertIn("executed_steps=1", output)  # only the retry's step, not the dropped episode's
        self.assertIn("actor=policy", output)
        # The dataset carries the expert's relabelled plan (a step of 0.5), not what the policy executed (0.4)
        _assert_label_is_the_plan_of_one_step(self, writer.frames[0], 0.5)

    def test_policy_solve_failure_falls_back_to_the_expert_instead_of_dropping_the_episode(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.5)
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        def failing_policy(observation: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("projector failed")

        metrics, output = _collect(
            simulator, planner, writer, beta=0.0, policy_action_fn=failing_policy, initial_states=[[0.0]]
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(planner.reset_count, 1)  # no retry
        self.assertEqual(writer.frames[0]["observation"], [0.0])
        self.assertIn("Policy action unavailable this step", output)
        self.assertNotIn("Discarding DAgger episode", output)
        self.assertEqual((metrics.discarded_episodes, metrics.planner_failures, metrics.policy_solve_fallbacks), (0, 0, 1))
        # The executed step was the expert's, and is recorded as such
        self.assertIn("realized_expert_fraction=1.000", output)
        self.assertIn("executed_steps=1", output)

    def test_an_episode_that_runs_out_of_steps_is_kept_as_a_failure_not_dropped(self) -> None:
        # The expert can stall short of the goal; such an episode is valid data (a timeout), not a retry
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        planner = _ConstantPlanner(action=[0.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, steps=3, initial_states=[[0.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual((metrics.min_steps, metrics.max_steps), (3, 3))
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(writer.episode_boundaries, [3])
        self.assertEqual(planner.reset_count, 1)
        self.assertNotIn("Discarding DAgger episode", output)

    def test_colliding_initial_state_is_skipped_and_the_next_attempt_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, initial_states=[[50.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertIn("Skipping DAgger episode due to colliding initial state", output)
        self.assertEqual(planner.reset_count, 1)  # the skipped start never reached the planner
        self.assertNotEqual(writer.frames[0]["observation"], [50.0])

    def test_initial_only_episode_still_randomizes_goal(self) -> None:
        """Regression guard: an explicit initial state without a paired goal must
        still honor randomize_goal_for_reset, matching what a fallback-sampled
        episode gets, instead of silently freezing at whatever goal the
        simulator started with for as many initial states as were provided.
        """
        simulator = _FakeSimulator(collision_threshold=100.0, goal_threshold=100.0)
        planner = _ConstantPlanner(action=[0.0])
        writer = _FakeDatasetWriter()
        self.assertEqual(simulator.goal.tolist(), [0.0])

        _collect(simulator, planner, writer, steps=1, initial_states=[[0.0]])

        self.assertNotEqual(simulator.goal.tolist(), [0.0])


class EvaluatePolicyRolloutsTorchSeedingTests(unittest.TestCase):
    """A flow/safeflow policy's inference draws from an unseeded torch.randn
    for its ODE initial noise, so without pinning torch's RNG per episode,
    the exact same checkpoint on the exact same scenario can succeed or fail
    from pure sampling luck -- evaluate_policy_rollouts must call
    torch.manual_seed with a value derived only from that episode's own seed
    spec, so re-evaluating the same scenario reproduces the same draw.
    """

    def test_manual_seed_is_called_once_per_episode_with_the_derived_seed(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        with mock.patch("learning.dagger.rollouts.torch.manual_seed") as manual_seed:
            evaluate_policy_rollouts(
                simulator=simulator,
                num_episodes=3,
                num_steps=2,
                seed_start=100,
                action_fn=lambda observation: np.array([0.0]),
            )
        # evaluation_seed_specs yields plain ints seed_start, seed_start+1,
        # ... ; the torch seed is the *derived* (stream-ID-mixed) value, not
        # that raw int -- see torch_inference_seed_for_rollout.
        manual_seed.assert_has_calls([
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=100)),
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=101)),
            mock.call(torch_inference_seed_for_rollout(0, seed_spec=102)),
        ])

    def test_torch_seed_is_decorrelated_from_the_action_noise_seed(self) -> None:
        """Regression guard for the exact bug this stream ID exists to
        prevent: both seeds are derived from the same seed_spec, but must
        not collide or trivially track each other.
        """
        seed_spec = 100
        torch_seed = torch_inference_seed_for_rollout(0, seed_spec=seed_spec)
        noise_seed = action_noise_seed_for_rollout(0, seed_spec=seed_spec)
        self.assertNotEqual(torch_seed, noise_seed)

    def test_reevaluating_the_same_scenario_reproduces_the_same_seed(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        seeds_seen: list[list[int]] = [[], []]
        for run_idx in range(2):
            with mock.patch("learning.dagger.rollouts.torch.manual_seed", side_effect=seeds_seen[run_idx].append):
                evaluate_policy_rollouts(
                    simulator=simulator,
                    num_episodes=2,
                    num_steps=2,
                    seed_start=7,
                    action_fn=lambda observation: np.array([0.0]),
                )
        self.assertEqual(seeds_seen[0], seeds_seen[1])

    def test_global_torch_rng_state_is_restored_after_evaluation(self) -> None:
        """Regression guard: torch.manual_seed(...) mutates the same
        process-wide generator(s) DAgger training's own compute_loss draws
        from (e.g. FlowPolicy.compute_loss's torch.randn_like/torch.rand) --
        pinning eval's own seed for reproducibility must not leak into the
        next round's training noise sequence.
        """
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        torch.manual_seed(12345)
        before_state = torch.get_rng_state().clone()
        evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=3,
            num_steps=2,
            seed_start=100,
            action_fn=lambda observation: np.array([0.0]),
        )
        after_state = torch.get_rng_state()
        self.assertTrue(torch.equal(before_state, after_state))


class RestartInitialStateRoundTests(unittest.TestCase):
    """Coverage for restart_initial_state_round: it must make initial-state/goal
    sampling reproducible across rounds that key off the same seed, without
    changing anything when left at its default.
    """

    @staticmethod
    def _sample_via_collect(*, round_index: int, restart_initial_state_round: bool) -> tuple[list[float], list[float]]:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        planner = _ConstantPlanner(action=[0.0])
        writer = _FakeDatasetWriter()
        collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=planner,
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=1,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=7,
            expert_mixing_beta=1.0,
            prediction_horizon=PREDICTION_HORIZON,
            policy_action_fn=None,
            frame_builder=_frame_builder,
            initial_states=None,
            goal_states=None,
            round_index=round_index,
            restart_initial_state_round=restart_initial_state_round,
        )
        # action=[0.0] never moves the state, so simulator.state is still exactly
        # the sampled initial state after the one recorded step.
        return simulator.goal.tolist(), simulator.state.tolist()

    def test_restart_true_makes_rounds_sharing_a_seed_sample_identically(self) -> None:
        round0 = self._sample_via_collect(round_index=0, restart_initial_state_round=True)
        round1 = self._sample_via_collect(round_index=1, restart_initial_state_round=True)
        self.assertEqual(round0, round1)

    def test_restart_false_preserves_todays_behavior_of_varying_by_round(self) -> None:
        round0 = self._sample_via_collect(round_index=0, restart_initial_state_round=False)
        round1 = self._sample_via_collect(round_index=1, restart_initial_state_round=False)
        self.assertNotEqual(round0, round1)


class EvaluatePolicyRolloutsSourceSplitTests(unittest.TestCase):
    """evaluate_policy_rollouts must report config- and random-sourced episode
    outcomes separately (DaggerEvalMetrics.config_*/random_*), not just the
    pooled success_rate -- the curriculum's explicit initial_states/goal_states
    are the leading `usable_count` episodes, everything after that falls back
    to simulator RNG sampling (see its own usable_count logic).
    """

    def test_config_and_random_successes_are_counted_separately(self) -> None:
        # goal_threshold=1000.0 with a constant zero action means:
        # - a config-sourced episode starting exactly at the threshold
        #   terminates instantly (reached_goal=True, 0 steps) every time.
        # - a random-sourced episode starts uniformly in [-1, 1] (see
        #   _FakeSimulator.random_initial_state) and a zero action never
        #   moves it, so it can never reach 1000.0 -- always a timeout.
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1000.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=5,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
            initial_states=[[1000.0], [1000.0]],
            goal_states=[[0.0], [0.0]],
        )
        assert metrics is not None
        self.assertEqual(metrics.num_episodes, 5)
        self.assertEqual(metrics.config_num_episodes, 2)
        self.assertEqual(metrics.config_successes, 2)
        self.assertEqual(metrics.random_num_episodes, 3)
        self.assertEqual(metrics.random_successes, 0)
        # Pooled rate still reflects both groups combined.
        self.assertAlmostEqual(metrics.success_rate, 2.0 / 5.0)

    def test_no_explicit_states_leaves_config_bucket_empty(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1000.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=3,
            num_steps=2,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
        )
        assert metrics is not None
        self.assertEqual(metrics.config_num_episodes, 0)
        self.assertEqual(metrics.config_successes, 0)
        self.assertEqual(metrics.random_num_episodes, 3)


class EvaluatePolicyRolloutsFailureBreakdownTests(unittest.TestCase):
    """evaluate_policy_rollouts must report *why* a failed episode failed --
    collision, timeout, or solve_failure -- not just a pooled success_rate.
    A policy that times out while actively avoiding is a different (and
    better) outcome than one that collides outright, which success_rate
    alone can't distinguish.
    """

    def test_collision_timeout_and_solve_failure_are_counted_separately(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1000.0)

        def action_fn(observation: np.ndarray) -> np.ndarray:
            # Sentinel initial state [5.0] marks the solve_failure episode;
            # everything else is a plain no-op action (never moves, so it
            # can only time out or start already past the collision
            # threshold).
            if float(np.asarray(observation)[0]) == 5.0:
                raise PlannerSolveError("forced failure for test")
            return np.array([0.0])

        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=3,
            num_steps=3,
            seed_start=0,
            action_fn=action_fn,
            # goal_threshold=1000.0 means none of these ever reach the goal;
            # no goal_states, so all 3 are sourced from this initial_states
            # list, indexed 0/1/2 in order.
            initial_states=[[15.0], [0.0], [5.0]],
        )

        assert metrics is not None
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.collision_failures, 1)
        self.assertEqual(metrics.timeout_failures, 1)
        self.assertEqual(metrics.solve_failures, 1)

    def test_successful_episodes_are_not_counted_as_any_failure(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=0.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=2,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation: np.array([0.0]),
            initial_states=[[0.0], [0.0]],
        )
        assert metrics is not None
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.collision_failures, 0)
        self.assertEqual(metrics.timeout_failures, 0)
        self.assertEqual(metrics.solve_failures, 0)


class RolloutCollisionTests(unittest.TestCase):
    def test_a_rollout_collision_is_reported_and_the_episode_discarded(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        simulator.first_step_disturbance = 20.0  # only the very first .step() call collides
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=planner,
                dataset_writer=writer,
                trajectories_per_iteration=1,
                steps_per_trajectory=2,
                action_noise_std=0.0,
                action_noise_seed=0,
                initial_state_seed=0,
                expert_mixing_beta=1.0,
                prediction_horizon=PREDICTION_HORIZON,
                policy_action_fn=None,
                frame_builder=_frame_builder,
                initial_states=[[0.0]],
                goal_states=None,
            )

        self.assertIn("DAgger rollout collision (attempt=1, step=1, actor=expert).", stdout.getvalue())


class ApplyExecutionNoiseTests(unittest.TestCase):
    """apply_execution_noise perturbs the executed action with seeded Gaussian
    noise and clips the result to the simulator's max_action.
    """

    def test_zero_noise_returns_the_action_unchanged(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        action = np.array([0.3])
        self.assertIs(apply_execution_noise(simulator, action, 0.0), action)

    def test_noise_perturbs_the_action_and_is_reproducible_for_the_same_rng_seed(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        action = np.array([0.3])

        first = apply_execution_noise(simulator, action, 0.1, np.random.default_rng(5))
        second = apply_execution_noise(simulator, action, 0.1, np.random.default_rng(5))

        np.testing.assert_array_equal(first, second)
        self.assertNotEqual(first[0], action[0])

    def test_noisy_action_is_clipped_to_max_action(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        simulator.max_action = np.array([1.0])
        # A noise std this large pushes the action far past +-max_action.
        noisy = apply_execution_noise(simulator, np.array([0.9]), 100.0, np.random.default_rng(0))
        np.testing.assert_array_equal(noisy, [1.0])


class _StubPolicy:
    """Records the policy input it is queried with and returns a fixed action tensor."""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.received: dict[str, torch.Tensor] | None = None
        self.queried_in_inference_mode: bool | None = None

    def select_action(self, observation_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.received = dict(observation_dict)
        self.queried_in_inference_mode = torch.is_inference_mode_enabled()
        return self.output


class BuildPolicyActionTests(unittest.TestCase):
    """build_policy_action queries the policy once, on a batch of one built
    from simulator.policy_observation, and hands back the first row of the
    predicted action chunk.
    """

    def test_returns_the_first_action_of_the_chunk(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        chunk = torch.tensor([[[0.5, 0.25], [3.0, 4.0], [5.0, 6.0]]])  # (batch, horizon, nu)

        action = build_policy_action(simulator, _StubPolicy(chunk), np.array([0.5]), torch.device("cpu"))

        self.assertEqual(action.shape, (2,))
        np.testing.assert_allclose(action, [0.5, 0.25])

    def test_policy_input_is_a_batch_of_one_with_exactly_the_two_observation_keys(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        simulator.set_goal(np.array([0.25]))
        policy = _StubPolicy(torch.zeros(1, 3, 2))

        build_policy_action(simulator, policy, np.array([0.5]), torch.device("cpu"))

        assert policy.received is not None
        self.assertEqual(set(policy.received), {"observation.environment_state", "observation.state"})
        for name, expected in (("observation.environment_state", [[0.25]]), ("observation.state", [[0.5]])):
            self.assertEqual(policy.received[name].dtype, torch.float32, name)
            np.testing.assert_allclose(policy.received[name].numpy(), expected, err_msg=name)
        self.assertTrue(policy.queried_in_inference_mode)

    def test_malformed_policy_output_is_rejected(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        # Not a batch of chunks, too many dimensions, or a chunk without an action
        for output in (torch.zeros(2), torch.zeros(1, 2), torch.zeros(1, 2, 3, 4), torch.zeros(1, 0, 2)):
            with self.subTest(shape=tuple(output.shape)):
                with self.assertRaises(ValueError):
                    build_policy_action(simulator, _StubPolicy(output), np.array([0.5]), torch.device("cpu"))


class DefaultFrameBuilderTests(unittest.TestCase):
    """Without a frame_builder, collect_dagger_rollouts uses the simulator's
    own format_dataset_frame -- ONE mapping per step (the arm is a single
    robot) -- and adds the LeRobot ``task`` string: the simulator's
    task_description if it has one, else "reach target".
    """

    @staticmethod
    def _collect(simulator: _FakeSimulator, **kwargs: object) -> _FakeDatasetWriter:
        writer = _FakeDatasetWriter()
        collect_dagger_rollouts(
            simulator=simulator,
            expert_planner=_ConstantPlanner(action=[1.0]),
            dataset_writer=writer,
            trajectories_per_iteration=1,
            steps_per_trajectory=1,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=1.0,
            prediction_horizon=PREDICTION_HORIZON,
            policy_action_fn=None,
            initial_states=[[0.0]],
            goal_states=None,
            **kwargs,
        )
        return writer

    def test_frames_come_from_the_simulators_dataset_frame_with_its_task_description(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        simulator.task_description = "reach fake target"

        writer = self._collect(simulator)

        self.assertEqual(len(writer.frames), 1)
        frame = writer.frames[0]
        self.assertEqual(set(frame), {"observation.environment_state", "observation.state", "action", "task"})
        self.assertEqual(frame["task"], "reach fake target")
        np.testing.assert_allclose(frame["observation.state"], [0.0])
        # The expert's planned actions (the action 1, then zeros), flattened row by row
        np.testing.assert_allclose(frame["action"], [1.0, 0.0])
        self.assertEqual(writer.episode_boundaries, [1])

    def test_task_defaults_to_reach_target_without_a_task_description(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        writer = self._collect(simulator)
        self.assertEqual(writer.frames[0]["task"], "reach target")

    def test_a_frame_builder_must_return_a_single_mapping(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        with self.assertRaises(TypeError):
            self._collect(
                simulator,
                frame_builder=lambda observation, action: [_frame_builder(observation, action)],
            )


class RolloutsWithTheArmTests(unittest.TestCase):
    """The same entry points on the real FrankaDynamics/CasadiPlanner instead of
    the 1-D fake: a few steps from a start near the ready pose towards a
    nearby goal pose, so this stays fast and deterministic -- what it checks
    is the interfaces (frame schema, the 12-dim goal, the policy input), not
    task success.
    """

    @classmethod
    def setUpClass(cls) -> None:
        with open(CONFIG_PATH) as f:
            cls.config = validate_system_config(raw_config=yaml.safe_load(f))
        cls.simulator = FrankaDynamics(cls.config)
        q = Q_READY.copy()
        q += np.array([0.12, -0.10, 0.08, 0.10, -0.12, 0.10, 0.25])
        cls.initial_state = np.concatenate([q, np.zeros(2 * cls.simulator.n)])
        goal_q = q.copy()
        goal_q += 0.1
        cls.goal = cls.simulator.pose_to_goal(cls.simulator.tcp_placement(goal_q))

    def test_collect_dagger_rollouts_writes_frames_matching_the_dataset_schema(self) -> None:
        simulator = self.simulator
        writer = _FakeDatasetWriter()

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=CasadiPlanner(simulator, dict(self.config)),
                dataset_writer=writer,
                trajectories_per_iteration=1,
                steps_per_trajectory=3,
                action_noise_std=0.0,
                action_noise_seed=0,
                initial_state_seed=0,
                expert_mixing_beta=1.0,
                prediction_horizon=PREDICTION_HORIZON,
                policy_action_fn=None,
                frame_builder=simulator.format_dataset_frame,
                initial_states=[self.initial_state.tolist()],
                goal_states=[self.goal.tolist()],
            )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(writer.episode_boundaries, [3])
        features = simulator.get_dataset_features(PREDICTION_HORIZON)
        for frame in writer.frames:
            self.assertEqual(set(frame), set(features) | {"task"})
            for name, feature in features.items():
                self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
            self.assertEqual(frame["task"], simulator.task_description)
            np.testing.assert_allclose(frame["observation.environment_state"], self.goal, atol=1e-6)
        np.testing.assert_allclose(writer.frames[0]["observation.state"], self.initial_state, atol=1e-6)
        # The label of a step is the expert's plan from its state: its jerks, the first of which is the action the
        # expert executed (it leads to the next frame's state)
        for step, frame in enumerate(writer.frames[:-1]):
            actions = frame["action"].reshape(PREDICTION_HORIZON, simulator.nu)
            np.testing.assert_allclose(
                simulator.predict_next_state(frame["observation.state"].astype(float), actions[0]),
                writer.frames[step + 1]["observation.state"],
                atol=1e-4,
            )

    def test_evaluate_policy_rollouts_queries_the_policy_through_build_policy_action(self) -> None:
        simulator = self.simulator
        policy = FlowPolicy(
            action_dim=simulator.nu, state_dim=simulator.obs_dim, hidden_dims=(16, 16),
            prediction_horizon=4, num_inference_steps=2,
        ).eval()
        actions: list[np.ndarray] = []

        def action_fn(observation: np.ndarray) -> np.ndarray:
            action = build_policy_action(simulator, policy, observation, torch.device("cpu"))
            actions.append(action)
            return action

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_policy_rollouts(
                simulator=simulator,
                num_episodes=2,
                num_steps=2,
                seed_start=0,
                action_fn=action_fn,
                reset_fn=policy.reset,
                initial_states=[self.initial_state.tolist()],
                goal_states=[self.goal.tolist()],
            )

        assert metrics is not None
        self.assertEqual(metrics.num_episodes, 2)
        # The first episode is sourced from the explicit initial state/goal pose, the second from
        # the simulator's own random sampling.
        self.assertEqual((metrics.config_num_episodes, metrics.random_num_episodes), (1, 1))
        failures = metrics.collision_failures + metrics.timeout_failures + metrics.solve_failures
        self.assertEqual(failures + round(metrics.success_rate * metrics.num_episodes), metrics.num_episodes)
        self.assertGreaterEqual(len(actions), 2)
        for action in actions:
            self.assertEqual(action.shape, (simulator.nu,))
            self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
