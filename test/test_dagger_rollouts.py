from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from learning.data_utils import build_dataset_features
from learning.dagger.collection import collect_dagger_rollouts
from learning.dagger.episode_starts import _setup_episode_start
from learning.dagger.evaluation import evaluate_policy_rollouts, rollout_policy_with_action_fn
from learning.dagger.policy_action import _policy_action_fn, apply_execution_noise, build_policy_actions
from learning.dagger.workers import RolloutWorkerSpec
from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.planner import Plan, Planner, PlannerSolveError
from planning.planner_factory import PlannerFactory
from test.planner_fixtures import pinned_config, pose_goal, rest_state
from systems.seed_utils import (
    action_noise_seed_for_rollout,
    initial_state_seed_for_rollout,
    torch_inference_seed_for_rollout,
)

PREDICTION_HORIZON = 2  # the chunk length of the fake system's labels (and of the arm's, in the arm tests)
HISTORY_LENGTH = 3
EXECUTION_HORIZON = 2  # control periods the first jerks of a chunk are executed for, in the arm tests


class _FakeSimulator:
    """Minimal 1-D stand-in exercising only what collect_dagger_rollouts needs."""

    task_description = "reach target"

    def __init__(self, collision_threshold: float = 1e9, goal_threshold: float = 1.0, done_hold_steps: int = 1) -> None:
        self.goal = np.array([0.0], dtype=float)
        self.nx = 1
        self.n = 1
        self.nu = 1
        self.dt = 0.05
        self.max_action = np.array([1e9], dtype=float)
        self.collision_threshold = collision_threshold
        self.goal_threshold = goal_threshold
        self.done_hold_steps = done_hold_steps
        self.state: np.ndarray | None = None
        self.step_call_count = 0

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
        return self.state.copy()

    def observe(self, state: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float).copy()

    def invert_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=float).copy()

    def predict_next_state(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        return np.asarray(state, dtype=float) + np.asarray(action, dtype=float)

    def step(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        self.step_call_count += 1
        return self.predict_next_state(state, action)

    def is_done(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.goal_threshold)

    def collides(self, state: np.ndarray) -> bool:
        return bool(np.asarray(state, dtype=float)[0] >= self.collision_threshold)

    def policy_observation(self, obs: np.ndarray, history: np.ndarray) -> dict[str, np.ndarray]:
        """Mirrors SystemSimulator.policy_observation: the dataset's three observation keys, float32."""
        return {
            "observation.environment_state": self.goal.astype(np.float32),
            "observation.state": np.asarray(obs, dtype=np.float32),
            "observation.history": np.asarray(history, dtype=np.float32),
        }


PLAN_ROWS = PREDICTION_HORIZON + 1  # the rows the fake expert plans, one more than the policy's chunk


class _FakePlantPlanner(Planner):
    """A planner of the 1-D fake plant (state += action) whose plan moves on at the scripted action of this call: row k
    is the state k + 1 steps ahead. A subclass chooses the action (or fails)."""

    def __init__(self) -> None:
        self.call_count = 0
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def plan(self, observation: np.ndarray) -> Plan:
        self.call_count += 1
        action = self._action_of_call()
        steps = np.arange(1, PLAN_ROWS + 1)[:, None]
        return Plan(positions=observation + steps * action, actions=np.tile(action, (PLAN_ROWS, 1)))

    def _action_of_call(self) -> np.ndarray:
        raise NotImplementedError


class _FailThenSucceedPlanner(_FakePlantPlanner):
    """Raises PlannerSolveError (a genuine solve failure) on its first call, then
    plans a constant safe action."""

    def __init__(self, safe_action: list[float]) -> None:
        super().__init__()
        self.safe_action = np.asarray(safe_action, dtype=float)

    def _action_of_call(self) -> np.ndarray:
        if self.call_count == 1:
            raise PlannerSolveError("forced solver failure for test")
        return self.safe_action


class _FailOnCallPlanner(_FakePlantPlanner):
    """Raises PlannerSolveError on the specified call number(s), otherwise plans a constant action."""

    def __init__(self, action: list[float], fail_calls: set[int]) -> None:
        super().__init__()
        self.action = np.asarray(action, dtype=float)
        self.fail_calls = fail_calls

    def _action_of_call(self) -> np.ndarray:
        if self.call_count in self.fail_calls:
            raise PlannerSolveError("forced solver failure for test")
        return self.action


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


def _frame_builder(observation: np.ndarray, history: np.ndarray, positions: np.ndarray) -> dict[str, object]:
    return {
        "observation": np.asarray(observation).tolist(),
        "history": np.asarray(history).tolist(),
        "action": np.asarray(positions).tolist(),
    }


def _assert_label_is_the_plan_of_the_action(test: unittest.TestCase, frame: dict[str, object], action: float) -> None:
    """The label of a frame is the first ``PREDICTION_HORIZON`` rows of the expert's plan from its state."""
    state = frame["observation"][0]
    test.assertEqual(len(frame["action"]), PREDICTION_HORIZON)
    np.testing.assert_allclose(frame["action"], [[state + (k + 1) * action] for k in range(PREDICTION_HORIZON)])


class _AlwaysFailingPlanner(_FakePlantPlanner):
    """Every solve raises PlannerSolveError (an expert that can never solve)."""

    def _action_of_call(self) -> np.ndarray:
        raise PlannerSolveError("forced solver failure for test")


def _collect(
    simulator, planner, writer, *, trajectories=1, steps=3, beta=1.0, policy_action_fn=None, initial_states=None, execution_horizon=1
):
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
            history_length=HISTORY_LENGTH,
            execution_horizon=execution_horizon,
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
    """An episode whose expert fails at its first state is dropped and retried from a fresh initial state; one that fails
    later ends there and keeps the states it labeled.

    The fake simulator's first attempt starts from the explicit initial state [0.0]; every later attempt
    starts from a sampled state in [-1, 1] and reaches the goal after one step (goal_threshold=-10), so a
    retry is recognisable by its start state and by the number of planner calls.
    """

    def test_planner_solve_failure_at_the_first_state_drops_the_episode_and_the_retry_is_saved(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _FailThenSucceedPlanner(safe_action=[1.0])
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, initial_states=[[0.0]])

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(planner.call_count, 2)
        self.assertEqual(len(writer.frames), 1)
        _assert_label_is_the_plan_of_the_action(self, writer.frames[0], 1.0)
        self.assertNotEqual(writer.frames[0]["observation"], [0.0])
        self.assertIn("Planner failed to solve", output)
        self.assertIn("Discarding DAgger episode", output)
        self.assertEqual(
            (metrics.attempted_episodes, metrics.discarded_episodes, metrics.planner_failures, metrics.policy_solve_fallbacks),
            (2, 1, 1, 0),
        )
        self.assertEqual((metrics.truncated_episodes, dict(metrics.expert_failure_causes)), (0, {"forced solver failure for test": 1}))

    def test_an_episode_whose_expert_fails_later_keeps_the_states_it_labeled(self) -> None:
        # The planner fails on its third call: the two states labeled before it stay, and the episode is not retried
        simulator = _FakeSimulator(collision_threshold=2.5, goal_threshold=100.0)
        planner = _FailOnCallPlanner(action=[1.0], fail_calls={3})
        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, steps=5, initial_states=[[0.0]])

        self.assertEqual(len(writer.frames), 2)
        self.assertEqual(writer.episode_boundaries, [2])
        self.assertEqual(planner.call_count, 3)
        self.assertEqual(
            (metrics.num_episodes, metrics.success_rate, metrics.attempted_episodes, metrics.discarded_episodes),
            (1, 0.0, 1, 0),
        )
        self.assertEqual((metrics.truncated_episodes, metrics.planner_failures), (1, 1))
        self.assertNotIn("Discarding DAgger episode", output)

    def test_the_label_of_a_state_is_the_first_rows_of_the_experts_plan(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        writer = _FakeDatasetWriter()

        _collect(simulator, _ConstantPlanner(action=[1.0]), writer, steps=2, initial_states=[[0.0]])

        self.assertLess(PREDICTION_HORIZON, PLAN_ROWS)
        self.assertEqual([frame["action"] for frame in writer.frames], [[[1.0], [2.0]], [[2.0], [3.0]]])

    def test_every_frame_carries_the_history_of_the_rollout_so_far(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        writer = _FakeDatasetWriter()

        _collect(simulator, _ConstantPlanner(action=[1.0]), writer, steps=4, initial_states=[[0.0]])

        self.assertEqual(
            [frame["history"] for frame in writer.frames],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
        )

    def test_the_history_of_the_policy_query_is_that_of_the_frame(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        writer = _FakeDatasetWriter()
        queried: list[list[float]] = []

        def policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            queried.append(history.tolist())
            return np.array([[1.0]])

        _collect(simulator, _ConstantPlanner(action=[1.0]), writer, steps=3, beta=0.0, policy_action_fn=policy, initial_states=[[0.0]])

        self.assertEqual(queried, [frame["history"] for frame in writer.frames])

    def test_gives_up_after_three_times_the_requested_episodes(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=-10.0)
        planner = _AlwaysFailingPlanner()
        writer = _FakeDatasetWriter()

        with self.assertRaisesRegex(RuntimeError, "Too many failed DAgger rollout attempts.*0/2"):
            _collect(simulator, planner, writer, trajectories=2)

        self.assertEqual(planner.call_count, 6)  # 3 * trajectories_per_iteration attempts, one solve each
        self.assertEqual(writer.frames, [])

    def test_the_steps_of_a_truncated_episode_are_counted_in_the_execution_mixing_stats(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.7)
        # Step 1: the policy moves 0.0 -> 0.4, short of the goal; the expert's solve at step 2 fails, which ends the episode
        planner = _FailOnCallPlanner(action=[0.5], fail_calls={2})
        policy_calls = 0

        def policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            nonlocal policy_calls
            policy_calls += 1
            return np.array([[0.4]])

        writer = _FakeDatasetWriter()

        metrics, output = _collect(simulator, planner, writer, beta=0.0, policy_action_fn=policy, initial_states=[[0.0]])

        self.assertEqual((metrics.num_episodes, metrics.truncated_episodes), (1, 1))
        self.assertEqual(policy_calls, 1)
        self.assertIn("realized_expert_fraction=0.000", output)
        self.assertIn("executed_steps=1", output)
        # The dataset carries the expert's relabelled plan (a step of 0.5), not what the policy executed (0.4)
        _assert_label_is_the_plan_of_the_action(self, writer.frames[0], 0.5)

    def test_policy_solve_failure_falls_back_to_the_expert_instead_of_dropping_the_episode(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=0.5)
        planner = _ConstantPlanner(action=[1.0])
        writer = _FakeDatasetWriter()

        def failing_policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("projector failed")

        metrics, output = _collect(
            simulator, planner, writer, beta=0.0, policy_action_fn=failing_policy, initial_states=[[0.0]]
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(planner.reset_count, 1)  # no retry
        self.assertEqual(writer.frames[0]["observation"], [0.0])
        self.assertIn("Policy actions unavailable, falling back to the expert for the block", output)
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


class CollectionBlockTests(unittest.TestCase):
    """The policy is asked once per block of ``execution_horizon`` steps and executes its jerks one after the other, the
    expert plans and labels at every step, and a block is the expert's or the policy's as a whole.

    On the fake plant (state += action) the expert's jerk is 1.0, so a block can be told from the step it takes.
    """

    BLOCK = 3
    POLICY_JERKS = np.array([[0.1], [0.2], [0.3]])

    def _policy(self, queries: list[list[float]]):
        def policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            queries.append(history.tolist())
            return self.POLICY_JERKS

        return policy

    def _run(self, *, steps=6, beta=0.0, policy=None, trajectories=1):
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        planner, writer = _ConstantPlanner(action=[1.0]), _FakeDatasetWriter()
        metrics, output = _collect(
            simulator, planner, writer, trajectories=trajectories, steps=steps, beta=beta, policy_action_fn=policy,
            initial_states=[[0.0]], execution_horizon=self.BLOCK,
        )
        return planner, writer, metrics, output

    def test_the_policy_is_asked_once_per_block_and_its_jerks_are_executed_in_order(self) -> None:
        queries: list[list[float]] = []

        _, writer, _, _ = self._run(policy=self._policy(queries))

        self.assertEqual(len(queries), 2)
        states = [frame["observation"][0] for frame in writer.frames]
        np.testing.assert_allclose(states, [0.0, 0.1, 0.3, 0.6, 0.7, 0.9])  # the jerks 0.1, 0.2, 0.3 twice

    def test_the_expert_labels_every_step_of_a_block_the_policy_executes(self) -> None:
        _, writer, _, _ = self._run(policy=self._policy([]))

        for frame in writer.frames:
            _assert_label_is_the_plan_of_the_action(self, frame, 1.0)

    def test_the_expert_plans_at_every_step(self) -> None:
        planner, _, _, _ = self._run(policy=self._policy([]))

        self.assertEqual(planner.call_count, 6)

    def test_the_history_slides_at_every_step_of_a_block(self) -> None:
        queries: list[list[float]] = []

        _, writer, _, _ = self._run(policy=self._policy(queries))

        histories = [frame["history"] for frame in writer.frames]
        np.testing.assert_allclose(histories[:4], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.1, 0.3], [0.1, 0.3, 0.6]])
        self.assertEqual(queries, [histories[0], histories[3]])  # asked at the first step of each block

    def test_a_block_is_the_experts_or_the_policys_as_a_whole_and_both_occur(self) -> None:
        _, writer, _, _ = self._run(steps=6, beta=0.5, policy=self._policy([]), trajectories=8)

        states = np.array([frame["observation"][0] for frame in writer.frames]).reshape(8, 6)
        by_the_expert = np.isclose(np.diff(states, axis=1), 1.0)  # the jerks of steps 1 to 5; the expert's is 1.0
        blocks = [by_the_expert[:, :3], by_the_expert[:, 3:]]
        for block in blocks:
            self.assertTrue(np.all(block == block[:, :1]))  # one actor per block
        self.assertTrue(any(block.any() for block in blocks) and not all(block.all() for block in blocks))

    def test_the_realized_expert_fraction_counts_every_step_of_the_blocks_the_expert_executes(self) -> None:
        _, _, _, output = self._run(steps=6, beta=1.0, policy=self._policy([]))

        self.assertIn("realized_expert_fraction=1.000", output)
        self.assertIn("executed_steps=6", output)

    def test_a_failed_policy_solve_makes_the_block_the_experts_and_the_episode_goes_on(self) -> None:
        calls = 0

        def policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise PlannerSolveError("projector failed")
            return self.POLICY_JERKS

        _, writer, metrics, output = self._run(policy=policy)

        self.assertEqual(calls, 2)  # asked again at the next block
        states = [frame["observation"][0] for frame in writer.frames]
        np.testing.assert_allclose(states, [0.0, 1.0, 2.0, 3.0, 3.1, 3.3])  # the expert's block, then the policy's
        self.assertEqual((metrics.num_episodes, metrics.discarded_episodes, metrics.policy_solve_fallbacks), (1, 0, 1))
        self.assertIn("realized_expert_fraction=0.500", output)

    def test_nothing_is_queried_with_only_the_expert_acting(self) -> None:
        def policy(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            raise AssertionError("the policy must not be queried")

        _, writer, _, _ = self._run(beta=1.0, policy=policy)

        np.testing.assert_allclose([frame["observation"][0] for frame in writer.frames], np.arange(6.0))

    def test_the_policy_is_not_asked_again_after_the_episode_ends_inside_a_block(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=0.25)
        queries: list[list[float]] = []
        writer = _FakeDatasetWriter()

        metrics, _ = _collect(
            simulator, _ConstantPlanner(action=[1.0]), writer, steps=6, beta=0.0, policy_action_fn=self._policy(queries),
            initial_states=[[0.0]], execution_horizon=self.BLOCK,
        )

        self.assertEqual(len(queries), 1)  # the goal is reached with the second jerk
        self.assertEqual((metrics.success_rate, metrics.max_steps), (1.0, 2))


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
        with mock.patch("learning.dagger.collection.torch.manual_seed") as manual_seed:
            evaluate_policy_rollouts(
                simulator=simulator,
                num_episodes=3,
                num_steps=2,
                seed_start=100,
                action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
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
            with mock.patch("learning.dagger.collection.torch.manual_seed", side_effect=seeds_seen[run_idx].append):
                evaluate_policy_rollouts(
                    simulator=simulator,
                    num_episodes=2,
                    num_steps=2,
                    seed_start=7,
                    action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
                )
        self.assertEqual(seeds_seen[0], seeds_seen[1])

    def test_the_action_fn_is_given_the_history_of_the_rollout(self) -> None:
        simulator = _FakeSimulator(collision_threshold=1e9, goal_threshold=1e9)
        seen: list[list[float]] = []

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            seen.append(history.tolist())
            return np.array([[1.0]])

        evaluate_policy_rollouts(
            simulator=simulator, num_episodes=1, num_steps=4, seed_start=0, action_fn=action_fn,
            history_length=HISTORY_LENGTH, initial_states=[[0.0]],
        )

        self.assertEqual(seen, [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])

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
            action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
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
            history_length=HISTORY_LENGTH,
            execution_horizon=1,
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
            action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
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
            action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
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

    def test_timeout_and_solve_failure_are_counted_separately(self) -> None:
        simulator = _FakeSimulator(goal_threshold=1000.0)

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            if float(np.asarray(observation)[0]) == 5.0:
                raise PlannerSolveError("forced failure for test")
            return np.array([[0.0]])

        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=2,
            num_steps=3,
            seed_start=0,
            action_fn=action_fn,
            history_length=HISTORY_LENGTH,
            initial_states=[[0.0], [5.0]],
        )

        assert metrics is not None
        self.assertEqual(metrics.success_rate, 0.0)
        self.assertEqual(metrics.timeout_failures, 1)
        self.assertEqual(metrics.solve_failures, 1)

    def test_the_mean_inference_time_pools_the_calls_that_returned_an_action(self) -> None:
        simulator = _FakeSimulator(goal_threshold=1000.0)
        durations = iter([0.010, 0.030, 0.020])  # the call that fails is not timed

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            if float(np.asarray(observation)[0]) == 5.0:
                raise PlannerSolveError("forced failure for test")
            return np.array([[0.0]])

        clock = mock.patch("learning.dagger.evaluation.time.perf_counter", side_effect=self._clock(durations))
        with clock:
            metrics = evaluate_policy_rollouts(
                simulator=simulator, num_episodes=2, num_steps=3, seed_start=0, action_fn=action_fn,
                history_length=HISTORY_LENGTH, initial_states=[[0.0], [5.0]],
            )

        assert metrics is not None
        self.assertAlmostEqual(metrics.mean_inference_ms, 20.0)

    @staticmethod
    def _clock(durations):
        """A perf_counter whose consecutive (start, end) readings are ``durations`` apart."""
        readings: list[float] = []
        now = 0.0
        for duration in durations:
            readings += [now, now + duration]
            now += duration + 1.0
        return iter(readings + [now] * 20).__next__

    def test_an_evaluation_in_which_every_call_fails_has_no_inference_time(self) -> None:
        def failing(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("forced failure for test")

        metrics = evaluate_policy_rollouts(
            simulator=_FakeSimulator(goal_threshold=1000.0), num_episodes=1, num_steps=2, seed_start=0,
            action_fn=failing, history_length=HISTORY_LENGTH, initial_states=[[0.0]],
        )

        assert metrics is not None
        self.assertIsNone(metrics.mean_inference_ms)

    def test_an_episode_that_touches_an_obstacle_fails_as_a_collision_even_at_the_goal(self) -> None:
        # the state steps to 1.0, which is both at the goal and inside the obstacle
        simulator = _FakeSimulator(goal_threshold=1.0, collision_threshold=1.0)

        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=2,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation, history: np.array([[1.0]]),
            history_length=HISTORY_LENGTH,
            initial_states=[[0.0], [-5.0]],
        )

        assert metrics is not None
        self.assertEqual(metrics.collision_failures, 1)
        self.assertEqual(metrics.timeout_failures, 1)  # the second never gets near
        self.assertEqual(metrics.success_rate, 0.0)

    def test_successful_episodes_are_not_counted_as_any_failure(self) -> None:
        simulator = _FakeSimulator(goal_threshold=0.0)
        metrics = evaluate_policy_rollouts(
            simulator=simulator,
            num_episodes=2,
            num_steps=3,
            seed_start=0,
            action_fn=lambda observation, history: np.array([[0.0]]),
            history_length=HISTORY_LENGTH,
            initial_states=[[0.0], [0.0]],
        )
        assert metrics is not None
        self.assertEqual(metrics.success_rate, 1.0)
        self.assertEqual(metrics.timeout_failures, 0)
        self.assertEqual(metrics.solve_failures, 0)


class RolloutPolicyWithActionFnBlockTests(unittest.TestCase):
    """All the actions an ``action_fn`` call returns are executed one after the other before it is called again."""

    @staticmethod
    def _rollout(action_fn, *, num_steps=6, goal_threshold=1e9, collision_threshold=1e9, call_times=None):
        simulator = _FakeSimulator(collision_threshold=collision_threshold, goal_threshold=goal_threshold)
        outcome = rollout_policy_with_action_fn(
            simulator, np.array([0.0]), num_steps, action_fn, HISTORY_LENGTH, call_times=call_times
        )
        return simulator, outcome

    def test_every_returned_action_is_executed_and_the_calls_are_counted_not_the_steps(self) -> None:
        histories: list[list[float]] = []

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            histories.append(history.tolist())
            return np.array([[1.0], [2.0], [3.0]])

        call_times: list[float] = []
        simulator, outcome = self._rollout(action_fn, call_times=call_times)

        self.assertEqual(outcome, (False, 6, "timeout"))
        self.assertEqual((simulator.step_call_count, len(call_times)), (6, 2))
        self.assertEqual(histories, [[0.0, 0.0, 0.0], [1.0, 3.0, 6.0]])  # the history slid through the block

    def test_the_rollout_stops_at_num_steps_inside_a_block(self) -> None:
        simulator, outcome = self._rollout(lambda observation, history: np.ones((3, 1)), num_steps=4)

        self.assertEqual(outcome, (False, 4, "timeout"))
        self.assertEqual(simulator.step_call_count, 4)

    def test_the_rollout_stops_at_a_collision_inside_a_block(self) -> None:
        simulator, outcome = self._rollout(lambda observation, history: np.ones((3, 1)), collision_threshold=2.0)

        self.assertEqual(outcome, (False, 2, "collision"))
        self.assertEqual(simulator.step_call_count, 2)

    def test_the_rollout_stops_at_the_goal_inside_a_block(self) -> None:
        simulator, outcome = self._rollout(lambda observation, history: np.ones((3, 1)), goal_threshold=2.0)

        self.assertEqual(outcome, (True, 2, None))
        self.assertEqual(simulator.step_call_count, 2)

    def test_a_solve_failure_reports_the_steps_executed_before_the_failing_call(self) -> None:
        calls = 0

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise PlannerSolveError("forced failure for test")
            return np.ones((3, 1))

        call_times: list[float] = []
        with contextlib.redirect_stdout(io.StringIO()):
            _, outcome = self._rollout(action_fn, num_steps=10, call_times=call_times)

        self.assertEqual(outcome, (False, 3, "solve_failure"))
        self.assertEqual(len(call_times), 1)  # the failed call is not timed

    def test_evaluation_metrics_count_the_calls_of_the_inference_time_not_the_steps(self) -> None:
        durations = iter([0.010, 0.030])
        clock = mock.patch("learning.dagger.evaluation.time.perf_counter", side_effect=EvaluatePolicyRolloutsFailureBreakdownTests._clock(durations)
        )

        with clock:
            metrics = evaluate_policy_rollouts(
                simulator=_FakeSimulator(goal_threshold=1e9), num_episodes=1, num_steps=6, seed_start=0,
                action_fn=lambda observation, history: np.ones((3, 1)), history_length=HISTORY_LENGTH, initial_states=[[0.0]],
            )

        assert metrics is not None
        self.assertAlmostEqual(metrics.mean_inference_ms, 20.0)  # two calls for six steps
        self.assertEqual(metrics.max_steps, 6)


class SetupEpisodeStartTests(unittest.TestCase):
    """Collection and evaluation start an episode through the one helper ``_setup_episode_start``."""

    def setUp(self) -> None:
        self.simulator = _FakeSimulator()
        self.baseline = np.array([0.25])
        self.simulator.set_goal(self.baseline)
        self.states = [np.array([0.5]), np.array([0.6])]
        self.goals = [np.array([0.7]), np.array([0.8])]

    def _start(self, index: int, usable: int, goals: list, seed: int = 3):
        return _setup_episode_start(self.simulator, index, usable, self.states, goals, self.baseline, seed)

    def test_a_provided_episode_takes_its_state_and_its_goal_from_the_lists(self) -> None:
        state = self._start(1, usable=2, goals=self.goals)

        np.testing.assert_array_equal(state, [0.6])
        np.testing.assert_array_equal(self.simulator.goal, [0.8])

    def test_a_provided_state_without_goals_draws_the_goal_from_the_seed(self) -> None:
        state = self._start(0, usable=2, goals=[], seed=3)
        first_goal = self.simulator.goal.copy()
        self._start(0, usable=2, goals=[], seed=3)

        np.testing.assert_array_equal(state, [0.5])
        np.testing.assert_array_equal(self.simulator.goal, first_goal)  # the same seed, the same goal
        self.assertNotEqual(float(first_goal[0]), float(self.baseline[0]))

    def test_an_episode_beyond_the_provided_ones_restores_the_baseline_goal_and_samples_a_start(self) -> None:
        self.simulator.set_goal(np.array([9.0]))  # what an earlier explicit episode left behind

        with mock.patch("learning.dagger.episode_starts.sample_initial_state", return_value=np.array([0.1])) as sample:
            state = self._start(2, usable=2, goals=self.goals, seed=5)

        np.testing.assert_array_equal(state, [0.1])
        np.testing.assert_array_equal(self.simulator.goal, self.baseline)
        self.assertEqual(sample.call_args.args[1], 5)


class ApplyExecutionNoiseTests(unittest.TestCase):
    """apply_execution_noise perturbs the executed action with seeded Gaussian noise."""

    def test_zero_noise_returns_the_action_unchanged(self) -> None:
        action = np.array([0.3])
        self.assertIs(apply_execution_noise(action, 0.0), action)

    def test_noise_perturbs_the_action_and_is_reproducible_for_the_same_rng_seed(self) -> None:
        action = np.array([0.3])

        first = apply_execution_noise(action, 0.1, np.random.default_rng(5))
        second = apply_execution_noise(action, 0.1, np.random.default_rng(5))

        np.testing.assert_array_equal(first, second)
        self.assertNotEqual(first[0], action[0])


class _StubPolicy:
    """Records the policy input it is queried with and returns a fixed action tensor."""

    def __init__(self, output: torch.Tensor, execution_horizon: int = 1) -> None:
        self.output = output
        self.execution_horizon = execution_horizon
        self.received: dict[str, torch.Tensor] | None = None
        self.queried_in_inference_mode: bool | None = None

    def select_action(self, observation_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        self.received = dict(observation_dict)
        self.queried_in_inference_mode = torch.is_inference_mode_enabled()
        return self.output


class BuildPolicyActionsTests(unittest.TestCase):
    """build_policy_actions queries the policy once, on a batch of one built from simulator.policy_observation, and hands
    back the first ``execution_horizon`` rows of the predicted action chunk.
    """

    HISTORY = np.array([0.1, 0.3, 0.5])

    def test_returns_the_first_execution_horizon_actions_of_the_chunk(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        chunk = torch.tensor([[[0.5, 0.25], [3.0, 4.0], [5.0, 6.0]]])  # (batch, horizon, nu)

        actions = build_policy_actions(
            simulator, _StubPolicy(chunk, execution_horizon=2), np.array([0.5]), self.HISTORY, torch.device("cpu")
        )

        self.assertEqual(actions.shape, (2, 2))
        np.testing.assert_allclose(actions, [[0.5, 0.25], [3.0, 4.0]])

    def test_an_execution_horizon_of_one_returns_a_single_row(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)

        actions = build_policy_actions(
            simulator, _StubPolicy(torch.ones(1, 3, 2)), np.array([0.5]), self.HISTORY, torch.device("cpu")
        )

        self.assertEqual(actions.shape, (1, 2))

    def test_policy_input_is_a_batch_of_one_with_exactly_the_three_observation_keys(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        simulator.set_goal(np.array([0.25]))
        policy = _StubPolicy(torch.zeros(1, 3, 2))

        build_policy_actions(simulator, policy, np.array([0.5]), self.HISTORY, torch.device("cpu"))

        assert policy.received is not None
        self.assertEqual(
            set(policy.received), {"observation.environment_state", "observation.state", "observation.history"}
        )
        expected = {
            "observation.environment_state": [[0.25]],
            "observation.state": [[0.5]],
            "observation.history": [self.HISTORY.tolist()],
        }
        for name, values in expected.items():
            self.assertEqual(policy.received[name].dtype, torch.float32, name)
            np.testing.assert_allclose(policy.received[name].numpy(), values, err_msg=name)
        self.assertTrue(policy.queried_in_inference_mode)

    def test_malformed_policy_output_is_rejected(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        # Not a batch of chunks, too many dimensions, or a chunk without an action
        for output in (torch.zeros(2), torch.zeros(1, 2), torch.zeros(1, 2, 3, 4), torch.zeros(1, 0, 2)):
            with self.subTest(shape=tuple(output.shape)):
                with self.assertRaises(ValueError):
                    build_policy_actions(
                        simulator, _StubPolicy(output), np.array([0.5]), self.HISTORY, torch.device("cpu")
                    )

    def test_a_chunk_shorter_than_the_execution_horizon_is_rejected(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)

        with self.assertRaises(ValueError):
            build_policy_actions(
                simulator, _StubPolicy(torch.zeros(1, 2, 2), execution_horizon=3), np.array([0.5]), self.HISTORY, torch.device("cpu")
            )

    def test_the_policy_action_function_is_the_same_query_on_the_cpu(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=1.0)
        chunk = torch.tensor([[[0.5], [1.5], [2.5]]])

        actions = _policy_action_fn(simulator, _StubPolicy(chunk, execution_horizon=2))(np.array([0.5]), self.HISTORY)

        np.testing.assert_allclose(actions, [[0.5], [1.5]])


class DefaultFrameBuilderTests(unittest.TestCase):
    """Without a frame_builder, collect_dagger_rollouts formats frames with
    data_utils.format_dataset_frame -- ONE mapping per step (the arm is a
    single robot) -- and adds the LeRobot ``task`` string: the simulator's task_description.
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
            history_length=HISTORY_LENGTH,
            execution_horizon=1,
            policy_action_fn=None,
            initial_states=[[0.0]],
            goal_states=None,
            **kwargs,
        )
        return writer

    def test_frames_are_the_formatted_dataset_frames_with_the_simulators_task_description(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        simulator.task_description = "reach fake target"

        writer = self._collect(simulator)

        self.assertEqual(len(writer.frames), 1)
        frame = writer.frames[0]
        self.assertEqual(
            set(frame),
            {"observation.environment_state", "observation.state", "observation.history", "action", "task"},
        )
        self.assertEqual(frame["task"], "reach fake target")
        np.testing.assert_allclose(frame["observation.state"], [0.0])
        np.testing.assert_allclose(frame["observation.history"], [0.0] * HISTORY_LENGTH)  # the start repeated
        # The first prediction_horizon rows of the expert's plan (the states 1 and 2 steps ahead), flattened row by row
        np.testing.assert_allclose(frame["action"], [1.0, 2.0])
        self.assertEqual(writer.episode_boundaries, [1])

    def test_a_frame_builder_must_return_a_single_mapping(self) -> None:
        simulator = _FakeSimulator(collision_threshold=10.0, goal_threshold=100.0)
        with self.assertRaises(TypeError):
            self._collect(
                simulator,
                frame_builder=lambda observation, history, positions: [_frame_builder(observation, history, positions)],
            )


class RolloutsWithTheArmTests(unittest.TestCase):
    """The same entry points on the real FrankaSystem and its OMPL + TOTG expert instead of the 1-D fake: a few steps
    from a start near the ready pose towards a nearby goal pose, so this stays fast and deterministic -- what it checks is
    the interfaces (frame schema, the 12-dim goal, the history, the policy input), not task success.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.initial_state = rest_state(cls.simulator)
        cls.goal = pose_goal(cls.simulator, cls.initial_state[: cls.simulator.n] + 0.1)

    def _collect(self, simulator: FrankaSystem, config: dict, writer: _FakeDatasetWriter, **kwargs) -> object:
        settings = dict(
            trajectories_per_iteration=1,
            steps_per_trajectory=3,
            action_noise_std=0.0,
            action_noise_seed=0,
            initial_state_seed=0,
            expert_mixing_beta=1.0,
            prediction_horizon=PREDICTION_HORIZON,
            history_length=HISTORY_LENGTH,
            execution_horizon=EXECUTION_HORIZON,
            policy_action_fn=None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return collect_dagger_rollouts(
                simulator=simulator,
                expert_planner=PlannerFactory.create(config["expert"], simulator, config),
                dataset_writer=writer,
                **{**settings, **kwargs},
            )

    def test_collect_dagger_rollouts_writes_frames_matching_the_dataset_schema(self) -> None:
        simulator, n = self.simulator, self.simulator.n
        writer = _FakeDatasetWriter()

        metrics = self._collect(
            simulator, self.config, writer, initial_states=[self.initial_state.tolist()], goal_states=[self.goal.tolist()]
        )

        self.assertEqual(metrics.num_episodes, 1)
        self.assertEqual(len(writer.frames), 3)
        self.assertEqual(writer.episode_boundaries, [3])
        features = build_dataset_features(simulator, PREDICTION_HORIZON, HISTORY_LENGTH)
        self.assertEqual(set(features), {"observation.environment_state", "observation.state", "observation.history", "action"})
        for frame in writer.frames:
            self.assertEqual(set(frame), set(features) | {"task"})
            for name, feature in features.items():
                self.assertEqual(frame[name].shape, tuple(feature["shape"]), name)
            self.assertEqual(frame["task"], simulator.task_description)
            # the goal in the TCP frame: the goal comes back from it and the state
            np.testing.assert_allclose(
                simulator.observation_from_policy(
                    frame["observation.environment_state"].astype(float), frame["observation.state"].astype(float)
                )[: simulator.goal_dim],
                self.goal,
                atol=1e-4,
            )
        np.testing.assert_allclose(writer.frames[0]["observation.state"], self.initial_state, atol=1e-6)

    def test_the_history_of_a_frame_ends_at_its_state_and_slides_with_the_rollout(self) -> None:
        simulator, n = self.simulator, self.simulator.n
        writer = _FakeDatasetWriter()

        self._collect(
            simulator, self.config, writer, initial_states=[self.initial_state.tolist()], goal_states=[self.goal.tolist()]
        )

        histories = [frame["observation.history"].astype(float).reshape(HISTORY_LENGTH, n) for frame in writer.frames]
        np.testing.assert_allclose(histories[0], np.tile(self.initial_state[:n], (HISTORY_LENGTH, 1)), atol=1e-6)
        for frame, history in zip(writer.frames, histories):
            np.testing.assert_allclose(history[-1], frame["observation.state"][:n], atol=1e-6)
        for previous, history in zip(histories, histories[1:]):
            np.testing.assert_allclose(history[:-1], previous[1:], atol=1e-6)

    def test_the_label_of_a_frame_is_the_experts_positions_plan_that_the_arm_follows(self) -> None:
        simulator, n = self.simulator, self.simulator.n
        writer = _FakeDatasetWriter()

        self._collect(
            simulator, self.config, writer, initial_states=[self.initial_state.tolist()], goal_states=[self.goal.tolist()]
        )

        for step, frame in enumerate(writer.frames[:-1]):
            plan = frame["action"].reshape(PREDICTION_HORIZON, n)
            self.assertTrue(np.all(np.isfinite(plan)))
            # the expert's step tracks the first row of its plan
            np.testing.assert_allclose(plan[0], writer.frames[step + 1]["observation.state"][:n], atol=0.05)

    def test_evaluate_policy_rollouts_queries_the_policy_through_build_policy_actions(self) -> None:
        simulator = self.simulator
        policy = PolicyFactory.create(
            PolicySpec(
                policy_type="flow", action_dim=simulator.n, cond_dim=simulator.environment_dim + HISTORY_LENGTH * simulator.n,
                history_length=HISTORY_LENGTH, execution_horizon=EXECUTION_HORIZON, prediction_horizon=4, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=2,
                action_scale=tuple(simulator.position_scale.tolist()),
            ),
            simulator,
            self.config,
        ).eval()
        actions: list[np.ndarray] = []
        histories: list[np.ndarray] = []

        def action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            histories.append(history)
            action = build_policy_actions(simulator, policy, observation, history, torch.device("cpu"))
            actions.append(action)
            return action

        with contextlib.redirect_stdout(io.StringIO()):
            metrics = evaluate_policy_rollouts(
                simulator=simulator,
                num_episodes=2,
                num_steps=2,
                seed_start=0,
                action_fn=action_fn,
                history_length=HISTORY_LENGTH,
                reset_fn=policy.reset,
                initial_states=[self.initial_state.tolist()],
                goal_states=[self.goal.tolist()],
            )

        assert metrics is not None
        self.assertEqual(metrics.num_episodes, 2)
        # The first episode is sourced from the explicit initial state/goal pose, the second from
        # the simulator's own random sampling.
        self.assertEqual((metrics.config_num_episodes, metrics.random_num_episodes), (1, 1))
        failures = metrics.timeout_failures + metrics.solve_failures
        self.assertEqual(failures + round(metrics.success_rate * metrics.num_episodes), metrics.num_episodes)
        self.assertGreaterEqual(len(actions), 2)
        for action in actions:
            self.assertEqual(action.shape, (EXECUTION_HORIZON, simulator.nu))
            self.assertTrue(np.all(np.isfinite(action)))
        np.testing.assert_allclose(histories[0], np.tile(self.initial_state[: simulator.n], HISTORY_LENGTH), atol=1e-6)

    def test_parallel_workers_produce_the_same_frames_and_metrics_as_one_process(self) -> None:
        config = {**self.config, "randomize_goal": True}
        worker_spec = RolloutWorkerSpec(config=config, need_expert=True)
        writers = [_FakeDatasetWriter(), _FakeDatasetWriter()]
        metrics_by_workers = []
        for num_workers, writer in zip((1, 2), writers):
            metrics_by_workers.append(
                self._collect(
                    FrankaSystem(config), config, writer,
                    trajectories_per_iteration=2, steps_per_trajectory=2, num_workers=num_workers, worker_spec=worker_spec,
                )
            )

        self.assertEqual(metrics_by_workers[0], metrics_by_workers[1])
        self.assertEqual(writers[0].episode_boundaries, writers[1].episode_boundaries)
        for frame_single, frame_multi in zip(writers[0].frames, writers[1].frames):
            for key in ("observation.environment_state", "observation.state", "observation.history", "action"):
                np.testing.assert_allclose(frame_single[key], frame_multi[key], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
