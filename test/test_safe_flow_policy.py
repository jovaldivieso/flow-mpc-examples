from __future__ import annotations

import os
import sys
import unittest
import warnings
from unittest import mock

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from systems.franka_system import Q_READY, FrankaSystem
from learning.dagger.policy_action import build_policy_actions
from learning.models.flow_policy import FlowPolicy
from learning.models.policy_factory import PolicyFactory, PolicySpec
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.planner import PlannerSolveError
from planning.planner_config import ProjectionConfig
from planning.casadi_projector import CasadiTrajectoryProjector, Projection
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_config.yaml")

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
    "pos_tol": 0.002,
    "rot_tol": 0.20,
    "vel_tol": 0.02,
}
HORIZON = 10
HISTORY = 3
EXECUTION = 2


def _validated_config(**overrides) -> dict:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


# Two goal poses (12,), of two configurations: which one the arm holds must never matter to the projection.
GOAL = list(pose_goal(FrankaSystem(_validated_config()), np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.5])))
OTHER_GOAL = list(pose_goal(FrankaSystem(_validated_config()), np.array([0.2, -0.7, 0.1, -1.8, -0.2, 1.3, 0.8])))


def _build_simulator(goal: list[float] = GOAL, **overrides) -> FrankaSystem:
    """The arm with an explicit, non-randomized goal pose."""
    return FrankaSystem(_validated_config(goal=goal, **overrides))


def _rest_state(simulator: FrankaSystem) -> np.ndarray:
    """Ready pose with the wrist turned by 0.3 rad, at rest."""
    q = Q_READY.copy()
    q[6] += 0.3
    return np.concatenate([q, np.zeros(2 * simulator.n)])


def _rest_observation_dict(simulator: FrankaSystem) -> dict[str, torch.Tensor]:
    """The policy input (a batch of one) for the arm at rest in the ready pose."""
    return _policy_input(simulator, _rest_state(simulator))


def _policy_input(simulator: FrankaSystem, state: np.ndarray) -> dict[str, torch.Tensor]:
    """The policy input (a batch of one) of ``state`` whose history stands still at its configuration."""
    policy_observation = simulator.policy_observation(simulator.observe(state), np.tile(state[: simulator.n], HISTORY))
    return {
        name: torch.as_tensor(np.expand_dims(value, axis=0), dtype=torch.float32)
        for name, value in policy_observation.items()
    }


class _NoGuidance:
    """Hands the proposal on as it is: the tests of the projection wiring need what the projector is handed unmoved."""

    def guide(self, environment_state, state, proposal):
        return proposal

    def reset(self) -> None:
        pass


def _policy_spec(simulator, policy_type: str = "safeflow", **options) -> PolicySpec:
    """The flow conditions on the environment_state (the goal in the TCP frame) and the history of positions; its chunk
    is positions, scaled by half the range of their joints exactly as DaggerTrainer passes it."""
    return PolicySpec(
        policy_type=policy_type, action_dim=simulator.n, cond_dim=simulator.environment_dim + simulator.n * HISTORY,
        history_length=HISTORY, execution_horizon=EXECUTION, prediction_horizon=HORIZON, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=2,
        action_scale=tuple(simulator.position_scale.tolist()), **options,
    )


def _build_safe_flow_policy(simulator, planner_config=None, *, guided=False, **policy_options) -> SafeFlowMPCPolicy:
    # planner_config defaults to the validated expert config, as it does in the trainer. The tests of the projection
    # wiring run unguided: the flow's guidance (test_pose_guidance.py) would move what the projector is handed.
    policy = PolicyFactory.create(
        _policy_spec(simulator, **policy_options), simulator, _validated_config() if planner_config is None else planner_config
    )
    assert isinstance(policy, SafeFlowMPCPolicy)
    if not guided:
        policy.inner_policy.guidance = _NoGuidance()
    return policy


def _hold_chunk(state: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """A chunk of positions (horizon, n) that stays in the configuration of ``state``: from rest, it is safe."""
    return np.tile(state[:7], (horizon, 1))


def _zero_the_network(policy: SafeFlowMPCPolicy) -> None:
    """With every weight and bias zeroed the predicted velocity is identically 0: the Euler iterate never leaves
    its start, so what the projector is handed is exactly the integration's initial chunk."""
    with torch.no_grad():
        for param in policy.inner_policy.velocity_net.unet.parameters():
            param.zero_()


class SharedSimulatorTests(unittest.TestCase):
    """The projector of a safe flow policy shares the simulator it was built for (no private copy): it reads no goal of
    it, because every constraint and cost it enforces -- dynamics, action/state bounds, terminal velocity, obstacles --
    is built from the system and from the state of the observation, which carries no goal.
    """

    def test_the_projector_uses_the_simulator_it_was_given(self) -> None:
        simulator = _build_simulator(goal=GOAL)

        policy = _build_safe_flow_policy(simulator)

        self.assertIs(policy.projector.sim, simulator)
        self.assertFalse(hasattr(policy, "simulator"))  # the policy only hands the projector the state it observes

    def test_construction_neither_copies_nor_changes_the_simulator(self) -> None:
        simulator = _build_simulator(goal=GOAL)
        kinematics_before = simulator.kinematics

        with mock.patch("copy.deepcopy", side_effect=AssertionError("the simulator must not be deep-copied")):
            _build_safe_flow_policy(simulator)

        np.testing.assert_allclose(simulator.goal, GOAL)
        self.assertIs(simulator.kinematics, kinematics_before)

    def test_select_action_is_invariant_to_which_goal_the_simulator_holds(self) -> None:
        # The actual mathematical property the whole redesign relies on,
        # checked end to end (test_casadi_projector.py checks it at the
        # projector level): invert_obs drops the goal, so the projection
        # must not depend on it -- policies with the same weights, built
        # from simulators holding different goals, must act identically on
        # the same observation.
        observation_dict = _rest_observation_dict(_build_simulator(goal=GOAL))
        actions = []
        for goal in (GOAL, OTHER_GOAL):
            torch.manual_seed(0)  # same weights
            policy = _build_safe_flow_policy(_build_simulator(goal=goal))
            torch.manual_seed(1)  # same flow noise draw
            actions.append(policy.select_action(observation_dict))

        torch.testing.assert_close(actions[0], actions[1], atol=1e-5, rtol=0.0)


class TerminalConditionSupportTests(unittest.TestCase):
    """The arm has a velocity state, so it gets a soft terminal-velocity cost
    instead of a hard equality constraint: a hard constraint is only
    satisfiable when the horizon leaves no margin beyond the system's own
    worst-case braking distance for anything else (u_ref tracking, collision
    avoidance) sharing that same horizon, which made the projector stall
    rather than progress even in ordinary scenarios -- so a too-short horizon
    here just weakens the soft cost, it doesn't risk outright infeasibility.
    """

    def test_short_horizon_for_a_fast_moving_arm_constructs_and_solves(self) -> None:
        # The shortest horizon (the current position and one planned step), started from a fast motion that the
        # arm cannot brake from within it, must not raise: the terminal velocity term is a soft cost, so it's
        # always solvable, just weakly enforced from far-from-rest states.
        horizon = 2
        simulator = _build_simulator()
        projector = CasadiTrajectoryProjector(simulator, ProjectionConfig.from_mapping(_validated_config()), horizon=horizon)
        state0 = _rest_state(simulator)
        state0[simulator.n] = 2.0
        state0[simulator.n + 4] = 3.0
        # sanity: braking takes more than the single planned step
        self.assertGreater(np.max(state0[simulator.n : 2 * simulator.n] / (simulator.max_acceleration * simulator.dt)), 1.0)
        projection = projector.project(state0, _hold_chunk(state0, horizon))
        self.assertEqual(projection.positions.shape, (horizon, simulator.n))
        self.assertEqual(projection.actions.shape, (horizon, simulator.nu))


class PolicySolveFailureRecoveryTests(unittest.TestCase):
    """A raised PlannerSolveError from a policy's projector must not crash a
    whole rollout/evaluation run over one cold-start hiccup with no safe
    trajectory to fall back on. Callers treat it as an episode-ending
    failure, exactly like an expert-side PlannerSolveError -- verified here
    against the generic evaluation-side entry point (rollout_policy_with_
    action_fn); the collection-side entry point (collect_dagger_rollouts)
    falls back to that step's already-computed expert action instead, since
    unlike evaluation it always has the expert available.
    """

    def test_rollout_policy_with_action_fn_ends_episode_on_policy_solve_failure(self) -> None:
        from learning.dagger.evaluation import rollout_policy_with_action_fn

        simulator = _build_simulator()
        initial_state = _rest_state(simulator)

        def failing_action_fn(observation: np.ndarray, history: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("forced failure for test")

        reached_goal, steps_taken, failure_reason = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=10,
            action_fn=failing_action_fn,
            history_length=HISTORY,
        )
        self.assertFalse(reached_goal)
        # failing_action_fn raises on the very first call, before
        # simulator.step() ever runs -- zero steps were actually executed.
        self.assertEqual(steps_taken, 0)
        self.assertEqual(failure_reason, "solve_failure")

    def _rollout_with_a_failing_projector(self):
        from learning.dagger.evaluation import rollout_policy_with_action_fn

        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        policy.projector._graph._solve_once = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        initial_state = _rest_state(simulator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = rollout_policy_with_action_fn(
                simulator=simulator,
                initial_state=initial_state,
                num_steps=40,
                action_fn=lambda observation, history: build_policy_actions(simulator, policy, observation, history, torch.device("cpu")),
                history_length=HISTORY,
                reset_fn=policy.reset,
            )
        return result, policy.projector.config.max_consecutive_fallbacks

    def test_projector_failures_at_rest_hold_still_for_the_fallback_budget_and_then_end_the_episode(self) -> None:
        (reached_goal, steps_taken, failure_reason), budget = self._rollout_with_a_failing_projector()

        self.assertFalse(reached_goal)
        self.assertEqual(steps_taken, budget * EXECUTION)  # the arm held still for every tick of the budget
        self.assertEqual(failure_reason, "solve_failure")

    def test_every_action_starts_a_new_control_tick_of_the_projector(self) -> None:

        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        with mock.patch.object(policy.projector, "begin_tick", wraps=policy.projector.begin_tick) as begin_tick:
            for _ in range(2):
                build_policy_actions(
                    simulator, policy, simulator.observe(_rest_state(simulator)), np.tile(Q_READY, HISTORY), torch.device("cpu")
                )

        self.assertEqual(begin_tick.call_count, 2)


class ExecutedActionsTests(unittest.TestCase):
    def test_the_first_execution_horizon_jerks_of_the_chunk_are_returned(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        state = _rest_state(simulator)
        calls = SelectActionProjectionTests._recording_projector(policy)

        actions = build_policy_actions(simulator, policy, simulator.observe(state), np.tile(state[: simulator.n], HISTORY), torch.device("cpu"))

        self.assertEqual(actions.shape, (EXECUTION, simulator.nu))
        np.testing.assert_allclose(actions, calls[-1][2].actions[:EXECUTION], atol=1e-4)


class SelectActionBatchSizeGuardTests(unittest.TestCase):
    """select_action serves the one row of the one system its projector belongs to (build_policy_actions, the only real
    caller, always builds exactly one). A batch of any other size must be rejected explicitly rather than silently
    leaving every row past the first unprojected (too large a batch) or crashing with an opaque IndexError (too small).
    """

    @staticmethod
    def _build_observation_dict(simulator) -> dict[str, torch.Tensor]:
        state = _rest_state(simulator)
        return _policy_input(simulator, state)

    def test_single_row_batch_succeeds(self) -> None:
        torch.manual_seed(0)
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        observation_dict = self._build_observation_dict(simulator)

        action = policy.select_action(observation_dict)
        self.assertEqual(tuple(action.shape), (1, HORIZON, simulator.nu))

    def test_mismatched_batch_size_raises_instead_of_silently_skipping_the_projection(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        observation_dict = self._build_observation_dict(simulator)
        doubled = {name: tensor.repeat(2, 1) for name, tensor in observation_dict.items()}
        emptied = {name: tensor[:0] for name, tensor in observation_dict.items()}

        for batch_size, batch in ((2, doubled), (0, emptied)):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    policy.select_action(batch)


class PolicyFactorySafeFlowTests(unittest.TestCase):
    """PolicyFactory.create(a safeflow spec, ...) wires an inner FlowPolicy and a CasadiTrajectoryProjector into one
    single-arm SafeFlowMPCPolicy.
    """

    def test_requires_the_planner_config(self) -> None:
        simulator = _build_simulator()
        with self.assertRaises(ValueError):
            PolicyFactory.create(_policy_spec(simulator), simulator)

    def test_projector_is_sized_by_the_prediction_horizon_and_keeps_the_arms_rest_indices(self) -> None:
        policy = _build_safe_flow_policy(_build_simulator())

        self.assertEqual(policy.projector.N, HORIZON)
        self.assertEqual(policy.projector.tracked_horizon, HORIZON)
        self.assertTrue(policy.projector.safe)
        self.assertEqual(policy.projector.tick_steps, EXECUTION)
        self.assertEqual(policy.history_length, HISTORY)
        self.assertEqual(policy.execution_horizon, EXECUTION)
        self.assertIsInstance(policy.projector.sim, FrankaSystem)
        # What the terminal come-to-rest cost reads: the arm's joint velocities and accelerations, state = [q, v, a].
        self.assertEqual(policy.projector.rest_idx, list(range(7, 21)))

    def test_projector_reads_its_tuning_from_the_validated_expert_config(self) -> None:
        planner_config = _validated_config(
            terminal_velocity_weight=42.0, projection_jerk_weight=0.5, max_consecutive_fallbacks=3,
        )
        policy = _build_safe_flow_policy(_build_simulator(), planner_config=planner_config)

        self.assertEqual(policy.projector.config.terminal_velocity_weight, 42.0)
        self.assertEqual(policy.projector.config.jerk_weight, 0.5)
        self.assertEqual(policy.projector.config.max_consecutive_fallbacks, 3)


class SelectActionProjectionTests(unittest.TestCase):
    """select_action integrates the flow over the chunk of positions: every Euler step hands the projector the observed
    state and the current proposal, the projected positions are the flow's next state, and the returned chunk is the
    jerks of the last projection."""

    @staticmethod
    def _recording_projector(policy: SafeFlowMPCPolicy) -> list[tuple[np.ndarray, np.ndarray, Projection]]:
        real_project = policy.projector.project
        calls: list[tuple[np.ndarray, np.ndarray, Projection]] = []

        def recording_project(state: np.ndarray, q_ref: np.ndarray) -> Projection:
            projection = real_project(state, q_ref)
            calls.append((state.copy(), q_ref.copy(), projection))
            return projection

        policy.projector.project = recording_project
        return calls

    def test_every_euler_step_projects_the_observed_state_and_the_proposed_chunk(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        observation_dict = _rest_observation_dict(simulator)
        # With a zero velocity the Euler iterate never leaves its start, the projection of the coasting arm (the first
        # call): the first flow reference must be exactly that, and since each projection is the flow's next state, the
        # second reference must be the positions of the first flow projection.
        _zero_the_network(policy)
        calls = self._recording_projector(policy)

        action = policy.select_action(observation_dict)

        steps = policy.inner_policy.velocity_net.num_inference_steps
        self.assertEqual(len(calls), steps + 1)
        expected_state = _rest_state(simulator)  # the projector gets the state the policy saw
        for state, q_ref, _ in calls:
            self.assertEqual(state.shape, (simulator.nx,))
            np.testing.assert_allclose(state, expected_state, atol=1e-5)
            self.assertEqual(q_ref.shape, (HORIZON, simulator.n))
        np.testing.assert_allclose(calls[1][1], calls[0][2].positions, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(calls[2][1], calls[1][2].positions, rtol=1e-5, atol=1e-5)
        # The returned chunk is the jerks of the last projection, as (batch, horizon, nu), within the action limits
        last_projection = calls[-1][2]
        self.assertEqual(tuple(action.shape), (1, HORIZON, simulator.nu))
        np.testing.assert_allclose(action[0].numpy(), last_projection.actions, atol=1e-4)
        self.assertTrue(np.all(np.abs(last_projection.actions) <= simulator.max_action + 1e-6))

    def test_the_projector_is_handed_the_guided_proposal(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator, guided=True)
        _zero_the_network(policy)
        calls = self._recording_projector(policy)
        goal_ward = np.full((HORIZON, simulator.n), 0.25)

        with mock.patch.object(policy.inner_policy.guidance, "guide", return_value=goal_ward) as guide:
            policy.select_action(_rest_observation_dict(simulator))

        self.assertEqual(guide.call_count, policy.inner_policy.velocity_net.num_inference_steps)
        np.testing.assert_allclose(calls[1][1], goal_ward)  # the first call projects the start, the next the guided proposal

    def test_the_projected_positions_are_safe_whatever_the_proposal(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        calls = self._recording_projector(policy)
        noise = np.random.default_rng(0).normal(size=(HORIZON, simulator.n))

        with mock.patch.object(policy, "_initial_chunk", return_value=noise):  # proposals far from anything safe
            policy.select_action(_rest_observation_dict(simulator))

        lower, upper = simulator.state_lower_bounds[: simulator.n], simulator.state_upper_bounds[: simulator.n]
        for _, _, projection in calls:
            self.assertTrue(np.all(projection.positions >= lower - 1e-6) and np.all(projection.positions <= upper + 1e-6))

    def test_reset_clears_the_projectors_warm_start(self) -> None:
        # Without this, a new episode's first solve would warm start from
        # the previous episode's unrelated final trajectory.
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        state = _rest_state(simulator)
        policy.projector.project(state, _hold_chunk(state))
        self.assertIsNotNone(policy.projector._prev_U_sol)

        with mock.patch.object(policy.inner_policy, "reset") as inner_reset:
            policy.reset()

        inner_reset.assert_called_once_with()
        self.assertIsNone(policy.projector._prev_U_sol)

    def test_compute_loss_delegates_to_the_inner_policy(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        observation = {name: value.repeat(4, 1) for name, value in _rest_observation_dict(simulator).items()}
        positions = torch.randn(4, HORIZON, simulator.n)

        torch.manual_seed(0)
        expected = policy.inner_policy.compute_loss(observation, positions)
        torch.manual_seed(0)
        torch.testing.assert_close(policy.compute_loss(observation, positions), expected)

    def test_the_inner_policy_has_no_tracker_and_the_safe_policy_executes_it(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)

        self.assertIsNone(policy.inner_policy.tracker)
        with self.assertRaises(RuntimeError):
            policy.inner_policy.select_action(_rest_observation_dict(simulator))
        self.assertEqual(tuple(policy.select_action(_rest_observation_dict(simulator)).shape), (1, HORIZON, simulator.nu))

    def test_the_paths_reach_the_inner_policy_loss(self) -> None:
        policy = _build_safe_flow_policy(_build_simulator())
        observation = {name: value.repeat(2, 1) for name, value in _rest_observation_dict(_build_simulator()).items()}
        positions, paths = torch.zeros(2, HORIZON, 7), torch.ones(2, 3, HORIZON, 7)

        with mock.patch.object(policy.inner_policy, "compute_loss", return_value=torch.zeros(())) as compute_loss:
            policy.compute_loss(observation, positions, paths)

        self.assertIs(compute_loss.call_args.args[2], paths)

    def test_the_history_length_and_the_execution_horizon_are_the_ones_of_the_inner_policy(self) -> None:
        policy = _build_safe_flow_policy(_build_simulator())

        self.assertEqual(policy.history_length, policy.inner_policy.history_length)
        self.assertEqual(policy.execution_horizon, policy.inner_policy.execution_horizon)
        self.assertEqual((policy.history_length, policy.execution_horizon), (HISTORY, EXECUTION))


class FlowInitializationTests(unittest.TestCase):
    """The integration starts from the previous tick's safe positions (or, on the first tick, from the safe positions
    nearest to the arm coasting) instead of from Gaussian noise."""

    STEPS = 2  # num_inference_steps of _build_safe_flow_policy

    def _policy_and_calls(self, simulator):
        policy = _build_safe_flow_policy(simulator)
        _zero_the_network(policy)
        return policy, SelectActionProjectionTests._recording_projector(policy)

    def test_the_first_tick_starts_from_the_projection_of_the_coasting_arm(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator)
        state = _rest_state(simulator)

        policy.select_action(_rest_observation_dict(simulator))

        self.assertEqual(len(calls), self.STEPS + 1)  # the projection of the start, then one per flow step
        np.testing.assert_allclose(calls[0][1], _hold_chunk(state), atol=1e-5)  # at rest, coasting stays in place
        np.testing.assert_allclose(calls[1][1], calls[0][2].positions, atol=1e-5)  # which is where the flow starts

    def test_a_moving_arm_starts_its_first_tick_from_safe_braking_positions(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator)
        state = _rest_state(simulator)
        state[simulator.n : 2 * simulator.n] = 0.5 * simulator.state_upper_bounds[simulator.n : 2 * simulator.n]

        policy.select_action(_policy_input(simulator, state))

        coasting = simulator.rollout_positions(state, np.zeros((HORIZON, simulator.nu)))[1:]
        np.testing.assert_allclose(calls[0][1], coasting, atol=1e-5)
        start = calls[0][2]
        braked = np.abs(simulator.rollout_states(state, start.actions)[-1][simulator.n :]).max()
        coasting_speed = np.abs(simulator.rollout_states(state, np.zeros_like(start.actions))[-1][simulator.n :]).max()
        self.assertGreater(np.abs(start.actions).max(), 0.0)
        self.assertLess(braked, 0.5 * coasting_speed)  # the start slows the arm down instead of coasting on

    def test_later_ticks_start_from_the_previous_safe_positions_one_execution_horizon_later(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator)
        state = _rest_state(simulator)
        first_chunk = policy.select_action(_policy_input(simulator, state))[0].numpy()
        positions = calls[-1][2].positions
        guess = policy.projector.next_chunk_guess()
        self.assertEqual(guess.shape, (HORIZON, simulator.n))
        np.testing.assert_allclose(guess[: HORIZON - EXECUTION], positions[EXECUTION:], atol=1e-9)  # the plan EXECUTION periods on
        np.testing.assert_allclose(guess[HORIZON - EXECUTION :], np.tile(positions[-1], (EXECUTION, 1)), atol=1e-9)  # the end held
        next_state = state
        for jerk in first_chunk[:EXECUTION]:  # execute the policy's actions, as the rollout does
            next_state = simulator.step(next_state, jerk)
        calls.clear()

        policy.select_action(_policy_input(simulator, next_state))

        self.assertEqual(len(calls), self.STEPS)  # no extra projection: the previous positions are the start
        np.testing.assert_allclose(calls[0][1], guess, atol=1e-5)

    def test_reset_returns_to_the_coasting_start(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator)
        state = _rest_state(simulator)
        policy.projector.project(state, _hold_chunk(state))
        policy.reset()
        calls.clear()

        policy.select_action(_policy_input(simulator, state))

        self.assertEqual(len(calls), self.STEPS + 1)
        np.testing.assert_allclose(calls[0][1], _hold_chunk(state), atol=1e-5)



class StateDictInterchangeTests(unittest.TestCase):
    """'flow' and 'safeflow' checkpoints are interchangeable: a SafeFlowMPCPolicy's state dict is its inner
    FlowPolicy's behind an ``inner_policy.`` prefix (the projector is no nn.Module, so it contributes nothing), and
    ``load_network_state_dict`` loads either kind of checkpoint into either kind of policy through PyTorch's own,
    strict, ``load_state_dict`` -- which neither policy overrides.
    """

    def _policies(self):
        simulator = _build_simulator()
        safe_policy = _build_safe_flow_policy(simulator)
        flow_policy = PolicyFactory.create(_policy_spec(simulator, "flow"), simulator, _validated_config())
        return safe_policy, flow_policy

    def test_neither_policy_overrides_load_state_dict(self) -> None:
        self.assertIs(SafeFlowMPCPolicy.load_state_dict, torch.nn.Module.load_state_dict)
        self.assertIs(FlowPolicy.load_state_dict, torch.nn.Module.load_state_dict)

    def test_flow_and_safeflow_state_dicts_load_into_each_other(self) -> None:
        safe_policy, flow_policy = self._policies()
        flow_state_dict = {name: value.clone() for name, value in flow_policy.state_dict().items()}
        safe_state_dict = {name: value.clone() for name, value in safe_policy.state_dict().items()}
        self.assertEqual(set(safe_state_dict), {f"inner_policy.{name}" for name in flow_state_dict})

        # a bare flow checkpoint into the safeflow policy ...
        safe_policy.load_network_state_dict(flow_state_dict)
        for name, value in flow_state_dict.items():
            torch.testing.assert_close(safe_policy.state_dict()[f"inner_policy.{name}"], value)
        # ... and a safeflow checkpoint into the bare flow policy
        flow_policy.load_network_state_dict(safe_state_dict)
        for name, value in safe_state_dict.items():
            torch.testing.assert_close(flow_policy.state_dict()[name.removeprefix("inner_policy.")], value)

    def test_the_loading_is_strict_and_a_checkpoint_of_its_own_keys_loads_natively(self) -> None:
        safe_policy, flow_policy = self._policies()
        own = {name: value.clone() for name, value in safe_policy.state_dict().items()}

        safe_policy.load_state_dict(own)  # PyTorch's own loading, no remapping needed
        with self.assertRaises(RuntimeError):  # strict: an unknown key is an error, not silently dropped
            flow_policy.load_network_state_dict({**own, "inner_policy.unexpected": torch.zeros(1)})
        with self.assertRaises(RuntimeError):  # and so is a missing one
            safe_policy.load_network_state_dict(dict(list(own.items())[1:]))

    def test_keys_saved_while_the_network_was_torch_compiled_and_projector_keys_are_mapped(self) -> None:
        safe_policy, _ = self._policies()
        saved = {name.replace("velocity_net.", "velocity_net._orig_mod.", 1): value.clone() for name, value in safe_policy.state_dict().items()}
        saved["projector.stale"] = torch.zeros(1)
        self.assertTrue(any("_orig_mod" in name for name in saved))

        safe_policy.load_network_state_dict(saved)


if __name__ == "__main__":
    unittest.main()
