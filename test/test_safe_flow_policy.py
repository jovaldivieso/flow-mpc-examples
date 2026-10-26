from __future__ import annotations

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
from systems.franka_system import Q_READY, FrankaSystem
from learning.models.flow_policy import FlowPolicy
from learning.models.policy_factory import PolicyFactory
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from planning.planner import PlannerSolveError
from planning.casadi_projector import CasadiTrajectoryProjector
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_casadi_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "pos_tol": 0.002,
    "vel_tol": 0.02,
}
HORIZON = 10


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
    policy_observation = simulator.policy_observation(simulator.observe(state))
    return {
        name: torch.as_tensor(np.expand_dims(value, axis=0), dtype=torch.float32)
        for name, value in policy_observation.items()
    }


def _build_safe_flow_policy(simulator, planner_config=None, **policy_options) -> SafeFlowMPCPolicy:
    # The flow policy conditions directly on observation_features:
    # environment_state(19) + state(21) = 40 = simulator.environment_dim + simulator.nx.
    # The chunk is jerks, scaled by their limits exactly as DaggerTrainer passes it;
    # planner_config defaults to the validated expert config, as it does there. policy_options are the
    # safeflow options (flow_init, use_guidance).
    policy = PolicyFactory.create(
        "safeflow",
        action_dim=simulator.nu, state_dim=simulator.environment_dim + simulator.nx, unet_dim=8, unet_dim_mults=(1, 2), prediction_horizon=HORIZON,
        num_inference_steps=2, action_scale=simulator.max_action.tolist(),
        simulator=simulator, planner_config=_validated_config() if planner_config is None else planner_config,
        **policy_options,
    )
    assert isinstance(policy, SafeFlowMPCPolicy)
    return policy


def _zero_chunk(horizon: int = HORIZON) -> np.ndarray:
    """A chunk of zero jerks (horizon, nu): from rest, it stays at rest."""
    return np.zeros((horizon, 7))


def _zero_the_network(policy: SafeFlowMPCPolicy) -> None:
    """With every weight and bias zeroed the predicted velocity is identically 0: the Euler iterate never leaves
    its start, so what the projector is handed is exactly the integration's initial chunk."""
    with torch.no_grad():
        for param in policy.inner_policy.velocity_net.unet.parameters():
            param.zero_()


class GoalAnchorIndependenceTests(unittest.TestCase):
    """Root-cause fix for the same staleness issue a since-removed
    sync_simulator() hook used to patch around: PolicyFactory.create now
    gives SafeFlowMPCPolicy its own private, goal-zeroed dynamics objects
    instead of live references into whatever simulator built it, so there
    is nothing left to keep in sync with rollout collection's or
    evaluate_current_policy's own separately-constructed simulators.

    This is correct, not just convenient, because every constraint/cost
    CasadiTrajectoryProjector enforces -- dynamics, action/state bounds,
    terminal velocity -- is built
    from the state invert_obs reconstructs from the observation, which
    carries no goal; the goal used there is never anything more than a
    fixed, self-consistent local anchor. See learning/models/policy_factory.py's
    comment for the one way this could stop being true (a future
    world-frame-referencing term).
    """

    def test_local_sim_has_zeroed_goal_regardless_of_source_simulator(self) -> None:
        simulator = _build_simulator(goal=GOAL)
        policy = _build_safe_flow_policy(simulator)

        np.testing.assert_allclose(policy.local_sim.goal, np.zeros(12))
        # projector.sim and local_sim must still be the same object
        # (both are PolicyFactory.create's own copy) -- a projector never
        # needs a goal of its own, separate from local_sim's.
        self.assertIs(policy.projector.sim, policy.local_sim)

    def test_construction_does_not_mutate_the_source_simulator(self) -> None:
        simulator = _build_simulator(goal=GOAL)
        policy = _build_safe_flow_policy(simulator)

        # The deep copy must fully isolate local_sim from the simulator
        # that built it -- a caller that passes the same simulator object
        # for both policy construction and its own rollout (some
        # evaluate_policy.py call sites do) must not find its goal silently
        # zeroed out from under it.
        self.assertIsNot(policy.local_sim, simulator)
        np.testing.assert_allclose(simulator.goal, GOAL)

    def test_select_action_is_invariant_to_which_goal_the_source_simulator_holds(self) -> None:
        # The actual mathematical property the whole redesign relies on,
        # checked end to end (test_casadi_projector.py checks it at the
        # projector level): invert_obs drops the goal, so the projection
        # must not depend on it -- policies with the same weights, built
        # from simulators holding different goals, must act identically on
        # the same observation. If this ever fails, zeroing the goal in
        # PolicyFactory.create is no longer a safe substitute for the real
        # one, and this whole approach needs revisiting.
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
        projector = CasadiTrajectoryProjector(simulator, _validated_config(), horizon=horizon)
        state0 = _rest_state(simulator)
        state0[simulator.n] = 2.0
        state0[simulator.n + 4] = 3.0
        # sanity: braking takes more than the single planned step
        self.assertGreater(np.max(state0[simulator.n : 2 * simulator.n] / (simulator.max_acceleration * simulator.dt)), 1.0)
        obs0 = simulator.observe(state0)
        q_safe = projector.project(obs0, _zero_chunk(horizon))
        self.assertEqual(q_safe.shape, (horizon, simulator.n))


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
        from learning.dagger.rollouts import rollout_policy_with_action_fn

        simulator = _build_simulator()
        initial_state = simulator.reset_random()

        def failing_action_fn(observation: np.ndarray) -> np.ndarray:
            raise PlannerSolveError("forced failure for test")

        reached_goal, steps_taken, failure_reason = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=10,
            action_fn=failing_action_fn,
        )
        self.assertFalse(reached_goal)
        # failing_action_fn raises on the very first call, before
        # simulator.step() ever runs -- zero steps were actually executed.
        self.assertEqual(steps_taken, 0)
        self.assertEqual(failure_reason, "solve_failure")

    def test_projector_cold_start_failure_ends_the_episode_with_a_solve_failure(self) -> None:
        # The same property through the real policy: with no previously
        # solved safe trajectory to fall back on, a failing projector solve
        # surfaces from select_action as a PlannerSolveError (see
        # CasadiTrajectoryProjector.project), which the evaluation-side
        # rollout turns into an episode-ending failure.
        from learning.dagger.rollouts import build_policy_action, rollout_policy_with_action_fn

        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        policy.projector.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        initial_state = simulator.reset_random()

        reached_goal, steps_taken, failure_reason = rollout_policy_with_action_fn(
            simulator=simulator,
            initial_state=initial_state,
            num_steps=10,
            action_fn=lambda observation: build_policy_action(simulator, policy, observation, torch.device("cpu")),
            reset_fn=policy.reset,
        )
        self.assertFalse(reached_goal)
        self.assertEqual(steps_taken, 0)
        self.assertEqual(failure_reason, "solve_failure")


class SelectActionBatchSizeGuardTests(unittest.TestCase):
    """select_action hands ego_obs_np[0] to the one projector this policy
    owns, assuming the batch *is* that one robot's single row (see
    build_policy_action, the only real caller, which always builds exactly
    one). A batch of any other size must be rejected explicitly rather than
    silently leaving every row past the first unprojected (too large a
    batch) or crashing with an opaque IndexError (too small).
    """

    @staticmethod
    def _build_observation_dict(simulator) -> dict[str, torch.Tensor]:
        state = simulator.reset_random()
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
    """PolicyFactory.create("safeflow", ...) wires an inner FlowPolicy, a
    CasadiTrajectoryProjector and the policy's own goal-zeroed copy of the
    simulator into one single-arm SafeFlowMPCPolicy.
    """

    def test_requires_simulator_and_planner_config(self) -> None:
        simulator = _build_simulator()
        flow_kwargs = dict(
            action_dim=simulator.nu, state_dim=simulator.environment_dim + simulator.nx, unet_dim=8, unet_dim_mults=(1, 2),
            prediction_horizon=HORIZON, num_inference_steps=2,
        )
        for given in ({}, {"simulator": simulator}, {"planner_config": {}}):
            with self.subTest(given=sorted(given)):
                with self.assertRaises(ValueError):
                    PolicyFactory.create("safeflow", **flow_kwargs, **given)

    def test_projector_is_sized_by_the_prediction_horizon_and_keeps_the_arms_rest_indices(self) -> None:
        policy = _build_safe_flow_policy(_build_simulator())

        self.assertEqual(policy.projector.N, HORIZON)
        self.assertEqual(policy.projector.tracked_horizon, HORIZON)
        self.assertIsInstance(policy.local_sim, FrankaSystem)
        # What the terminal come-to-rest cost reads: the arm's joint velocities and accelerations, state = [q, v, a].
        self.assertEqual(policy.projector.rest_idx, list(range(7, 21)))

    def test_projector_reads_its_tuning_from_the_validated_expert_config(self) -> None:
        planner_config = _validated_config(
            terminal_velocity_weight=42.0, fallback_terminal_velocity_tol=0.007, max_consecutive_fallbacks=3,
        )
        policy = _build_safe_flow_policy(_build_simulator(), planner_config=planner_config)

        self.assertEqual(policy.projector.terminal_velocity_weight, 42.0)
        self.assertEqual(policy.projector.fallback_terminal_velocity_tol, 0.007)
        self.assertEqual(policy.projector.max_consecutive_fallbacks, 3)


class SelectActionProjectionTests(unittest.TestCase):
    """select_action integrates the flow over the chunk of actions:
    every Euler step hands the projector the flat observation and the
    current proposal, the projection is the flow's next state, and the
    returned chunk is the last projection.
    """

    @staticmethod
    def _recording_projector(policy: SafeFlowMPCPolicy) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        real_project = policy.projector.project
        calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def recording_project(obs: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
            q_safe = real_project(obs, q_ref)
            calls.append((obs.copy(), q_ref.copy(), q_safe.copy()))
            return q_safe

        policy.projector.project = recording_project
        return calls

    def test_every_euler_step_projects_the_flat_observation_and_the_proposed_chunk(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator, flow_init="gaussian")
        observation_dict = _rest_observation_dict(simulator)
        # With a zero velocity the Euler iterate never leaves its initial noise draw x_0: the first reference
        # must be exactly x_0, and since each projection is the flow's next state, the second reference must be
        # the first projection itself.
        _zero_the_network(policy)
        torch.manual_seed(7)
        expected_x0 = torch.randn(1, HORIZON, simulator.nu)
        calls = self._recording_projector(policy)

        torch.manual_seed(7)
        action = policy.select_action(observation_dict)

        self.assertEqual(len(calls), policy.inner_policy.velocity_net.num_inference_steps)
        # the projector gets the observation [goal, state] rebuilt from what the policy saw
        expected_observation = simulator.observe(_rest_state(simulator))
        for obs, q_ref, _ in calls:
            self.assertEqual(obs.shape, (simulator.obs_dim,))
            np.testing.assert_allclose(obs, expected_observation, atol=1e-5)
            self.assertEqual(q_ref.shape, (HORIZON, simulator.nu))
        np.testing.assert_allclose(calls[0][1], expected_x0[0].numpy() * simulator.max_action, rtol=1e-5, atol=1e-4)
        np.testing.assert_allclose(calls[1][1], calls[0][2], rtol=1e-5, atol=1e-5)
        # The returned chunk is the last projection itself, as (batch, horizon, nu), within the action limits
        last_projection = calls[-1][2]
        self.assertEqual(tuple(action.shape), (1, HORIZON, simulator.nu))
        np.testing.assert_allclose(action[0].numpy(), last_projection, atol=1e-4)
        self.assertTrue(np.all(np.abs(last_projection) <= simulator.max_action + 1e-6))

    def test_reset_clears_the_projectors_warm_start(self) -> None:
        # Without this, a new episode's first solve would warm start from
        # the previous episode's unrelated final trajectory.
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        state = _rest_state(simulator)
        policy.projector.project(simulator.observe(state), _zero_chunk())
        self.assertIsNotNone(policy.projector._prev_U_sol)

        with mock.patch.object(policy.inner_policy, "reset") as inner_reset:
            policy.reset()

        inner_reset.assert_called_once_with()
        self.assertIsNone(policy.projector._prev_U_sol)

    def test_compute_loss_delegates_to_the_inner_policy(self) -> None:
        simulator = _build_simulator()
        policy = _build_safe_flow_policy(simulator)
        observation = {name: value.repeat(4, 1) for name, value in _rest_observation_dict(simulator).items()}
        actions = torch.randn(4, HORIZON, simulator.nu)

        torch.manual_seed(0)
        expected = policy.inner_policy.compute_loss(observation, actions)
        torch.manual_seed(0)
        torch.testing.assert_close(policy.compute_loss(observation, actions), expected)


class FlowInitializationTests(unittest.TestCase):
    """With flow_init="previous_safe" the integration starts from the previous
    tick's safe actions (or, on the first tick, from zero actions) instead of
    from Gaussian noise.
    """

    STEPS = 2  # num_inference_steps of _build_safe_flow_policy

    def _policy_and_calls(self, simulator, **options):
        policy = _build_safe_flow_policy(simulator, **options)
        _zero_the_network(policy)
        return policy, SelectActionProjectionTests._recording_projector(policy)

    def test_gaussian_init_needs_no_extra_projection(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator, flow_init="gaussian")

        policy.select_action(_rest_observation_dict(simulator))

        self.assertEqual(len(calls), self.STEPS)

    def test_the_first_tick_starts_from_zero_actions(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator, flow_init="previous_safe")

        policy.select_action(_rest_observation_dict(simulator))

        self.assertEqual(len(calls), self.STEPS)  # no extra projection of the start
        np.testing.assert_array_equal(calls[0][1], np.zeros((HORIZON, simulator.nu)))

    def test_later_ticks_start_from_the_previous_safe_actions_shifted_by_one_step(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator, flow_init="previous_safe")
        state = _rest_state(simulator)
        first_chunk = policy.select_action(_policy_input(simulator, state))[0].numpy()
        guess = policy.projector.next_chunk_guess()
        self.assertIsNotNone(guess)
        next_state = simulator.step(state, first_chunk[0])  # execute the first action, as the rollout does
        calls.clear()

        policy.select_action(_policy_input(simulator, next_state))

        self.assertEqual(len(calls), self.STEPS)  # no extra projection: the previous actions are the start
        np.testing.assert_allclose(calls[0][1], guess, atol=1e-6)

    def test_reset_returns_to_the_zero_start(self) -> None:
        simulator = _build_simulator()
        policy, calls = self._policy_and_calls(simulator, flow_init="previous_safe")
        state = _rest_state(simulator)
        policy.projector.project(simulator.observe(state), _zero_chunk())
        policy.reset()
        calls.clear()

        policy.select_action(_policy_input(simulator, state))

        np.testing.assert_array_equal(calls[0][1], np.zeros((HORIZON, simulator.nu)))

    def test_an_unknown_flow_init_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_safe_flow_policy(_build_simulator(), flow_init="zeros")


class StateDictInterchangeTests(unittest.TestCase):
    """'flow' and 'safeflow' checkpoints are interchangeable (see
    scripts/evaluate_policy.py's _POLICY_TYPE_EQUIVALENCE): a SafeFlowMPCPolicy's
    state dict is its inner FlowPolicy's behind an ``inner_policy.`` prefix
    (the projector is no nn.Module, so it contributes nothing), and either
    kind of policy loads either kind of checkpoint.
    """

    def test_flow_and_safeflow_state_dicts_load_into_each_other(self) -> None:
        simulator = _build_simulator()
        safe_policy = _build_safe_flow_policy(simulator)
        flow_policy = FlowPolicy(
            action_dim=simulator.nu, state_dim=simulator.environment_dim + simulator.nx, unet_dim=8, unet_dim_mults=(1, 2),
            prediction_horizon=HORIZON, num_inference_steps=2,
        )
        flow_state_dict = {name: value.clone() for name, value in flow_policy.state_dict().items()}
        safe_state_dict = {name: value.clone() for name, value in safe_policy.state_dict().items()}
        self.assertEqual(set(safe_state_dict), {f"inner_policy.{name}" for name in flow_state_dict})

        # a bare flow checkpoint into the safeflow policy ...
        safe_policy.load_state_dict(flow_state_dict)
        for name, value in flow_state_dict.items():
            torch.testing.assert_close(safe_policy.state_dict()[f"inner_policy.{name}"], value)
        # ... and a safeflow checkpoint into the bare flow policy
        flow_policy.load_state_dict(safe_state_dict)
        for name, value in safe_state_dict.items():
            torch.testing.assert_close(flow_policy.state_dict()[name.removeprefix("inner_policy.")], value)


if __name__ == "__main__":
    unittest.main()
