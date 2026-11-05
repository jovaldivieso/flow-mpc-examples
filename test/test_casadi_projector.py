from __future__ import annotations

import os
import sys
import unittest
import warnings

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.casadi_graph import GraphSolution
from planning.casadi_projector import CasadiTrajectoryProjector, Projection
from planning.planner import PlannerSolveError
from planning.planner_config import ProjectionConfig
from systems.franka_system import FrankaSystem
from test.planner_fixtures import pinned_config, rest_state

HORIZON = 10


def _hold_reference(state: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """The reference (horizon, n) that stays at the configuration of ``state``."""
    return np.tile(state[:7], (horizon, 1))


def _moving_reference(simulator: FrankaSystem, state: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """A reference that starts a motion of joint 1 and does not come to rest within the horizon."""
    jerks = np.zeros((horizon, simulator.nu))
    jerks[:3, 0] = 8.0
    return simulator.rollout_positions(state, jerks)[1:]


def _rest_to_rest_reference(state: np.ndarray, amplitude: float = 0.2, horizon: int = HORIZON) -> np.ndarray:
    """A quintic move of joint 1 by ``amplitude`` rad over the horizon, from rest to rest: within the plant's limits."""
    tau = np.arange(1, horizon + 1) / horizon
    reference = _hold_reference(state, horizon)
    reference[:, 0] += amplitude * (10 * tau**3 - 15 * tau**4 + 6 * tau**5)
    return reference


def _forced_failure():
    raise RuntimeError("forced failure")


class ProjectorFailClosedFallbackTests(unittest.TestCase):
    """A failed solve never returns the unprojected reference. The projector falls back to the last known-safe plan
    (SafeFlowMPC Theorem 2) whenever one exists, and at the start of an episode (Assumption 2) to holding still, which is
    safe when the arm is at rest and clear of the obstacles. Where there is no safe plan it raises PlannerSolveError.
    """

    def setUp(self) -> None:
        # A large terminal_velocity_weight makes the projector settle at rest: the terminal rest is a soft cost, and
        # these tests need a genuinely resting cached plan to fall back on
        config = pinned_config(terminal_velocity_weight=500.0)
        self.simulator = FrankaSystem(config)
        self.projector = CasadiTrajectoryProjector(self.simulator, ProjectionConfig.from_mapping(config), horizon=HORIZON)
        self.state0 = rest_state(self.simulator)
        self.q_ref = _hold_reference(self.state0)

    def _perturbed_state(self, offset: float) -> np.ndarray:
        # Joint positions only: the terminal velocity of the cached plan stays at rest
        state = self.state0.copy()
        state[: self.simulator.n] += offset
        return state

    def test_falls_back_to_last_safe_plan_when_one_exists(self) -> None:
        self.projector.project(self.state0, self.q_ref)
        self.assertIsNotNone(self.projector._prev_U_sol)
        # the shifted plan ends with a zero jerk, which holds the rest state it ends in
        shifted_actions = np.hstack([self.projector._prev_U_sol[:, 1:], np.zeros((self.simulator.nu, 1))])
        perturbed_state = self._perturbed_state(0.005)
        expected = self.projector._revalidate_fallback(perturbed_state, shifted_actions)
        self.assertIsNotNone(expected)

        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fallback = self.projector.project(perturbed_state, self.q_ref)

        np.testing.assert_allclose(fallback.actions, shifted_actions.T)
        np.testing.assert_allclose(fallback.positions, expected[: self.simulator.n, 1:].T)
        self.assertTrue(any("last known-safe" in str(w.message) for w in caught))

    def test_a_cold_start_failure_holds_still_when_at_rest_and_clear_of_the_obstacles(self) -> None:
        # No solved plan exists (first call, or right after reset): the safe plan of Assumption 2 is holding still
        self.projector.reset()
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = self.projector.project(self.state0, _moving_reference(self.simulator, self.state0))

        np.testing.assert_array_equal(plan.actions, np.zeros((HORIZON, self.simulator.nu)))
        np.testing.assert_allclose(plan.positions, _hold_reference(self.state0), atol=1e-12)
        self.assertTrue(any("last known-safe" in str(w.message) for w in caught))
        self.assertIsNotNone(self.projector._prev_U_sol)  # the next tick shifts the plan that was held

    def test_a_cold_start_failure_raises_when_holding_still_is_not_safe(self) -> None:
        # In motion, holding still (zero jerk) does not come to rest within the horizon: no safe plan
        moving = self.state0.copy()
        moving[self.simulator.n] = 0.5
        self.projector.reset()
        self.projector._graph._solve_once = _forced_failure
        with self.assertRaises(PlannerSolveError):
            self.projector.project(moving, self.q_ref)

    def test_a_cold_start_failure_is_not_solved_a_second_time_from_the_same_guess(self) -> None:
        self.projector.reset()
        solves = []

        def failing_solve():
            solves.append(1)
            raise RuntimeError("forced failure")

        self.projector._graph._solve_once = failing_solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.projector.project(self.state0, self.q_ref)

        self.assertEqual(len(solves), 1)

    def test_consecutive_fallbacks_stay_at_rest_instead_of_drifting_away(self) -> None:
        # Across ticks with a new x0 each time, every fallback pads its plan with zero jerk, so the terminal velocity
        # stays at rest however many fallbacks follow one another
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                self.projector.project(self._perturbed_state(0.001 * i), self.q_ref)
        terminal_velocity = self.projector._prev_X_sol[self.projector.rest_idx, self.projector.N]
        # the terminal rest is a soft cost: a plan settles close to rest, within the bound the revalidation enforces
        np.testing.assert_allclose(terminal_velocity, 0.0, atol=self.projector.config.fallback_terminal_velocity_tol)

    def test_exceeding_max_consecutive_fallbacks_raises_instead_of_coasting_indefinitely(self) -> None:
        # A fallback accepted within fallback_terminal_velocity_tol coasts at a small residual velocity; an unbroken run of
        # them would let that accumulate, so one more than max_consecutive_fallbacks raises
        self.assertEqual(self.projector.config.max_consecutive_fallbacks, 5)
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(1, 6):
                self.projector.project(self._perturbed_state(0.001 * i), self.q_ref)
            with self.assertRaises(PlannerSolveError):
                self.projector.project(self._perturbed_state(0.006), self.q_ref)

    def test_repeated_same_tick_calls_do_not_each_consume_the_fallback_budget(self) -> None:
        # The flow calls project once per Euler step, several times per control tick at the same x0 (only the reference
        # is refined): none of them is more elapsed time or drift
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(20):  # far more than max_consecutive_fallbacks (5)
                self.projector.project(self.state0, self.q_ref)
        self.assertEqual(self.projector._consecutive_fallback_count, 0)

    def test_a_held_arm_that_keeps_failing_falls_back_for_the_budget_and_then_raises(self) -> None:
        # An arm held at rest has the same state tick after tick, so only begin_tick tells the ticks apart: each of them
        # consumes the budget, and one more than max_consecutive_fallbacks raises instead of holding still forever
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(self.projector.config.max_consecutive_fallbacks):
                self.projector.begin_tick()
                plan = self.projector.project(self.state0, self.q_ref)
                self.assertEqual((plan.positions.shape, plan.actions.shape), ((HORIZON, 7), (HORIZON, self.simulator.nu)))
            self.projector.begin_tick()
            with self.assertRaises(PlannerSolveError):
                self.projector.project(self.state0, self.q_ref)

    def test_the_flow_steps_of_a_tick_after_begin_tick_do_not_each_consume_the_budget(self) -> None:
        self.projector.project(self.state0, self.q_ref)
        self.projector._graph._solve_once = _forced_failure
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.projector.begin_tick()
            for _ in range(20):  # the flow steps of one tick, the state the same
                self.projector.project(self.state0, self.q_ref)
        self.assertEqual(self.projector._consecutive_fallback_count, 1)

    def test_a_tick_that_starts_from_the_hold_still_plan_is_not_solved_a_second_time_from_it(self) -> None:
        self.projector.reset()
        solves = []

        def failing_solve():
            solves.append(1)
            raise RuntimeError("forced failure")

        self.projector._graph._solve_once = failing_solve
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.projector.project(self.state0, self.q_ref)  # the cold start holds still, and falls back to it
            solves.clear()
            self.projector.begin_tick()
            self.projector.project(self.state0, self.q_ref)  # the warm guess now is the hold-still plan

        self.assertEqual(len(solves), 1)

    def test_a_warm_start_that_is_not_the_hold_still_plan_gets_a_cold_restart(self) -> None:
        self.projector.project(self.state0, _moving_reference(self.simulator, self.state0))  # a plan that moves
        solves = []
        real_solve = self.projector._graph._solve_once

        def failing_then_real():
            solves.append(1)
            if len(solves) == 1:
                raise RuntimeError("the warm start fails")
            return real_solve()

        self.projector._graph._solve_once = failing_then_real
        self.projector.begin_tick()
        self.projector.project(self._perturbed_state(0.001), self.q_ref)

        self.assertEqual(len(solves), 2)


class FallbackRevalidationTests(unittest.TestCase):
    """_revalidate_fallback re-checks a reused plan against what the live solve enforces."""

    def setUp(self) -> None:
        config = pinned_config(terminal_velocity_weight=500.0)
        self.simulator = FrankaSystem(config)
        self.projector = CasadiTrajectoryProjector(self.simulator, ProjectionConfig.from_mapping(config), horizon=HORIZON)
        self.state0 = rest_state(self.simulator)
        self.u_rest = np.zeros((self.simulator.nu, HORIZON))

    def test_accepts_a_resting_plan(self) -> None:
        trajectory = self.projector._revalidate_fallback(self.state0, self.u_rest)
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.shape, (self.simulator.nx, HORIZON + 1))

    def test_rejects_a_plan_that_does_not_end_at_rest(self) -> None:
        moving = self.u_rest.copy()
        moving[0, :] = 5.0
        self.assertIsNone(self.projector._revalidate_fallback(self.state0, moving))

    def test_rejects_a_plan_that_penetrates_an_obstacle(self) -> None:
        simulator = FrankaSystem(
            pinned_config(obstacles=[{"position": self.simulator.tcp_position(self.state0[:7]).tolist(), "radius": 0.05}])
        )
        projector = CasadiTrajectoryProjector(simulator, ProjectionConfig(), horizon=HORIZON)

        self.assertFalse(simulator.collision.is_collision_free(self.state0[:7]))
        self.assertIsNone(projector._revalidate_fallback(self.state0, self.u_rest))

    def test_without_the_safety_filter_a_plan_need_not_end_at_rest(self) -> None:
        no_rest = CasadiTrajectoryProjector(self.simulator, ProjectionConfig(), horizon=HORIZON, safe=False)
        moving = self.u_rest.copy()
        moving[0, :3] = 5.0  # ends in motion

        self.assertIsNotNone(no_rest._revalidate_fallback(self.state0, moving))
        self.assertIsNone(self.projector._revalidate_fallback(self.state0, moving))

    def test_without_the_safety_filter_a_plan_through_an_obstacle_is_accepted(self) -> None:
        simulator = FrankaSystem(
            pinned_config(obstacles=[{"position": self.simulator.tcp_position(self.state0[:7]).tolist(), "radius": 0.05}])
        )
        no_obstacles = CasadiTrajectoryProjector(simulator, ProjectionConfig(), horizon=HORIZON, safe=False)

        self.assertIsNotNone(no_obstacles._revalidate_fallback(self.state0, self.u_rest))


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.projection_config = ProjectionConfig.from_mapping(cls.config)
        cls.simulator = FrankaSystem(cls.config)
        cls.projector = CasadiTrajectoryProjector(cls.simulator, cls.projection_config, horizon=HORIZON)

    def setUp(self) -> None:
        self.projector.reset()

    def test_a_projection_holds_the_positions_and_actions_of_the_plan(self) -> None:
        state = rest_state(self.simulator)

        projection = self.projector.project(state, _rest_to_rest_reference(state))

        self.assertIsInstance(projection, Projection)
        self.assertEqual((projection.positions.shape, projection.actions.shape), ((HORIZON, 7), (HORIZON, self.simulator.nu)))
        np.testing.assert_allclose(projection.positions, self.simulator.rollout_positions(state, projection.actions)[1:], atol=1e-5)

    def test_the_projection_of_a_holding_reference_at_rest_stays_put(self) -> None:
        state = rest_state(self.simulator)

        projection = self.projector.project(state, _hold_reference(state))

        np.testing.assert_allclose(projection.positions, _hold_reference(state), atol=1e-4)
        np.testing.assert_allclose(projection.actions, 0.0, atol=1e-2)

    def test_a_feasible_rest_to_rest_reference_is_tracked_closely(self) -> None:
        state = rest_state(self.simulator)
        reference = _rest_to_rest_reference(state)

        projection = self.projector.project(state, reference)

        np.testing.assert_allclose(projection.positions, reference, atol=0.05)  # the soft rest at the end gives a little way

    def test_a_reference_far_from_reachable_gives_the_closest_reachable_plan_within_the_limits(self) -> None:
        simulator, tolerance = self.simulator, 1e-5
        state = rest_state(simulator)
        reference = _hold_reference(state)
        reference[:, 0] += 2.0  # a jump no jerk-limited arm makes

        projection = self.projector.project(state, reference)

        self.assertTrue(np.all(np.abs(projection.actions) <= simulator.max_action + tolerance))
        self.assertGreater(projection.positions[-1, 0], state[0])  # it goes towards the reference
        self.assertLess(projection.positions[-1, 0], reference[-1, 0] - 0.5)  # but cannot reach it
        for q in self.simulator.rollout_states(state, projection.actions)[1:]:
            self.assertTrue(np.all(q >= simulator.state_lower_bounds - tolerance) and np.all(q <= simulator.state_upper_bounds + tolerance))

    def test_projected_plans_are_feasible_for_the_plant_even_for_unreachable_references(self) -> None:
        simulator = self.simulator
        tolerance = 1e-5  # the constraints hold up to the solver's tolerance
        for seed, scale in enumerate((0.05, 0.5, 3.0)):
            rng = np.random.default_rng(seed)
            x0 = simulator.random_initial_state(rng)
            self.projector.reset()
            reference = x0[: simulator.n] + rng.normal(0.0, scale, (HORIZON, simulator.n))

            projection = self.projector.project(x0, reference)

            self.assertEqual(projection.actions.shape, (HORIZON, simulator.nu))
            self.assertTrue(np.all(np.abs(projection.actions) <= simulator.max_action + tolerance), scale)
            state = x0.copy()
            for action in projection.actions:  # the plant follows the projected actions
                state = simulator.predict_next_state(state, action)
                self.assertTrue(np.all(state >= simulator.state_lower_bounds - tolerance), scale)
                self.assertTrue(np.all(state <= simulator.state_upper_bounds + tolerance), scale)
            np.testing.assert_allclose(state[: simulator.n], projection.positions[-1], atol=1e-4)

    def test_noise_around_the_arm_is_projected_without_failing(self) -> None:
        # The flow starts from noise like this: the cold start must not begin inside the constraints, where the solver
        # ends in a locally infeasible point
        simulator = self.simulator
        state = rest_state(simulator)
        for seed in range(6):
            self.projector.reset()
            noise = np.random.default_rng(seed).normal(size=(HORIZON, simulator.n))

            projection = self.projector.project(state, state[: simulator.n] + noise)

            self.assertEqual(projection.positions.shape, noise.shape)

    def test_the_projection_stays_inside_the_joint_limits_the_reference_ignores(self) -> None:
        joint = 4
        simulator = self.simulator
        state = rest_state(simulator)
        upper = simulator.state_upper_bounds[joint]
        state[joint] = upper - 0.3
        reference = _hold_reference(state)
        reference[:, joint] = upper + 0.5  # beyond the limit

        projection = self.projector.project(state, reference)

        self.assertLessEqual(projection.positions[:, joint].max(), upper + 1e-6)

    def test_the_safety_filter_makes_the_plan_slow_down_and_the_unsafe_one_not(self) -> None:
        state = rest_state(self.simulator)
        reference = _hold_reference(state)
        reference[:, 0] += 0.15 * np.arange(1, HORIZON + 1)  # keeps moving: 1.5 rad/s
        no_rest = CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON, safe=False)

        rest_plan = self.projector.project(state, reference)
        free_plan = no_rest.project(state, reference)

        n = self.simulator.n
        rest_velocity = abs(self.projector._prev_X_sol[n, -1])
        free_velocity = abs(no_rest._prev_X_sol[n, -1])
        self.assertLess(rest_velocity, 0.5 * free_velocity)
        self.assertGreater(free_plan.positions[-1, 0], rest_plan.positions[-1, 0])

    def test_without_the_safety_filter_the_limits_are_the_only_constraints(self) -> None:
        simulator, tolerance = self.simulator, 1e-5
        limits_only = CasadiTrajectoryProjector(simulator, self.projection_config, horizon=HORIZON, safe=False)
        state = rest_state(simulator)

        plan = limits_only.project(state, _moving_reference(simulator, state))

        self.assertFalse(limits_only.safe)
        self.assertTrue(np.all(np.abs(plan.actions) <= simulator.max_action + tolerance))
        np.testing.assert_allclose(plan.positions, simulator.rollout_positions(state, plan.actions)[1:], atol=1e-5)

    def test_the_next_chunk_guess_is_the_positions_one_control_period_later(self) -> None:
        projector = CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON)
        self.assertIsNone(projector.next_chunk_guess())
        state = rest_state(self.simulator)

        projection = projector.project(state, _rest_to_rest_reference(state))
        guess = projector.next_chunk_guess()

        self.assertEqual(guess.shape, (HORIZON, self.simulator.n))
        np.testing.assert_allclose(guess[:-1], projection.positions[1:])
        np.testing.assert_allclose(guess[-1], projection.positions[-1])  # the last row is held
        projector.reset()
        self.assertIsNone(projector.next_chunk_guess())

    def test_the_next_chunk_guess_is_shifted_by_the_tick_steps(self) -> None:
        tick_steps = 3
        projector = CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON, tick_steps=tick_steps)
        state = rest_state(self.simulator)

        projection = projector.project(state, _rest_to_rest_reference(state))
        guess = projector.next_chunk_guess()

        self.assertEqual(guess.shape, (HORIZON, self.simulator.n))
        np.testing.assert_allclose(guess[: HORIZON - tick_steps], projection.positions[tick_steps:])
        np.testing.assert_allclose(guess[HORIZON - tick_steps :], np.tile(projection.positions[-1], (tick_steps, 1)))

    def test_the_tick_steps_are_between_one_and_the_horizon(self) -> None:
        for tick_steps in (0, -1, HORIZON + 1):
            with self.subTest(tick_steps=tick_steps), self.assertRaises(ValueError):
                CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON, tick_steps=tick_steps)
        self.assertEqual(CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON, tick_steps=HORIZON).tick_steps, HORIZON)

    def test_the_warm_start_of_a_new_tick_is_shifted_by_the_tick_steps(self) -> None:
        tick_steps = 3
        projector = CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=HORIZON, tick_steps=tick_steps)
        state = rest_state(self.simulator)
        reference = _rest_to_rest_reference(state)
        projector.project(state, reference)
        previous_states, previous_actions = projector._prev_X_sol, projector._prev_U_sol
        guesses = []
        real_set_initial = projector._graph.set_initial
        projector._graph.set_initial = lambda states, actions: guesses.append((states, actions)) or real_set_initial(states, actions)

        projector.begin_tick()
        projector.project(state, reference)

        states, actions = guesses[0]
        np.testing.assert_allclose(states[:, : HORIZON + 1 - tick_steps], previous_states[:, tick_steps:])
        np.testing.assert_allclose(states[:, HORIZON + 1 - tick_steps :], np.tile(previous_states[:, -1:], (1, tick_steps)))
        np.testing.assert_allclose(actions[:, : HORIZON - tick_steps], previous_actions[:, tick_steps:])
        np.testing.assert_array_equal(actions[:, HORIZON - tick_steps :], np.zeros((self.simulator.nu, tick_steps)))

    def test_a_reference_of_the_wrong_shape_is_rejected(self) -> None:
        state = rest_state(self.simulator)
        with self.assertRaises(ValueError):
            self.projector.project(state, _hold_reference(state).T)
        with self.assertRaises(ValueError):
            self.projector.project(state[:7], _hold_reference(state))

    def test_short_horizon_constructs_and_solves(self) -> None:
        projector = CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=2)
        state = rest_state(self.simulator)

        projection = projector.project(state, _hold_reference(state, 2))

        self.assertEqual((projection.positions.shape, projection.actions.shape), ((2, 7), (2, self.simulator.nu)))

    def test_the_projector_ignores_the_goal_of_the_simulator_it_was_built_on(self) -> None:
        # No projector cost or constraint reads the goal, so whatever goal the simulator holds must not change the plan
        other = FrankaSystem(self.config)
        other.set_goal(np.zeros_like(other.goal))
        projector = CasadiTrajectoryProjector(other, self.projection_config, horizon=HORIZON)
        state = rest_state(self.simulator)
        reference = state[:7] + np.random.default_rng(5).normal(0.0, 0.1, (HORIZON, 7))

        np.testing.assert_allclose(
            projector.project(state, reference).positions, self.projector.project(state, reference).positions, atol=1e-6
        )

    def test_a_projector_needs_a_horizon(self) -> None:
        # Its scalar settings are validated when the config is built (test_planner_config)
        with self.assertRaises(ValueError):
            CasadiTrajectoryProjector(self.simulator, self.projection_config, horizon=0)

    def test_the_weights_of_the_config_shape_the_plan(self) -> None:
        state = rest_state(self.simulator)
        reference = _rest_to_rest_reference(state, amplitude=1.0)
        heavy = CasadiTrajectoryProjector(
            self.simulator, ProjectionConfig(jerk_weight=1.0, velocity_weight=1.0, acceleration_weight=1.0), horizon=HORIZON
        )

        light_plan = self.projector.project(state, reference)
        heavy_plan = heavy.project(state, reference)

        self.assertGreater(np.abs(light_plan.positions[-1, 0] - state[0]), np.abs(heavy_plan.positions[-1, 0] - state[0]))


class ObstacleProjectionTests(unittest.TestCase):
    """With obstacles, the plan of every solve is a collision-free one, linearized along its own reference."""

    def setUp(self) -> None:
        free = FrankaSystem(pinned_config())
        self.state = rest_state(free)
        reference = _moving_reference(free, self.state)
        self.reference = reference
        self.config = pinned_config(
            terminal_velocity_weight=500.0, obstacles=[{"position": free.tcp_position(reference[-1]).tolist(), "radius": 0.05}]
        )
        self.simulator = FrankaSystem(self.config)
        self.projector = CasadiTrajectoryProjector(self.simulator, ProjectionConfig.from_mapping(self.config), horizon=HORIZON)

    def test_the_reference_runs_into_the_obstacle_and_the_plan_does_not(self) -> None:
        self.assertTrue(self.simulator.collision.is_collision_free(self.state[:7]))
        self.assertFalse(self.simulator.collision.is_collision_free(self.reference[-1]))

        plan = self.projector.project(self.state, self.reference)

        self.assertTrue(self.simulator.collision.trajectory_is_collision_free(plan.positions))
        for q in self.simulator.rollout_positions(self.state, plan.actions)[1:]:
            self.assertTrue(self.simulator.collision.is_collision_free(q))

    def test_without_the_safety_filter_the_obstacles_are_ignored(self) -> None:
        no_obstacles = CasadiTrajectoryProjector(
            self.simulator, ProjectionConfig.from_mapping(self.config), horizon=HORIZON, safe=False
        )

        plan = no_obstacles.project(self.state, self.reference)

        self.assertFalse(self.simulator.collision.trajectory_is_collision_free(plan.positions))

    def test_the_obstacles_are_linearized_at_every_step(self) -> None:
        real_sets_along = self.simulator.collision.sets_along
        calls: list[int] = []
        self.simulator.collision.sets_along = lambda q_current, q: calls.append(len(q)) or real_sets_along(q_current, q)

        for _ in range(3):  # the Euler steps of one tick
            self.projector.project(self.state, self.reference)

        self.assertGreaterEqual(len(calls), 3)

    def test_every_projection_of_a_tick_clears_the_obstacles(self) -> None:
        for _ in range(3):
            plan = self.projector.project(self.state, self.reference)
            self.assertTrue(self.simulator.collision.trajectory_is_collision_free(plan.positions))

    def test_a_solution_that_still_penetrates_an_obstacle_is_a_failed_projection(self) -> None:
        simulator = self.simulator
        simulator.collision.obstacles[:] = [(simulator.tcp_position(self.state[: simulator.n]), 0.05)]  # on the TCP of the start
        at_start = np.tile(self.state[:, None], (1, HORIZON + 1))
        self.projector._graph.solve = lambda: GraphSolution(at_start, np.zeros((simulator.nu, HORIZON)), penetrates=True)

        with self.assertRaises(PlannerSolveError):
            self.projector.project(self.state, self.reference)


if __name__ == "__main__":
    unittest.main()
