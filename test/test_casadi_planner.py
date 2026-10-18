from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import validate_system_config
from dynamics.franka_dynamics import Q_READY, FrankaDynamics
from planning.casadi_planner import CasadiPlanner
from planning.planner import Plan, PlannerSolveError

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

# Behaviour tests pin the settings they depend on, so tuning scripts/config/franka_casadi_config.yaml cannot break them
PINNED_EXPERT_SETTINGS = {
    "expert": "casadi",
    "horizon": 20,
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "pos_tol": 0.002,
    "vel_tol": 0.02,
}


def _build(**overrides) -> tuple[FrankaDynamics, CasadiPlanner, dict]:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    validated = validate_system_config(raw_config=raw_config)
    simulator = FrankaDynamics(validated)
    return simulator, CasadiPlanner(simulator, dict(validated)), validated


def _first_action(planner: CasadiPlanner, simulator: FrankaDynamics, state: np.ndarray) -> np.ndarray:
    """The action of the planner's first step: the first jerk of its plan."""
    return planner.plan(simulator.observe(state)).actions[0]


def _initial_guess(planner: CasadiPlanner, observation: np.ndarray, initial_plan: Plan | None = None):
    """The (states (nx, N + 1), actions (nu, N)) the planner starts its solve from."""
    guessed = {}
    original = planner.opti.set_initial

    def recording_set_initial(variable, value):
        guessed[id(variable)] = np.asarray(value)
        return original(variable, value)

    planner.opti.set_initial = recording_set_initial
    planner.plan(observation, initial_plan)
    return guessed[id(planner.X)], guessed[id(planner.U)]


def _start_near_the_ready_pose(simulator: FrankaDynamics) -> np.ndarray:
    q = Q_READY.copy()
    q += np.array([0.12, -0.10, 0.08, 0.10, -0.12, 0.10, 0.25])
    return np.concatenate([q, np.zeros(2 * simulator.n)])


class MpcWarmStartFailureRecoveryTests(unittest.TestCase):
    """Regression test for a real training failure: the warm start (the previous solution shifted) can leave IPOPT stuck
    at a locally infeasible point once noisy execution has pushed the actual state away from what the previous solve
    predicted, even though a feasible trajectory exists (the same x0 and goal solve from a cold start). The planner
    must recover through a cold restart instead of raising.

    The failing warm-started solve is injected (the first solve after a successful one raises), so the test does not
    depend on a particular IPOPT trajectory.
    """

    def test_warm_start_failure_recovers_via_cold_restart_instead_of_raising(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.reset()

        first_action = _first_action(planner, simulator, state)
        self.assertIsNotNone(planner.last_X_sol)

        real_solve = planner.opti.solve
        calls = {"count": 0}

        def failing_once():
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("forced warm-start failure")
            return real_solve()

        planner.opti.solve = failing_once
        state = simulator.step(state, first_action)
        try:
            second_action = _first_action(planner, simulator, state)
        except PlannerSolveError as exc:
            self.fail(f"Planner failed instead of recovering via cold restart: {exc}")

        self.assertEqual(calls["count"], 2, "expected exactly one failed warm solve and one cold restart")
        self.assertTrue(np.all(np.isfinite(second_action)))

    def test_failure_of_both_warm_and_cold_start_raises_planner_solve_error(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.reset()
        planner.opti.solve = lambda: (_ for _ in ()).throw(RuntimeError("forced failure"))
        with self.assertRaises(PlannerSolveError):
            planner.plan(simulator.observe(state))


class PlannerBehaviorTests(unittest.TestCase):
    def test_planner_takes_an_observation_and_plans_the_horizon(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))

        plan = planner.plan(simulator.observe(state))

        self.assertEqual(plan.positions.shape, (planner.N + 1, simulator.n))
        self.assertEqual(_first_action(planner, simulator, state).shape, (simulator.nu,))

    def test_planned_trajectory_respects_every_constraint_of_the_problem(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.reset()
        planner.plan(simulator.observe(state))
        X, U = planner.last_X_sol, planner.last_U_sol
        tolerance = 1e-6

        np.testing.assert_allclose(X[:, 0], state, atol=1e-9)
        self.assertTrue(np.all(np.abs(U) <= simulator.max_action[:, None] + tolerance))
        self.assertTrue(np.all(X[:, 1:] >= simulator.state_lower_bounds[:, None] - tolerance))
        self.assertTrue(np.all(X[:, 1:] <= simulator.state_upper_bounds[:, None] + tolerance))
        for k in range(planner.N):
            # the planned next state is the model's next state
            expected = np.array(simulator.casadi_dynamics(X[:, k], U[:, k])).ravel()
            np.testing.assert_allclose(X[:, k + 1], expected, atol=1e-6)

    def test_closed_loop_mpc_reaches_a_nearby_goal_and_comes_to_rest(self) -> None:
        simulator, planner, _ = _build(pos_tol=0.005, vel_tol=0.05)
        simulator.set_goal(Q_READY)

        def first_action(observation: np.ndarray) -> np.ndarray:
            return planner.plan(observation).actions[0]

        trajectory = simulator.simulate(_start_near_the_ready_pose(simulator), first_action, num_steps=80)

        self.assertTrue(simulator.is_done(trajectory["states"][-1]), "MPC did not reach the goal")
        self.assertLess(len(trajectory["actions"]), 80, "episode should terminate early on the goal")

    def test_the_plan_starts_at_the_state_and_its_first_step_is_the_optimal_action(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))

        plan = planner.plan(simulator.observe(state))

        np.testing.assert_allclose(plan.positions[0], state[: simulator.n], atol=1e-9)
        # The plan's actions are the optimal jerks of the solve (what a DAgger step executes and labels with), and
        # the first one takes the arm to the plan's next row
        np.testing.assert_allclose(plan.actions, planner.last_U_sol.T)
        np.testing.assert_allclose(
            plan.positions[1], simulator.predict_next_state(state, plan.actions[0])[: simulator.n], atol=1e-6
        )

    def test_the_first_solve_from_rest_starts_from_holding_still(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))

        states_guess, actions_guess = _initial_guess(planner, simulator.observe(state))

        np.testing.assert_allclose(states_guess, np.tile(state[:, None], (1, planner.N + 1)))
        np.testing.assert_allclose(actions_guess, 0.0)

    def test_the_next_solve_starts_from_the_previous_solution_shifted_by_one_step(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.plan(simulator.observe(state))
        previous_states, previous_actions = planner.last_X_sol.copy(), planner.last_U_sol.copy()

        states_guess, actions_guess = _initial_guess(planner, simulator.observe(state))

        np.testing.assert_allclose(states_guess[:, :-1], previous_states[:, 1:])
        np.testing.assert_allclose(actions_guess[:, :-1], previous_actions[:, 1:])

    def test_reset_clears_the_warm_start(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.plan(simulator.observe(state))
        self.assertIsNotNone(planner.last_X_sol)
        planner.reset()
        self.assertIsNone(planner.last_X_sol)
        self.assertIsNone(planner.last_U_sol)
        self.assertIsNone(planner._prev_lam_g)
        self.assertIsNone(planner.goal_configuration)

    def test_the_plan_stays_inside_the_joint_limits_however_far_the_goal_is_beyond_them(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        beyond = simulator.state_upper_bounds[: simulator.n] + 0.5

        with mock.patch.object(planner, "_goal_configuration", return_value=beyond):
            plan = planner.plan(simulator.observe(state))

        self.assertTrue(np.all(plan.positions <= simulator.state_upper_bounds[: simulator.n] + 1e-6))

    def test_statistics_count_the_solves(self) -> None:
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))

        planner.plan(simulator.observe(state))
        planner.plan(simulator.observe(state))

        self.assertEqual(planner.statistics, {"trajectory_optimizations": 2})

    def test_invalid_planner_config_is_rejected(self) -> None:
        simulator, _, validated = _build()
        for key, value in (
            ("horizon", 0),
            ("terminal_cost_multiplier", 0.0),
            ("Q_diag", [1.0] * 13),
            ("R_diag", [1.0] * 6),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    CasadiPlanner(simulator, {**validated, key: value})


class GoalConfigurationTests(unittest.TestCase):
    """The goal of the MPC is the goal configuration in the observation."""

    def test_the_goal_configuration_matches_the_goal_in_the_observation(self) -> None:
        simulator, planner, _ = _build()
        start = simulator.reset(_start_near_the_ready_pose(simulator))

        planner.plan(simulator.observe(start))

        np.testing.assert_allclose(planner.goal_configuration, simulator.goal, atol=1e-9)

        simulator.set_goal(start[: simulator.n])
        planner.plan(simulator.observe(start))
        np.testing.assert_allclose(planner.goal_configuration, start[: simulator.n], atol=1e-9)


class InitialPlanTests(unittest.TestCase):
    """``plan`` with an ``initial_plan`` starts the solve from it, and nothing else: it does not hold the solution."""

    def _setup(self):
        simulator, planner, _ = _build()
        state = simulator.reset(_start_near_the_ready_pose(simulator))
        planner.reset()
        return simulator, planner, state

    @staticmethod
    def _shifted_plan(state: np.ndarray, planner: CasadiPlanner, shift: float) -> Plan:
        """The plan that holds the current configuration, with joint 7 shifted by ``shift`` from the first step on."""
        q = state[: planner.n_joints]
        positions = np.tile(q, (planner.N + 1, 1))
        positions[1:, -1] += shift
        return Plan(positions=positions)

    def test_the_solve_starts_from_the_configurations_of_the_plan_and_the_motion_through_them(self) -> None:
        simulator, planner, state = self._setup()
        n = simulator.n
        rng = np.random.default_rng(0)
        positions = np.vstack([state[:n], state[:n] + np.cumsum(rng.uniform(-0.02, 0.02, (planner.N, n)), axis=0)])

        states, actions = _initial_guess(planner, simulator.observe(state), Plan(positions=positions))

        np.testing.assert_allclose(states[:, 0], state)
        np.testing.assert_allclose(states[:n, 1:], positions[1:].T)
        # velocities and accelerations are the finite differences of the positions, the jerks those of the accelerations
        dt = simulator.dt
        np.testing.assert_allclose(states[n : 2 * n, 1:], np.diff(states[:n], axis=1) / dt)
        np.testing.assert_allclose(states[2 * n :, 1:], np.diff(states[n : 2 * n], axis=1) / dt)
        np.testing.assert_allclose(actions, np.diff(states[2 * n :], axis=1) / dt)

    def test_the_initial_plan_does_not_hold_the_solution(self) -> None:
        # A plan that holds a configuration 0.3 rad off the goal's: the optimizer does not stay near it, it ends on the goal pose
        simulator, planner, state = self._setup()
        initial_plan = self._shifted_plan(state, planner, 0.3)

        plan = planner.plan(simulator.observe(state), initial_plan)

        goal_distance = sum(simulator.goal_errors(plan.positions[-1]))
        guess_distance = sum(simulator.goal_errors(initial_plan.positions[-1]))
        self.assertLess(goal_distance, 0.1 * guess_distance)

    def test_a_plan_shorter_than_the_horizon_is_held_at_its_last_configuration(self) -> None:
        simulator, planner, state = self._setup()
        q = state[: simulator.n]
        last = q + np.array([0, 0, 0, 0, 0, 0, 0.3])
        short = Plan(positions=np.vstack([q, last]))  # one step, then it stays there
        full = Plan(positions=np.vstack([q, np.tile(last, (planner.N, 1))]))

        held, _ = _initial_guess(planner, simulator.observe(state), short)
        explicit, _ = _initial_guess(planner, simulator.observe(state), full)

        np.testing.assert_allclose(held, explicit)

    def test_a_longer_plan_is_cut_at_the_horizon(self) -> None:
        simulator, planner, state = self._setup()
        q = state[: simulator.n]
        long = Plan(positions=np.vstack([q, np.tile(q + 0.01, (3 * planner.N, 1))]))

        states, _ = _initial_guess(planner, simulator.observe(state), long)

        self.assertEqual(states.shape, (simulator.nx, planner.N + 1))

    def test_a_plan_with_the_wrong_joints_or_without_a_step_is_rejected(self) -> None:
        simulator, planner, state = self._setup()
        for positions in (np.zeros((planner.N + 1, simulator.n + 1)), np.zeros((1, simulator.n))):
            with self.subTest(shape=positions.shape):
                with self.assertRaises(ValueError):
                    planner.plan(simulator.observe(state), Plan(positions=positions))


if __name__ == "__main__":
    unittest.main()
