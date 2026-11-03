from __future__ import annotations

import os
import sys
import unittest

import casadi as ca
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.casadi_graph import SLACK_TOLERANCE, CasadiKinematicGraph
from planning.casadi_planner import CasadiPlanner
from planning.planner import PlannerSolveError
from planning.planner_config import PlannerConfig
from systems.franka_system import FrankaSystem
from systems.free_space_sets import MAX_SET_SIZE
from test.planner_fixtures import pinned_config, rest_state

OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
    {"position": [0.45, -0.40, 0.48], "radius": 0.08},
]


def _planner(num_obstacles: int) -> tuple[FrankaSystem, CasadiPlanner]:
    config = pinned_config(expert="casadi", obstacles=OBSTACLES[:num_obstacles])
    simulator = FrankaSystem(config)
    return simulator, CasadiPlanner(simulator, PlannerConfig.from_mapping(config))


class CasadiKinematicGraphTests(unittest.TestCase):
    def test_the_stages_do_not_grow_with_the_number_of_obstacles_beyond_the_set_size(self) -> None:
        # One slack per sphere and node, shared by its half-spaces, and a set holds at most MAX_SET_SIZE of them
        horizon = 5
        for count in (1, 3):
            simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:count]))
            graph = CasadiKinematicGraph(simulator, horizon)
            num_spheres = len(simulator.collision_spheres(np.zeros(simulator.n))[2])
            self.assertEqual(graph.num_halfspaces, num_spheres * count)
            self.assertEqual(graph.opti.nx, (horizon + 1) * simulator.nx + horizon * simulator.nu + horizon * num_spheres)
        many = [{"position": [0.45, -0.5 + 0.1 * i, 0.5], "radius": 0.04} for i in range(12)]
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=many))
        graph = CasadiKinematicGraph(simulator, horizon)
        self.assertEqual(graph.num_halfspaces, num_spheres * MAX_SET_SIZE)
        self.assertEqual(graph.opti.nx, (horizon + 1) * simulator.nx + horizon * simulator.nu + horizon * num_spheres)

    def test_a_solve_the_solver_reports_as_failed_raises(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        state[simulator.n : 2 * simulator.n] = 100.0  # no jerk brings that velocity within its limit in a step
        graph = CasadiKinematicGraph(simulator, 5)
        graph.minimize(ca.sumsqr(graph.U))
        graph.set_state(state)
        graph.set_initial(np.tile(state[:, None], (1, 6)), np.zeros((simulator.nu, 5)))
        graph.update_collision_halfspaces(np.tile(state[: simulator.n], (5, 1)))

        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            graph.solve()

    def test_a_good_solve_after_a_failed_one_succeeds_again(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        graph = CasadiKinematicGraph(simulator, 5)
        graph.minimize(ca.sumsqr(graph.U))
        fast = state.copy()
        fast[simulator.n : 2 * simulator.n] = 100.0
        for x0, succeeds in ((state, True), (fast, False), (state, True)):
            graph.set_state(x0)
            graph.set_initial(np.tile(x0[:, None], (1, 6)), np.zeros((simulator.nu, 5)))
            graph.update_collision_halfspaces(np.tile(x0[: simulator.n], (5, 1)))
            if succeeds:
                np.testing.assert_allclose(graph.solve().states[:, 0], x0)
            else:
                with self.assertRaises(RuntimeError):
                    graph.solve()

    def test_a_parameter_takes_a_value_of_its_shape_only_and_gives_it_back(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        graph = CasadiKinematicGraph(simulator, 5)
        owner = graph.parameter(3, 2)

        graph.set_value(owner, np.arange(6.0).reshape(3, 2))
        graph.set_state(np.arange(simulator.nx, dtype=float))

        np.testing.assert_array_equal(graph.value(owner), np.arange(6.0).reshape(3, 2))
        np.testing.assert_array_equal(graph.value(graph.x0_param), np.arange(simulator.nx, dtype=float))  # a column as a vector
        for wrong in (np.zeros(6), np.zeros((2, 3)), np.zeros((3, 2, 1))):
            with self.subTest(shape=wrong.shape), self.assertRaises(ValueError):
                graph.set_value(owner, wrong)

    def test_the_initial_values_are_kept_between_solves_until_a_reset(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        graph = CasadiKinematicGraph(simulator, 5)
        states, actions = np.ones((simulator.nx, 6)), 2.0 * np.ones((simulator.nu, 5))

        np.testing.assert_array_equal(graph.initial_guess[0], np.zeros((simulator.nx, 6)))  # zero before anything is set
        graph.set_initial(states, actions)
        np.testing.assert_array_equal(graph.initial_guess[0], states)
        np.testing.assert_array_equal(graph.initial_guess[1], actions)
        graph.reset()
        np.testing.assert_array_equal(graph.initial_guess[1], np.zeros((simulator.nu, 5)))

    def test_a_solution_with_a_slack_above_the_tolerance_is_rejected(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        # an obstacle on the TCP of the rest state: its half-space cannot be satisfied from there within a step
        state = rest_state(simulator)
        simulator.collision.obstacles[:] = [(simulator.tcp_position(state[: simulator.n]), 0.05)]
        graph = CasadiKinematicGraph(simulator, 5)
        graph.minimize(ca.sumsqr(graph.U))
        graph.set_state(state)
        graph.set_initial(np.tile(state[:, None], (1, 6)), np.zeros((simulator.nu, 5)))
        graph.update_collision_halfspaces(np.tile(state[: simulator.n], (5, 1)))

        with self.assertRaisesRegex(RuntimeError, "slack"):
            graph.solve()

    def test_a_solution_is_accepted_up_to_the_slack_tolerance_it_is_asked_for(self) -> None:
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        simulator.collision.obstacles[:] = [(simulator.tcp_position(state[: simulator.n]), 0.05)]
        graph = CasadiKinematicGraph(simulator, 5)
        graph.minimize(ca.sumsqr(graph.U))
        graph.set_state(state)
        graph.set_initial(np.tile(state[:, None], (1, 6)), np.zeros((simulator.nu, 5)))
        graph.update_collision_halfspaces(np.tile(state[: simulator.n], (5, 1)))

        solution = graph.solve(slack_tolerance=1.0)

        self.assertGreater(solution.max_slack, SLACK_TOLERANCE)
        self.assertLessEqual(solution.max_slack, 1.0)

    def test_a_planner_labels_the_state_a_strict_solve_rejects_when_its_tolerance_allows_it(self) -> None:
        for tolerance, labelled in ((SLACK_TOLERANCE, False), (1.0, True)):
            with self.subTest(expert_slack_tolerance=tolerance):
                config = pinned_config(expert="casadi", obstacles=OBSTACLES[:1], expert_slack_tolerance=tolerance)
                simulator = FrankaSystem(config)
                planner = CasadiPlanner(simulator, PlannerConfig.from_mapping(config))
                state = rest_state(simulator)
                planner.plan(simulator.observe(state))
                # an obstacle on the TCP: the plan cannot clear it from there within a step
                simulator.collision.obstacles[:] = [(simulator.tcp_position(state[: simulator.n]), 0.05)]

                if labelled:
                    planner.plan(simulator.observe(state))
                    self.assertEqual(planner.statistics["relaxed_solutions"], 1)
                else:
                    with self.assertRaises(PlannerSolveError):
                        planner.plan(simulator.observe(state))

    def test_the_solution_follows_the_dynamics_and_the_limits_among_obstacles(self) -> None:
        simulator, planner = _planner(3)
        state = simulator.reset(rest_state(simulator))
        planner.plan(simulator.observe(state))
        states, actions = planner.last_X_sol, planner.last_U_sol
        np.testing.assert_allclose(states[:, 0], state)
        for k in range(planner.N):
            np.testing.assert_allclose(states[:, k + 1], simulator.predict_next_state(states[:, k], actions[:, k]), atol=1e-5)
        self.assertLessEqual(np.abs(actions).max(), np.max(simulator.max_action) + 1e-6)

    def test_a_warm_started_solve_takes_fewer_iterations_than_the_cold_first_one(self) -> None:
        simulator, planner = _planner(2)
        state = simulator.reset(rest_state(simulator))
        iterations = []
        for _ in range(4):
            plan = planner.plan(simulator.observe(state))
            iterations.append(planner._graph.iterations)
            state = simulator.step(state, plan.actions[0])
        self.assertLess(iterations[-1], iterations[0])


if __name__ == "__main__":
    unittest.main()
