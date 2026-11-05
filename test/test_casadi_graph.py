from __future__ import annotations

import os
import sys
import unittest

import casadi as ca
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.casadi_graph import SLACK_TOLERANCE, CasadiKinematicGraph
from systems.franka_system import FrankaSystem
from systems.free_space_sets import MAX_SET_SIZE
from test.planner_fixtures import pinned_config, rest_state

OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
    {"position": [0.45, -0.40, 0.48], "radius": 0.08},
]


def _graph(num_obstacles: int, horizon: int = 5, with_obstacles: bool = True) -> tuple[FrankaSystem, CasadiKinematicGraph]:
    """A graph whose cost is the squared jerk, on the Franka system with the first ``num_obstacles`` obstacles."""
    simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:num_obstacles]))
    graph = CasadiKinematicGraph(simulator, horizon, with_obstacles=with_obstacles)
    graph.minimize(ca.sumsqr(graph.U))
    return simulator, graph


def _prepare(simulator: FrankaSystem, graph: CasadiKinematicGraph, state: np.ndarray) -> None:
    """Start the horizon from ``state``, held still as the initial guess, and linearize the obstacles along it."""
    graph.set_state(state)
    graph.set_initial(np.tile(state[:, None], (1, graph.N + 1)), np.zeros((simulator.nu, graph.N)))
    graph.update_collision_halfspaces(np.tile(state[: simulator.n], (graph.N, 1)))


class CasadiKinematicGraphTests(unittest.TestCase):
    def test_the_stages_do_not_grow_with_the_number_of_obstacles_beyond_the_set_size(self) -> None:
        # One slack per sphere and node, shared by its half-spaces, and a set holds at most MAX_SET_SIZE of them
        horizon = 5
        for count in (1, 3):
            simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:count]))
            graph = CasadiKinematicGraph(simulator, horizon)
            num_spheres = len(simulator.collision_spheres(np.zeros(simulator.n))[2])
            self.assertEqual(graph.num_halfspaces, num_spheres * count)
            self.assertEqual(graph.opti.nx, (horizon + 1) * simulator.nx + horizon * simulator.nu + horizon * num_spheres)
        many = [{"position": [0.45, -0.5 + 0.1 * i, 0.5], "radius": 0.04} for i in range(12)]
        simulator = FrankaSystem(pinned_config(obstacles=many))
        graph = CasadiKinematicGraph(simulator, horizon)
        self.assertEqual(graph.num_halfspaces, num_spheres * MAX_SET_SIZE)
        self.assertEqual(graph.opti.nx, (horizon + 1) * simulator.nx + horizon * simulator.nu + horizon * num_spheres)

    def test_a_solve_the_solver_reports_as_failed_raises(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
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
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
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
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
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
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        graph = CasadiKinematicGraph(simulator, 5)
        states, actions = np.ones((simulator.nx, 6)), 2.0 * np.ones((simulator.nu, 5))

        np.testing.assert_array_equal(graph.initial_guess[0], np.zeros((simulator.nx, 6)))  # zero before anything is set
        graph.set_initial(states, actions)
        np.testing.assert_array_equal(graph.initial_guess[0], states)
        np.testing.assert_array_equal(graph.initial_guess[1], actions)
        graph.reset()
        np.testing.assert_array_equal(graph.initial_guess[1], np.zeros((simulator.nu, 5)))

    def test_a_solution_with_a_slack_above_the_tolerance_is_rejected(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
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
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
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

    def test_the_solution_follows_the_dynamics_and_the_limits_among_obstacles(self) -> None:
        simulator, graph = _graph(3)
        state = rest_state(simulator)
        _prepare(simulator, graph, state)

        solution = graph.solve()

        np.testing.assert_allclose(solution.states[:, 0], state)
        for k in range(graph.N):
            np.testing.assert_allclose(
                solution.states[:, k + 1], simulator.predict_next_state(solution.states[:, k], solution.actions[:, k]), atol=1e-5
            )
        self.assertLessEqual(np.abs(solution.actions).max(), np.max(simulator.max_action) + 1e-6)
        self.assertFalse(solution.penetrates)

    def test_the_state_limits_bound_every_node(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=[]))
        joint, upper = 4, simulator.state_upper_bounds[4]
        state = rest_state(simulator)
        state[joint] = upper - 0.1
        graph = CasadiKinematicGraph(simulator, 5, with_obstacles=False)
        graph.minimize(ca.sumsqr(graph.X[joint, 1:] - (upper + 1.0)))  # a cost that pulls the joint past its limit
        graph.set_state(state)
        graph.set_initial(np.tile(state[:, None], (1, 6)), np.zeros((simulator.nu, 5)))

        solution = graph.solve()

        self.assertLessEqual(solution.states[joint].max(), upper + 1e-6)
        self.assertGreater(solution.states[joint, -1], state[joint])

    def test_the_half_spaces_keep_the_spheres_clear_of_the_obstacles(self) -> None:
        horizon = 10
        simulator = FrankaSystem(pinned_config(obstacles=[]))
        state = rest_state(simulator)
        jerks = np.zeros((horizon, simulator.nu))
        jerks[:3, 0] = 8.0  # starts a motion of joint 1 that comes to rest only after the horizon
        rollout = simulator.rollout_positions(state, jerks)
        target = simulator.tcp_position(rollout[-1]).tolist()
        simulator = FrankaSystem(pinned_config(obstacles=[{"position": target, "radius": 0.05}]))
        graph = CasadiKinematicGraph(simulator, horizon)
        graph.minimize(ca.sumsqr(graph.X[: simulator.n, 1:] - rollout[1:].T))  # a tracking that runs into the obstacle
        self.assertFalse(simulator.collision.trajectory_is_collision_free(rollout[1:]))
        _prepare(simulator, graph, state)
        graph.update_collision_halfspaces(rollout[1:])  # the sets along the tracked reference

        solution = graph.solve()

        self.assertTrue(simulator.collision.trajectory_is_collision_free(solution.states[: simulator.n, 1:].T))
        self.assertFalse(solution.penetrates)

    def test_a_graph_without_obstacles_has_no_half_spaces_and_no_slack(self) -> None:
        simulator, graph = _graph(3, with_obstacles=False)
        state = rest_state(simulator)
        horizon = 5

        self.assertEqual(graph.num_halfspaces, 0)
        self.assertEqual(graph.opti.nx, (horizon + 1) * simulator.nx + horizon * simulator.nu)
        graph.set_state(state)
        graph.set_initial(np.tile(state[:, None], (1, horizon + 1)), np.zeros((simulator.nu, horizon)))
        graph.update_collision_halfspaces(np.tile(state[: simulator.n], (horizon, 1)))
        solution = graph.solve()
        self.assertEqual((solution.max_slack, solution.penetrates), (0.0, False))

    def test_a_solve_started_from_its_own_solution_returns_it(self) -> None:
        simulator, graph = _graph(2)
        state = rest_state(simulator)
        _prepare(simulator, graph, state)
        first = graph.solve()
        graph.set_initial(first.states, first.actions)

        again = graph.solve()

        np.testing.assert_allclose(again.states, first.states, atol=1e-5)
        np.testing.assert_allclose(again.actions, first.actions, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
