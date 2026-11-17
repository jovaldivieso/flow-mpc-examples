from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.acados_graph import SLACK_TOLERANCE, AcadosKinematicGraph, TrackingCost
from systems.franka_system import FrankaSystem
from systems.free_space_sets import MAX_SET_SIZE
from test.planner_fixtures import pinned_config, rest_state

OBSTACLES = [
    {"position": [0.45, -0.12, 0.40], "radius": 0.08},
    {"position": [0.45, 0.15, 0.48], "radius": 0.08},
    {"position": [0.45, -0.40, 0.48], "radius": 0.08},
]
COST = TrackingCost(velocity_weight=1e-3, acceleration_weight=1e-3, jerk_weight=1e-4, terminal_weight=10.0)


def _graph(simulator: FrankaSystem, horizon: int = 5, with_obstacles: bool = True) -> AcadosKinematicGraph:
    return AcadosKinematicGraph(simulator, horizon, COST, with_obstacles=with_obstacles)


def _prepare(simulator: FrankaSystem, graph: AcadosKinematicGraph, state: np.ndarray, reference: np.ndarray | None = None) -> None:
    """Start the horizon from ``state``, held still as the initial guess and (unless given) as the reference, and linearize
    the obstacles along the reference."""
    reference = np.tile(state[: simulator.n], (graph.N, 1)) if reference is None else reference
    graph.set_state(state)
    graph.set_reference(reference)
    graph.set_initial(np.tile(state[:, None], (1, graph.N + 1)), np.zeros((simulator.nu, graph.N)))
    graph.update_collision_halfspaces(reference)


class AcadosKinematicGraphTests(unittest.TestCase):
    def test_the_problem_does_not_grow_with_the_number_of_obstacles_beyond_the_set_size(self) -> None:
        # One slack per sphere and node, shared by its half-spaces, and a set holds at most MAX_SET_SIZE of them
        for count in (1, 3):
            simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:count]))
            graph = _graph(simulator)
            num_spheres = len(simulator.collision_spheres(np.zeros(simulator.n))[2])
            self.assertEqual(graph.num_halfspaces, num_spheres * count)
            self.assertEqual(graph.num_slacks, num_spheres)
        many = [{"position": [0.45, -0.5 + 0.1 * i, 0.5], "radius": 0.04} for i in range(12)]
        graph = _graph(FrankaSystem(pinned_config(obstacles=many)))
        self.assertEqual(graph.num_halfspaces, num_spheres * MAX_SET_SIZE)
        self.assertEqual(graph.num_slacks, num_spheres)

    def test_a_solve_the_solver_reports_as_failed_raises(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        state[simulator.n : 2 * simulator.n] = 100.0  # no jerk brings that velocity within its limit in a step
        graph = _graph(simulator)
        _prepare(simulator, graph, state)

        with self.assertRaisesRegex(RuntimeError, "solver failed"):
            graph.solve()

    def test_a_good_solve_after_a_failed_one_succeeds_again(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        graph = _graph(simulator)
        fast = state.copy()
        fast[simulator.n : 2 * simulator.n] = 100.0
        for x0, succeeds in ((state, True), (fast, False), (state, True)):
            _prepare(simulator, graph, x0)
            if succeeds:
                np.testing.assert_allclose(graph.solve().states[:, 0], x0)
            else:
                with self.assertRaises(RuntimeError):
                    graph.solve()

    def test_a_reference_of_the_wrong_shape_is_rejected(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        graph = _graph(simulator)
        for wrong in (np.zeros((graph.N + 1, simulator.n)), np.zeros((graph.N, simulator.n - 1)), np.zeros(graph.N * simulator.n)):
            with self.subTest(shape=wrong.shape), self.assertRaises(ValueError):
                graph.set_reference(wrong)

    def test_the_initial_values_are_kept_between_solves_until_a_reset(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        graph = _graph(simulator)
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
        graph = _graph(simulator)
        _prepare(simulator, graph, state)

        with self.assertRaisesRegex(RuntimeError, "slack"):
            graph.solve()

    def test_a_solution_is_accepted_up_to_the_slack_tolerance_it_is_asked_for(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        simulator.collision.obstacles[:] = [(simulator.tcp_position(state[: simulator.n]), 0.05)]
        graph = _graph(simulator)
        _prepare(simulator, graph, state)

        solution = graph.solve(slack_tolerance=1.0)

        self.assertGreater(solution.max_slack, SLACK_TOLERANCE)
        self.assertLessEqual(solution.max_slack, 1.0)

    def test_the_solution_follows_the_dynamics_and_the_limits_among_obstacles(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES))
        graph = _graph(simulator)
        state = rest_state(simulator)
        _prepare(simulator, graph, state)

        solution = graph.solve()

        np.testing.assert_allclose(solution.states[:, 0], state)
        for k in range(graph.N):
            np.testing.assert_allclose(
                solution.states[:, k + 1], simulator.predict_next_state(solution.states[:, k], solution.actions[:, k]), atol=1e-9
            )
        self.assertLessEqual(np.abs(solution.actions).max(), np.max(simulator.max_action))
        self.assertFalse(solution.penetrates)

    def test_the_state_limits_bound_every_node(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=[]))
        joint, upper = 4, simulator.state_upper_bounds[4]
        state = rest_state(simulator)
        state[joint] = upper - 0.1
        graph = _graph(simulator, with_obstacles=False)
        reference = np.tile(state[: simulator.n], (graph.N, 1))
        reference[:, joint] = upper + 1.0  # a reference that pulls the joint past its limit
        _prepare(simulator, graph, state, reference)

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
        graph = _graph(simulator, horizon)
        self.assertFalse(simulator.collision.trajectory_is_collision_free(rollout[1:]))
        _prepare(simulator, graph, state, rollout[1:])  # a tracking that runs into the obstacle, linearized along it

        solution = graph.solve()

        self.assertTrue(simulator.collision.trajectory_is_collision_free(solution.states[: simulator.n, 1:].T))
        self.assertFalse(solution.penetrates)

    def test_a_graph_without_obstacles_has_no_half_spaces_and_no_slack(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES))
        graph = _graph(simulator, with_obstacles=False)
        state = rest_state(simulator)

        self.assertEqual((graph.num_halfspaces, graph.num_slacks), (0, 0))
        _prepare(simulator, graph, state)
        solution = graph.solve()
        self.assertEqual((solution.max_slack, solution.penetrates), (0.0, False))

    def test_a_solve_started_from_its_own_solution_returns_it(self) -> None:
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:2]))
        graph = _graph(simulator)
        state = rest_state(simulator)
        _prepare(simulator, graph, state)
        first = graph.solve()
        graph.set_initial(first.states, first.actions)

        again = graph.solve()

        np.testing.assert_allclose(again.states, first.states, atol=1e-5)
        np.testing.assert_allclose(again.actions, first.actions, atol=1e-5)

    def test_graphs_of_one_problem_share_the_built_solver_and_keep_their_own_settings(self) -> None:
        # Two graphs of the same dimensions load the same generated code, but their weights and data stay their own
        simulator = FrankaSystem(pinned_config(obstacles=OBSTACLES[:1]))
        state = rest_state(simulator)
        light = AcadosKinematicGraph(simulator, 5, COST._replace(jerk_weight=1e-6))
        heavy = AcadosKinematicGraph(simulator, 5, COST._replace(jerk_weight=1.0))
        reference = np.tile(state[: simulator.n], (5, 1))
        reference[:, 0] += 0.2
        for graph in (light, heavy):
            _prepare(simulator, graph, state, reference)

        light_jerk = np.abs(light.solve().actions).sum()
        heavy_jerk = np.abs(heavy.solve().actions).sum()

        self.assertLess(heavy_jerk, light_jerk)


if __name__ == "__main__":
    unittest.main()
