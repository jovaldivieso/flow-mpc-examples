from __future__ import annotations

import os
import sys
import unittest

import casadi as ca
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.casadi_graph import CasadiKinematicGraph
from planning.casadi_planner import CasadiPlanner
from planning.planner_config import PlannerConfig
from systems.franka_system import FrankaSystem
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
    def test_the_stages_do_not_grow_with_the_number_of_obstacles(self) -> None:
        # One slack per sphere and node, shared by all its obstacles: only the rows of the half-spaces grow with them
        horizon = 5
        simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:1]))
        num_spheres = CasadiKinematicGraph(simulator, horizon).num_collision_pairs
        expected = (horizon + 1) * simulator.nx + horizon * simulator.nu + horizon * num_spheres
        for count in (1, 3):
            simulator = FrankaSystem(pinned_config(expert="casadi", obstacles=OBSTACLES[:count]))
            self.assertEqual(CasadiKinematicGraph(simulator, horizon).opti.nx, expected)

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
            iterations.append(planner._graph.opti.stats()["iter_count"])
            state = simulator.step(state, plan.actions[0])
        self.assertLess(iterations[-1], iterations[0])


if __name__ == "__main__":
    unittest.main()
