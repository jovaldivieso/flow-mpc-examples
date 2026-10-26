from __future__ import annotations

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from learning.dagger.utils import sample_initial_state
from planning.casadi_planner import CasadiPlanner
from planning.planner_factory import PlannerFactory
from planning.ompl_initial_guess import OmplInitialGuess
from planning.ompl_planner import OmplPlanner
from planning.planner import Plan, Planner
from test.planner_fixtures import pinned_config

N = 7


class PlannerInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.start = sample_initial_state(cls.simulator, 3)  # also sets the goal of the sample
        cls.observation = cls.simulator.observe(cls.start)

    def _planners(self) -> list[Planner]:
        return [
            CasadiPlanner(self.simulator, self.config),
            OmplPlanner(self.simulator, self.config),
            PlannerFactory.create("ompl_casadi", self.simulator, self.config),
        ]

    def test_the_abstract_interface_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            Planner()

    def test_every_planner_plans_from_the_observed_configuration_to_a_plan_of_joint_positions(self) -> None:
        for planner in self._planners():
            with self.subTest(planner=type(planner).__name__):
                self.assertIsInstance(planner, Planner)

                plan = planner.plan(self.observation)

                self.assertIsInstance(plan, Plan)
                self.assertEqual(plan.positions.shape[1], N)
                self.assertGreaterEqual(len(plan.positions), 2)
                np.testing.assert_allclose(plan.positions[0], self.start[:N], atol=1e-9)

    def test_every_planner_accepts_an_initial_plan_and_can_be_reset(self) -> None:
        initial_plan = Plan(positions=np.tile(self.start[:N], (5, 1)))
        for planner in self._planners():
            with self.subTest(planner=type(planner).__name__):
                planner.plan(self.observation, initial_plan)
                planner.reset()

    def test_the_factory_builds_the_planner_of_every_name(self) -> None:
        expected = {"casadi": CasadiPlanner, "ompl_casadi": CasadiPlanner}

        self.assertEqual(set(PlannerFactory.names()), set(expected))
        for name, planner_class in expected.items():
            with self.subTest(name=name):
                self.assertIsInstance(PlannerFactory.create(name, self.simulator, self.config), planner_class)

    def test_the_factory_rejects_an_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            PlannerFactory.create("rrt", self.simulator, self.config)

    def test_every_planner_can_be_the_expert_of_a_rollout_label(self) -> None:
        # whatever the length of its plan, a label of prediction_horizon actions can be read off it
        for name in PlannerFactory.names():
            with self.subTest(name=name):
                planner = PlannerFactory.create(name, self.simulator, self.config)

                label = planner.plan(self.observation).first_actions(16)

                self.assertEqual(label.shape, (16, self.simulator.nu))

    def test_a_plan_shorter_than_a_label_goes_on_with_zero_actions(self) -> None:
        plan = Plan(positions=np.zeros((3, 2)), actions=np.array([[0.0, 1.0], [2.0, 3.0]]))

        np.testing.assert_array_equal(plan.first_actions(4), [[0.0, 1.0], [2.0, 3.0], [0.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(plan.first_actions(1), [[0.0, 1.0]])

    def test_a_path_has_no_actions_to_label_with(self) -> None:
        with self.assertRaises(ValueError):
            Plan(positions=np.zeros((3, 2))).first_actions(2)

    def test_a_planner_keeps_no_statistics_unless_it_says_so(self) -> None:
        self.assertEqual(OmplPlanner(self.simulator, self.config).statistics, {})
        self.assertEqual(CasadiPlanner(self.simulator, self.config).statistics, {"trajectory_optimizations": 0})
        ompl_casadi = PlannerFactory.create("ompl_casadi", self.simulator, self.config)
        self.assertIn("ompl_paths", ompl_casadi.statistics)
        self.assertIsInstance(ompl_casadi.initial_guess_generator, OmplInitialGuess)
        self.assertIsNone(CasadiPlanner(self.simulator, self.config).initial_guess_generator)


if __name__ == "__main__":
    unittest.main()
