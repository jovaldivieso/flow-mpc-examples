from __future__ import annotations

import dataclasses
import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import FrankaSystem
from learning.dagger.utils import sample_initial_state
from planning.ompl_casadi_planner import OmplCasadiPlanner
from planning.planner import Plan, Planner, PlannerSolveError
from planning.planner_factory import PlannerFactory
from test.planner_fixtures import pinned_config

N = 7
HORIZON = 20


class _HoldStill(Planner):
    """The smallest expert: the plan holds the configuration of the observation."""

    def __init__(self, simulator: FrankaSystem) -> None:
        self.sim = simulator

    def reset(self) -> None:
        pass

    def plan(self, observation: np.ndarray) -> Plan:
        q = self.sim.invert_obs(observation)[:N]
        return Plan(positions=np.tile(q, (3, 1)), actions=np.zeros((3, N)))


class PlannerInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = pinned_config()
        cls.simulator = FrankaSystem(cls.config)
        cls.start = sample_initial_state(cls.simulator, 3)  # also sets the goal of the sample
        cls.observation = cls.simulator.observe(cls.start)

    def test_the_abstract_interface_cannot_be_instantiated(self) -> None:
        with self.assertRaises(TypeError):
            Planner()

    def test_a_planner_keeps_no_statistics_unless_it_says_so(self) -> None:
        self.assertEqual(_HoldStill(self.simulator).statistics, {})

    def test_a_planner_plans_from_the_observation_to_a_plan_of_joint_positions(self) -> None:
        planner = _HoldStill(self.simulator)

        plan = planner.plan(self.observation)

        self.assertIsInstance(plan, Plan)
        np.testing.assert_allclose(plan.positions, np.tile(self.start[:N], (3, 1)))

    def test_a_plan_is_immutable(self) -> None:
        plan = Plan(positions=np.zeros((3, 2)), actions=np.zeros((3, 2)))

        with self.assertRaises(dataclasses.FrozenInstanceError):
            plan.actions = np.zeros((3, 2))

    def test_the_factory_has_the_ompl_casadi_planner_only(self) -> None:
        self.assertEqual(PlannerFactory.names(), ("ompl_casadi",))

    def test_the_factory_builds_the_planner_of_every_name(self) -> None:
        for name in PlannerFactory.names():
            with self.subTest(name=name):
                planner = PlannerFactory.create(name, self.simulator, self.config)

                self.assertIsInstance(planner, Planner)
                self.assertIsInstance(planner, OmplCasadiPlanner)

    def test_the_factory_rejects_an_unknown_name(self) -> None:
        with self.assertRaises(ValueError):
            PlannerFactory.create("rrt", self.simulator, self.config)

    def test_the_planner_of_the_factory_takes_its_settings_from_the_config(self) -> None:
        config = pinned_config(horizon=7, replan_distance=0.3)

        planner = PlannerFactory.create("ompl_casadi", self.simulator, config)

        self.assertEqual((planner.config.horizon, planner.config.replan_distance), (7, 0.3))

    def test_every_planner_can_be_the_expert_of_a_rollout_label(self) -> None:
        for name in PlannerFactory.names():
            with self.subTest(name=name):
                planner = PlannerFactory.create(name, self.simulator, self.config)

                plan = planner.plan(self.observation)

                self.assertEqual(plan.positions.shape, (HORIZON, N))
                self.assertEqual(plan.actions.shape, (HORIZON, self.simulator.nu))
                self.assertEqual(plan.actions[0].shape, (self.simulator.nu,))

    def test_a_planner_can_be_reset(self) -> None:
        for name in PlannerFactory.names():
            with self.subTest(name=name):
                planner = PlannerFactory.create(name, self.simulator, self.config)
                planner.plan(self.observation)

                planner.reset()

                self.assertEqual(planner.statistics["ompl_paths"], 1)  # the counters are cumulative


if __name__ == "__main__":
    unittest.main()
