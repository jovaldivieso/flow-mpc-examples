from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dynamics.franka_dynamics import Q_READY, FrankaDynamics
from learning.finetune_safe import generate_paths, path_batch, projected_path
from planning.casadi_projector import CasadiTrajectoryProjector
from planning.config import validate_system_config

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")
HORIZON = 8
PATH_STEPS = 3


def _config() -> dict:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update({"horizon": HORIZON})
    return validate_system_config(raw_config=raw_config)


def _rest_state(simulator: FrankaDynamics) -> np.ndarray:
    q = Q_READY.copy()
    return np.concatenate([q, np.zeros(2 * simulator.n)])


class ProjectedPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _config()
        cls.simulator = FrankaDynamics(cls.config)
        cls.projector = CasadiTrajectoryProjector(cls.simulator, cls.config, horizon=HORIZON)
        cls.state = _rest_state(cls.simulator)
        cls.observation = cls.simulator.observe(cls.state)
        cls.plan = np.zeros((HORIZON, cls.simulator.nu))  # a gentle jerk on the first joint, then rest
        cls.plan[:2, 0] = [2.0, -2.0]

    def test_the_path_runs_from_the_projected_noise_to_the_projected_plan(self) -> None:
        noise = np.random.default_rng(0).normal(size=self.plan.shape) * self.simulator.max_action

        path = projected_path(self.projector, self.observation, self.plan, noise, PATH_STEPS)

        self.assertEqual(path.shape, (PATH_STEPS + 1, HORIZON, self.simulator.nu))
        # The last point is the projection of the plan itself, which a safe plan stays close to
        self.projector.reset()
        np.testing.assert_allclose(path[-1], self.projector.project(self.observation, self.plan), atol=1e-4)
        # while the first one (the projected noise) is far from it
        self.assertGreater(np.max(np.abs(path[0] - path[-1])), 1.0)

    def test_every_point_of_the_path_is_collision_free(self) -> None:
        noise = np.random.default_rng(1).normal(size=self.plan.shape) * self.simulator.max_action

        path = projected_path(self.projector, self.observation, self.plan, noise, PATH_STEPS)

        for chunk in path:
            for q in self.simulator.rollout_positions(self.state, chunk):
                self.assertFalse(self.simulator.is_collision(q))

    def test_generate_paths_returns_one_path_per_frame_and_noise_draw_deterministically(self) -> None:
        observations = np.stack([self.observation, self.observation])
        plans = np.stack([self.plan, self.plan])

        first = generate_paths(self.config, observations, plans, noise_samples=2, num_steps=PATH_STEPS, seed=3, num_workers=1)
        second = generate_paths(self.config, observations, plans, noise_samples=2, num_steps=PATH_STEPS, seed=3, num_workers=1)

        frames, paths, rows, failures = first
        self.assertEqual(failures, 0)
        self.assertEqual(frames.tolist(), [0, 0, 1, 1])
        self.assertEqual(paths.shape, (4, PATH_STEPS + 1, HORIZON, self.simulator.nu))
        self.assertEqual(rows.shape, (4, self.simulator.obs_dim))
        np.testing.assert_allclose(paths, second[1])
        # Different noise draws of the same frame give different paths
        self.assertFalse(np.allclose(paths[0, 0], paths[1, 0]))


class PathBatchTests(unittest.TestCase):
    def test_the_batch_holds_grid_points_with_their_finite_difference_velocity(self) -> None:
        num_steps = 4
        # path l is the constant chunk l * (1, ..., 1) so that the velocity is num_steps everywhere
        paths = torch.arange(num_steps + 1, dtype=torch.float32).view(1, -1, 1, 1).repeat(3, 1, 2, 2)
        rows = torch.arange(3, dtype=torch.float32).view(3, 1).repeat(1, 4)
        torch.manual_seed(0)

        conditioning, x_t, dx_t, t = path_batch(paths, rows, 64, torch.Generator().manual_seed(0))

        self.assertEqual(tuple(x_t.shape), (64, 2, 2))
        self.assertEqual(tuple(conditioning.shape), (64, 4))
        torch.testing.assert_close(dx_t, torch.full_like(dx_t, float(num_steps)))
        # t is the grid time of x_t (index / num_steps), never the last grid point
        torch.testing.assert_close(t * num_steps, x_t[:, 0, 0])
        self.assertTrue(bool((t < 1.0).all()))
        # the conditioning row belongs to the sampled path
        self.assertTrue(bool((conditioning[:, 0] == conditioning[:, 1]).all()))


if __name__ == "__main__":
    unittest.main()
