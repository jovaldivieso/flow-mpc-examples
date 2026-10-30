from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import meshcat.geometry as g
import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from systems.franka_system import Q_READY, FrankaSystem
from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.casadi_planner import CasadiPlanner
from planning.config import validate_system_config
from planning.planner_config import PlannerConfig
from scripts.evaluate_policy import load_policy
from scripts.meshcat_dataset import episode_summary, load_episodes, plan_positions, read_frames
from scripts.meshcat_utils import (
    TUBE_SIDES,
    draw_collision_spheres,
    draw_plan,
    move_collision_spheres,
    plan_tcp_points,
    tube_geometry,
)
from scripts.meshcat_policy import rollout_policy
from test.planner_fixtures import pose_goal

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")
PLAN_HORIZON = 4  # of the synthetic dataset
CHUNK = 8  # of the policies built here
N = 7


def _config(**overrides) -> dict:
    with open(CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


def _rest_state(simulator: FrankaSystem) -> np.ndarray:
    q = Q_READY.copy()
    return np.concatenate([q, np.zeros(2 * simulator.n)])


def _episode_rows(index: int, frames: int, environment: np.ndarray, q: np.ndarray, moving_plan: bool) -> pd.DataFrame:
    """Frames of a synthetic episode: the arm stays at ``q``; the labelled plan holds still or applies a jerk of
    15 rad/s^3 for ``PLAN_HORIZON`` steps."""
    rows = []
    for frame in range(frames):
        plan = np.full((PLAN_HORIZON, N), 15.0 if moving_plan else 0.0)
        rows.append(
            {
                "episode_index": index,
                "frame_index": frame,
                "observation.state": np.concatenate([q, np.zeros(2 * N)]).astype(np.float32),
                "observation.environment_state": environment.astype(np.float32),
                "action": plan.reshape(-1).astype(np.float32),
            }
        )
    return pd.DataFrame(rows)


class DatasetViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.simulator = FrankaSystem(_config())
        cls.q = _rest_state(cls.simulator)[:N]
        cls.goal = pose_goal(cls.simulator, cls.q + 0.1)
        cls.simulator.set_goal(cls.goal)
        state = np.concatenate([cls.q, np.zeros(2 * N)])
        cls.environment = cls.simulator.policy_observation(cls.simulator.observe(state))["observation.environment_state"]

    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        chunk = self.root / "data" / "chunk-000"
        chunk.mkdir(parents=True)
        _episode_rows(0, 5, self.environment, self.q, moving_plan=True).to_parquet(chunk / "file-000.parquet")
        pd.concat(
            [
                _episode_rows(1, 4, self.environment, self.q, moving_plan=False),
                _episode_rows(2, 6, self.environment, self.q, moving_plan=True),
            ]
        ).to_parquet(chunk / "file-001.parquet")
        # the file a running DAgger job is still writing: no parquet footer yet
        (chunk / "file-002.parquet").write_bytes(b"PAR1 not finished")

    def _load(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_episodes(self.root, self.simulator, *args, **kwargs)

    def test_an_unfinished_file_is_skipped_and_the_rest_read(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            frames = read_frames(self.root)

        self.assertEqual(sorted(frames["episode_index"].unique().tolist()), [0, 1, 2])
        self.assertIn("file-002.parquet", output.getvalue())

    def test_a_dataset_without_a_readable_file_is_an_error(self) -> None:
        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (empty / "data" / "chunk-000").mkdir(parents=True)
        with self.assertRaises(FileNotFoundError):
            read_frames(empty)

    def test_the_first_episodes_come_in_order_with_their_chunks_reshaped(self) -> None:
        episodes = self._load(2, seed=0, first=True)

        self.assertEqual([episode.index for episode in episodes], [0, 1])
        episode = episodes[0]
        self.assertEqual(episode.positions.shape, (5, N))
        self.assertEqual(episode.actions.shape, (5, PLAN_HORIZON, N))
        self.assertEqual(episode.velocities.shape, (5, N))
        np.testing.assert_allclose(episode.goal, self.goal, atol=1e-5)  # recovered from the goal in the TCP frame
        plans = plan_positions(self.simulator, episode)
        self.assertEqual(plans.shape, (5, PLAN_HORIZON + 1, N))
        np.testing.assert_allclose(plans[0, 0], episode.positions[0], atol=1e-6)  # row 0 is the frame's own q
        np.testing.assert_allclose(plans[0, -1], self.simulator.rollout_positions(episode.states[0], episode.actions[0])[-1])
        self.assertGreater(np.abs(plans[0, -1] - plans[0, 0]).max(), 0.0)

    def test_explicit_indices_are_loaded_and_unknown_ones_rejected(self) -> None:
        self.assertEqual([episode.index for episode in self._load(10, seed=0, indices=[2, 0])], [2, 0])
        with self.assertRaises(ValueError):
            self._load(10, seed=0, indices=[7])

    def test_random_episodes_are_reproducible_by_seed(self) -> None:
        first = [episode.index for episode in self._load(2, seed=5)]
        self.assertEqual(first, [episode.index for episode in self._load(2, seed=5)])
        self.assertEqual(len(first), 2)

    def test_the_summary_tells_a_moving_episode_from_a_stalled_one(self) -> None:
        moving, stalled = self._load(3, seed=0, first=True)[0], self._load(3, seed=0, first=True)[1]

        moving_summary = episode_summary(self.simulator, moving)
        stalled_summary = episode_summary(self.simulator, stalled)

        self.assertEqual(moving_summary["frames"], 5)
        self.assertEqual(moving_summary["stay_label_fraction"], 0.0)
        moving_plan = self.simulator.rollout_positions(moving.states[0], moving.actions[0])
        self.assertAlmostEqual(moving_summary["plan_excursion"], np.abs(moving_plan[1:] - moving_plan[:1]).max(), places=6)
        self.assertGreater(moving_summary["plan_excursion"], 0.05)
        self.assertEqual(stalled_summary["stay_label_fraction"], 1.0)
        self.assertEqual(stalled_summary["at_rest_fraction"], 1.0)
        self.assertGreater(stalled_summary["final_position_error"], 0.0)  # it never got to the goal

    def test_plan_tcp_points_are_the_forward_kinematics_of_the_plan(self) -> None:
        plan = np.stack([self.q, self.q + 0.1])
        points = plan_tcp_points(self.simulator, plan)

        self.assertEqual(points.shape, (2, 3))
        np.testing.assert_allclose(points[1], self.simulator.tcp_placement(self.q + 0.1).translation)


class _RecordingNode:
    """Stands in for a Meshcat scene node: records what is set at each path."""

    def __init__(self, objects: dict, transforms: dict, path: str = "") -> None:
        self.objects, self.transforms, self.path = objects, transforms, path

    def __getitem__(self, key) -> "_RecordingNode":
        return _RecordingNode(self.objects, self.transforms, f"{self.path}/{key}")

    def set_object(self, geometry, material=None) -> None:
        self.objects[self.path] = geometry

    def set_transform(self, matrix) -> None:
        self.transforms[self.path] = np.asarray(matrix)

    def delete(self) -> None:
        self.objects.clear()


class CollisionGeometryDrawingTests(unittest.TestCase):
    def test_the_spheres_are_created_with_their_radii_and_move_to_the_centers(self) -> None:
        objects, transforms = {}, {}
        viz = type("Viewer", (), {"viewer": _RecordingNode(objects, transforms)})()
        simulator = FrankaSystem(_config())
        centers, _, radii = simulator.collision_spheres(Q_READY)

        draw_collision_spheres(viz, radii)
        move_collision_spheres(viz, centers)

        self.assertEqual(len(objects), len(radii))
        self.assertEqual(sorted(round(sphere.radius, 6) for sphere in objects.values()), sorted(np.round(radii, 6)))
        for index, center in enumerate(centers):
            np.testing.assert_allclose(transforms[f"/collision_spheres/{index}"][:3, 3], center)


class ThickPathDrawingTests(unittest.TestCase):
    def test_a_tube_surrounds_each_segment_at_its_radius_and_skips_empty_segments(self) -> None:
        starts = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        ends = np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 2.0]])

        geometry = tube_geometry(starts, ends, 0.01)

        self.assertEqual(len(geometry.vertices), 2 * 2 * TUBE_SIDES)
        self.assertEqual(len(geometry.faces), 2 * 2 * TUBE_SIDES)
        first = geometry.vertices[: 2 * TUBE_SIDES].astype(float)
        axis = np.array([1.0, 0.0, 0.0])
        offsets = first - np.where(np.arange(2 * TUBE_SIDES)[:, None] < TUBE_SIDES, starts[0], ends[0])
        np.testing.assert_allclose(offsets @ axis, 0.0, atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(offsets, axis=1), 0.01, atol=1e-6)

    def test_segment_colors_follow_the_kept_segments(self) -> None:
        starts, ends = np.zeros((2, 3)), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        geometry = tube_geometry(starts, ends, 0.01, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

        np.testing.assert_allclose(geometry.color, np.tile([0.0, 1.0, 0.0], (2 * TUBE_SIDES, 1)))

    def test_a_plan_is_one_thick_mesh_and_a_single_point_removes_it(self) -> None:
        objects, transforms = {}, {}
        viz = type("Viewer", (), {"viewer": _RecordingNode(objects, transforms)})()

        draw_plan(viz, "plan", np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0]]), 0x00D0FF)
        self.assertIsInstance(objects["/plan"], g.TriangularMeshGeometry)

        draw_plan(viz, "plan", np.zeros((1, 3)), 0x00D0FF)
        self.assertEqual(objects, {})


class PolicyRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _config()
        cls.simulator = FrankaSystem(cls.config)
        cls.state = _rest_state(cls.simulator)
        cls.simulator.set_goal(pose_goal(cls.simulator, cls.state[:N] + 0.3))
        torch.manual_seed(0)
        cls.policy = PolicyFactory.create(
            PolicySpec(
                policy_type="flow", action_dim=N, state_dim=cls.simulator.environment_dim + cls.simulator.nx,
                prediction_horizon=CHUNK, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=2,
                action_scale=tuple(cls.simulator.max_action.tolist()),
            ),
            cls.simulator,
        ).eval()

    def test_a_rollout_records_the_visited_states_and_the_chunk_planned_at_each(self) -> None:
        rollout = rollout_policy(self.simulator, self.policy, self.state, 3, torch.device("cpu"))

        steps = len(rollout.chunks)
        self.assertEqual(rollout.states.shape, (steps + 1, 3 * N))
        self.assertEqual(rollout.chunks.shape, (steps, CHUNK, N))
        self.assertIsNone(rollout.expert_plans)
        self.assertIn(rollout.outcome, {"timeout", "collision", "reached", "solve failure"})
        np.testing.assert_allclose(rollout.states[0], self.state)

    def test_the_executed_step_applies_the_first_row_of_the_chunk(self) -> None:
        rollout = rollout_policy(self.simulator, self.policy, self.state, 1, torch.device("cpu"))

        np.testing.assert_allclose(rollout.states[1], self.simulator.predict_next_state(self.state, rollout.chunks[0][0]))

    def test_the_expert_plans_from_every_visited_state_when_asked_to(self) -> None:
        planner = CasadiPlanner(self.simulator, PlannerConfig.from_mapping(self.config))

        rollout = rollout_policy(self.simulator, self.policy, self.state, 2, torch.device("cpu"), planner)

        self.assertEqual(rollout.expert_plans.shape, rollout.chunks.shape)
        first_action = CasadiPlanner(self.simulator, PlannerConfig.from_mapping(self.config)).plan(self.simulator.observe(self.state)).actions[0]
        np.testing.assert_allclose(rollout.expert_plans[0][0], first_action, atol=1e-6)  # the plan starts at the state

    def test_a_rollout_that_starts_at_the_goal_is_reached_without_steps(self) -> None:
        simulator = FrankaSystem(_config(done_hold_steps=1))  # one check at the goal and at rest is enough
        simulator.set_goal(pose_goal(simulator, self.state[:N]))

        rollout = rollout_policy(simulator, self.policy, self.state, 5, torch.device("cpu"))

        self.assertEqual((rollout.outcome, len(rollout.chunks)), ("reached", 0))


class LoadPolicyTests(unittest.TestCase):
    def test_the_policy_of_a_checkpoint_runs_with_its_ema_weights_in_eval_mode(self) -> None:
        simulator = FrankaSystem(_config())
        state_dim = simulator.environment_dim + simulator.nx
        template = PolicyFactory.create(
            PolicySpec(
                policy_type="flow", action_dim=N, state_dim=state_dim, prediction_horizon=CHUNK, unet_dim=8,
                unet_dim_mults=(1, 2), num_inference_steps=2, action_scale=(1.0,) * N,
            ),
            simulator,
        )
        averaged = {name: torch.full_like(tensor, 0.25) for name, tensor in template.state_dict().items()}
        checkpoint = {
            "model_state_dict": template.state_dict(),
            "ema_model_state_dict": averaged,
            "state_dim": state_dim,
            "action_dim": N,
            "prediction_horizon": CHUNK,
            "unet": {"dim": 8, "dim_mults": [1, 2]},
            "policy_type": "flow",
            "flow_config": {"num_inference_steps": 2, "safeflow_init": "previous_safe", "action_scale": [1.0] * N},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save(checkpoint, path)

            policy = load_policy("flow", path, simulator, {}, torch.device("cpu"))

        self.assertFalse(policy.training)
        for name, tensor in policy.state_dict().items():
            torch.testing.assert_close(tensor, averaged[name])


if __name__ == "__main__":
    unittest.main()
