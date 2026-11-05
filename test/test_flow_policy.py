from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import observation_features
from learning.models.policy_factory import PolicyFactory, PolicySpec
from learning.models.pose_guidance import PoseGuidance
from planning.casadi_projector import CasadiTrajectoryProjector, Projection
from systems.franka_system import Q_READY, FrankaSystem
from systems.point_mass_system import PointMassSystem
from test.planner_fixtures import pinned_config


HISTORY = 3
EXECUTION = 2
COND_DIM = 2 + 2 * HISTORY  # the relative goal of the point mass and its last HISTORY positions
STATE_DIM = 6


def _flow_policy(horizon: int = 4, **kwargs) -> FlowPolicy:
    """A flow policy without tracker, guided on a point mass (its guidance is stored, the tests of the network never call it)."""
    kwargs.setdefault("history_length", HISTORY)
    kwargs.setdefault("execution_horizon", EXECUTION)
    return FlowPolicy(guidance=PoseGuidance(PointMassSystem({}), horizon), tracker=None, prediction_horizon=horizon, **kwargs)


def _build_policy(action_scale=None) -> FlowPolicy:
    return _flow_policy(
        action_dim=2, cond_dim=COND_DIM, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=4, action_scale=action_scale,
    )


def _spec(policy_type: str = "flow", **overrides) -> PolicySpec:
    fields = dict(
        policy_type=policy_type, action_dim=2, cond_dim=COND_DIM, history_length=HISTORY, execution_horizon=EXECUTION,
        prediction_horizon=4, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=4, action_scale=(1.0, 2.0),
    )
    return PolicySpec(**{**fields, **overrides})


def _tracked_policy(simulator: PointMassSystem, **overrides) -> FlowPolicy:
    """A flow policy of the factory: guided toward the goal of the point mass and tracked within its limits."""
    return PolicyFactory.create(_spec(**overrides), simulator, {}).eval()


def _unguided(proposal):
    return proposal


def _fake_observation(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "observation.environment_state": torch.zeros(batch_size, 2),
        "observation.state": torch.zeros(batch_size, STATE_DIM),
        "observation.history": torch.zeros(batch_size, 2 * HISTORY),
    }


def _point_mass_batch(simulator: PointMassSystem, state: np.ndarray) -> dict[str, torch.Tensor]:
    """The policy observation (a batch of one) of the point mass in ``state`` with a history standing still at its position."""
    history = np.tile(state[:2], HISTORY)
    observation = simulator.policy_observation(simulator.observe(state), history)
    return {name: torch.as_tensor(value)[None] for name, value in observation.items()}


class ActionScaleValidationTests(unittest.TestCase):
    def test_wrong_length_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[1.0, 2.0, 3.0])

    def test_non_positive_entry_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[1.0, 0.0])
        with self.assertRaises(ValueError):
            _build_policy(action_scale=[-1.0, 1.0])

    def test_non_finite_entry_rejected(self) -> None:
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad_value=bad_value):
                with self.assertRaises(ValueError):
                    _build_policy(action_scale=[1.0, bad_value])

    def test_omitted_defaults_to_all_ones(self) -> None:
        policy = _build_policy(action_scale=None)
        torch.testing.assert_close(policy.velocity_net.action_scale, torch.ones(2))


class FlowTimeSamplingTests(unittest.TestCase):
    """The training flow time is t = 1 - Beta(1.5, 1): its density rises toward t = 0, so the noisy start of the
    flow gets more of the training samples (mean 1 - 1.5 / 2.5 = 0.4, against 0.5 for a uniform t)."""

    def test_flow_times_follow_one_minus_beta_1_5_1(self) -> None:
        policy = _build_policy()
        recorded: list[torch.Tensor] = []
        real_predict = policy.velocity_net.predict_velocity

        def recording_predict(action_state_flat, obs_cond, timesteps):
            recorded.append(timesteps.clone())
            return real_predict(action_state_flat, obs_cond, timesteps)

        torch.manual_seed(0)
        with mock.patch.object(policy.velocity_net, "predict_velocity", recording_predict):
            policy.compute_loss(_fake_observation(20000), torch.zeros(20000, 4, 2))

        t = recorded[0]
        self.assertEqual(tuple(t.shape), (20000,))
        self.assertTrue(bool(((t >= 0.0) & (t <= 1.0)).all()))
        self.assertAlmostEqual(float(t.mean()), 0.4, delta=0.01)
        # P(t < 0.25) = P(Beta > 0.75) = 1 - 0.75^1.5 = 0.351 (uniform: 0.25)
        self.assertAlmostEqual(float((t < 0.25).float().mean()), 1.0 - 0.75**1.5, delta=0.015)


class PathLossTests(unittest.TestCase):
    """With paths every row is learned along its path (piecewise linear in the flow time); without, from Gaussian noise."""

    SEGMENTS = 4

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.net = _build_policy(action_scale=[1.0, 2.0]).velocity_net
        self.actions = torch.randn(4, 4, 2)
        self.paths = torch.randn(4, self.SEGMENTS + 1, 4, 2)

    def _flow(self, paths: torch.Tensor | None, flow_times: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The point and the flow time the network is shown and the velocity it is trained towards (normalized units)."""
        shown: list[tuple[torch.Tensor, torch.Tensor]] = []

        def recording_predict(chunk_flat, obs_cond, timesteps):
            shown.append((chunk_flat.view(-1, 4, 2).clone(), timesteps.clone()))
            return torch.zeros_like(chunk_flat)

        targets: list[torch.Tensor] = []
        real_mse = torch.nn.functional.mse_loss

        def recording_mse(prediction, target):
            targets.append(target.view(-1, 4, 2).clone())
            return real_mse(prediction, target)

        with mock.patch.object(self.net, "predict_velocity", recording_predict), mock.patch.object(torch.nn.functional, "mse_loss", recording_mse):
            if flow_times is None:
                self.net.compute_loss(torch.zeros(4, COND_DIM), self.actions, paths)
            else:
                with mock.patch.object(self.net._flow_time_distribution, "sample", return_value=1.0 - flow_times):
                    self.net.compute_loss(torch.zeros(4, COND_DIM), self.actions, paths)
        return shown[0][0], shown[0][1], targets[0]

    def test_a_straight_path_is_interpolated_with_the_velocity_of_its_line(self) -> None:
        start, end = torch.randn(4, 4, 2), torch.randn(4, 4, 2)
        along = torch.linspace(0.0, 1.0, self.SEGMENTS + 1).view(1, -1, 1, 1)
        straight = start.unsqueeze(1) + along * (end - start).unsqueeze(1)

        x_t, t, velocity = self._flow(straight)

        scale = self.net.action_scale
        torch.testing.assert_close(x_t, (start + t.view(-1, 1, 1) * (end - start)) / scale, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(velocity, (end - start) / scale, atol=1e-5, rtol=1e-5)

    def test_a_bent_path_has_the_slope_of_the_segment_of_the_flow_time_and_its_point_lies_on_the_polyline(self) -> None:
        flow_times = torch.tensor([0.1, 0.3, 0.6, 0.95])  # the segments 0, 1, 2 and 3

        x_t, t, velocity = self._flow(self.paths, flow_times)

        scale = self.net.action_scale
        for row, segment in enumerate(range(4)):
            start, end = self.paths[row, segment] / scale, self.paths[row, segment + 1] / scale
            along = flow_times[row] * self.SEGMENTS - segment
            torch.testing.assert_close(x_t[row], (1 - along) * start + along * end, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(velocity[row], (end - start) * self.SEGMENTS, atol=1e-4, rtol=1e-4)

    def test_the_point_is_the_first_and_the_last_value_of_the_path_at_the_flow_times_zero_and_one(self) -> None:
        for time, index in ((0.0, 0), (1.0, self.SEGMENTS)):
            with self.subTest(time=time):
                x_t, _, _ = self._flow(self.paths, torch.full((4,), time))

                torch.testing.assert_close(x_t, self.paths[:, index] / self.net.action_scale, atol=1e-5, rtol=1e-5)

    def test_without_paths_every_row_is_learned_from_noise_towards_its_chunk(self) -> None:
        x_t, t, velocity = self._flow(None)

        x_1 = self.actions / self.net.action_scale
        x_0 = x_1 - velocity
        torch.testing.assert_close(x_t, t.view(-1, 1, 1) * x_1 + (1 - t.view(-1, 1, 1)) * x_0, atol=1e-5, rtol=1e-5)
        self.assertGreater(float(x_0.std()), 0.3)  # the start is noise

    def test_every_row_is_learned_along_its_path(self) -> None:
        x_t, t, velocity = self._flow(self.paths)

        scale = self.net.action_scale
        for row in range(4):
            segment = min(int(t[row] * self.SEGMENTS), self.SEGMENTS - 1)
            slope = (self.paths[row, segment + 1] - self.paths[row, segment]) / scale * self.SEGMENTS
            torch.testing.assert_close(velocity[row], slope, atol=1e-4, rtol=1e-4)

    def test_the_policy_hands_the_paths_to_the_network(self) -> None:
        policy = _build_policy()

        with mock.patch.object(policy.velocity_net, "compute_loss", return_value=torch.zeros(())) as compute_loss:
            policy.compute_loss(_fake_observation(4), self.actions, self.paths)

        self.assertIs(compute_loss.call_args.args[2], self.paths)


class VelocityNetworkTests(unittest.TestCase):
    """The network is SafeFlowMPC's temporal U-Net, with the flow time passed as it is."""

    def test_the_velocity_is_predicted_by_the_unet_on_the_chunk_the_condition_and_the_time(self) -> None:
        policy = _build_policy()
        net = policy.velocity_net
        chunk, condition, time = torch.randn(3, 4, 2), torch.randn(3, COND_DIM), torch.rand(3)

        velocity = net.predict_velocity(chunk.flatten(1), condition, time)

        torch.testing.assert_close(velocity, net.unet(chunk, time, condition).flatten(1))
        self.assertEqual(tuple(velocity.shape), (3, 8))

    def test_the_horizon_must_survive_the_downsampling_of_the_unet(self) -> None:
        with self.assertRaises(ValueError):
            _flow_policy(6, action_dim=2, cond_dim=COND_DIM, unet_dim=8, unet_dim_mults=(1, 2, 4))

    def test_the_channel_count_must_suit_the_group_norm(self) -> None:
        with self.assertRaises(ValueError):
            _flow_policy(8, action_dim=2, cond_dim=COND_DIM, unet_dim=12)

    def test_the_default_network_has_the_size_of_the_one_of_safeflowmpc(self) -> None:
        net = _flow_policy(16, action_dim=7, cond_dim=26).velocity_net

        self.assertEqual(sum(p.numel() for p in net.unet.parameters()), 3_962_311)  # dim 32, mults (1, 2, 4, 8)


class ComputeLossActionScaleInvarianceTests(unittest.TestCase):
    """compute_loss divides actions by action_scale before ever touching the
    x_0 ~ N(0, 1) noise prior (see FlowPolicy's own docstring for why: an
    isotropic prior trained against anisotropic raw action bounds lets
    whichever dimension has the largest physical range dominate the loss,
    regardless of task relevance). That means feeding in actions already
    pre-multiplied by a given scale, under that same scale, must reduce to
    exactly the unscaled case: both compute x_1 = raw_actions internally.
    """

    def test_prescaled_actions_reproduce_the_unscaled_loss(self) -> None:
        policy = _build_policy(action_scale=None)
        obs = _fake_observation(batch_size=5)
        raw_actions = torch.randn(5, policy.velocity_net.prediction_horizon, policy.velocity_net.action_dim)

        torch.manual_seed(0)
        baseline_loss = policy.compute_loss(obs, raw_actions)

        scale = torch.tensor([2.0, 5.0])
        with torch.no_grad():
            policy.velocity_net.action_scale.copy_(scale)
        scaled_actions = raw_actions * scale

        torch.manual_seed(0)
        rescaled_loss = policy.compute_loss(obs, scaled_actions)

        torch.testing.assert_close(rescaled_loss, baseline_loss)

    def test_skipping_normalization_gives_a_different_loss(self) -> None:
        """Sanity check that the test above isn't vacuously true -- omitting
        the division by action_scale for the same prescaled input must land
        on a genuinely different loss value, confirming the test can fail.
        """
        policy = _build_policy(action_scale=None)
        obs = _fake_observation(batch_size=5)
        raw_actions = torch.randn(5, policy.velocity_net.prediction_horizon, policy.velocity_net.action_dim)

        torch.manual_seed(0)
        baseline_loss = policy.compute_loss(obs, raw_actions)

        scale = torch.tensor([2.0, 5.0])
        scaled_actions = raw_actions * scale  # deliberately NOT re-pointing action_scale

        torch.manual_seed(0)
        unnormalized_loss = policy.compute_loss(obs, scaled_actions)

        self.assertFalse(torch.allclose(unnormalized_loss, baseline_loss))


class IntegrateRescalesToPhysicalUnitsTests(unittest.TestCase):
    """integrate runs entirely in the normalized space compute_loss trains in, then must rescale back to physical
    action units before returning -- every downstream consumer (apply_execution_noise, simulator.step, ...) expects
    physical units, not the network's internal representation.
    """

    def test_zero_network_output_returns_exactly_the_scaled_noise_draw(self) -> None:
        # With every weight and bias zeroed, predict_velocity is
        # identically 0 regardless of input (Linear(0)=0, Mish(0)=0
        # propagates through every layer), so the Euler loop never changes x away from its initial torch.randn
        # draw -- the only transformation left to verify is the final "* action_scale" rescale.
        scale = [1.0, 100.0]
        policy = _build_policy(action_scale=scale)
        with torch.no_grad():
            for param in policy.velocity_net.unet.parameters():
                param.zero_()

        torch.manual_seed(42)
        expected_x0 = torch.randn(1, policy.velocity_net.prediction_horizon, policy.velocity_net.action_dim)

        torch.manual_seed(42)
        action = policy.velocity_net.integrate(observation_features(_fake_observation(batch_size=1)), _unguided)

        torch.testing.assert_close(action, expected_x0 * torch.tensor(scale))


class ObservationConditioningTests(unittest.TestCase):
    """The flow is conditioned on [observation.environment_state, observation.history] in that order, not on the state
    (velocity and acceleration), which only the guidance and the tracking read."""

    @staticmethod
    def _distinguishable_observation() -> dict[str, torch.Tensor]:
        return {
            "observation.environment_state": torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
            "observation.history": torch.tensor([[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]]),
            "observation.state": torch.full((2, STATE_DIM), -1.0),
        }

    def test_observation_features_concatenates_environment_state_then_history(self) -> None:
        torch.testing.assert_close(
            observation_features(self._distinguishable_observation()),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]]),
        )

    def test_the_features_do_not_depend_on_the_state(self) -> None:
        observation = self._distinguishable_observation()
        other = {**observation, "observation.state": torch.randn(2, STATE_DIM)}

        torch.testing.assert_close(observation_features(other), observation_features(observation))

    def test_velocity_net_conditioning_width_is_the_environment_plus_the_history(self) -> None:
        self.assertEqual(_build_policy().velocity_net.obs_cond_dim, COND_DIM)

    def test_the_history_length_and_the_execution_horizon_are_the_ones_the_policy_was_built_for(self) -> None:
        policy = _build_policy()

        self.assertEqual(policy.history_length, HISTORY)
        self.assertEqual(policy.execution_horizon, EXECUTION)

    def test_compute_loss_and_select_action_condition_on_observation_features(self) -> None:
        policy = _tracked_policy(PointMassSystem({"goal": [0.5, 0.0]}))
        observation = self._distinguishable_observation()
        expected = observation_features(observation)
        positions = torch.randn(2, policy.velocity_net.prediction_horizon, policy.velocity_net.action_dim)
        single = {key: value[:1] for key, value in observation.items()}

        with mock.patch.object(
            policy.velocity_net, "compute_loss", wraps=policy.velocity_net.compute_loss
        ) as compute_loss:
            policy.compute_loss(observation, positions)
        torch.testing.assert_close(compute_loss.call_args.args[0], expected)
        torch.testing.assert_close(compute_loss.call_args.args[1], positions)
        self.assertIsNone(compute_loss.call_args.args[2])

        with mock.patch.object(policy, "guide", side_effect=lambda _, proposal: proposal), mock.patch.object(
            policy.tracker, "project", side_effect=lambda state, positions: Projection(positions, np.zeros((4, 2)))
        ), mock.patch.object(policy.velocity_net, "integrate", wraps=policy.velocity_net.integrate) as integrate:
            policy.select_action(single)
        torch.testing.assert_close(integrate.call_args.args[0], expected[:1])


class FlowPolicyLearnsAConditionalMappingTests(unittest.TestCase):
    """End-to-end check of the flow-matching math (loss direction, time convention, Euler integration):
    a small policy trained for a few hundred steps on a deterministic observation -> position-chunk
    mapping must reproduce it at inference time, and because of its conditioning, not because it
    matched the marginal statistics."""

    def test_trained_policy_follows_the_mapping_and_depends_on_the_observation(self) -> None:
        torch.manual_seed(0)
        policy = _flow_policy(4, action_dim=2, cond_dim=COND_DIM, unet_dim=16, unet_dim_mults=(1, 2), num_inference_steps=8)
        mixing = torch.randn(COND_DIM, 8) * 0.5

        def make_batch(size: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            features = torch.rand(size, COND_DIM) * 2 - 1
            observation = {
                "observation.environment_state": features[:, :2],
                "observation.history": features[:, 2:],
                "observation.state": torch.zeros(size, STATE_DIM),
            }
            return observation, torch.tanh(features @ mixing).reshape(size, 4, 2)

        optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)
        for _ in range(200):
            observation, target = make_batch(64)
            loss = policy.compute_loss(observation, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        def sample(observation: dict[str, torch.Tensor]) -> torch.Tensor:
            features = observation_features(observation)
            return torch.cat([policy.velocity_net.integrate(row[None], _unguided) for row in features])

        observation, target = make_batch(64)
        matched_error = torch.mean((sample(observation) - target) ** 2).item()
        permutation = torch.randperm(64)
        shuffled = {key: value[permutation] for key, value in observation.items()}
        shuffled_error = torch.mean((sample(shuffled) - target) ** 2).item()

        self.assertLess(matched_error, 0.3 * target.var().item())
        self.assertLess(matched_error, 0.4 * shuffled_error)


class ResetContractTests(unittest.TestCase):
    def test_a_policy_reset_goes_down_to_the_unet(self) -> None:
        policy = _build_policy()

        with mock.patch.object(policy.velocity_net, "reset", wraps=policy.velocity_net.reset) as net_reset, \
                mock.patch.object(policy.velocity_net.unet, "reset", wraps=policy.velocity_net.unet.reset) as unet_reset:
            policy.reset()

        net_reset.assert_called_once_with()
        unet_reset.assert_called_once_with()

    def test_every_level_has_a_reset_that_returns_nothing(self) -> None:
        policy = _build_policy()
        for module in (policy, policy.velocity_net, policy.velocity_net.unet):
            self.assertIsNone(module.reset())


class FlowTimeDeviceTests(unittest.TestCase):
    def test_the_flow_times_are_sampled_on_the_cpu_and_the_copy_to_the_device_blocks(self) -> None:
        net = _build_policy().velocity_net
        copies: list[dict] = []
        original_to = torch.Tensor.to

        def spy(tensor, *args, **kwargs):
            copies.append(kwargs)
            return original_to(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", spy):
            net.compute_loss(torch.zeros(5, COND_DIM), torch.zeros(5, 4, 2))

        # A non_blocking copy of the temporary reads freed host memory on MPS and yields flow times of ~1e22
        self.assertTrue(copies)
        self.assertFalse(any(kwargs.get("non_blocking") for kwargs in copies))

    def test_the_sampler_is_built_once_and_the_loss_stays_reproducible(self) -> None:
        net = _build_policy().velocity_net
        sampler = net._flow_time_distribution
        obs, actions = torch.zeros(5, COND_DIM), torch.randn(5, 4, 2)

        torch.manual_seed(3)
        first = net.compute_loss(obs, actions)
        torch.manual_seed(3)
        second = net.compute_loss(obs, actions)

        self.assertIs(net._flow_time_distribution, sampler)
        torch.testing.assert_close(first, second)


class CompiledCheckpointTests(unittest.TestCase):
    def test_keys_of_a_torch_compiled_network_load(self) -> None:
        policy = _build_policy()
        compiled = {name.replace("net.", "net._orig_mod.", 1): value.clone() for name, value in policy.state_dict().items()}
        self.assertTrue(any("_orig_mod" in name for name in compiled))
        _build_policy().load_network_state_dict(compiled)


class SelectActionTests(unittest.TestCase):
    """select_action integrates the positions of the flow, guided at every Euler step, and returns the jerks with which the
    tracker executes the final chunk."""

    def setUp(self) -> None:
        self.simulator = PointMassSystem({"goal": [0.5, 0.0]})
        self.batch = _point_mass_batch(self.simulator, np.zeros(6))

    def test_it_returns_a_chunk_of_jerks_within_the_limits(self) -> None:
        policy = _tracked_policy(self.simulator, action_scale=(1.0, 1.0))

        chunk = policy.select_action(self.batch)

        self.assertEqual(tuple(chunk.shape), (1, 4, 2))
        self.assertTrue(np.all(np.abs(chunk[0].numpy()) <= self.simulator.max_action + 1e-6))

    def test_the_tracker_is_handed_the_integrated_positions_and_its_jerks_are_returned(self) -> None:
        policy = _tracked_policy(self.simulator)
        with torch.no_grad():
            for param in policy.velocity_net.unet.parameters():
                param.zero_()  # the iterate never leaves its noise draw
        torch.manual_seed(5)
        draw = torch.randn(1, 4, 2) * policy.velocity_net.action_scale
        jerks = np.arange(8, dtype=float).reshape(4, 2)

        with mock.patch.object(policy, "guide", side_effect=lambda _, proposal: proposal), mock.patch.object(
            policy.tracker, "project", return_value=Projection(np.zeros((4, 2)), jerks)
        ) as project:
            torch.manual_seed(5)
            chunk = policy.select_action(self.batch)

        state, positions = project.call_args.args
        np.testing.assert_allclose(state, np.zeros(6))
        np.testing.assert_allclose(positions, draw[0].numpy(), atol=1e-5)
        np.testing.assert_allclose(chunk[0].numpy(), jerks, atol=1e-6)

    def test_every_euler_step_is_guided_by_the_observation_it_serves(self) -> None:
        policy = _tracked_policy(self.simulator, num_inference_steps=3)

        with mock.patch.object(policy, "guide", wraps=policy.guide) as guide:
            policy.select_action(self.batch)

        self.assertEqual(guide.call_count, 3)
        for call in guide.call_args_list:
            torch.testing.assert_close(call.args[0]["observation.state"], self.batch["observation.state"])
            self.assertEqual(call.args[1].shape, (4, 2))

    def test_every_action_starts_a_new_tick_of_the_tracker(self) -> None:
        policy = _tracked_policy(self.simulator)

        with mock.patch.object(policy.tracker, "begin_tick", wraps=policy.tracker.begin_tick) as begin_tick:
            policy.select_action(self.batch)

        begin_tick.assert_called_once_with()

    def test_a_policy_without_a_tracker_refuses_to_select_an_action(self) -> None:
        with self.assertRaises(RuntimeError):
            _build_policy().select_action(self.batch)

    def test_a_flow_policy_serves_one_system(self) -> None:
        policy = _tracked_policy(self.simulator)
        doubled = {name: value.repeat(2, 1) for name, value in self.batch.items()}

        with self.assertRaises(ValueError):
            policy.select_action(doubled)

    def test_the_guidance_and_the_tracker_start_a_new_episode_with_a_reset(self) -> None:
        policy = _tracked_policy(self.simulator)

        with mock.patch.object(policy.guidance, "reset") as guidance_reset, mock.patch.object(policy.tracker, "reset") as tracker_reset:
            policy.reset()

        guidance_reset.assert_called_once_with()
        tracker_reset.assert_called_once_with()


class FlowIteratesTests(unittest.TestCase):
    """The iterates of the flow are only guided: the final chunk is what the tracker gets, whatever its positions."""

    def test_the_final_chunk_reaches_the_tracker_without_any_clipping(self) -> None:
        simulator = FrankaSystem(pinned_config())
        n, horizon = simulator.n, 4
        spec = _spec(
            action_dim=n, cond_dim=simulator.environment_dim + n * HISTORY, prediction_horizon=horizon,
            num_inference_steps=1, action_scale=tuple(simulator.position_scale),
        )
        policy = PolicyFactory.create(spec, simulator, pinned_config()).eval()
        state = np.concatenate([Q_READY, np.zeros(2 * n)])
        observation = simulator.policy_observation(simulator.observe(state), np.tile(Q_READY, HISTORY))
        batch = {name: torch.as_tensor(value)[None] for name, value in observation.items()}

        with mock.patch.object(policy, "guide", side_effect=lambda _, proposal: np.full_like(proposal, 100.0)), mock.patch.object(
            policy.tracker, "project", return_value=Projection(np.zeros((horizon, n)), np.zeros((horizon, n)))
        ) as project:
            policy.select_action(batch)

        np.testing.assert_allclose(project.call_args.args[1], np.full((horizon, n), 100.0))


class PolicyFactoryTests(unittest.TestCase):
    def test_creates_flow_policy(self) -> None:
        policy = PolicyFactory.create(_spec(), PointMassSystem({}), {})
        self.assertIsInstance(policy, FlowPolicy)
        self.assertIsInstance(policy.guidance, PoseGuidance)  # the flow is always guided toward the goal
        self.assertIsInstance(policy.tracker, CasadiTrajectoryProjector)
        self.assertFalse(policy.tracker.safe)  # the tracking within the limits alone
        self.assertEqual(policy.tracker.tracked_horizon, 4)
        self.assertEqual(policy.tracker.tick_steps, EXECUTION)
        self.assertEqual(policy.history_length, HISTORY)
        self.assertEqual(policy.execution_horizon, EXECUTION)
        self.assertEqual(policy.velocity_net.prediction_horizon, 4)
        self.assertEqual(policy.velocity_net.num_inference_steps, 4)
        self.assertEqual(policy.velocity_net.obs_cond_dim, COND_DIM)
        torch.testing.assert_close(policy.velocity_net.action_scale, torch.tensor([1.0, 2.0]))

    def test_the_names_are_the_registered_policies(self) -> None:
        self.assertEqual(PolicyFactory.names(), ("flow", "safeflow"))

    def test_a_policy_needs_the_planner_config(self) -> None:
        for policy_type in ("flow", "safeflow"):
            with self.subTest(policy_type=policy_type):
                with self.assertRaises(ValueError):
                    PolicyFactory.create(_spec(policy_type), PointMassSystem({}))

    def test_unknown_policy_types_are_rejected(self) -> None:
        for policy_type in ("mlp", "wam", "not_a_real_policy_type"):
            with self.subTest(policy_type=policy_type):
                with self.assertRaises(ValueError) as ctx:
                    PolicyFactory.create(_spec(policy_type), PointMassSystem({}), {})
                self.assertIn("flow", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
