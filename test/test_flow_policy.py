from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.models.flow_policy import FlowPolicy
from learning.models.policy import observation_features
from learning.models.policy_factory import PolicyFactory, PolicySpec
from learning.models.pose_guidance import PoseGuidance
from systems.point_mass_system import PointMassSystem


def _flow_policy(horizon: int = 4, **kwargs) -> FlowPolicy:
    """A flow policy guided on a point mass (its guidance is stored, the tests of the network never call it)."""
    return FlowPolicy(guidance=PoseGuidance(PointMassSystem({}), horizon), prediction_horizon=horizon, **kwargs)


def _build_policy(action_scale=None) -> FlowPolicy:
    return _flow_policy(
        action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2), num_inference_steps=4, action_scale=action_scale,
    )


def _unguided(proposal):
    return proposal


def _fake_observation(batch_size: int) -> dict[str, torch.Tensor]:
    return {
        "observation.environment_state": torch.zeros(batch_size, 2),
        "observation.state": torch.zeros(batch_size, 4),
    }


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


class VelocityNetworkTests(unittest.TestCase):
    """The network is SafeFlowMPC's temporal U-Net, with the flow time passed as it is."""

    def test_the_velocity_is_predicted_by_the_unet_on_the_chunk_the_condition_and_the_time(self) -> None:
        policy = _build_policy()
        net = policy.velocity_net
        chunk, condition, time = torch.randn(3, 4, 2), torch.randn(3, 6), torch.rand(3)

        velocity = net.predict_velocity(chunk.flatten(1), condition, time)

        torch.testing.assert_close(velocity, net.unet(chunk, time, condition).flatten(1))
        self.assertEqual(tuple(velocity.shape), (3, 8))

    def test_the_horizon_must_survive_the_downsampling_of_the_unet(self) -> None:
        with self.assertRaises(ValueError):
            _flow_policy(6, action_dim=2, state_dim=6, unet_dim=8, unet_dim_mults=(1, 2, 4))

    def test_the_channel_count_must_suit_the_group_norm(self) -> None:
        with self.assertRaises(ValueError):
            _flow_policy(8, action_dim=2, state_dim=6, unet_dim=12)

    def test_the_default_network_has_the_size_of_the_one_of_safeflowmpc(self) -> None:
        net = _flow_policy(16, action_dim=7, state_dim=26).velocity_net

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
    """FlowPolicy conditions its velocity network directly on
    observation_features, i.e. [observation.environment_state,
    observation.state] in that order (the same flat vector
    CasadiTrajectoryProjector's invert_obs takes), so the network's
    conditioning width is simply environment_state_dim + state_dim.
    """

    @staticmethod
    def _distinguishable_observation() -> dict[str, torch.Tensor]:
        return {
            "observation.environment_state": torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
            "observation.state": torch.tensor([[3.0, 4.0, 5.0, 6.0], [9.0, 10.0, 11.0, 12.0]]),
        }

    def test_observation_features_concatenates_environment_state_then_state(self) -> None:
        torch.testing.assert_close(
            observation_features(self._distinguishable_observation()),
            torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]),
        )

    def test_velocity_net_conditioning_width_is_state_dim(self) -> None:
        self.assertEqual(_build_policy().velocity_net.obs_cond_dim, 6)

    def test_compute_loss_and_select_action_condition_on_observation_features(self) -> None:
        policy = _build_policy()
        observation = self._distinguishable_observation()
        expected = observation_features(observation)
        actions = torch.randn(2, policy.velocity_net.prediction_horizon, policy.velocity_net.action_dim)
        single = {key: value[:1] for key, value in observation.items()}

        with mock.patch.object(
            policy.velocity_net, "compute_loss", wraps=policy.velocity_net.compute_loss
        ) as compute_loss:
            policy.compute_loss(observation, actions)
        torch.testing.assert_close(compute_loss.call_args.args[0], expected)

        with mock.patch.object(policy, "guide", side_effect=lambda _, proposal: proposal), mock.patch.object(
            policy.velocity_net, "integrate", wraps=policy.velocity_net.integrate
        ) as integrate:
            policy.select_action(single)
        torch.testing.assert_close(integrate.call_args.args[0], expected[:1])


class FlowPolicyLearnsAConditionalMappingTests(unittest.TestCase):
    """End-to-end check of the flow-matching math (loss direction, time convention, Euler integration):
    a small policy trained for a few hundred steps on a deterministic observation -> action-chunk
    mapping must reproduce it at inference time, and because of its conditioning, not because it
    matched the marginal action statistics."""

    def test_trained_policy_follows_the_mapping_and_depends_on_the_observation(self) -> None:
        torch.manual_seed(0)
        policy = _flow_policy(2, action_dim=2, state_dim=6, unet_dim=16, unet_dim_mults=(1, 2), num_inference_steps=8)
        mixing = torch.randn(6, 4) * 0.5

        def make_batch(size: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            features = torch.rand(size, 6) * 2 - 1
            observation = {"observation.environment_state": features[:, :2], "observation.state": features[:, 2:]}
            return observation, torch.tanh(features @ mixing).reshape(size, 2, 2)

        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
        for _ in range(800):
            observation, target = make_batch(256)
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


class FlowTimeSamplingTests(unittest.TestCase):
    def test_the_flow_times_are_sampled_on_the_cpu_and_the_copy_to_the_device_blocks(self) -> None:
        net = _build_policy().velocity_net
        copies: list[dict] = []
        original_to = torch.Tensor.to

        def spy(tensor, *args, **kwargs):
            copies.append(kwargs)
            return original_to(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", spy):
            net.compute_loss(torch.zeros(5, 6), torch.zeros(5, 4, 2))

        # A non_blocking copy of the temporary reads freed host memory on MPS and yields flow times of ~1e22
        self.assertTrue(copies)
        self.assertFalse(any(kwargs.get("non_blocking") for kwargs in copies))

    def test_the_sampler_is_built_once_and_the_loss_stays_reproducible(self) -> None:
        net = _build_policy().velocity_net
        sampler = net._flow_time_distribution
        obs, actions = torch.zeros(5, 6), torch.randn(5, 4, 2)

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


class PolicyFactoryTests(unittest.TestCase):
    @staticmethod
    def _spec(policy_type: str = "flow") -> PolicySpec:
        return PolicySpec(
            policy_type=policy_type, action_dim=2, state_dim=6, prediction_horizon=4, unet_dim=8, unet_dim_mults=(1, 2),
            num_inference_steps=4, action_scale=(1.0, 2.0),
        )

    def test_creates_flow_policy(self) -> None:
        policy = PolicyFactory.create(self._spec(), PointMassSystem({}))
        self.assertIsInstance(policy, FlowPolicy)
        self.assertIsInstance(policy.guidance, PoseGuidance)  # the flow is always guided toward the goal
        self.assertEqual(policy.velocity_net.prediction_horizon, 4)
        self.assertEqual(policy.velocity_net.num_inference_steps, 4)
        torch.testing.assert_close(policy.velocity_net.action_scale, torch.tensor([1.0, 2.0]))

    def test_the_names_are_the_registered_policies(self) -> None:
        self.assertEqual(PolicyFactory.names(), ("flow", "safeflow"))

    def test_unknown_policy_types_are_rejected(self) -> None:
        for policy_type in ("mlp", "wam", "not_a_real_policy_type"):
            with self.subTest(policy_type=policy_type):
                with self.assertRaises(ValueError) as ctx:
                    PolicyFactory.create(self._spec(policy_type), PointMassSystem({}))
                self.assertIn("flow", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
