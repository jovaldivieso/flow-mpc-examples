from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from learning.config_loaders import (
    DEFAULT_DAGGER_TRAINING_CONFIG,
    FlowConfig,
    default_checkpoint_dir,
    default_dataset_root,
    default_repo_id,
    load_dagger_training_config,
    load_flow_config,
    load_policy_type,
    load_unet_config,
    UnetConfig,
    load_prediction_horizon,
)


def _write_policy_config(training_yaml_lines: list[str]) -> Path:
    """Write a minimal policy config with the given 'training:' body to a
    uniquely-named temp file (load_dagger_training_config's yaml cache keys
    off the path, so each test needs its own file to avoid a stale hit)."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    body = "model:\n  policy_type: flow\ntraining:\n" + "\n".join(
        f"  {line}" for line in training_yaml_lines
    )
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return Path(path)


def _write_model_config(model_yaml_lines: list[str]) -> Path:
    """Same as _write_policy_config, but with the given 'model:' body instead."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    body = "model:\n" + "\n".join(f"  {line}" for line in model_yaml_lines)
    with os.fdopen(fd, "w") as f:
        f.write(body)
    return Path(path)


class LoadDaggerTrainingConfigTests(unittest.TestCase):
    """Regression coverage for the 'training:' section's key allowlist --
    DEFAULT_DAGGER_TRAINING_CONFIG is the single source of truth for which
    keys a policy YAML may set; DaggerConfig/collect_dagger_rollouts tests
    never exercise this loader directly, so a field added everywhere else
    but here still fails at CLI startup with 'Unknown key(s) in policy
    config' -- exactly what happened before this file existed.
    """

    def test_every_dagger_config_field_used_by_train_dagger_is_a_known_key(self) -> None:
        # Mirrors the exact fields learning/train_dagger.py's CLI option
        # parsing reads out of this dict via option(...). If a name is added
        # there without a matching DEFAULT_DAGGER_TRAINING_CONFIG entry, the
        # yaml loader rejects it as "unknown" before it ever reaches
        # train_dagger.py -- this test would catch that mismatch here
        # instead of at CLI runtime.
        expected_keys = {
            "dagger_iterations", "trajectories_per_iteration",
            "steps_per_trajectory", "action_noise_std", "training_curriculum",
            "round_seeds", "restart_round_seed", "initial_states", "goal_states",
            "tolerance_overrides", "eval_tolerance_overrides",
            "expert_beta_per_iteration",
            "target_epochs_per_round", "eval_episodes", "eval_steps",
            "eval_seed_start", "eval_action_noise_std", "batch_size",
            "learning_rate", "seed", "max_train_steps", "num_workers",
        }
        self.assertEqual(expected_keys, set(DEFAULT_DAGGER_TRAINING_CONFIG))

    def test_removed_beta_keys_are_rejected(self) -> None:
        # Legacy decay/recovery keys were replaced by explicit expert_beta_per_iteration;
        # a policy YAML that still sets them must fail loudly instead of being silently ignored.
        for key in (
            "expert_mix_beta_start",
            "expert_mix_beta_end",
            "expert_mix_beta_decay_rate",
            "expert_mix_decay_after_eval_success",
            "adaptive_beta_recovery",
            "expert_mix_beta_recovery",
            "expert_mix_beta_recovery_increment",
        ):
            with self.subTest(key=key):
                path = _write_policy_config([f"{key}: 0.5"])
                try:
                    with self.assertRaisesRegex(ValueError, "Unknown key"):
                        load_dagger_training_config(path)
                finally:
                    path.unlink()

    def test_eval_tolerance_overrides_is_accepted_and_independent_of_tolerance_overrides(self) -> None:
        path = _write_policy_config([
            "tolerance_overrides:",
            "  pos_tol: 0.1",
            "eval_tolerance_overrides:",
            "  pos_tol: 0.2",
            "  vel_tol: 0.05",
        ])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertEqual(config["tolerance_overrides"], {"pos_tol": 0.1})
        self.assertEqual(config["eval_tolerance_overrides"], {"pos_tol": 0.2, "vel_tol": 0.05})

    def test_eval_tolerance_overrides_defaults_to_none_when_omitted(self) -> None:
        path = _write_policy_config(["dagger_iterations: 3"])
        try:
            config = load_dagger_training_config(path)
        finally:
            path.unlink()
        self.assertIsNone(config["eval_tolerance_overrides"])

    def test_unknown_key_is_still_rejected(self) -> None:
        path = _write_policy_config(["not_a_real_field: 1"])
        try:
            with self.assertRaises(ValueError):
                load_dagger_training_config(path)
        finally:
            path.unlink()


class LoadModelSectionTests(unittest.TestCase):
    """The 'model:' section's loaders -- unet, policy_type, the flow
    num_inference_steps and the safeflow options -- each fall back to their own
    default when their key (or the whole policy config) is omitted.
    policy_type is restricted to the only two policies there are: flow and
    safeflow.
    """

    def _write(self, *model_yaml_lines: str) -> Path:
        path = _write_model_config(list(model_yaml_lines))
        self.addCleanup(path.unlink)
        return path

    def test_unet_defaults_to_the_safeflowmpc_network_when_omitted(self) -> None:
        self.assertEqual(load_unet_config(self._write("policy_type: flow")), UnetConfig(dim=32, dim_mults=(1, 2, 4, 8)))

    def test_unet_is_read_as_a_dim_and_a_tuple_of_ints(self) -> None:
        path = self._write("unet:", "  dim: 16", "  dim_mults: [1, 2]")
        self.assertEqual(load_unet_config(path), UnetConfig(dim=16, dim_mults=(1, 2)))

    def test_a_unet_that_is_not_a_mapping_or_has_no_levels_is_rejected(self) -> None:
        for lines in (["unet: 3"], ["unet:", "  dim_mults: []"], ["unet:", "  dim_mults: 4"]):
            with self.subTest(lines=lines):
                with self.assertRaises(ValueError):
                    load_unet_config(self._write(*lines))

    def test_the_chunk_length_is_a_policy_setting_with_a_default_of_16(self) -> None:
        self.assertEqual(load_prediction_horizon(None), 16)
        self.assertEqual(load_prediction_horizon(self._write("policy_type: flow")), 16)
        self.assertEqual(load_prediction_horizon(self._write("prediction_horizon: 8")), 8)
        for lines in (["prediction_horizon: 0"], ["prediction_horizon: 2.5"], ["prediction_horizon: true"]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                load_prediction_horizon(self._write(*lines))

    def test_policy_type_defaults_to_flow(self) -> None:
        self.assertEqual(load_policy_type(None), "flow")
        self.assertEqual(load_policy_type(self._write("unet:", "  dim: 8")), "flow")

    def test_policy_type_accepts_flow_and_safeflow_in_any_case(self) -> None:
        for raw, expected in (("flow", "flow"), ("safeflow", "safeflow"), (' " SafeFlow " ', "safeflow")):
            with self.subTest(raw=raw):
                self.assertEqual(load_policy_type(self._write(f"policy_type: {raw}")), expected)

    def test_policy_type_rejects_unknown_types(self) -> None:
        for bad_type in ("mlp", "wam", "not_a_real_policy_type"):
            with self.subTest(policy_type=bad_type):
                with self.assertRaises(ValueError):
                    load_policy_type(self._write(f"policy_type: {bad_type}"))

    def test_flow_config_is_read_with_defaults(self) -> None:
        path = self._write("flow:", "  num_inference_steps: 5")
        self.assertEqual(load_flow_config(path), FlowConfig(num_inference_steps=5))
        self.assertEqual(load_flow_config(None), FlowConfig())
        self.assertEqual(FlowConfig().safeflow_init, "previous_safe")

    def test_safeflow_options_are_read(self) -> None:
        path = self._write("safeflow:", "  init: gaussian")
        config = load_flow_config(path)
        self.assertEqual(config.safeflow_init, "gaussian")

    def test_invalid_safeflow_options_are_rejected(self) -> None:
        for lines in (("safeflow:", "  init: zeros"), ("safeflow: 3",)):
            with self.subTest(lines=lines):
                with self.assertRaises(ValueError):
                    load_flow_config(self._write(*lines))


class DefaultPathHelperTests(unittest.TestCase):
    def test_defaults_follow_the_franka_naming(self) -> None:
        self.assertEqual(default_checkpoint_dir(), Path("outputs/train_dagger"))
        self.assertEqual(default_repo_id(123), "local/franka_dagger_123")
        self.assertEqual(default_dataset_root(123), Path("data/lerobot_dataset_franka_dagger_123"))


class ShippedPolicyConfigTests(unittest.TestCase):
    """Regression coverage for the policy configs shipped under learning/config:
    train_dagger.py's default --policy-config must go through every loader
    without tripping the 'training:' key allowlist above, and each config
    must name the policy type its file name says.
    """

    def test_shipped_policy_configs_load_through_every_loader(self) -> None:
        for file_name, policy_type, expected_unet, expected_horizon in (
            ("franka_casadi_flow_config.yaml", "flow", UnetConfig(dim=32, dim_mults=(1, 2)), 10),
            ("franka_casadi_safeflow_config.yaml", "safeflow", UnetConfig(dim=32, dim_mults=(1, 2)), 10),
            ("triple_integrator_casadi_flow_config.yaml", "flow", UnetConfig(dim=32, dim_mults=(1, 2)), 10),
            ("triple_integrator_casadi_safeflow_config.yaml", "safeflow", UnetConfig(), 16),
        ):
            path = Path(PROJECT_ROOT) / "learning" / "config" / file_name
            with self.subTest(config=file_name):
                self.assertEqual(set(load_dagger_training_config(path)), set(DEFAULT_DAGGER_TRAINING_CONFIG))
                self.assertEqual(load_policy_type(path), policy_type)
                self.assertEqual(load_unet_config(path), expected_unet)
                self.assertEqual(load_prediction_horizon(path), expected_horizon)
                self.assertGreater(load_flow_config(path).num_inference_steps, 0)


if __name__ == "__main__":
    unittest.main()
