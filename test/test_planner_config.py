from __future__ import annotations

import math
import os
import sys
import unittest
from dataclasses import asdict

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import ConfigurationError, OmplConfig, load_and_validate_system_config, validate_system_config
from planning.factory import PlannerFactory

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")


def _load_raw_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class ProjectorTuningKeysReachValidatedConfigTests(unittest.TestCase):
    """Regression coverage for a real gap: CasadiTrajectoryProjector reads
    terminal_velocity_weight/fallback_terminal_velocity_tol/
    max_consecutive_fallbacks off its planner_config (see
    planning/casadi_projector.py), but validate_system_config's key
    allowlist never listed them, so any expert YAML setting one was
    silently either rejected outright or (if it slipped past that) just
    never made it into the dict PolicyFactory hands the projector -- the
    projector always fell back to its own hardcoded defaults regardless of
    what the config said.
    """

    def test_overridden_values_are_accepted_and_pass_through(self) -> None:
        raw_config = _load_raw_config()
        raw_config["terminal_velocity_weight"] = 42.0
        raw_config["fallback_terminal_velocity_tol"] = 0.007
        raw_config["max_consecutive_fallbacks"] = 3

        validated = validate_system_config(raw_config=raw_config)

        self.assertEqual(validated["terminal_velocity_weight"], 42.0)
        self.assertEqual(validated["fallback_terminal_velocity_tol"], 0.007)
        self.assertEqual(validated["max_consecutive_fallbacks"], 3)

    def test_omitted_values_default_to_the_projectors_own_hardcoded_fallbacks(self) -> None:
        # These three literals must stay in sync with
        # CasadiTrajectoryProjector.__init__'s own config.get(..., <default>)
        # calls -- a config that omits these keys must behave identically to
        # before they were configurable.
        raw_config = _load_raw_config()

        validated = validate_system_config(raw_config=raw_config)

        self.assertEqual(validated["terminal_velocity_weight"], 1.0)
        self.assertEqual(validated["fallback_terminal_velocity_tol"], 1e-2)
        self.assertEqual(validated["max_consecutive_fallbacks"], 5)

    def test_the_keys_of_removed_features_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        for key in ("joint_limit_margin", "collision_slack_penalty_weight", "task_weights", "mode", "path_length_weight"):
            with self.subTest(key=key):
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config={**raw_config, key: 1.0})

    def test_ompl_settings_default_when_omitted_and_reach_the_validated_config(self) -> None:
        raw_config = _load_raw_config()
        for key in OmplConfig.keys():
            raw_config.pop(key, None)

        validated = validate_system_config(raw_config=raw_config)

        for key, default in asdict(OmplConfig()).items():
            self.assertEqual(validated[key], default, key)

    def test_ompl_settings_of_the_config_are_passed_on(self) -> None:
        raw_config = {**_load_raw_config(), "ompl_speed": 3.0, "ompl_seed": 7}

        validated = validate_system_config(raw_config=raw_config)

        self.assertEqual((validated["ompl_speed"], validated["ompl_seed"]), (3.0, 7))

    def test_the_expert_defaults_to_casadi_and_must_be_one_the_factory_has(self) -> None:
        raw_config = _load_raw_config()
        raw_config.pop("expert", None)
        self.assertEqual(validate_system_config(raw_config=raw_config)["expert"], "casadi")
        for name in PlannerFactory.names():
            self.assertEqual(validate_system_config(raw_config={**raw_config, "expert": name})["expert"], name)
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config={**raw_config, "expert": "rrt"})

    def test_invalid_ompl_settings_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        for key, value in (
            ("ompl_speed", -1.0),
            ("ompl_replan_distance", 0.0),
            ("ompl_timeout", 0.0),
            ("ompl_ik_solutions", 0),
        ):
            with self.subTest(key=key):
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config={**raw_config, key: value})

    def test_the_chunk_length_is_not_an_expert_setting(self) -> None:
        raw_config = _load_raw_config()
        raw_config["plan_horizon"] = 16
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_non_positive_terminal_velocity_weight_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["terminal_velocity_weight"] = 0.0
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_negative_fallback_terminal_velocity_tol_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["fallback_terminal_velocity_tol"] = -0.01
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_non_positive_max_consecutive_fallbacks_is_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["max_consecutive_fallbacks"] = 0
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)


class ExpertConfigValidationTests(unittest.TestCase):
    def test_shipped_expert_config_validates_and_revalidation_is_idempotent(self) -> None:
        # DaggerTrainer re-validates an already-validated config after merging
        # runtime overrides (workspace bounds, tolerances), so the validator
        # must accept its own output unchanged.
        validated = load_and_validate_system_config(CONFIG_PATH)
        self.assertEqual(validate_system_config(raw_config=validated), validated)

    def test_a_yaml_string_like_1e_minus_3_is_reported_with_a_hint(self) -> None:
        raw_config = _load_raw_config()
        raw_config["R_diag"] = ["10e-3"] * 7  # PyYAML parses a number without a dot in the mantissa as a string
        with self.assertRaisesRegex(ConfigurationError, "write 1.0e-3"):
            validate_system_config(raw_config=raw_config)

    def test_unknown_keys_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["max_linear_accel"] = 2.0  # not a key of the arm's config
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_q_and_r_diagonals_must_match_the_state_and_action_dimensions(self) -> None:
        raw_config = _load_raw_config()
        raw_config["Q_diag"] = [1.0] * 13
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)
        raw_config = _load_raw_config()
        raw_config["R_diag"] = [1.0] * 6
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)
        raw_config = _load_raw_config()
        raw_config["R_diag"] = [1.0] * 6 + [0.0]
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_r_diag_defaults_to_the_scalar_r_weight(self) -> None:
        raw_config = _load_raw_config()
        del raw_config["R_diag"]
        raw_config["R_weight"] = 0.3
        self.assertEqual(validate_system_config(raw_config=raw_config)["R_diag"], [0.3] * 7)

    def test_floor_clearance_is_numeric_and_not_negative(self) -> None:
        raw_config = _load_raw_config()
        del raw_config["floor_clearance"]
        self.assertEqual(validate_system_config(raw_config=raw_config)["floor_clearance"], 0.05)

        for invalid in (-0.01, "low"):
            raw_config["floor_clearance"] = invalid
            with self.assertRaises(ConfigurationError):
                validate_system_config(raw_config=raw_config)

    def test_joint_limits_are_broadcast_from_a_scalar_and_their_list_length_is_checked(self) -> None:
        for key in ("max_acceleration", "max_jerk"):
            with self.subTest(key=key):
                raw_config = _load_raw_config()
                raw_config[key] = 7.0
                self.assertEqual(validate_system_config(raw_config=raw_config)[key], [7.0] * 7)

                raw_config[key] = [7.0] * 6
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)

                raw_config[key] = [7.0] * 6 + [0.0]
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)

    def test_tolerances_must_be_positive(self) -> None:
        for key in ("pos_tol", "rot_tol", "vel_tol"):
            with self.subTest(key=key):
                raw_config = _load_raw_config()
                raw_config[key] = 0.0
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)

    def test_tolerance_overrides_merge_like_the_dagger_trainer_applies_them(self) -> None:
        validated = load_and_validate_system_config(CONFIG_PATH)
        merged = validate_system_config({**validated, "pos_tol": 0.05, "rot_tol": 0.5, "vel_tol": 0.2})
        self.assertEqual((merged["pos_tol"], merged["rot_tol"], merged["vel_tol"]), (0.05, 0.5, 0.2))

    def test_workspace_bounds_are_three_ordered_finite_pairs(self) -> None:
        bad_bounds = (
            [[0.2, 1.0], [-1.0, 1.0]],  # two axes only
            [[0.2, 1.0], [1.0, -1.0], [0.1, 0.8]],  # reversed
            [[0.2, 1.0], [-1.0, 1.0], [0.1, 0.1]],  # empty
            [[0.2, 1.0], [-1.0, math.inf], [0.1, 0.8]],  # not finite
            [[0.2, math.nan], [-1.0, 1.0], [0.1, 0.8]],
            [-1.0, 1.0],  # a single [min, max] pair
        )
        for workspace_bounds in bad_bounds:
            with self.subTest(workspace_bounds=workspace_bounds):
                raw_config = _load_raw_config()
                raw_config["workspace_bounds"] = workspace_bounds
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)

    def test_goal_is_a_twelve_vector_and_controls_the_randomize_goal_default(self) -> None:
        raw_config = _load_raw_config()
        self.assertTrue(validate_system_config(raw_config=raw_config)["randomize_goal"])
        self.assertNotIn("goal", validate_system_config(raw_config=raw_config))

        raw_config["goal"] = [0.4, 0.0, 0.4] + [1, 0, 0, 0, -1, 0, 0, 0, -1]
        validated = validate_system_config(raw_config=raw_config)
        self.assertFalse(validated["randomize_goal"])
        self.assertEqual(len(validated["goal"]), 12)

        raw_config["goal"] = [0.4, 0.0, 0.4]
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_action_noise_seed_falls_back_to_the_initial_state_seed_then_zero(self) -> None:
        raw_config = _load_raw_config()
        del raw_config["action_noise_seed"]
        raw_config["initial_state_seed"] = 17
        self.assertEqual(validate_system_config(raw_config=raw_config)["action_noise_seed"], 17)

        del raw_config["initial_state_seed"]
        self.assertEqual(validate_system_config(raw_config=raw_config)["action_noise_seed"], 0)

    def test_done_hold_steps_is_optional_and_positive(self) -> None:
        raw_config = _load_raw_config()
        self.assertNotIn("done_hold_steps", validate_system_config(raw_config=raw_config))
        raw_config["done_hold_steps"] = 3
        self.assertEqual(validate_system_config(raw_config=raw_config)["done_hold_steps"], 3)
        raw_config["done_hold_steps"] = 0
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)


if __name__ == "__main__":
    unittest.main()
