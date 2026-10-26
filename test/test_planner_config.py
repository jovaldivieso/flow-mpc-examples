from __future__ import annotations

import math
import os
import sys
import unittest
from dataclasses import asdict

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import (
    ConfigurationError,
    OmplConfig,
    PlannerConfig,
    load_and_validate_system_config,
    validate_system_config,
)
from planning.planner_factory import PlannerFactory

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


class PlannerConfigTests(unittest.TestCase):
    """PlannerConfig parses and validates like OmplConfig: ``from_mapping`` reads, ``__post_init__`` checks."""

    def test_defaults_and_the_r_diag_that_follows_the_r_weight(self) -> None:
        planner = PlannerConfig.from_mapping({})

        self.assertEqual((planner.expert, planner.horizon, planner.r_weight), ("casadi", 20, 0.1))
        self.assertEqual((planner.q_diag, planner.r_diag), ((), ()))  # the planner completes them for its system
        self.assertEqual(PlannerConfig.from_mapping({"R_weight": 0.3}).r_weight, 0.3)
        self.assertEqual(PlannerConfig.from_mapping({"R_diag": [1.0, 2.0]}).r_diag, (1.0, 2.0))

    def test_what_the_mapping_sets_is_read(self) -> None:
        planner = PlannerConfig.from_mapping(
            {"expert": "ompl_casadi", "horizon": 7, "Q_diag": [1.0] * 6, "terminal_cost_multiplier": 3.0,
             "terminal_velocity_weight": 2.0, "fallback_terminal_velocity_tol": 0.5, "max_consecutive_fallbacks": 9},
        )

        self.assertEqual((planner.expert, planner.horizon, planner.q_diag), ("ompl_casadi", 7, (1.0,) * 6))
        self.assertEqual(
            (planner.terminal_cost_multiplier, planner.terminal_velocity_weight, planner.fallback_terminal_velocity_tol,
             planner.max_consecutive_fallbacks),
            (3.0, 2.0, 0.5, 9),
        )

    def test_wrong_types_and_values_are_rejected(self) -> None:
        for bad in ({"Q_diag": [1.0, 0.0]}, {"Q_diag": []}, {"Q_diag": 1.0}, {"R_diag": [1.0, 0.0]}, {"horizon": 2.5}, {"R_weight": "x"},
                    {"R_weight": 0.0}, {"horizon": 0}, {"expert": "rrt"}, {"terminal_cost_multiplier": 0.0},
                    {"terminal_velocity_weight": -1.0}, {"fallback_terminal_velocity_tol": -0.1},
                    {"max_consecutive_fallbacks": 0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                PlannerConfig.from_mapping(bad)

    def test_a_planner_config_built_directly_is_validated_too(self) -> None:
        for bad in ({"horizon": 0}, {"expert": "rrt"}, {"r_weight": 0.0}, {"max_consecutive_fallbacks": 0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                PlannerConfig(**bad)


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

    def test_q_and_r_diagonals_are_positive_lists_whatever_their_length(self) -> None:
        # Their lengths depend on the system, which is the planner's to check (test_casadi_planner)
        raw_config = _load_raw_config()
        raw_config["Q_diag"] = [1.0] * 13
        self.assertEqual(validate_system_config(raw_config=raw_config)["Q_diag"], [1.0] * 13)
        raw_config = _load_raw_config()
        raw_config["R_diag"] = [1.0] * 6 + [0.0]
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)

    def test_omitted_diagonals_stay_out_of_the_validated_config(self) -> None:
        raw_config = _load_raw_config()
        del raw_config["Q_diag"], raw_config["R_diag"]
        raw_config["R_weight"] = 0.3
        validated = validate_system_config(raw_config=raw_config)
        self.assertEqual((validated["R_weight"], "Q_diag" in validated, "R_diag" in validated), (0.3, False, False))

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
        for key in ("pos_tol", "vel_tol"):
            with self.subTest(key=key):
                raw_config = _load_raw_config()
                raw_config[key] = 0.0
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)

    def test_tolerance_overrides_merge_like_the_dagger_trainer_applies_them(self) -> None:
        validated = load_and_validate_system_config(CONFIG_PATH)
        merged = validate_system_config({**validated, "pos_tol": 0.05, "vel_tol": 0.2})
        self.assertEqual((merged["pos_tol"], merged["vel_tol"]), (0.05, 0.2))

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

    def test_goal_is_a_pose_and_controls_the_randomize_goal_default(self) -> None:
        raw_config = _load_raw_config()
        self.assertTrue(validate_system_config(raw_config=raw_config)["randomize_goal"])
        self.assertNotIn("goal", validate_system_config(raw_config=raw_config))

        pose = [0.4, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]
        raw_config["goal"] = pose
        validated = validate_system_config(raw_config=raw_config)
        self.assertFalse(validated["randomize_goal"])
        self.assertEqual(validated["goal"], pose)

        raw_config["goal"] = [0.4, 0.0, 0.4]  # a position alone is no pose
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config=raw_config)
        raw_config["goal"] = [0.4, 0.0, 0.4] + [1.0] * 9  # the rotation block is no rotation matrix
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

    def test_spherical_obstacles_are_validated(self) -> None:
        raw_config = _load_raw_config()
        raw_config.pop("obstacles", None)
        self.assertEqual(validate_system_config(raw_config=raw_config)["obstacles"], [])

        raw_config["obstacles"] = [{"position": [0.5, 0.0, 0.4], "radius": 0.08}]
        validated = validate_system_config(raw_config=raw_config)
        self.assertEqual(validated["obstacles"], [{"position": [0.5, 0.0, 0.4], "radius": 0.08}])

        for bad_obstacles in (
            [{"position": [0.5, 0.0], "radius": 0.08}],  # wrong position length
            [{"position": [0.5, 0.0, 0.4], "radius": 0.0}],  # non-positive radius
            [{"position": [0.5, 0.0, 0.4], "radius": -0.1}],
            [{"position": [0.5, 0.0, 0.4]}],  # missing radius
            [{"position": [0.5, 0.0, 0.4], "radius": 0.08, "extra": 1}],  # unknown key
            "not-a-list",
        ):
            with self.subTest(bad_obstacles=bad_obstacles):
                raw_config["obstacles"] = bad_obstacles
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=raw_config)


if __name__ == "__main__":
    unittest.main()
