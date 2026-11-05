from __future__ import annotations

import math
import os
import sys
import unittest
from dataclasses import asdict

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import load_and_validate_system_config, validate_system_config
from planning.planner_config import ExpertConfig, ProjectionConfig
from planning.planner_factory import PlannerFactory
from systems.config_fields import ConfigurationError

CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_config.yaml")


def _load_raw_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class ExpertAndProjectionKeysReachValidatedConfigTests(unittest.TestCase):
    """The expert and projection settings of a config are validated and reach the config the planners are built from."""

    def test_overridden_values_are_accepted_and_pass_through(self) -> None:
        raw_config = _load_raw_config()
        raw_config.update(
            {
                "terminal_velocity_weight": 42.0,
                "projection_velocity_weight": 0.5,
                "projection_acceleration_weight": 0.25,
                "projection_jerk_weight": 0.125,
                "fallback_terminal_velocity_tol": 0.007,
                "max_consecutive_fallbacks": 3,
            }
        )

        validated = validate_system_config(raw_config=raw_config)

        self.assertEqual(validated["terminal_velocity_weight"], 42.0)
        self.assertEqual(
            (validated["projection_velocity_weight"], validated["projection_acceleration_weight"], validated["projection_jerk_weight"]),
            (0.5, 0.25, 0.125),
        )
        self.assertEqual(validated["fallback_terminal_velocity_tol"], 0.007)
        self.assertEqual(validated["max_consecutive_fallbacks"], 3)

    def test_omitted_values_default_to_the_dataclass_defaults(self) -> None:
        raw_config = _load_raw_config()
        for key in ExpertConfig.keys() | ProjectionConfig.keys():
            raw_config.pop(key, None)

        validated = validate_system_config(raw_config=raw_config)

        for name, default in asdict(ExpertConfig()).items():
            self.assertEqual(validated[name], default, name)
        for name, default in asdict(ProjectionConfig()).items():
            self.assertEqual(validated[ProjectionConfig.key(name)], default, name)

    def test_the_expert_settings_of_the_config_are_passed_on(self) -> None:
        raw_config = {**_load_raw_config(), "ompl_seed": 7, "horizon": 12, "totg_max_deviation": 0.1, "replan_distance": 0.3}

        validated = validate_system_config(raw_config=raw_config)

        self.assertEqual((validated["ompl_seed"], validated["horizon"]), (7, 12))
        self.assertEqual((validated["totg_max_deviation"], validated["replan_distance"]), (0.1, 0.3))

    def test_keys_outside_the_schema_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        for key in ("joint_limit_margin", "task_weights", "mode", "Q_diag", "R_diag", "R_weight", "ompl_speed", "plan_horizon"):
            with self.subTest(key=key):
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config={**raw_config, key: 1.0})

    def test_the_expert_defaults_to_ompl_casadi_and_must_be_one_the_factory_has(self) -> None:
        raw_config = _load_raw_config()
        raw_config.pop("expert", None)
        self.assertEqual(validate_system_config(raw_config=raw_config)["expert"], "ompl_casadi")
        for name in PlannerFactory.names():
            self.assertEqual(validate_system_config(raw_config={**raw_config, "expert": name})["expert"], name)
        with self.assertRaises(ConfigurationError):
            validate_system_config(raw_config={**raw_config, "expert": "rrt"})

    def test_invalid_expert_settings_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        for key, value in (
            ("horizon", 0),
            ("ompl_timeout", 0.0),
            ("replan_distance", 0.0),
            ("progress_window", -1.0),
            ("totg_max_deviation", 0.0),
            ("totg_velocity_scale", 0.0),
            ("totg_velocity_scale", 1.5),
            ("totg_acceleration_scale", 0.0),
            ("totg_acceleration_scale", 1.5),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config={**raw_config, key: value})

    def test_invalid_projection_settings_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        for key, value in (
            ("projection_velocity_weight", 0.0),
            ("projection_acceleration_weight", -1.0),
            ("projection_jerk_weight", 0.0),
            ("terminal_velocity_weight", 0.0),
            ("fallback_terminal_velocity_tol", -0.01),
            ("max_consecutive_fallbacks", 0),
        ):
            with self.subTest(key=key, value=value):
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config={**raw_config, key: value})


class ExpertConfigTests(unittest.TestCase):
    """ExpertConfig parses and validates: ``from_mapping`` reads, ``__post_init__`` checks."""

    def test_defaults(self) -> None:
        expert = ExpertConfig.from_mapping({})

        self.assertEqual((expert.expert, expert.horizon, expert.ompl_timeout, expert.ompl_seed), ("ompl_casadi", 20, 5.0, 0))
        self.assertEqual((expert.replan_distance, expert.progress_window), (0.5, 1.5))
        self.assertEqual((expert.totg_velocity_scale, expert.totg_acceleration_scale, expert.totg_max_deviation), (0.8, 0.5, 0.05))

    def test_what_the_mapping_sets_is_read(self) -> None:
        expert = ExpertConfig.from_mapping({"expert": "ompl_casadi", "horizon": 7, "ompl_seed": 3, "totg_velocity_scale": 1.0})

        self.assertEqual((expert.horizon, expert.ompl_seed, expert.totg_velocity_scale), (7, 3, 1.0))

    def test_the_keys_are_the_field_names(self) -> None:
        self.assertEqual(ExpertConfig.keys(), set(asdict(ExpertConfig())))

    def test_wrong_types_and_values_are_rejected(self) -> None:
        for bad in ({"horizon": 2.5}, {"horizon": 0}, {"ompl_timeout": "x"}, {"ompl_timeout": 0.0}, {"totg_acceleration_scale": 2.0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ExpertConfig.from_mapping(bad)

    def test_a_config_built_directly_is_validated_too(self) -> None:
        for bad in ({"horizon": 0}, {"replan_distance": 0.0}, {"totg_velocity_scale": 0.0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ExpertConfig(**bad)


class ProjectionConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        projection = ProjectionConfig.from_mapping({})

        self.assertEqual((projection.velocity_weight, projection.acceleration_weight, projection.jerk_weight), (1e-3, 1e-3, 1e-4))
        self.assertEqual(
            (projection.terminal_velocity_weight, projection.fallback_terminal_velocity_tol, projection.max_consecutive_fallbacks),
            (10.0, 1e-2, 5),
        )

    def test_the_yaml_keys_carry_the_projection_prefix_except_the_terminal_and_fallback_ones(self) -> None:
        self.assertEqual(
            ProjectionConfig.keys(),
            {
                "projection_velocity_weight",
                "projection_acceleration_weight",
                "projection_jerk_weight",
                "terminal_velocity_weight",
                "fallback_terminal_velocity_tol",
                "max_consecutive_fallbacks",
            },
        )
        self.assertEqual(ProjectionConfig.key("jerk_weight"), "projection_jerk_weight")
        self.assertEqual(ProjectionConfig.key("max_consecutive_fallbacks"), "max_consecutive_fallbacks")

    def test_what_the_mapping_sets_is_read(self) -> None:
        projection = ProjectionConfig.from_mapping(
            {"projection_velocity_weight": 2.0, "terminal_velocity_weight": 3.0, "fallback_terminal_velocity_tol": 0.5, "max_consecutive_fallbacks": 9}
        )

        self.assertEqual((projection.velocity_weight, projection.terminal_velocity_weight), (2.0, 3.0))
        self.assertEqual((projection.fallback_terminal_velocity_tol, projection.max_consecutive_fallbacks), (0.5, 9))

    def test_wrong_types_and_values_are_rejected(self) -> None:
        for bad in (
            {"terminal_velocity_weight": -1.0},
            {"projection_jerk_weight": 0.0},
            {"projection_velocity_weight": "x"},
            {"fallback_terminal_velocity_tol": -0.1},
            {"max_consecutive_fallbacks": 0},
            {"max_consecutive_fallbacks": 1.5},
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ProjectionConfig.from_mapping(bad)

    def test_a_config_built_directly_is_validated_too(self) -> None:
        for bad in ({"velocity_weight": 0.0}, {"max_consecutive_fallbacks": 0}):
            with self.subTest(bad=bad), self.assertRaises(ConfigurationError):
                ProjectionConfig(**bad)


class ExpertConfigValidationTests(unittest.TestCase):
    def test_shipped_expert_config_validates_and_revalidation_is_idempotent(self) -> None:
        # DaggerTrainer re-validates an already-validated config after merging
        # runtime overrides (workspace bounds, tolerances), so the validator
        # must accept its own output unchanged.
        validated = load_and_validate_system_config(CONFIG_PATH)
        self.assertEqual(validate_system_config(raw_config=validated), validated)

    def test_unknown_keys_are_rejected(self) -> None:
        raw_config = _load_raw_config()
        raw_config["max_linear_accel"] = 2.0  # not a key of the arm's config
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

    def test_the_sampling_modes_are_validated_and_default_to_the_workspace(self) -> None:
        raw_config = _load_raw_config()
        for key in ("sampling", "sampling_modes"):
            raw_config.pop(key, None)
        validated = validate_system_config(raw_config=raw_config)
        self.assertEqual(list(validated["sampling"]), ["workspace"])
        self.assertEqual(validated["sampling_modes"], ["workspace"])
        workspace = validated["sampling"]["workspace"]
        self.assertEqual((workspace["weight"], workspace["start_speed"]), (1.0, 0.0))
        self.assertEqual(workspace["start"], workspace["goal"])
        self.assertEqual(workspace["start"]["tool_axis"], [1.0, 0.0, -1.0])

        box = {"x": [0.3, 0.6], "y": [-0.2, 0.2], "z": [0.1, 0.3]}
        raw_config["sampling"] = {
            "under": {"weight": 2, "start": box, "goal": {**box, "tool_axis": [1, 0, 0], "tool_tilt": 0.4}, "start_speed": 0.3},
            "anywhere": {},
        }
        validated = validate_system_config(raw_config=raw_config)
        under = validated["sampling"]["under"]
        self.assertEqual((under["weight"], under["start_speed"]), (2.0, 0.3))
        self.assertEqual(under["start"], {**box, "tool_axis": [1.0, 0.0, -1.0], "tool_tilt": 0.9})
        self.assertEqual(under["goal"], {**box, "tool_axis": [1.0, 0.0, 0.0], "tool_tilt": 0.4})
        self.assertEqual(validated["sampling_modes"], ["under", "anywhere"])
        self.assertEqual(validate_system_config(raw_config=validated), validated)

        raw_config["sampling_modes"] = ["anywhere"]
        self.assertEqual(validate_system_config(raw_config=raw_config)["sampling_modes"], ["anywhere"])

    def test_invalid_sampling_is_rejected(self) -> None:
        box = {"x": [0.3, 0.6], "y": [-0.2, 0.2], "z": [0.1, 0.3]}
        for key, bad in (
            ("sampling", {}),  # no mode
            ("sampling", [box]),  # not a mapping
            ("sampling", {"a": None}),  # a mode that is not a mapping
            ("sampling", {"a": {"extra": 1}}),  # unknown key
            ("sampling", {"a": {"weight": 0}}),
            ("sampling", {"a": {"start_speed": 1.5}}),
            ("sampling", {"a": {"start": {"x": [0.3, 0.6], "y": [-0.2, 0.2]}}}),  # missing axis
            ("sampling", {"a": {"start": {**box, "z": [0.3, 0.1]}}}),  # min above max
            ("sampling", {"a": {"start": {**box, "z": [0.1]}}}),  # not a pair
            ("sampling", {"a": {"start": {**box, "extra": 1}}}),  # unknown key
            ("sampling", {"a": {"start": [0.3, 0.6]}}),  # not a mapping
            ("sampling", {"a": {"goal": {**box, "tool_axis": [0, 0, 0]}}}),  # no direction
            ("sampling", {"a": {"goal": {**box, "tool_axis": [1, 0]}}}),  # wrong length
            ("sampling", {"a": {"goal": {**box, "tool_tilt": -0.1}}}),
            ("sampling", {"a": {"goal": {**box, "tool_tilt": 4.0}}}),
            ("sampling_modes", []),
            ("sampling_modes", ["missing"]),
        ):
            with self.subTest(key=key, bad=bad):
                invalid = {**_load_raw_config(), key: bad}
                with self.assertRaises(ConfigurationError):
                    validate_system_config(raw_config=invalid)

if __name__ == "__main__":
    unittest.main()
