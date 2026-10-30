from __future__ import annotations

import contextlib
import copy
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from planning.config import load_and_validate_system_config
from systems.franka_system import FrankaSystem
from learning.config_loaders import FlowConfig, UnetConfig
from learning.data_utils import build_dataset_features
from learning.dagger.dagger_trainer import MIN_LEARNING_RATE, WEIGHT_DECAY, _validate_resumable_dataset_schema
from learning.dagger.metrics import DaggerEvalMetrics
from learning.models.flow_policy import FlowPolicy
from learning.models.policy_factory import PolicyFactory, PolicySpec
from learning.models.safe_flow_policy import SafeFlowMPCPolicy
from learning.train_dagger import DaggerConfig, DaggerTrainer

EXPERT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")


def _minimal_dagger_config(**overrides: object) -> DaggerConfig:
    """Build a DaggerConfig with every required field filled by an inert placeholder.

    start_with_aggregation=True skips the on-disk dataset_root existence check, so
    none of these values need to point at anything real -- only the fields under
    test (round_seeds / restart_round_seed / dagger_iterations here) matter.
    """
    defaults: dict[str, object] = dict(
        experiment_config={},
        repo_id="test/repo",
        dataset_root=Path("unused"),
        start_with_aggregation=True,
        dagger_iterations=3,
        trajectories_per_iteration=[10],
        steps_per_trajectory=50,
        action_noise_std=0.0,
        expert_beta_per_iteration=[0.8],
        target_epochs_per_round=[10.0],
        eval_episodes=0,
        eval_steps=None,
        eval_seed_start=0,
        eval_action_noise_std=0.0,
        batch_size=64,
        learning_rate=1e-3,
        unet=UnetConfig(dim=8, dim_mults=(1, 2)),
        prediction_horizon=6,
        policy_type="flow",
        flow_config=FlowConfig(),
        checkpoint_dir=Path("unused"),
        seed=0,
        max_train_steps=None,
    )
    defaults.update(overrides)
    return DaggerConfig(**defaults)


class DaggerConfigRoundSeedsTests(unittest.TestCase):
    def test_round_seeds_matching_dagger_iterations_is_accepted(self) -> None:
        cfg = _minimal_dagger_config(dagger_iterations=3, round_seeds=[0, 0, 1])
        self.assertEqual(cfg.round_seeds, [0, 0, 1])

    def test_round_seeds_length_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(dagger_iterations=3, round_seeds=[0, 1])

    def test_round_seeds_and_restart_default_to_unset(self) -> None:
        cfg = _minimal_dagger_config()
        self.assertIsNone(cfg.round_seeds)
        self.assertFalse(cfg.restart_round_seed)


class DaggerConfigExpertBetaPerIterationTests(unittest.TestCase):
    def test_single_or_matching_length_beta_list_is_accepted(self) -> None:
        cfg_single = _minimal_dagger_config(dagger_iterations=3, expert_beta_per_iteration=[0.9])
        self.assertEqual(cfg_single.expert_beta_per_iteration, [0.9, 0.9, 0.9])
        cfg_multi = _minimal_dagger_config(dagger_iterations=3, expert_beta_per_iteration=[0.9, 0.6, 0.3])
        self.assertEqual(cfg_multi.expert_beta_per_iteration, [0.9, 0.6, 0.3])

    def test_mismatched_length_or_out_of_range_beta_is_rejected(self) -> None:
        for bad_betas in ([], [0.9, 0.6], [-0.1], [1.1], [0.9, 1.2, 0.3]):
            with self.subTest(bad_betas=bad_betas):
                with self.assertRaises(ValueError):
                    _minimal_dagger_config(dagger_iterations=3, expert_beta_per_iteration=bad_betas)


class DaggerConfigEvalToleranceOverridesTests(unittest.TestCase):
    def test_defaults_to_none_and_is_independent_of_tolerance_overrides(self) -> None:
        cfg = _minimal_dagger_config(tolerance_overrides={"pos_tol": 0.1})
        self.assertIsNone(cfg.eval_tolerance_overrides)
        self.assertEqual(cfg.tolerance_overrides, {"pos_tol": 0.1})

    def test_can_differ_from_tolerance_overrides(self) -> None:
        cfg = _minimal_dagger_config(
            tolerance_overrides={"pos_tol": 0.1},
            eval_tolerance_overrides={"pos_tol": 0.2, "vel_tol": 0.05},
        )
        self.assertEqual(cfg.tolerance_overrides, {"pos_tol": 0.1})
        self.assertEqual(cfg.eval_tolerance_overrides, {"pos_tol": 0.2, "vel_tol": 0.05})

    def test_non_positive_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _minimal_dagger_config(eval_tolerance_overrides={"pos_tol": 0.0})
        with self.assertRaises(ValueError):
            _minimal_dagger_config(eval_tolerance_overrides={"pos_tol": -0.1})


class DaggerConfigPerRoundExpansionTests(unittest.TestCase):
    """The per-round lists are expanded by DaggerConfig itself: one value is repeated for every round."""

    def test_length_one_lists_are_repeated_for_every_round(self) -> None:
        cfg = _minimal_dagger_config(dagger_iterations=3, round_seeds=[5], training_curriculum=["config"])

        self.assertEqual(cfg.trajectories_per_iteration, [10, 10, 10])
        self.assertEqual(cfg.target_epochs_per_round, [10.0, 10.0, 10.0])
        self.assertEqual(cfg.expert_beta_per_iteration, [0.8, 0.8, 0.8])
        self.assertEqual(cfg.round_seeds, [5, 5, 5])
        self.assertEqual(cfg.training_curriculum, ["config", "config", "config"])

    def test_exact_length_lists_are_preserved(self) -> None:
        cfg = _minimal_dagger_config(
            dagger_iterations=3, expert_beta_per_iteration=[0.9, 0.6, 0.3], round_seeds=[0, 0, 1],
            trajectories_per_iteration=[4, 5, 6], target_epochs_per_round=[1.0, 2.0, 3.0],
        )

        self.assertEqual(cfg.expert_beta_per_iteration, [0.9, 0.6, 0.3])
        self.assertEqual(cfg.round_seeds, [0, 0, 1])
        self.assertEqual((cfg.trajectories_per_iteration, cfg.target_epochs_per_round), ([4, 5, 6], [1.0, 2.0, 3.0]))

    def test_a_list_of_another_length_is_rejected_whichever_it_is(self) -> None:
        for name, value in (
            ("trajectories_per_iteration", [1, 2]),
            ("expert_beta_per_iteration", [0.5, 0.5]),
            ("target_epochs_per_round", [1.0, 2.0]),
            ("round_seeds", [0, 1]),
            ("training_curriculum", ["random", "config"]),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                _minimal_dagger_config(dagger_iterations=3, **{name: value})

    def test_omitted_round_seeds_and_curriculum_stay_none(self) -> None:
        cfg = _minimal_dagger_config(dagger_iterations=3)
        self.assertIsNone(cfg.round_seeds)
        self.assertIsNone(cfg.training_curriculum)

    def test_the_trainer_has_no_schedule_expansion_of_its_own(self) -> None:
        self.assertFalse(hasattr(DaggerTrainer, "schedules"))
        self.assertFalse(hasattr(DaggerTrainer, "persist_round_summary"))


class DaggerEvalMetricsSummaryTests(unittest.TestCase):
    """DaggerEvalMetrics structures its own entry of training_summary.yaml."""

    def _metrics(self, **overrides) -> DaggerEvalMetrics:
        fields = dict(
            success_rate=0.5, mean_steps=40.0, min_steps=20, max_steps=60, num_episodes=4,
            config_successes=1, config_num_episodes=2, random_successes=1, random_num_episodes=2,
            timeout_failures=1, solve_failures=0, collision_failures=1, mean_inference_ms=12.5,
        )
        return DaggerEvalMetrics(**{**fields, **overrides})

    def test_to_dict_is_the_evaluation_part_of_the_entry(self) -> None:
        self.assertEqual(
            self._metrics().to_dict(),
            {
                "eval_success_rate": 0.5, "eval_mean_steps": 40.0, "eval_min_steps": 20, "eval_max_steps": 60,
                "eval_num_episodes": 4,
                "eval_success_by_source": {
                    "config": {"successes": 1, "total": 2, "rate": 50.0},
                    "random": {"successes": 1, "total": 2, "rate": 50.0},
                },
                "eval_failure_breakdown": {"timeout": 1, "solve_failure": 0, "collision": 1},
                "eval_mean_inference_ms": 12.5,
            },
        )

    def test_a_source_with_no_episodes_has_no_rate(self) -> None:
        by_source = self._metrics(config_successes=0, config_num_episodes=0).to_dict()["eval_success_by_source"]
        self.assertEqual(by_source["config"], {"successes": 0, "total": 0, "rate": None})

    def test_a_round_that_was_not_evaluated_has_the_same_keys_without_rates_and_with_zero_counts(self) -> None:
        empty = DaggerEvalMetrics.not_evaluated()

        self.assertEqual(set(empty), set(self._metrics().to_dict()))
        self.assertIsNone(empty["eval_success_rate"])
        self.assertEqual(empty["eval_failure_breakdown"], {"timeout": 0, "solve_failure": 0, "collision": 0})
        self.assertEqual(empty["eval_success_by_source"]["random"], {"successes": 0, "total": 0, "rate": None})

    def test_the_collection_summary_has_the_failures_and_the_expert_statistics_when_there_are_any(self) -> None:
        collected = self._metrics(
            attempted_episodes=12, discarded_episodes=2, planner_failures=2, policy_solve_fallbacks=1,
            expert_statistics={"ompl_paths": 10},
        )

        self.assertEqual(
            collected.collection_summary(),
            {
                "aggregation_failures": {
                    "attempted_episodes": 12, "discarded_episodes": 2, "planner_failures": 2, "policy_solve_fallbacks": 1,
                },
                "expert_statistics": {"ompl_paths": 10},
            },
        )
        self.assertNotIn("expert_statistics", self._metrics().collection_summary())


class DaggerTrainerSummaryYamlTests(unittest.TestCase):
    def test_save_summary_yaml_persists_round_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            cfg = _minimal_dagger_config(checkpoint_dir=checkpoint_dir)
            trainer = DaggerTrainer(cfg)
            metrics = trainer.__class__.__module__
            del metrics

            from learning.dagger.metrics import DaggerEvalMetrics

            eval_metrics = DaggerEvalMetrics(
                success_rate=0.7,
                mean_steps=42.5,
                min_steps=10,
                max_steps=80,
                num_episodes=20,
                config_successes=3,
                config_num_episodes=4,
                random_successes=11,
                random_num_episodes=16,
                timeout_failures=4,
                solve_failures=1,
            )
            trainer.save_round_summary(
                round_index=2,
                train_loss=0.33,
                eval_metrics=eval_metrics,
                latest_checkpoint=checkpoint_dir / "flow_dagger_checkpoint.pt",
                iteration_checkpoint=checkpoint_dir / "flow_dagger_iter_002.pt",
            )
            for round_index, (attempts, discarded, planner_failures, fallbacks) in ((3, (12, 2, 1, 5)), (4, (9, 1, 1, 0))):
                trainer.save_round_summary(
                    round_index=round_index,
                    train_loss=0.3,
                    eval_metrics=eval_metrics,
                    aggregation_metrics=DaggerEvalMetrics(
                        success_rate=0.8, mean_steps=50.0, min_steps=5, max_steps=90, num_episodes=10,
                        attempted_episodes=attempts, discarded_episodes=discarded,
                        planner_failures=planner_failures, policy_solve_fallbacks=fallbacks,
                    ),
                )
            summary_path = checkpoint_dir / "training_summary.yaml"
            self.assertTrue(summary_path.exists())
            summary = yaml.safe_load(summary_path.read_text())
            self.assertEqual(summary["rounds"][0]["round"], 2)
            self.assertEqual(summary["rounds"][0]["eval_success_rate"], 0.7)
            self.assertEqual(summary["rounds"][0]["eval_failure_breakdown"]["timeout"], 4)
            self.assertEqual(summary["best_round"], 2)
            self.assertEqual(summary["best_success_rate"], 0.7)
            self.assertNotIn("aggregation_failures", summary["rounds"][0])
            self.assertEqual(
                summary["rounds"][1]["aggregation_failures"],
                {"attempted_episodes": 12, "discarded_episodes": 2, "planner_failures": 1, "policy_solve_fallbacks": 5},
            )
            self.assertEqual(
                summary["total_aggregation_failures"],
                {"attempted_episodes": 21, "discarded_episodes": 3, "planner_failures": 2, "policy_solve_fallbacks": 5},
            )
            # a run whose expert keeps no statistics reports none
            self.assertNotIn("expert_statistics", summary["rounds"][1])
            self.assertNotIn("total_expert_statistics", summary)

    def test_the_safeflow_evaluation_of_a_flow_policy_is_reported_next_to_its_own(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = DaggerTrainer(_minimal_dagger_config(checkpoint_dir=Path(directory)))
            flow = DaggerEvalMetrics(success_rate=0.1, mean_steps=80.0, min_steps=4, max_steps=100, num_episodes=10, timeout_failures=9)
            safeflow = DaggerEvalMetrics(success_rate=0.9, mean_steps=60.0, min_steps=40, max_steps=100, num_episodes=10, timeout_failures=1)
            trainer.save_round_summary(round_index=0, train_loss=0.3, eval_metrics=flow, safeflow_eval_metrics=safeflow)
            trainer.save_round_summary(round_index=1, train_loss=0.3, eval_metrics=flow)

            summary = yaml.safe_load((Path(directory) / "training_summary.yaml").read_text())

        self.assertEqual(summary["rounds"][0]["eval_success_rate"], 0.1)
        self.assertEqual(summary["rounds"][0]["safeflow_eval"]["eval_success_rate"], 0.9)
        self.assertEqual(summary["rounds"][0]["safeflow_eval"]["eval_failure_breakdown"]["timeout"], 1)
        self.assertNotIn("safeflow_eval", summary["rounds"][1])
        self.assertEqual(summary["best_success_rate"], 0.1)  # the best round is still the one of the trained policy

    def test_the_experts_statistics_are_reported_per_round_and_in_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = DaggerTrainer(_minimal_dagger_config(checkpoint_dir=Path(directory)))
            for round_index, statistics in ((1, {"ompl_paths": 10, "ompl_replans": 0}), (2, {"ompl_paths": 14, "ompl_replans": 4})):
                trainer.save_round_summary(
                    round_index=round_index,
                    train_loss=0.3,
                    eval_metrics=None,
                    aggregation_metrics=DaggerEvalMetrics(
                        success_rate=0.8, mean_steps=50.0, min_steps=5, max_steps=90, num_episodes=10,
                        expert_statistics=statistics,
                    ),
                )

            summary = yaml.safe_load((Path(directory) / "training_summary.yaml").read_text())

        self.assertEqual(summary["rounds"][1]["expert_statistics"], {"ompl_paths": 14, "ompl_replans": 4})
        self.assertEqual(summary["total_expert_statistics"], {"ompl_paths": 24, "ompl_replans": 4})


class DaggerTrainerRuntimeConfigOverridesTests(unittest.TestCase):
    """_apply_runtime_config_overrides injects the training config's convergence
    tolerances into the (already validated) expert config and validates the result again.
    """

    def setUp(self) -> None:
        self.expert_config = load_and_validate_system_config(EXPERT_CONFIG_PATH)

    def test_tolerances_are_injected(self) -> None:
        trainer = DaggerTrainer(
            _minimal_dagger_config(tolerance_overrides={"pos_tol": 0.01})
        )

        config = trainer._apply_runtime_config_overrides(copy.deepcopy(self.expert_config))

        self.assertEqual(config["pos_tol"], 0.01)
        # Not overridden: stays whatever the expert config says.
        self.assertEqual(config["vel_tol"], self.expert_config["vel_tol"])

    def test_explicit_tolerance_overrides_take_precedence_over_the_collection_ones(self) -> None:
        # evaluate_current_policy passes cfg.eval_tolerance_overrides here, so eval can
        # use a more relaxed pass/fail standard than collection/training does.
        trainer = DaggerTrainer(
            _minimal_dagger_config(
                tolerance_overrides={"pos_tol": 0.01},
                eval_tolerance_overrides={"pos_tol": 0.02},
            )
        )

        collection_config = trainer._apply_runtime_config_overrides(copy.deepcopy(self.expert_config))
        eval_config = trainer._apply_runtime_config_overrides(
            copy.deepcopy(self.expert_config), tolerance_overrides=trainer.cfg.eval_tolerance_overrides
        )

        self.assertEqual(collection_config["pos_tol"], 0.01)
        self.assertEqual(eval_config["pos_tol"], 0.02)

    def test_without_overrides_the_validated_config_is_unchanged(self) -> None:
        trainer = DaggerTrainer(_minimal_dagger_config())
        config = trainer._apply_runtime_config_overrides(copy.deepcopy(self.expert_config))
        self.assertEqual(config, self.expert_config)


class ResumableDatasetSchemaTests(unittest.TestCase):
    """A dataset being resumed must carry exactly the arm's current
    observation schema, or DAgger collection fails fast instead of mixing
    layouts.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.features = build_dataset_features(FrankaSystem(load_and_validate_system_config(EXPERT_CONFIG_PATH)), 16)

    def test_matching_schema_is_accepted(self) -> None:
        _validate_resumable_dataset_schema(copy.deepcopy(self.features), self.features)

    def test_a_different_feature_shape_is_rejected(self) -> None:
        existing = copy.deepcopy(self.features)
        existing["observation.state"]["shape"] = (self.features["observation.state"]["shape"][0] + 1,)
        with self.assertRaises(ValueError):
            _validate_resumable_dataset_schema(existing, self.features)

    def test_a_missing_feature_is_rejected(self) -> None:
        existing = copy.deepcopy(self.features)
        del existing["observation.environment_state"]
        with self.assertRaises(ValueError):
            _validate_resumable_dataset_schema(existing, self.features)


class DaggerTrainerSetupAndCheckpointTests(unittest.TestCase):
    """setup() builds the arm simulator and sizes the policy from its dataset
    schema (state_dim = environment_state + state, action_dim = action);
    save_checkpoints() writes the metadata checkpoint scripts/evaluate_policy.py
    rebuilds a policy from (it loads it with weights_only=True).
    """

    UNET = UnetConfig(dim=8, dim_mults=(1, 2))
    PREDICTION_HORIZON = 6
    NUM_INFERENCE_STEPS = 2

    def _expert_config(self) -> dict:
        return load_and_validate_system_config(EXPERT_CONFIG_PATH)

    def _setup_trainer(self, policy_type: str, **overrides: object) -> DaggerTrainer:
        # setup() seeds the process-wide RNGs and turns on deterministic algorithms (warn only).
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trainer = DaggerTrainer(
            _minimal_dagger_config(
                **{
                    "experiment_config": self._expert_config(),
                    "dataset_root": tmpdir / "dataset",
                    "checkpoint_dir": tmpdir / "checkpoints",
                    "unet": self.UNET,
                    "prediction_horizon": self.PREDICTION_HORIZON,
                    "policy_type": policy_type,
                    "flow_config": FlowConfig(num_inference_steps=self.NUM_INFERENCE_STEPS),
                    **overrides,
                }
            )
        )
        # Keeps the test on the CPU whatever accelerator the host has.
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trainer.setup()
        return trainer

    def test_setup_sizes_a_flow_policy_from_the_dataset_schema(self) -> None:
        trainer = self._setup_trainer("flow")

        self.assertIsInstance(trainer.simulator, FrankaSystem)
        self.assertEqual((trainer.state_dim, trainer.action_dim), (12 + 69 + 21, 7))  # goal in the TCP frame, sphere centers, state
        self.assertEqual(trainer.prediction_horizon, self.PREDICTION_HORIZON)
        self.assertEqual(trainer.obs_feature_names, ["observation.environment_state", "observation.state"])
        self.assertIsInstance(trainer.policy, FlowPolicy)
        velocity_net = trainer.policy.velocity_net
        self.assertEqual((velocity_net.obs_cond_dim, velocity_net.action_dim), (102, 7))
        self.assertEqual(velocity_net.prediction_horizon, self.PREDICTION_HORIZON)
        self.assertEqual(velocity_net.num_inference_steps, self.NUM_INFERENCE_STEPS)
        # The chunk is jerks, scaled by their limits
        torch.testing.assert_close(velocity_net.action_scale, torch.full((7,), 15.0))
        self.assertIsNotNone(trainer.ema)

    def test_each_round_decays_the_learning_rate_by_a_cosine_over_twice_its_steps(self) -> None:
        trainer = self._setup_trainer("flow")
        self.assertIsInstance(trainer.optimizer, torch.optim.AdamW)
        self.assertEqual(trainer.optimizer.defaults["weight_decay"], WEIGHT_DECAY)
        batch = (
            {"observation.environment_state": torch.zeros(2, 81), "observation.state": torch.zeros(2, 21)},
            torch.zeros(2, self.PREDICTION_HORIZON, 7),
        )
        base = trainer.cfg.learning_rate

        for _ in range(2):  # the second round starts from the base rate again
            with contextlib.redirect_stdout(io.StringIO()):
                trainer.train_policy_steps([batch], 4)
            # half a cosine period (4 of 8 steps) leaves half the rate above the floor
            self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], (base + MIN_LEARNING_RATE) / 2, delta=1e-9)

    def test_rollouts_use_a_cpu_copy_of_the_policy_that_holds_the_averaged_weights(self) -> None:
        for policy_type in ("flow", "safeflow"):
            with self.subTest(policy_type=policy_type):
                trainer = self._setup_trainer(policy_type)
                self.assertIsNot(trainer.rollout_policy, trainer.policy)
                with torch.no_grad():
                    for parameter in trainer.policy.parameters():
                        parameter.add_(1.0)  # the training weights move on, the averaged ones do not
                trainer.ema.shadow = {name: tensor + 0.5 for name, tensor in trainer.ema.shadow.items()}

                trainer._sync_rollout_policy()

                expected = trainer.ema.state_dict()
                for name, value in trainer.rollout_policy.state_dict().items():
                    self.assertEqual(value.device.type, "cpu")
                    torch.testing.assert_close(value, expected[name].cpu())

    def _saved_checkpoint(self) -> tuple[DaggerTrainer, Path]:
        """A trainer that has taken optimizer steps, and the checkpoint it saved."""
        trainer = self._setup_trainer("flow")
        batch = (
            {"observation.environment_state": torch.randn(4, 81), "observation.state": torch.randn(4, 21)},
            torch.randn(4, self.PREDICTION_HORIZON, 7),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train_policy_steps([batch], 3)
            trainer.save_checkpoints(training_round=2)
        return trainer, trainer.cfg.checkpoint_dir / "flow_dagger_iter_002.pt"

    def test_a_run_can_start_from_the_weights_the_averages_and_the_optimizer_of_a_checkpoint(self) -> None:
        saved, path = self._saved_checkpoint()

        resumed = self._setup_trainer("flow", init_checkpoint=path)

        for name, value in saved.policy.state_dict().items():
            torch.testing.assert_close(resumed.policy.state_dict()[name], value)
            torch.testing.assert_close(resumed.ema.state_dict()[name], saved.ema.state_dict()[name])
        self.assertEqual(resumed.optimizer.state_dict()["state"].keys(), saved.optimizer.state_dict()["state"].keys())
        self.assertGreater(len(resumed.optimizer.state_dict()["state"]), 0)

    def test_a_checkpoint_of_another_architecture_is_rejected(self) -> None:
        _, path = self._saved_checkpoint()

        with self.assertRaisesRegex(ValueError, "unet"):
            self._setup_trainer("flow", init_checkpoint=path, unet=UnetConfig(dim=16, dim_mults=(1, 2)))

    def test_a_checkpoint_with_non_finite_weights_is_rejected(self) -> None:
        _, path = self._saved_checkpoint()
        checkpoint = torch.load(path, weights_only=False)
        next(iter(checkpoint["model_state_dict"].values())).fill_(float("nan"))
        torch.save(checkpoint, path)

        with self.assertRaisesRegex(ValueError, "non-finite"):
            self._setup_trainer("flow", init_checkpoint=path)

    def test_a_missing_checkpoint_is_rejected_by_the_config(self) -> None:
        with self.assertRaises(FileNotFoundError):
            _minimal_dagger_config(init_checkpoint=Path("does_not_exist.pt"))

    def test_a_flow_trainer_evaluates_its_weights_as_a_safeflow_policy_too(self) -> None:
        trainer = self._setup_trainer("flow", eval_episodes=2, eval_steps=2, num_workers=1)

        with mock.patch.object(trainer, "_build_policy", wraps=trainer._build_policy) as build, contextlib.redirect_stdout(io.StringIO()):
            flow_metrics = trainer.evaluate_current_policy("flow evaluation")
            safeflow_metrics = trainer.evaluate_current_policy("safeflow evaluation", "safeflow")

        self.assertEqual([call.args[0] for call in build.call_args_list], ["safeflow"])  # the flow one is the trainer's own
        self.assertEqual((flow_metrics.num_episodes, safeflow_metrics.num_episodes), (2, 2))

    def test_an_offline_start_also_evaluates_a_flow_policy_as_safeflow_and_labels_both(self) -> None:
        trainer = self._setup_trainer("flow", eval_episodes=1, dagger_iterations=0)
        object.__setattr__(trainer.cfg, "start_with_aggregation", False)  # the config would want an existing dataset
        labels: list[tuple[str, str | None]] = []
        evaluation = DaggerEvalMetrics(success_rate=0.5, mean_steps=1.0, min_steps=1, max_steps=1, num_episodes=1)

        def record(label: str, policy_type: str | None = None) -> DaggerEvalMetrics:
            labels.append((label, policy_type))
            return evaluation

        with (
            mock.patch.object(trainer, "setup"),
            mock.patch.object(trainer, "train_on_aggregate"),
            mock.patch.object(trainer, "evaluate_current_policy", side_effect=record),
            mock.patch.object(trainer, "save_checkpoints"),
            mock.patch.object(trainer, "save_round_summary") as save,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            trainer.run()

        self.assertEqual(labels, [("Round 0 flow evaluation", None), ("Round 0 safeflow evaluation", "safeflow")])
        self.assertIs(save.call_args.kwargs["safeflow_eval_metrics"], evaluation)

    def test_setup_builds_the_safeflow_projector_on_the_trainers_simulator(self) -> None:
        trainer = self._setup_trainer("safeflow")

        self.assertIsInstance(trainer.policy, SafeFlowMPCPolicy)
        self.assertEqual(trainer.policy.projector.N, self.PREDICTION_HORIZON)
        self.assertIs(trainer.policy.projector.sim, trainer.simulator)

    def test_checkpoints_follow_the_documented_schema_and_rebuild_a_flow_policy(self) -> None:
        for policy_type in ("flow", "safeflow"):
            with self.subTest(policy_type=policy_type):
                trainer = self._setup_trainer(policy_type)
                checkpoint_dir = trainer.cfg.checkpoint_dir
                with contextlib.redirect_stdout(io.StringIO()):
                    trainer.save_checkpoints(training_round=2)

                # Both files evaluate_policy.py / resumption look for, always under the flow_dagger prefix.
                self.assertTrue((checkpoint_dir / "flow_dagger_iter_002.pt").exists())
                checkpoint = torch.load(checkpoint_dir / "flow_dagger_checkpoint.pt", map_location="cpu", weights_only=True)
                self.assertEqual(
                    set(checkpoint),
                    {
                        "iteration", "model_state_dict", "ema_model_state_dict", "optimizer_state_dict",
                        "state_dim", "action_dim", "prediction_horizon", "unet", "obs_feature_names",
                        "policy_type", "flow_config",
                    },
                )
                self.assertEqual(checkpoint["iteration"], 2)
                self.assertEqual((checkpoint["state_dim"], checkpoint["action_dim"]), (102, 7))
                self.assertEqual(checkpoint["prediction_horizon"], self.PREDICTION_HORIZON)
                self.assertEqual(checkpoint["unet"], {"dim": self.UNET.dim, "dim_mults": list(self.UNET.dim_mults)})
                self.assertEqual(checkpoint["obs_feature_names"], ["observation.environment_state", "observation.state"])
                self.assertEqual(checkpoint["policy_type"], policy_type)
                self.assertEqual(
                    checkpoint["flow_config"],
                    {
                        "num_inference_steps": self.NUM_INFERENCE_STEPS,
                        "safeflow_init": "previous_safe",
                        "action_scale": [15.0] * 7,
                    },
                )

                # 'flow' and 'safeflow' checkpoints are interchangeable: a bare FlowPolicy rebuilt
                # from the checkpoint's own metadata loads either one (through the loading utility).
                rebuilt = PolicyFactory.create(
                    PolicySpec(
                        policy_type="flow",
                        action_dim=checkpoint["action_dim"],
                        state_dim=checkpoint["state_dim"],
                        prediction_horizon=checkpoint["prediction_horizon"],
                        unet_dim=checkpoint["unet"]["dim"],
                        unet_dim_mults=tuple(checkpoint["unet"]["dim_mults"]),
                        num_inference_steps=checkpoint["flow_config"]["num_inference_steps"],
                        action_scale=tuple(checkpoint["flow_config"]["action_scale"]),
                    ),
                    trainer.simulator,
                )
                rebuilt.load_network_state_dict(checkpoint["ema_model_state_dict"])
                prefix = "inner_policy." if policy_type == "safeflow" else ""
                self.assertEqual({prefix + name for name in rebuilt.state_dict()}, set(checkpoint["ema_model_state_dict"]))
                for name, value in rebuilt.state_dict().items():
                    torch.testing.assert_close(value, checkpoint["ema_model_state_dict"][prefix + name])


if __name__ == "__main__":
    unittest.main()
