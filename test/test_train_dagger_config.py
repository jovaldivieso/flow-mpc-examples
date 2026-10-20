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
from dynamics.franka_dynamics import FrankaDynamics
from learning.config_loaders import FlowConfig, UnetConfig
from learning.dagger.dagger_trainer import MIN_LEARNING_RATE, WEIGHT_DECAY, _validate_resumable_dataset_schema
from learning.dagger.metrics import DaggerEvalMetrics
from learning.models.flow_policy import FlowPolicy
from learning.models.policy import PolicyFactory
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
        self.assertEqual(cfg_single.expert_beta_per_iteration, [0.9])
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


class DaggerTrainerSchedulesRoundSeedsTests(unittest.TestCase):
    def test_length_one_round_seeds_broadcasts_to_every_round(self) -> None:
        _, _, betas, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            betas=[0.8],
            rounds=3,
            round_seeds=[5],
        )
        self.assertEqual(betas, [0.8, 0.8, 0.8])
        self.assertEqual(round_seeds, [5, 5, 5])

    def test_exact_length_round_seeds_is_preserved(self) -> None:
        _, _, betas, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            betas=[0.9, 0.6, 0.3],
            rounds=3,
            round_seeds=[0, 0, 1],
        )
        self.assertEqual(betas, [0.9, 0.6, 0.3])
        self.assertEqual(round_seeds, [0, 0, 1])

    def test_mismatched_round_seeds_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            DaggerTrainer.schedules(
                trajectories=[10],
                epochs=[10.0],
                betas=[0.8],
                rounds=3,
                round_seeds=[0, 1],
            )

    def test_omitted_round_seeds_stays_none(self) -> None:
        _, _, _, _, round_seeds = DaggerTrainer.schedules(
            trajectories=[10],
            epochs=[10.0],
            betas=[0.8],
            rounds=3,
        )
        self.assertIsNone(round_seeds)


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
        cls.features = FrankaDynamics(load_and_validate_system_config(EXPERT_CONFIG_PATH)).get_dataset_features(16)

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

    def _setup_trainer(self, policy_type: str) -> DaggerTrainer:
        # setup() seeds the process-wide RNGs and turns on deterministic algorithms (warn only).
        self.addCleanup(
            torch.use_deterministic_algorithms,
            torch.are_deterministic_algorithms_enabled(),
            warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
        )
        tmpdir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trainer = DaggerTrainer(
            _minimal_dagger_config(
                experiment_config=self._expert_config(),
                dataset_root=tmpdir / "dataset",
                checkpoint_dir=tmpdir / "checkpoints",
                unet=self.UNET,
                prediction_horizon=self.PREDICTION_HORIZON,
                policy_type=policy_type,
                flow_config=FlowConfig(num_inference_steps=self.NUM_INFERENCE_STEPS),
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

        self.assertIsInstance(trainer.simulator, FrankaDynamics)
        self.assertEqual((trainer.state_dim, trainer.action_dim), (7 + 21, 7))
        self.assertEqual(trainer.prediction_horizon, self.PREDICTION_HORIZON)
        self.assertEqual(trainer.obs_feature_names, ["observation.environment_state", "observation.state"])
        self.assertIsInstance(trainer.policy, FlowPolicy)
        velocity_net = trainer.policy.velocity_net
        self.assertEqual((velocity_net.obs_cond_dim, velocity_net.action_dim), (28, 7))
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
            {"observation.environment_state": torch.zeros(2, 7), "observation.state": torch.zeros(2, 21)},
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

    def test_setup_builds_the_safeflow_projector_on_a_goal_zeroed_copy_of_the_simulator(self) -> None:
        trainer = self._setup_trainer("safeflow")

        self.assertIsInstance(trainer.policy, SafeFlowMPCPolicy)
        self.assertEqual(trainer.policy.projector.N, self.PREDICTION_HORIZON)
        self.assertIsNot(trainer.policy.local_sim, trainer.simulator)
        self.assertTrue((trainer.policy.local_sim.goal == 0.0).all())

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
                self.assertEqual((checkpoint["state_dim"], checkpoint["action_dim"]), (28, 7))
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
                # from the checkpoint's own metadata loads either one.
                rebuilt = PolicyFactory.create(
                    "flow",
                    action_dim=checkpoint["action_dim"],
                    state_dim=checkpoint["state_dim"],
                    unet_dim=checkpoint["unet"]["dim"],
                    unet_dim_mults=tuple(checkpoint["unet"]["dim_mults"]),
                    prediction_horizon=checkpoint["prediction_horizon"],
                    num_inference_steps=checkpoint["flow_config"]["num_inference_steps"],
                    action_scale=checkpoint["flow_config"]["action_scale"],
                )
                rebuilt.load_state_dict(checkpoint["ema_model_state_dict"])
                prefix = "inner_policy." if policy_type == "safeflow" else ""
                self.assertEqual({prefix + name for name in rebuilt.state_dict()}, set(checkpoint["ema_model_state_dict"]))
                for name, value in rebuilt.state_dict().items():
                    torch.testing.assert_close(value, checkpoint["ema_model_state_dict"][prefix + name])


if __name__ == "__main__":
    unittest.main()
