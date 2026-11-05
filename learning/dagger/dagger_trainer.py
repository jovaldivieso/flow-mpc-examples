from __future__ import annotations

import copy
import gc
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from planning.config import validate_system_config
from systems.position_history import PositionHistory
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from systems.seed_utils import default_action_noise_seed_for_config
from learning.data_utils import build_dataset_features, create_collate_fn
from learning.models.exponential_moving_average import ExponentialMovingAverage
from learning.models.policy import ActionPolicy
from learning.models.policy_factory import PolicyFactory, PolicySpec
from learning.models.safe_source import SafeSource
from planning.planner_config import ProjectionConfig
from planning.planner_factory import PlannerFactory

from .coverage import CoverageMetrics, DatasetCoverage
from .dagger_config import DaggerConfig
from .episode_starts import EpisodeStarts
from .metrics import DaggerEvalMetrics
from .collection import collect_dagger_rollouts
from .evaluation import evaluate_policy_rollouts
from .policy_action import build_policy_actions
from .round_summary import RoundTiming, append_round_summary
from .safe_pairs import PooledPairs, SafePairs
from .workers import RolloutWorkerSpec
from .utils import (
    apply_config_overrides,
    evaluation_seed_specs,
    print_rollout_metrics,
    resolve_initial_state_seed,
    resolve_round_steps,
    set_seed,
    with_seeded_initial_state_config,
)

# SafeFlowMPC's training recipe besides the config's learning rate: AdamW, and per round a cosine decay of the
# learning rate over twice the round's steps (half a cosine period) down to MIN_LEARNING_RATE
WEIGHT_DECAY = 1e-7
MIN_LEARNING_RATE = 1e-7
ROLLOUT_DEVICE = torch.device("cpu")
COVERAGE_EPISODES = 50  # evaluation starts whose coverage by the dataset is measured, per set of modes


def _validate_resumable_dataset_schema(
    existing_features: Mapping[str, Any],
    expected_features: Mapping[str, Any],
) -> None:
    """Fail fast when a dataset being resumed doesn't match the current observation schema."""
    for name, expected_info in expected_features.items():
        expected_shape = tuple(int(value) for value in expected_info.get("shape", ()))
        expected_dtype = expected_info.get("dtype")
        expected_names = list(expected_info.get("names") or [])
        existing_info = existing_features.get(name)
        if existing_info is None:
            existing_shape = None
            existing_dtype = None
            existing_names = None
        else:
            existing_shape = tuple(int(value) for value in existing_info.get("shape", ()))
            existing_dtype = existing_info.get("dtype")
            existing_names = list(existing_info.get("names") or [])
        if (existing_shape, existing_dtype, existing_names) != (expected_shape, expected_dtype, expected_names):
            raise ValueError(
                "Cannot resume DAgger collection: the on-disk dataset's "
                f"'{name}' feature is (shape={existing_shape}, dtype={existing_dtype}, names={existing_names}), "
                f"but the current run's observation schema expects "
                f"(shape={expected_shape}, dtype={expected_dtype}, names={expected_names}). "
                "The dataset was likely recorded with a different observation feature layout. "
                "Start a fresh dataset (omit --repo-id/--dataset-root) or resume a dataset recorded "
                "with the current schema."
            )


class DaggerTrainer:
    def __init__(self, cfg: DaggerConfig) -> None:
        self.cfg = cfg
        self.device: torch.device | None = None
        self.simulator: SystemSimulator | None = None
        self.seeded_config: Mapping[str, Any] | None = None
        self.policy: ActionPolicy | None = None
        self.rollout_policy: ActionPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.ema: ExponentialMovingAverage | None = None
        self.action_noise_seed = 0
        self.initial_state_seed = 0
        self.obs_feature_names: list[str] = []
        self.cond_dim = self.action_dim = self.prediction_horizon = 0
        self.timing = RoundTiming()  # the phases of the round under way
        self._round_started = time.perf_counter()
        self._safe_source: SafeSource | None = None  # built when a round first trains from safe sources

    def setup(self) -> None:
        set_seed(self.cfg.seed)
        self.seeded_config = with_seeded_initial_state_config(
            self.cfg.experiment_config,
            self.cfg.seed,
        )
        self.simulator = SystemFactory.create(self.seeded_config["system"], self.seeded_config)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.action_noise_seed = default_action_noise_seed_for_config(self.seeded_config)
        self.initial_state_seed = resolve_initial_state_seed(self.seeded_config, self.cfg.seed)
        # The policy predicts prediction_horizon planned positions per state, which DAgger labels with the expert's plan
        self.prediction_horizon = self.cfg.prediction_horizon
        expert_horizon = int(self.seeded_config["horizon"])
        if self.prediction_horizon > expert_horizon:
            raise ValueError(
                f"The policy's 'prediction_horizon' ({self.prediction_horizon}) must not exceed the expert's 'horizon' "
                f"({expert_horizon}): the expert's plan labels the policy's targets."
            )
        features = build_dataset_features(self.simulator, self.prediction_horizon, self.cfg.flow_config.history_length)
        if self.cfg.dataset_root.exists():
            existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
            _validate_resumable_dataset_schema(existing_meta.features, features)
        self.obs_feature_names = [n for n in features if n.startswith("observation.")]
        # The policy is conditioned on observation.environment_state (the goal and the body) and observation.history
        environment_state_dim = int(features["observation.environment_state"]["shape"][0])
        history_dim = int(features["observation.history"]["shape"][0])
        self.action_dim = self.simulator.n  # the policy's chunk has prediction_horizon rows of action_dim joint positions

        self.cond_dim = environment_state_dim + history_dim

        self.policy = self._build_policy(self.cfg.policy_type, self.device, self.simulator, self.seeded_config)
        # Single-sample inference is faster on the CPU than on an accelerator, so rollouts use a CPU copy
        self.rollout_policy = self._build_policy(self.cfg.policy_type, ROLLOUT_DEVICE, self.simulator, self.seeded_config)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.learning_rate, weight_decay=WEIGHT_DECAY
        )
        # The rollouts and checkpoints for inference use the averaged weights
        self.ema = ExponentialMovingAverage(self.policy)
        if self.cfg.init_checkpoint is not None:
            self._load_init_checkpoint(self.cfg.init_checkpoint)

        for name in self.cfg.regions:  # a region the system cannot sample in fails now, not rounds into the run
            self._apply_runtime_config_overrides(copy.deepcopy(dict(self.seeded_config)), region=name)

        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Keep startup logs brief and non-redundant; the full config summary is already printed above.
        print(f"Starting DAgger training for franka ({self.cfg.policy_type})")
        print(f"Device: {self.device} (rollouts on {ROLLOUT_DEVICE}); dataset_root={self.cfg.dataset_root}")
        print(
            "Schedule: "
            f"rounds={self.cfg.dagger_iterations}, "
            f"trajectories_per_round={self.cfg.trajectories_per_iteration}, "
            f"steps_per_trajectory={self.cfg.steps_per_trajectory}, "
            f"target_epochs={self.cfg.target_epochs_per_round}, "
            f"expert_beta={self.cfg.expert_beta_per_iteration}"
        )
        print(
            "Noise: "
            f"aggregation_std={self.cfg.action_noise_std:.6f}, "
            f"eval_std={self.cfg.eval_action_noise_std:.6f}, "
            f"seed={self.action_noise_seed}"
        )
        if self.cfg.start_with_aggregation:
            print("Mode: fresh DAgger collection before any offline pretraining")
        else:
            print("Mode: resume from existing expert dataset")

    def _load_init_checkpoint(self, path: Path) -> None:
        """Start from the weights, EMA and optimizer state of a checkpoint of the same architecture. The policy type may
        differ: flow and safeflow share their weights, so a flow checkpoint initializes a safeflow run."""
        assert self.policy is not None and self.optimizer is not None and self.ema is not None
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "cond_dim": self.cond_dim,
            "action_dim": self.action_dim,
            "prediction_horizon": self.prediction_horizon,
            "history_length": self.cfg.flow_config.history_length,
            "unet": {"dim": self.cfg.unet.dim, "dim_mults": list(self.cfg.unet.dim_mults)},
        }
        for key, value in expected.items():
            if checkpoint[key] != value:
                raise ValueError(f"Cannot start from {path}: its '{key}' is {checkpoint[key]}, this run's is {value}.")
        for name in ("model_state_dict", "ema_model_state_dict"):
            if not all(torch.isfinite(t).all() for t in checkpoint[name].values() if t.is_floating_point()):
                raise ValueError(f"Cannot start from {path}: its '{name}' has non-finite weights.")
        self.policy.load_network_state_dict(checkpoint["ema_model_state_dict"])
        self.ema = ExponentialMovingAverage(self.policy)
        self.policy.load_network_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Initialized from {path} (iteration {checkpoint['iteration']})")

    def train_policy_steps(self, dataloader: DataLoader, num_steps: int) -> float:
        if self.policy is None or self.optimizer is None or self.device is None:
            raise RuntimeError("Trainer is not set up.")
        self.policy.train()
        for group in self.optimizer.param_groups:
            group["lr"] = self.cfg.learning_rate  # every round starts the decay afresh
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=2 * num_steps, eta_min=MIN_LEARNING_RATE
        )
        running_loss = 0.0
        iterator = iter(dataloader)
        progress = tqdm(
            range(1, num_steps + 1),
            desc="Train steps",
            leave=False,
            dynamic_ncols=True,
        ) if tqdm is not None else None
        step_iterator = progress if progress is not None else range(1, num_steps + 1)
        if progress is None:
            print("tqdm not installed; showing periodic step progress.")

        for step in step_iterator:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(dataloader)
                batch = next(iterator)
            observations, actions, paths = batch
            if isinstance(observations, dict):
                observations = {
                    key: value.to(self.device)
                    for key, value in observations.items()
                }
            else:
                observations = observations.to(self.device)
            actions = actions.to(self.device)
            paths = paths.to(self.device) if paths is not None else None
            self.optimizer.zero_grad()
            loss = self.policy.compute_loss(observations, actions, paths)
            loss.backward()
            self.optimizer.step()
            scheduler.step()
            self.ema.update(self.policy)

            step_loss = float(loss.item())
            running_loss += step_loss
            running_mean = running_loss / float(step)
            if progress is not None:
                progress.set_postfix(loss=f"{step_loss:.6f}", mean=f"{running_mean:.6f}")
            elif step == 1 or step == num_steps or step % max(1, num_steps // 10) == 0:
                print(f"  step {step}/{num_steps} loss={step_loss:.6f} mean_loss={running_mean:.6f}")

        if progress is not None:
            progress.close()
        self.policy.eval()
        return running_loss / float(num_steps)

    def train_on_aggregate(self, label: str, training_round: int, target_epochs: float) -> float:
        assert self.simulator is not None
        assert self.policy is not None
        assert self.optimizer is not None
        assert self.device is not None
        print(f"\n=== {label} ===")
        started = time.perf_counter()
        dataset = LeRobotDataset(
            repo_id=self.cfg.repo_id,
            root=self.cfg.dataset_root,
        )
        generator = torch.Generator().manual_seed(self.cfg.seed + training_round)
        pairs = self._safe_pairs(dataset)
        training_set = dataset
        if pairs is not None:
            with_path = np.flatnonzero(np.isfinite(pairs.paths).reshape(len(pairs), -1).all(axis=1))
            if len(with_path) == 0:
                raise RuntimeError("No frame of the dataset has a safe path to train the safeflow policy on.")
            training_set = Subset(dataset, with_path.tolist())
        loader = DataLoader(
            training_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=create_collate_fn(self.prediction_horizon, pairs),
        )
        steps, approx = resolve_round_steps(
            len(training_set),
            self.cfg.batch_size,
            target_epochs,
            self.cfg.max_train_steps,
        )
        print(f"Training on {len(training_set)} of {len(dataset)} aggregated frames")
        print(
            f"  optimizer_steps={steps} "
            f"(~{approx:.2f} epochs at batch_size={self.cfg.batch_size})"
        )
        mean_loss = self.train_policy_steps(loader, steps)
        print(f"  mean_step_loss={mean_loss:.6f}")
        self.timing.add_training(time.perf_counter() - started)
        return mean_loss

    def _safe_pairs(self, dataset: LeRobotDataset) -> SafePairs | None:
        """The safe path from the safe source to the safe target of every frame of ``dataset``, which the safeflow policy
        trains on (the frames without one are left out), or None for a flow policy, which trains from Gaussian noise."""
        if self.cfg.policy_type != "safeflow":
            return None
        assert self.simulator is not None and self.seeded_config is not None
        pairs = SafePairs(self.cfg.dataset_root, self.prediction_horizon, self.action_dim)

        def load(start: int) -> tuple[np.ndarray, np.ndarray]:
            rows = dataset.hf_dataset.select(range(start, len(dataset)))
            states = np.asarray([row.numpy() for row in rows["observation.state"]], dtype=float)
            labels = np.asarray([row.numpy() for row in rows["action"]], dtype=float)
            return states, labels.reshape(-1, self.prediction_horizon, self.action_dim)

        started, before = time.perf_counter(), len(pairs)
        if self.cfg.num_workers > 1 and len(dataset) - before > PooledPairs.CHUNK:
            with PooledPairs(self.seeded_config, self.prediction_horizon, self.cfg.seed, self.cfg.num_workers) as pairs_of:
                pairs.update(len(dataset), load, pairs_of)
        else:
            if self._safe_source is None:
                self._safe_source = SafeSource(
                    self.simulator, ProjectionConfig.from_mapping(self.seeded_config), self.prediction_horizon, self.cfg.seed
                )
            pairs.update(len(dataset), load, self._safe_source.paths)
        usable = float(np.isfinite(pairs.paths).reshape(len(pairs), -1).all(axis=1).mean())
        print(
            f"Safe sources: {len(pairs) - before} new frames, {len(pairs)} in all ({100 * usable:.0f}% with a path), "
            f"{time.perf_counter() - started:.0f} s"
        )
        return pairs

    def _policy_spec(self, policy_type: str) -> PolicySpec:
        assert self.simulator is not None
        return PolicySpec(
            policy_type=policy_type,
            action_dim=self.action_dim,
            cond_dim=self.cond_dim,
            history_length=self.cfg.flow_config.history_length,
            execution_horizon=self.cfg.execution_horizon,
            prediction_horizon=self.prediction_horizon,
            unet_dim=self.cfg.unet.dim,
            unet_dim_mults=tuple(self.cfg.unet.dim_mults),
            num_inference_steps=self.cfg.flow_config.num_inference_steps,
            action_scale=tuple(self.simulator.position_scale.tolist()),
            flow_init=self.cfg.flow_config.safeflow_init,
        )

    def _build_policy(
        self, policy_type: str, device: torch.device, simulator: SystemSimulator, planner_config: dict[str, Any]
    ) -> ActionPolicy:
        """A policy of ``policy_type`` for ``simulator`` (whose projector is built from ``planner_config``)."""
        return PolicyFactory.create(self._policy_spec(policy_type), simulator, planner_config).to(device).eval()

    def _apply_runtime_config_overrides(
        self,
        config: dict[str, Any],
        *,
        tolerance_overrides: Mapping[str, float] | None = None,
        region: str | None = None,
    ) -> dict[str, Any]:
        """Inject the training config's solver/dynamics tuning knobs (tolerances), if set, and the sampling boxes of
        the named ``region``.

        tolerance_overrides defaults to self.cfg.tolerance_overrides (the
        collection/training value) when not given -- evaluate_current_policy
        passes self.cfg.eval_tolerance_overrides here instead, so eval can
        use a more relaxed pass/fail standard without touching what
        collection/training actually demonstrates against.
        """
        effective_tolerance_overrides = (
            tolerance_overrides if tolerance_overrides is not None else self.cfg.tolerance_overrides
        )
        merged_config = apply_config_overrides(config, effective_tolerance_overrides or {})
        if region is not None:
            merged_config = apply_config_overrides(merged_config, self.cfg.regions[region])
        return validate_system_config(merged_config)

    def _sync_rollout_policy(self) -> None:
        """Give the CPU rollout policy the averaged weights of the policy being trained."""
        assert self.rollout_policy is not None and self.ema is not None
        self.rollout_policy.load_state_dict({name: tensor.cpu() for name, tensor in self.ema.state_dict().items()})

    def _worker_spec(
        self, config: dict[str, Any], *, need_expert: bool, need_policy: bool
    ) -> RolloutWorkerSpec:
        assert self.simulator is not None and self.rollout_policy is not None
        policy_spec = None
        policy_state_dict = None
        if need_policy:
            policy_spec = self._policy_spec(self.cfg.policy_type)
            policy_state_dict = {name: tensor.cpu() for name, tensor in self.rollout_policy.state_dict().items()}
        return RolloutWorkerSpec(
            config=dict(config),
            policy_spec=policy_spec,
            policy_state_dict=policy_state_dict,
            need_expert=need_expert,
        )

    def evaluate_current_policy(self, label: str, region: str | None = None) -> DaggerEvalMetrics | None:
        """Evaluate the policy being trained. The episodes start and end as the system config samples them, or in the
        named ``region``."""
        assert self.rollout_policy is not None
        assert self.seeded_config is not None
        if self.cfg.eval_episodes == 0:
            return None
        started = time.perf_counter()
        eval_config = self._apply_runtime_config_overrides(
            copy.deepcopy(dict(self.seeded_config)),
            tolerance_overrides=self.cfg.eval_tolerance_overrides,
            region=region,
        )
        simulator = SystemFactory.create(eval_config["system"], eval_config)

        self._sync_rollout_policy()
        policy = self.rollout_policy

        def action_fn(obs: np.ndarray, history: np.ndarray) -> np.ndarray:
            return build_policy_actions(simulator, policy, obs, history, ROLLOUT_DEVICE)

        metrics = evaluate_policy_rollouts(
            simulator,
            self.cfg.eval_episodes,
            self.cfg.eval_steps or self.cfg.steps_per_trajectory,
            self.cfg.eval_seed_start,
            action_fn,
            self.cfg.flow_config.history_length,
            reset_fn=policy.reset,
            action_noise_std=self.cfg.eval_action_noise_std,
            action_noise_seed=self.action_noise_seed,
            initial_states=self.cfg.initial_states if region is None else None,
            goal_states=self.cfg.goal_states if region is None else None,
            num_workers=self.cfg.num_workers,
            worker_spec=self._worker_spec(eval_config, need_expert=False, need_policy=True),
        )
        if metrics is not None:
            print_rollout_metrics(label, "eval", metrics)
        self.timing.add_evaluation(self.cfg.policy_type, time.perf_counter() - started, region)
        return metrics

    def evaluate_regions(self, stage: str) -> dict[str, dict[str, DaggerEvalMetrics | None]]:
        """The evaluation in every region, by region and policy type."""
        return {
            name: {
                self.cfg.policy_type: self.evaluate_current_policy(
                    f"{stage} {self.cfg.policy_type} evaluation in region '{name}'", region=name
                )
            }
            for name in self.cfg.regions
        }

    def measure_coverage(self) -> dict[str, CoverageMetrics] | None:
        """How well the dataset covers the starts of the evaluation episodes, on all the modes ('random') and in every region."""
        if self.cfg.eval_episodes == 0:
            return None
        assert self.seeded_config is not None
        started = time.perf_counter()
        coverage = DatasetCoverage.from_dataset(self.cfg.dataset_root)
        seeds = evaluation_seed_specs(min(self.cfg.eval_episodes, COVERAGE_EPISODES), self.cfg.eval_seed_start)
        results: dict[str, CoverageMetrics] = {}
        for name in ("random", *self.cfg.regions):
            region = None if name == "random" else name
            config = self._apply_runtime_config_overrides(
                copy.deepcopy(dict(self.seeded_config)), tolerance_overrides=self.cfg.eval_tolerance_overrides, region=region
            )
            simulator = SystemFactory.create(config["system"], config)
            starts = EpisodeStarts.from_specs(
                simulator,
                self.cfg.initial_states if region is None else None,
                self.cfg.goal_states if region is None else None,
            )
            queries = np.array([self._start_features(simulator, starts.begin(simulator, index, seed)) for index, seed in enumerate(seeds)])
            results[name] = coverage.measure(queries)
            print(
                f"Coverage of the {name} starts: median distance {results[name].median:.2f}, "
                f"{100 * results[name].far_fraction:.0f}% farther than the dataset's own episodes (p90)"
            )
        self.timing.coverage = time.perf_counter() - started
        return results

    def _start_features(self, simulator: SystemSimulator, initial_state: np.ndarray) -> np.ndarray:
        """What the policy is conditioned on at the start of an episode (the history is the start configuration repeated)."""
        history = PositionHistory(simulator.n, self.cfg.flow_config.history_length)
        history.reset(initial_state)
        observed = simulator.policy_observation(simulator.observe(initial_state), history.flat)
        return np.concatenate([observed["observation.environment_state"], observed["observation.history"]])

    def save_round_summary(
        self,
        round_index: int,
        train_loss: float | None,
        eval_metrics: DaggerEvalMetrics | None,
        aggregation_metrics: DaggerEvalMetrics | None = None,
        latest_checkpoint: str | None = None,
        iteration_checkpoint: str | None = None,
        region_eval_metrics: Mapping[str, Mapping[str, DaggerEvalMetrics | None]] | None = None,
        timing: RoundTiming | None = None,
        coverage: Mapping[str, CoverageMetrics] | None = None,
    ) -> None:
        """Persist a round-by-round YAML summary with success and failure stats."""

        append_round_summary(
            self.cfg.checkpoint_dir / "training_summary.yaml",
            round_index,
            train_loss,
            eval_metrics,
            aggregation_metrics,
            latest_checkpoint,
            iteration_checkpoint,
            region_eval_metrics,
            timing,
            coverage,
        )

    def _start_round(self) -> None:
        self.timing = RoundTiming()
        self._round_started = time.perf_counter()

    def _finish_round(self) -> RoundTiming:
        """The timing of the round that is over."""
        self.timing.total = time.perf_counter() - self._round_started
        return self.timing

    def save_checkpoints(self, training_round: int) -> None:
        assert self.policy is not None
        assert self.optimizer is not None
        data: dict[str, Any] = {
            "iteration": training_round,
            "model_state_dict": self.policy.state_dict(),
            # What inference runs with (evaluate_policy.py prefers it over model_state_dict)
            "ema_model_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "cond_dim": self.cond_dim,
            "action_dim": self.action_dim,
            "prediction_horizon": self.prediction_horizon,
            "history_length": self.cfg.flow_config.history_length,
            "execution_horizon": self.cfg.execution_horizon,
            "unet": {"dim": self.cfg.unet.dim, "dim_mults": list(self.cfg.unet.dim_mults)},
            "obs_feature_names": self.obs_feature_names,
            "policy_type": self.cfg.policy_type,
        }
        data["flow_config"] = {
            "num_inference_steps": self.cfg.flow_config.num_inference_steps,
            "safeflow_init": self.cfg.flow_config.safeflow_init,
            # Saved explicitly so a checkpoint's normalized action space stays what trained it -- see FlowPolicy's own
            # docstring for why training and inference must agree on this exactly.
            "action_scale": self.simulator.position_scale.tolist(),
        }
        prefix = "flow_dagger"
        latest_checkpoint = self.cfg.checkpoint_dir / f"{prefix}_checkpoint.pt"
        iteration_checkpoint = self.cfg.checkpoint_dir / f"{prefix}_iter_{training_round:03d}.pt"
        torch.save(data, latest_checkpoint)
        torch.save(data, iteration_checkpoint)
        print(f"Saved checkpoints: {latest_checkpoint} and {iteration_checkpoint}")

    def run(self) -> None:
        self.setup()
        assert self.simulator is not None
        assert self.policy is not None
        assert self.device is not None
        assert self.seeded_config is not None

        if not self.cfg.start_with_aggregation:
            self._start_round()
            self.train_on_aggregate(
                "Initial offline training pass",
                0,
                self.cfg.target_epochs_per_round[0],
            )
            initial = self.evaluate_current_policy(f"Round 0 {self.cfg.policy_type} evaluation")
            initial_regions = self.evaluate_regions("Round 0")
            initial_coverage = self.measure_coverage()

            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_iter_000.pt"
            self.save_checkpoints(0)
            self.save_round_summary(
                round_index=0,
                train_loss=None,
                eval_metrics=initial,
                latest_checkpoint=str(latest_checkpoint),
                iteration_checkpoint=str(iteration_checkpoint),
                region_eval_metrics=initial_regions,
                timing=self._finish_round(),
                coverage=initial_coverage,
            )
            if self.cfg.dagger_iterations == 0:
                print("No DAgger refinements requested (--dagger-iterations 0).")
                return
            rounds = range(1, self.cfg.dagger_iterations + 1)
        else:
            if self.cfg.dagger_iterations == 0:
                raise ValueError(
                    "Fresh DAgger mode requires at least one aggregation round; "
                    "set --dagger-iterations to a positive value."
                )
            rounds = range(self.cfg.dagger_iterations)

        for index in rounds:
            self._start_round()
            if self.cfg.start_with_aggregation:
                schedule = index
                display = index + 1
                print(f"\n=== DAgger round {display}/{self.cfg.dagger_iterations}: aggregate ===")
            else:
                schedule = index - 1
                display = index
                print(f"\n=== DAgger refinement {display}/{self.cfg.dagger_iterations}: aggregate ===")

            mode = (
                self.cfg.training_curriculum[schedule]
                if self.cfg.training_curriculum is not None
                else ("config" if self.cfg.initial_states or self.cfg.goal_states else "random")
            )
            round_initial_states = self.cfg.initial_states if mode == "config" else None
            round_goal_states = self.cfg.goal_states if mode == "config" else None
            round_initial_state_seed = (
                self.cfg.round_seeds[schedule]
                if self.cfg.round_seeds is not None
                else self.initial_state_seed
            )
            collection_config = self._apply_runtime_config_overrides(
                copy.deepcopy(dict(self.seeded_config)), region=mode if mode in self.cfg.regions else None
            )
            collection_config = apply_config_overrides(
                collection_config, {"randomize_goal": True}
            )
            print(f"Aggregation goal source: {mode}")

            collection_started = time.perf_counter()
            simulator = SystemFactory.create(collection_config["system"], collection_config)
            planner = PlannerFactory.create(collection_config["expert"], simulator, collection_config)

            round_beta = self.cfg.expert_beta_per_iteration[schedule]
            print(f"Aggregation execution policy: expert_beta={round_beta:.3f}")

            if self.cfg.start_with_aggregation and not self.cfg.dataset_root.exists():
                features = build_dataset_features(simulator, self.prediction_horizon, self.cfg.flow_config.history_length)
                writer = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=int(1 / simulator.dt),
                    root=self.cfg.dataset_root,
                    features=features,
                )
            else:
                existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
                _validate_resumable_dataset_schema(
                    existing_meta.features,
                    build_dataset_features(simulator, self.prediction_horizon, self.cfg.flow_config.history_length),
                )
                writer = LeRobotDataset.resume(
                    repo_id=self.cfg.repo_id,
                    root=self.cfg.dataset_root,
                )

            try:
                def action_fn(obs: np.ndarray, history: np.ndarray) -> np.ndarray:
                    return build_policy_actions(simulator, self.rollout_policy, obs, history, ROLLOUT_DEVICE)

                self._sync_rollout_policy()
                metrics = collect_dagger_rollouts(
                    simulator=simulator,
                    expert_planner=planner,
                    dataset_writer=writer,
                    trajectories_per_iteration=self.cfg.trajectories_per_iteration[schedule],
                    steps_per_trajectory=self.cfg.steps_per_trajectory,
                    action_noise_std=self.cfg.action_noise_std,
                    action_noise_seed=self.action_noise_seed,
                    initial_state_seed=round_initial_state_seed,
                    initial_states=round_initial_states,
                    goal_states=round_goal_states,
                    expert_mixing_beta=round_beta,
                    prediction_horizon=self.prediction_horizon,
                    history_length=self.cfg.flow_config.history_length,
                    execution_horizon=self.cfg.execution_horizon,
                    round_index=schedule,
                    restart_initial_state_round=self.cfg.restart_round_seed,
                    policy_action_fn=action_fn,
                    policy_reset_fn=self.rollout_policy.reset,
                    num_workers=self.cfg.num_workers,
                    worker_spec=self._worker_spec(
                        collection_config,
                        need_expert=True,
                        need_policy=round_beta < 1.0,
                    ),
                )
                if metrics.num_episodes == 0:
                    raise RuntimeError(
                        "DAgger aggregation produced no valid episodes; skipping retraining."
                    )
            finally:
                writer.finalize()
                del writer
                gc.collect()
            self.timing.collection = time.perf_counter() - collection_started

            print_rollout_metrics(
                label=f"Round {display} aggregation"
                if self.cfg.start_with_aggregation
                else f"Refinement {display} aggregation",
                prefix="aggregation",
                metrics=metrics,
            )
            print(f"aggregation_goal_source: {mode}")

            train_loss = self.train_on_aggregate(
                f"DAgger round {display}/{self.cfg.dagger_iterations}: retrain"
                if self.cfg.start_with_aggregation
                else f"DAgger refinement {display}/{self.cfg.dagger_iterations}: retrain",
                training_round=index,
                target_epochs=self.cfg.target_epochs_per_round[schedule],
            )
            stage = "Round" if self.cfg.start_with_aggregation else "Refinement"
            eval_metrics = self.evaluate_current_policy(f"{stage} {display} {self.cfg.policy_type} evaluation")
            region_eval_metrics = self.evaluate_regions(f"{stage} {display}")
            coverage = self.measure_coverage()

            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / f"flow_dagger_iter_{index:03d}.pt"
            self.save_checkpoints(training_round=index)
            self.save_round_summary(
                round_index=index,
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                aggregation_metrics=metrics,
                latest_checkpoint=str(latest_checkpoint),
                iteration_checkpoint=str(iteration_checkpoint),
                region_eval_metrics=region_eval_metrics,
                timing=self._finish_round(),
                coverage=coverage,
            )

