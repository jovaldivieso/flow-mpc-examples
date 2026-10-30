from __future__ import annotations

import copy
import gc
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from planning.config import validate_system_config
from systems.system import SystemSimulator
from systems.system_factory import SystemFactory
from systems.seed_utils import default_action_noise_seed_for_config
from learning.data_utils import build_dataset_features, create_collate_fn
from learning.models.exponential_moving_average import ExponentialMovingAverage
from learning.models.policy import ActionPolicy
from learning.models.policy_factory import PolicyFactory, PolicySpec
from planning.planner_factory import PlannerFactory

from .dagger_config import DaggerConfig
from .metrics import DaggerEvalMetrics
from .collection import collect_dagger_rollouts
from .evaluation import evaluate_policy_rollouts
from .policy_action import build_policy_action
from .round_summary import append_round_summary
from .workers import RolloutWorkerSpec
from .utils import (
    apply_config_overrides,
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
        self.state_dim = self.action_dim = self.prediction_horizon = 0

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
        # The policy predicts prediction_horizon planned actions per state, which DAgger labels with the expert's plan
        self.prediction_horizon = self.cfg.prediction_horizon
        expert_horizon = int(self.seeded_config["horizon"])
        if self.prediction_horizon > expert_horizon:
            raise ValueError(
                f"The policy's 'prediction_horizon' ({self.prediction_horizon}) must not exceed the expert's 'horizon' "
                f"({expert_horizon}): the expert's plan labels the policy's targets."
            )
        features = build_dataset_features(self.simulator, self.prediction_horizon)
        if self.cfg.dataset_root.exists():
            existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
            _validate_resumable_dataset_schema(existing_meta.features, features)
        self.obs_feature_names = [n for n in features if n.startswith("observation.")]
        # observation.environment_state (the goal) and observation.state (proprioception) are both single-frame.
        environment_state_dim = int(features["observation.environment_state"]["shape"][0])
        proprioception_dim = int(features["observation.state"]["shape"][0])
        self.action_dim = self.simulator.nu  # the policy's chunk has prediction_horizon rows of action_dim jerks

        self.state_dim = environment_state_dim + proprioception_dim

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
        """Start from the weights, EMA and optimizer state of a checkpoint of the same architecture."""
        assert self.policy is not None and self.optimizer is not None and self.ema is not None
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        expected = {
            "policy_type": self.cfg.policy_type,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "prediction_horizon": self.prediction_horizon,
            "unet": {"dim": self.cfg.unet.dim, "dim_mults": list(self.cfg.unet.dim_mults)},
        }
        for key, value in expected.items():
            if checkpoint[key] != value:
                raise ValueError(f"Cannot start from {path}: its '{key}' is {checkpoint[key]}, this run's is {value}.")
        for name in ("model_state_dict", "ema_model_state_dict"):
            if not all(torch.isfinite(t).all() for t in checkpoint[name].values() if t.is_floating_point()):
                raise ValueError(f"Cannot start from {path}: its '{name}' has non-finite weights.")
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.ema.load_state_dict(checkpoint["ema_model_state_dict"])
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
            observations, actions = batch
            if isinstance(observations, dict):
                observations = {
                    key: value.to(self.device)
                    for key, value in observations.items()
                }
            else:
                observations = observations.to(self.device)
            actions = actions.to(self.device)
            self.optimizer.zero_grad()
            loss = self.policy.compute_loss(observations, actions)
            loss.backward()
            # Generic safety net, not tuned tight: catches a sudden pathological
            # spike without perturbing the normal gradient norms of the
            # already-stable policy types.
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.0)
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
        dataset = LeRobotDataset(
            repo_id=self.cfg.repo_id,
            root=self.cfg.dataset_root,
        )
        generator = torch.Generator().manual_seed(self.cfg.seed + training_round)
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=create_collate_fn(self.prediction_horizon),
        )
        steps, approx = resolve_round_steps(
            len(dataset),
            self.cfg.batch_size,
            target_epochs,
            self.cfg.max_train_steps,
        )
        print(f"Training on {len(dataset)} aggregated frames")
        print(
            f"  optimizer_steps={steps} "
            f"(~{approx:.2f} epochs at batch_size={self.cfg.batch_size})"
        )
        mean_loss = self.train_policy_steps(loader, steps)
        print(f"  mean_step_loss={mean_loss:.6f}")
        return mean_loss

    def _policy_spec(self, policy_type: str) -> PolicySpec:
        assert self.simulator is not None
        return PolicySpec(
            policy_type=policy_type,
            action_dim=self.action_dim,
            state_dim=self.state_dim,
            prediction_horizon=self.prediction_horizon,
            unet_dim=self.cfg.unet.dim,
            unet_dim_mults=tuple(self.cfg.unet.dim_mults),
            num_inference_steps=self.cfg.flow_config.num_inference_steps,
            action_scale=tuple(np.asarray(self.simulator.max_action, dtype=float).tolist()),
            flow_init=self.cfg.flow_config.safeflow_init,
        )

    def _build_policy(
        self, policy_type: str, device: torch.device, simulator: SystemSimulator, planner_config: dict[str, Any]
    ) -> ActionPolicy:
        """A policy of ``policy_type`` for ``simulator`` (whose projector is built from ``planner_config``)."""
        return PolicyFactory.create(self._policy_spec(policy_type), simulator, planner_config).to(device).eval()

    def _apply_runtime_config_overrides(
        self, config: dict[str, Any], *, tolerance_overrides: Mapping[str, float] | None = None
    ) -> dict[str, Any]:
        """Inject the training config's solver/dynamics tuning knobs (tolerances), if set.

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
        return validate_system_config(merged_config)

    def _sync_rollout_policy(self) -> None:
        """Give the CPU rollout policy the averaged weights of the policy being trained."""
        assert self.rollout_policy is not None and self.ema is not None
        self.rollout_policy.load_state_dict({name: tensor.cpu() for name, tensor in self.ema.state_dict().items()})

    def _worker_spec(
        self, config: dict[str, Any], *, need_expert: bool, need_policy: bool, policy_type: str | None = None
    ) -> RolloutWorkerSpec:
        assert self.simulator is not None and self.rollout_policy is not None
        policy_spec = None
        policy_state_dict = None
        policy_type = policy_type or self.cfg.policy_type
        if need_policy:
            policy_spec = self._policy_spec(policy_type)
            policy_state_dict = {name: tensor.cpu() for name, tensor in self.rollout_policy.state_dict().items()}
        return RolloutWorkerSpec(
            config=dict(config),
            policy_spec=policy_spec,
            policy_state_dict=policy_state_dict,
            need_expert=need_expert,
        )

    def evaluate_current_policy(self, label: str, policy_type: str | None = None) -> DaggerEvalMetrics | None:
        """Evaluate the weights being trained as a policy of ``policy_type`` (the trained one by default): the weights
        of a flow policy also make a safe flow policy, which runs the same episodes with its projector."""
        assert self.rollout_policy is not None
        assert self.seeded_config is not None
        if self.cfg.eval_episodes == 0:
            return None
        policy_type = policy_type or self.cfg.policy_type
        eval_config = self._apply_runtime_config_overrides(
            copy.deepcopy(dict(self.seeded_config)),
            tolerance_overrides=self.cfg.eval_tolerance_overrides,
        )
        simulator = SystemFactory.create(eval_config["system"], eval_config)

        self._sync_rollout_policy()
        policy = self.rollout_policy
        if policy_type != self.cfg.policy_type:
            policy = self._build_policy(policy_type, ROLLOUT_DEVICE, simulator, eval_config)
            policy.load_network_state_dict(self.rollout_policy.state_dict())

        def action_fn(obs: np.ndarray) -> np.ndarray:
            return build_policy_action(simulator, policy, obs, ROLLOUT_DEVICE)

        metrics = evaluate_policy_rollouts(
            simulator,
            self.cfg.eval_episodes,
            self.cfg.eval_steps or self.cfg.steps_per_trajectory,
            self.cfg.eval_seed_start,
            action_fn,
            reset_fn=policy.reset,
            action_noise_std=self.cfg.eval_action_noise_std,
            action_noise_seed=self.action_noise_seed,
            initial_states=self.cfg.initial_states,
            goal_states=self.cfg.goal_states,
            num_workers=self.cfg.num_workers,
            worker_spec=self._worker_spec(eval_config, need_expert=False, need_policy=True, policy_type=policy_type),
        )
        if metrics is not None:
            print_rollout_metrics(label, "eval", metrics)
        return metrics

    def save_round_summary(
        self,
        round_index: int,
        train_loss: float | None,
        eval_metrics: DaggerEvalMetrics | None,
        aggregation_metrics: DaggerEvalMetrics | None = None,
        safeflow_eval_metrics: DaggerEvalMetrics | None = None,
        latest_checkpoint: str | None = None,
        iteration_checkpoint: str | None = None,
    ) -> None:
        """Persist a round-by-round YAML summary with success and failure stats."""

        append_round_summary(
            self.cfg.checkpoint_dir / "training_summary.yaml",
            round_index,
            train_loss,
            eval_metrics,
            aggregation_metrics,
            safeflow_eval_metrics,
            latest_checkpoint,
            iteration_checkpoint,
        )

    def save_checkpoints(self, training_round: int) -> None:
        assert self.policy is not None
        assert self.optimizer is not None
        data: dict[str, Any] = {
            "iteration": training_round,
            "model_state_dict": self.policy.state_dict(),
            # What inference runs with (evaluate_policy.py prefers it over model_state_dict)
            "ema_model_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "prediction_horizon": self.prediction_horizon,
            "unet": {"dim": self.cfg.unet.dim, "dim_mults": list(self.cfg.unet.dim_mults)},
            "obs_feature_names": self.obs_feature_names,
            "policy_type": self.cfg.policy_type,
        }
        data["flow_config"] = {
            "num_inference_steps": self.cfg.flow_config.num_inference_steps,
            "safeflow_init": self.cfg.flow_config.safeflow_init,
            # Saved explicitly so a checkpoint's normalized action space stays what trained it -- see FlowPolicy's own
            # docstring for why training and inference must agree on this exactly.
            "action_scale": np.asarray(self.simulator.max_action, dtype=float).tolist(),
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
            self.train_on_aggregate(
                "Initial offline training pass",
                0,
                self.cfg.target_epochs_per_round[0],
            )
            initial = self.evaluate_current_policy(f"Round 0 {self.cfg.policy_type} evaluation")
            initial_safeflow = (
                self.evaluate_current_policy("Round 0 safeflow evaluation", "safeflow") if self.cfg.policy_type == "flow" else None
            )

            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_iter_000.pt"
            self.save_checkpoints(0)
            self.save_round_summary(
                round_index=0,
                train_loss=None,
                eval_metrics=initial,
                safeflow_eval_metrics=initial_safeflow,
                latest_checkpoint=str(latest_checkpoint),
                iteration_checkpoint=str(iteration_checkpoint),
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
                copy.deepcopy(dict(self.seeded_config))
            )
            collection_config = apply_config_overrides(
                collection_config, {"randomize_goal": True}
            )
            print(f"Aggregation goal source: {mode}")

            simulator = SystemFactory.create(collection_config["system"], collection_config)
            planner = PlannerFactory.create(collection_config["expert"], simulator, collection_config)

            round_beta = self.cfg.expert_beta_per_iteration[schedule]
            print(f"Aggregation execution policy: expert_beta={round_beta:.3f}")

            if self.cfg.start_with_aggregation and not self.cfg.dataset_root.exists():
                features = build_dataset_features(simulator, self.prediction_horizon)
                writer = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=int(1 / simulator.dt),
                    root=self.cfg.dataset_root,
                    features=features,
                )
            else:
                existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
                _validate_resumable_dataset_schema(existing_meta.features, build_dataset_features(simulator, self.prediction_horizon))
                writer = LeRobotDataset.resume(
                    repo_id=self.cfg.repo_id,
                    root=self.cfg.dataset_root,
                )

            try:
                def action_fn(obs: np.ndarray) -> np.ndarray:
                    return build_policy_action(simulator, self.rollout_policy, obs, ROLLOUT_DEVICE)

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

            safeflow_eval_metrics = (
                self.evaluate_current_policy(f"{stage} {display} safeflow evaluation", "safeflow")
                if self.cfg.policy_type == "flow"
                else None
            )

            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / f"flow_dagger_iter_{index:03d}.pt"
            self.save_checkpoints(training_round=index)
            self.save_round_summary(
                round_index=index,
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                aggregation_metrics=metrics,
                safeflow_eval_metrics=safeflow_eval_metrics,
                latest_checkpoint=str(latest_checkpoint),
                iteration_checkpoint=str(iteration_checkpoint),
            )

