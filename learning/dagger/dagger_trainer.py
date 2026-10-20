from __future__ import annotations

import copy
import gc
from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from planning.config import validate_system_config
from dynamics.dynamics import DynamicsSimulator
from dynamics.dynamics_factory import DynamicsFactory
from dynamics.seed_utils import default_action_noise_seed_for_config
from learning.data_utils import create_collate_fn
from learning.models.ema import ExponentialMovingAverage
from learning.models.policy import ActionPolicy, PolicyFactory
from planning.planner_factory import PlannerFactory

from .dagger_config import DaggerConfig
from .metrics import DaggerEvalMetrics
from .rollouts import build_policy_action, collect_dagger_rollouts, evaluate_policy_rollouts
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
        self.simulator: DynamicsSimulator | None = None
        self.seeded_config: Mapping[str, Any] | None = None
        self.policy: ActionPolicy | None = None
        self.rollout_policy: ActionPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.ema: ExponentialMovingAverage | None = None
        self.action_noise_seed = 0
        self.initial_state_seed = 0
        self.obs_feature_names: list[str] = []
        self.state_dim = self.action_dim = self.prediction_horizon = 0

    @staticmethod
    def schedules(
        trajectories: list[int],
        epochs: list[float],
        betas: list[float],
        rounds: int,
        training_curriculum: list[str] | None = None,
        round_seeds: list[int] | None = None,
    ) -> tuple[list[int], list[float], list[float], list[str] | None, list[int] | None]:
        def expand(values: list[Any], name: str) -> list[Any]:
            if len(values) == 1:
                return values if rounds == 0 else values * rounds
            if len(values) != rounds:
                raise ValueError(
                    f"{name} must contain one or exactly {rounds} values."
                )
            return values
        return (
            expand(trajectories, "trajectories-per-iteration"),
            expand(epochs, "target-epochs-per-round"),
            expand(betas, "expert-beta-per-iteration"),
            expand(training_curriculum, "training-curriculum") if training_curriculum is not None else None,
            expand(round_seeds, "round-seeds") if round_seeds is not None else None,
        )

    def setup(self) -> None:
        set_seed(self.cfg.seed)
        self.seeded_config = with_seeded_initial_state_config(
            self.cfg.experiment_config,
            self.cfg.seed,
        )
        self.simulator = DynamicsFactory.create(self.seeded_config["system"], self.seeded_config)
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
        features = self.simulator.get_dataset_features(self.prediction_horizon)
        if self.cfg.dataset_root.exists():
            existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
            _validate_resumable_dataset_schema(existing_meta.features, features)
        self.obs_feature_names = [n for n in features if n.startswith("observation.")]
        # observation.environment_state (the goal) and observation.state (proprioception) are both single-frame.
        environment_state_dim = int(features["observation.environment_state"]["shape"][0])
        proprioception_dim = int(features["observation.state"]["shape"][0])
        self.action_dim = self.simulator.nu  # the policy's chunk has prediction_horizon rows of action_dim jerks

        self.state_dim = environment_state_dim + proprioception_dim

        flow = {
            "num_inference_steps": self.cfg.flow_config.num_inference_steps,
            # The jerks are scaled by their limits, to the order of the unit-scale noise prior
            "action_scale": np.asarray(self.simulator.max_action, dtype=float).tolist(),
        }
        safeflow = (
            {
                "simulator": self.simulator,
                "planner_config": self.seeded_config,
                "flow_init": self.cfg.flow_config.safeflow_init,
            }
            if self.cfg.policy_type == "safeflow"
            else {}
        )
        def build_policy(device: torch.device) -> ActionPolicy:
            return PolicyFactory.create(
                self.cfg.policy_type,
                action_dim=self.action_dim,
                state_dim=self.state_dim,
                unet_dim=self.cfg.unet.dim,
                unet_dim_mults=self.cfg.unet.dim_mults,
                prediction_horizon=self.prediction_horizon,
                **flow,
                **safeflow,
            ).to(device).eval()

        self.policy = build_policy(self.device)
        # Single-sample inference is faster on the CPU than on an accelerator, so rollouts use a CPU copy
        self.rollout_policy = build_policy(ROLLOUT_DEVICE)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=self.cfg.learning_rate, weight_decay=WEIGHT_DECAY
        )
        # The rollouts and checkpoints for inference use the averaged weights
        self.ema = ExponentialMovingAverage(self.policy)

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

    def evaluate_current_policy(self, label: str) -> DaggerEvalMetrics | None:
        assert self.rollout_policy is not None
        assert self.seeded_config is not None
        if self.cfg.eval_episodes == 0:
            return None
        eval_config = self._apply_runtime_config_overrides(
            copy.deepcopy(dict(self.seeded_config)),
            tolerance_overrides=self.cfg.eval_tolerance_overrides,
        )
        simulator = DynamicsFactory.create(eval_config["system"], eval_config)

        def action_fn(obs: np.ndarray) -> np.ndarray:
            return build_policy_action(simulator, self.rollout_policy, obs, ROLLOUT_DEVICE)

        self._sync_rollout_policy()
        metrics = evaluate_policy_rollouts(
            simulator,
            self.cfg.eval_episodes,
            self.cfg.eval_steps or self.cfg.steps_per_trajectory,
            self.cfg.eval_seed_start,
            action_fn,
            reset_fn=self.rollout_policy.reset,
            action_noise_std=self.cfg.eval_action_noise_std,
            action_noise_seed=self.action_noise_seed,
            initial_states=self.cfg.initial_states,
            goal_states=self.cfg.goal_states,
        )
        if metrics is not None:
            print_rollout_metrics(label, "eval", metrics)
        return metrics

    def persist_round_summary(
        self,
        round_index: int,
        train_loss: float | None,
        eval_metrics: DaggerEvalMetrics | None,
        aggregation_metrics: DaggerEvalMetrics | None = None,
        latest_checkpoint: str | None = None,
        iteration_checkpoint: str | None = None,
    ) -> None:
        self.save_round_summary(
            round_index=round_index,
            train_loss=train_loss,
            eval_metrics=eval_metrics,
            aggregation_metrics=aggregation_metrics,
            latest_checkpoint=latest_checkpoint,
            iteration_checkpoint=iteration_checkpoint,
        )

    def save_round_summary(
        self,
        round_index: int,
        train_loss: float | None,
        eval_metrics: DaggerEvalMetrics | None,
        aggregation_metrics: DaggerEvalMetrics | None = None,
        latest_checkpoint: str | None = None,
        iteration_checkpoint: str | None = None,
    ) -> None:
        """Persist a round-by-round YAML summary with success and failure stats."""

        summary_path = self.cfg.checkpoint_dir / "training_summary.yaml"
        summary: dict[str, Any]
        if summary_path.exists():
            with summary_path.open("r") as f:
                summary = yaml.safe_load(f) or {}
        else:
            summary = {}

        rounds = summary.get("rounds", [])
        entry: dict[str, Any] = {
            "round": int(round_index),
            "train_loss": train_loss,
            "aggregation_success_rate": (
                aggregation_metrics.success_rate
                if aggregation_metrics is not None
                else None
            ),
            "eval_success_rate": (
                eval_metrics.success_rate
                if eval_metrics is not None
                else None
            ),
            "eval_mean_steps": (
                eval_metrics.mean_steps
                if eval_metrics is not None
                else None
            ),
            "eval_min_steps": (
                eval_metrics.min_steps
                if eval_metrics is not None
                else None
            ),
            "eval_max_steps": (
                eval_metrics.max_steps
                if eval_metrics is not None
                else None
            ),
            "eval_num_episodes": (
                eval_metrics.num_episodes
                if eval_metrics is not None
                else None
            ),
            "eval_success_by_source": {
                "config": {
                    "successes": (
                        eval_metrics.config_successes
                        if eval_metrics is not None
                        else 0
                    ),
                    "total": (
                        eval_metrics.config_num_episodes
                        if eval_metrics is not None
                        else 0
                    ),
                    "rate": (
                        100.0 * eval_metrics.config_successes / eval_metrics.config_num_episodes
                        if eval_metrics is not None and eval_metrics.config_num_episodes > 0
                        else None
                    ),
                },
                "random": {
                    "successes": (
                        eval_metrics.random_successes
                        if eval_metrics is not None
                        else 0
                    ),
                    "total": (
                        eval_metrics.random_num_episodes
                        if eval_metrics is not None
                        else 0
                    ),
                    "rate": (
                        100.0 * eval_metrics.random_successes / eval_metrics.random_num_episodes
                        if eval_metrics is not None and eval_metrics.random_num_episodes > 0
                        else None
                    ),
                },
            },
            "eval_failure_breakdown": {
                "timeout": (
                    eval_metrics.timeout_failures
                    if eval_metrics is not None
                    else 0
                ),
                "solve_failure": (
                    eval_metrics.solve_failures
                    if eval_metrics is not None
                    else 0
                ),
            },
        }
        if aggregation_metrics is not None:
            entry["aggregation_failures"] = {
                "attempted_episodes": aggregation_metrics.attempted_episodes,
                "discarded_episodes": aggregation_metrics.discarded_episodes,
                "planner_failures": aggregation_metrics.planner_failures,
                "policy_solve_fallbacks": aggregation_metrics.policy_solve_fallbacks,
            }
            if aggregation_metrics.expert_statistics:
                entry["expert_statistics"] = dict(aggregation_metrics.expert_statistics)
        if latest_checkpoint is not None or iteration_checkpoint is not None:
            entry["checkpoints"] = {
                "latest": str(latest_checkpoint) if latest_checkpoint is not None else None,
                "iteration": str(iteration_checkpoint) if iteration_checkpoint is not None else None,
            }
        rounds.append(entry)
        summary["rounds"] = rounds
        valid_rates = [
            round_entry["eval_success_rate"]
            for round_entry in rounds
            if round_entry["eval_success_rate"] is not None
        ]
        if valid_rates:
            best_round = max(
                rounds,
                key=lambda round_entry: round_entry["eval_success_rate"]
                if round_entry["eval_success_rate"] is not None
                else float("-inf"),
            )
            summary["best_round"] = best_round["round"]
            summary["best_success_rate"] = best_round["eval_success_rate"]
        else:
            summary["best_round"] = None
            summary["best_success_rate"] = None
        summary["latest_round"] = int(round_index)
        collection_rounds = [r["aggregation_failures"] for r in rounds if "aggregation_failures" in r]
        summary["total_aggregation_failures"] = {
            key: sum(r[key] for r in collection_rounds)
            for key in ("attempted_episodes", "discarded_episodes", "planner_failures", "policy_solve_fallbacks")
        }
        total_expert_statistics = Counter()
        for round_entry in rounds:
            total_expert_statistics.update(round_entry.get("expert_statistics", {}))
        if total_expert_statistics:
            summary["total_expert_statistics"] = dict(total_expert_statistics)
        with summary_path.open("w") as f:
            yaml.safe_dump(summary, f, sort_keys=False)

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
            initial = self.evaluate_current_policy("Round 0 evaluation")
            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_iter_000.pt"
            self.save_checkpoints(0)
            self.persist_round_summary(
                round_index=0,
                train_loss=None,
                eval_metrics=initial,
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

            simulator = DynamicsFactory.create(collection_config["system"], collection_config)
            planner = PlannerFactory.create(collection_config["expert"], simulator, collection_config)

            round_beta = self.cfg.expert_beta_per_iteration[schedule]
            print(f"Aggregation execution policy: expert_beta={round_beta:.3f}")

            if self.cfg.start_with_aggregation and not self.cfg.dataset_root.exists():
                features = simulator.get_dataset_features(self.prediction_horizon)
                writer = LeRobotDataset.create(
                    repo_id=self.cfg.repo_id,
                    fps=int(1 / simulator.dt),
                    root=self.cfg.dataset_root,
                    features=features,
                )
            else:
                existing_meta = LeRobotDatasetMetadata(repo_id=self.cfg.repo_id, root=self.cfg.dataset_root)
                _validate_resumable_dataset_schema(existing_meta.features, simulator.get_dataset_features(self.prediction_horizon))
                writer = LeRobotDataset.resume(
                    repo_id=self.cfg.repo_id,
                    root=self.cfg.dataset_root,
                )

            try:
                def action_fn(obs: np.ndarray) -> np.ndarray:
                    return build_policy_action(simulator, self.rollout_policy, obs, ROLLOUT_DEVICE)

                frames = simulator.format_dataset_frame
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
                    frame_builder=frames,
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
            eval_metrics = self.evaluate_current_policy(
                f"Round {display} evaluation"
                if self.cfg.start_with_aggregation
                else f"Refinement {display} evaluation"
            )

            latest_checkpoint = self.cfg.checkpoint_dir / "flow_dagger_checkpoint.pt"
            iteration_checkpoint = self.cfg.checkpoint_dir / f"flow_dagger_iter_{index:03d}.pt"
            self.save_checkpoints(training_round=index)
            self.persist_round_summary(
                round_index=index,
                train_loss=train_loss,
                eval_metrics=eval_metrics,
                aggregation_metrics=metrics,
                latest_checkpoint=str(latest_checkpoint),
                iteration_checkpoint=str(iteration_checkpoint),
            )

