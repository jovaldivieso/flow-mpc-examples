from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Mapping

from planning.config import load_yaml_config

DEFAULT_DAGGER_TRAINING_CONFIG: dict[str, object] = {
    "dagger_iterations": 4,
    "trajectories_per_iteration": [20],
    "steps_per_trajectory": 150,
    "action_noise_std": 0.0,
    "expert_mix_beta_start": 0.8,
    "expert_mix_beta_end": 0.0,
    "expert_mix_beta_decay_rate": None,
    "expert_mix_decay_after_eval_success": None,
    "adaptive_beta_recovery": False,
    "target_epochs_per_round": [30.0],
    "eval_episodes": 10,
    "eval_steps": None,
    "eval_seed_start": 10000,
    "eval_action_noise_std": 0.0,
    "batch_size": 1024,
    "learning_rate": 1e-4,
    "seed": 99,
    "max_train_steps": None,
    "training_curriculum": None,
    "round_seeds": None,
    "restart_round_seed": False,
    "initial_states": None,
    "goal_states": None,
    "workspace_bounds": None,
    "tolerance_overrides": None,
    "eval_tolerance_overrides": None,
}


DEFAULT_HIDDEN_DIMS = (256, 256, 256)
DEFAULT_PREDICTION_HORIZON = 16


@dataclass(frozen=True)
class FlowConfig:
    num_inference_steps: int = 10
    # Only used by the safeflow policy: where the flow integration starts ("previous_safe" or "gaussian")
    safeflow_init: str = "previous_safe"


@lru_cache(maxsize=None)
def _cached_yaml_config(policy_config_path: Path) -> dict[str, object]:
    """Avoid re-parsing the same policy YAML once per load_* call in a training run."""
    return load_yaml_config(policy_config_path)


def load_dagger_training_config(policy_config_path: Path | None) -> dict[str, object]:
    """Load optional DAgger CLI defaults from a policy configuration file."""
    config = dict(DEFAULT_DAGGER_TRAINING_CONFIG)
    if policy_config_path is None:
        return config

    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Policy config model section must be a mapping.")
    training_section = raw_config.get("training", model_section.get("training", {}))
    if not isinstance(training_section, Mapping):
        raise ValueError("Policy config 'training' section must be a mapping.")
    unknown_keys = sorted(set(training_section) - set(DEFAULT_DAGGER_TRAINING_CONFIG))
    if unknown_keys:
        raise ValueError(
            f"Unknown key(s) in policy config 'training' section: {unknown_keys}. "
            f"Supported keys are: {sorted(DEFAULT_DAGGER_TRAINING_CONFIG)}."
        )
    config.update(training_section)
    return config


def load_mlp_hidden_dims(policy_config_path: Path) -> tuple[int, ...]:
    """The ``model.hidden_dims`` widths of the flow network's MLP; the default if the config omits them."""
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    hidden_dims = model_section.get("hidden_dims", DEFAULT_HIDDEN_DIMS) if isinstance(model_section, Mapping) else DEFAULT_HIDDEN_DIMS
    if not isinstance(hidden_dims, (list, tuple)) or not hidden_dims or any(int(width) <= 0 for width in hidden_dims):
        raise ValueError("'model.hidden_dims' must be a non-empty list of positive integers.")
    return tuple(int(width) for width in hidden_dims)


def load_prediction_horizon(policy_config_path: Path | None) -> int:
    """The ``model.prediction_horizon``: how many actions the policy predicts per state (the length of the chunk DAgger
    labels with the expert's plan); ``DEFAULT_PREDICTION_HORIZON`` if the config omits it."""
    if policy_config_path is None:
        return DEFAULT_PREDICTION_HORIZON
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    horizon = model_section.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON) if isinstance(model_section, Mapping) else DEFAULT_PREDICTION_HORIZON
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("'model.prediction_horizon' must be a positive integer.")
    return horizon


def load_policy_type(policy_config_path: Path | None) -> str:
    if policy_config_path is None:
        return "flow"
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")
    policy_type_raw = model_section.get("policy_type", "flow")
    if (
        not isinstance(policy_type_raw, str)
        or policy_type_raw.strip().lower() not in {"flow", "safeflow"}
    ):
        raise ValueError(
            "'model.policy_type' must be one of {'flow', 'safeflow'}."
        )
    return policy_type_raw.strip().lower()


def load_flow_config(policy_config_path: Path | None) -> FlowConfig:
    if policy_config_path is None:
        return FlowConfig()

    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")

    flow_section = model_section.get("flow", {})
    if not isinstance(flow_section, Mapping):
        raise ValueError("Model config 'model.flow' must be a mapping.")

    safeflow_section = model_section.get("safeflow", {})
    if not isinstance(safeflow_section, Mapping):
        raise ValueError("Model config 'model.safeflow' must be a mapping.")
    safeflow_init = str(safeflow_section.get("init", FlowConfig.safeflow_init))
    if safeflow_init not in {"previous_safe", "gaussian"}:
        raise ValueError("'model.safeflow.init' must be one of {'previous_safe', 'gaussian'}.")

    return FlowConfig(
        num_inference_steps=int(flow_section.get("num_inference_steps", 10)),
        safeflow_init=safeflow_init,
    )


def default_checkpoint_dir() -> Path:
    return Path("outputs/train_dagger")


def default_repo_id(timestamp: int) -> str:
    return f"local/franka_dagger_{timestamp}"


def default_dataset_root(timestamp: int) -> Path:
    return Path(f"data/lerobot_dataset_franka_dagger_{timestamp}")
