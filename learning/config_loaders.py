from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from planning.config import load_yaml_config

DEFAULT_DAGGER_TRAINING_CONFIG: dict[str, object] = {
    "dagger_iterations": 4,
    "trajectories_per_iteration": [20],
    "steps_per_trajectory": 150,
    "action_noise_std": 0.0,
    "expert_beta_per_iteration": [0.8],
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
    "tolerance_overrides": None,
    "eval_tolerance_overrides": None,
}

DEFAULT_PREDICTION_HORIZON = 16


@dataclass(frozen=True)
class UnetConfig:
    """The size of the flow network, SafeFlowMPC's temporal U-Net: the base channel count and the channel
    multipliers of its levels (every level but the first halves the horizon)."""

    dim: int = 32
    dim_mults: tuple[int, ...] = (1, 2, 4, 8)


@dataclass(frozen=True)
class FlowConfig:
    num_inference_steps: int = 10
    # Only used by the safeflow policy: where the flow integration starts ("previous_safe" or "gaussian")
    safeflow_init: str = "previous_safe"


@lru_cache(maxsize=None)
def _cached_yaml_config(policy_config_path: Path) -> dict[str, object]:
    """Avoid re-parsing the same policy YAML once per load_* call in a training run."""
    return load_yaml_config(policy_config_path)


def _mapping_section(parent: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    section = parent.get(key, {})
    if not isinstance(section, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return section


def _model_section(policy_config_path: Path) -> tuple[dict[str, object], Mapping[str, object]]:
    raw_config = _cached_yaml_config(policy_config_path)
    model_section = raw_config.get("model", raw_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("Model config 'model' section must be a mapping.")
    return raw_config, model_section


def load_dagger_training_config(policy_config_path: Path | None) -> dict[str, object]:
    """Load optional DAgger CLI defaults from a policy configuration file."""
    config = dict(DEFAULT_DAGGER_TRAINING_CONFIG)
    if policy_config_path is None:
        return config

    raw_config, model_section = _model_section(policy_config_path)
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


def load_unet_config(policy_config_path: Path) -> UnetConfig:
    """The ``model.unet`` section (``dim``, ``dim_mults``); what it omits keeps the SafeFlowMPC defaults."""
    _, model_section = _model_section(policy_config_path)
    unet_section = _mapping_section(model_section, "unet", "Model config 'model.unet'")
    dim_mults = unet_section.get("dim_mults", UnetConfig.dim_mults)
    if not isinstance(dim_mults, (list, tuple)) or not dim_mults:
        raise ValueError("'model.unet.dim_mults' must be a non-empty list of integers.")
    return UnetConfig(dim=int(unet_section.get("dim", UnetConfig.dim)), dim_mults=tuple(int(m) for m in dim_mults))


def load_prediction_horizon(policy_config_path: Path | None) -> int:
    """The ``model.prediction_horizon``: how many actions the policy predicts per state (the length of the chunk DAgger
    labels with the expert's plan); ``DEFAULT_PREDICTION_HORIZON`` if the config omits it."""
    if policy_config_path is None:
        return DEFAULT_PREDICTION_HORIZON
    _, model_section = _model_section(policy_config_path)
    horizon = model_section.get("prediction_horizon", DEFAULT_PREDICTION_HORIZON)
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("'model.prediction_horizon' must be a positive integer.")
    return horizon


def load_policy_type(policy_config_path: Path | None) -> str:
    if policy_config_path is None:
        return "flow"
    _, model_section = _model_section(policy_config_path)
    policy_type_raw = model_section.get("policy_type", "flow")
    if not isinstance(policy_type_raw, str) or policy_type_raw.strip().lower() not in {"flow", "safeflow"}:
        raise ValueError("'model.policy_type' must be one of {'flow', 'safeflow'}.")
    return policy_type_raw.strip().lower()


def load_flow_config(policy_config_path: Path | None) -> FlowConfig:
    if policy_config_path is None:
        return FlowConfig()

    _, model_section = _model_section(policy_config_path)
    flow_section = _mapping_section(model_section, "flow", "Model config 'model.flow'")
    safeflow_section = _mapping_section(model_section, "safeflow", "Model config 'model.safeflow'")
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
