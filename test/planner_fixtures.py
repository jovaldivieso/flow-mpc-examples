"""Settings and helpers the planner tests share.

Behaviour tests pin the settings they depend on, so tuning ``scripts/config/franka_casadi_config.yaml`` cannot break them.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from dynamics.franka_dynamics import Q_READY, FrankaDynamics
from planning.config import validate_system_config

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

PINNED_EXPERT_SETTINGS = {
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "floor_clearance": 0.05,
    "expert": "ompl_casadi",
    "horizon": 20,
    "ompl_speed": 2.0,
    "ompl_replan_distance": 0.5,
    "ompl_timeout": 5.0,
    "ompl_seed": 0,
    "ompl_ik_solutions": 12,
    "pos_tol": 0.005,
    "rot_tol": 0.05,
    "vel_tol": 0.05,
}


def pinned_config(**overrides) -> dict:
    """The expert config with the pinned settings, validated."""
    with open(EXPERT_CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


def rest_state(simulator: FrankaDynamics) -> np.ndarray:
    """The ready pose with the wrist turned by 0.3 rad, at rest."""
    q = Q_READY.copy()
    q[6] += 0.3
    return np.concatenate([q, np.zeros(2 * simulator.n)])
