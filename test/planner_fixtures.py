"""Settings and helpers the planner tests share.

Behaviour tests pin the settings they depend on, so tuning ``scripts/config/franka_casadi_config.yaml`` cannot break them.
"""

from __future__ import annotations

import os

import numpy as np
import yaml

from systems.franka_system import Q_READY, FrankaSystem
from planning.config import validate_system_config
from planning.ompl_planner import initialize_ompl

initialize_ompl(0)  # the entry point of the planner tests: OMPL's seed and log level are set once per process

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "scripts", "config", "franka_casadi_config.yaml")

PINNED_EXPERT_SETTINGS = {
    "Q_diag": [20.0] * 7 + [2.0] * 7 + [0.2] * 7,
    "R_diag": [4.4e-3] * 7,
    "terminal_cost_multiplier": 10.0,
    "expert": "ompl_casadi",
    "horizon": 20,
    "ompl_speed": 2.0,
    "ompl_replan_distance": 0.5,
    "ompl_timeout": 5.0,
    "ompl_seed": 0,
    "pos_tol": 0.005,
    "vel_tol": 0.05,
}


def pinned_config(**overrides) -> dict:
    """The expert config with the pinned settings, validated."""
    with open(EXPERT_CONFIG_PATH) as f:
        raw_config = yaml.safe_load(f)
    raw_config.update(PINNED_EXPERT_SETTINGS)
    raw_config.update(overrides)
    return validate_system_config(raw_config=raw_config)


def rest_state(simulator: FrankaSystem) -> np.ndarray:
    """The ready pose with the wrist turned by 0.3 rad, at rest."""
    q = Q_READY.copy()
    q[6] += 0.3
    return np.concatenate([q, np.zeros(2 * simulator.n)])


def pose_goal(simulator: FrankaSystem, q: np.ndarray) -> np.ndarray:
    """The goal pose of the TCP in the configuration ``q``."""
    return simulator.pose_vector(simulator.tcp_placement(q))


def initial_guess(planner, observation: np.ndarray, initial_plan=None) -> tuple[np.ndarray, np.ndarray]:
    """The (states (nx, N + 1), actions (nu, N)) a ``CasadiPlanner`` starts its first solve of ``plan`` from."""
    graph = planner._graph
    guessed = {}
    solve = graph.opti.solve

    def recording_solve():
        if not guessed:
            guessed["states"], guessed["actions"] = (np.asarray(graph.opti.debug.value(v, graph.opti.initial())) for v in (graph.X, graph.U))
        return solve()

    graph.opti.solve = recording_solve
    planner.plan(observation, initial_plan)
    return guessed["states"], guessed["actions"]
