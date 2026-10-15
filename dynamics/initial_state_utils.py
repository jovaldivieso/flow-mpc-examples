from __future__ import annotations

import ast
from numbers import Real
from typing import Any

import numpy as np

from dynamics.dynamics import DynamicsProtocol


def _parse_state_spec_argument(raw: str | None, flag_name: str, examples: str) -> Any | None:
    if raw is None:
        return None

    text = raw.strip()
    if text == "":
        return None

    try:
        return ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse {flag_name}. Use Python-literal syntax like {examples}"
        ) from exc


def parse_initial_states_argument(raw_initial_states: str | None) -> Any | None:
    return _parse_state_spec_argument(
        raw_initial_states,
        "--initial-states",
        "'[q1, ..., q7, v1, ..., v7]' (single rollout) or "
        "'[[q1, ..., v7], [q1, ..., v7]]' (multiple rollouts).",
    )


def parse_goal_states_argument(raw_goal_states: str | None) -> Any | None:
    return _parse_state_spec_argument(
        raw_goal_states,
        "--goal-states",
        "'[x, y, z, R00, R01, ..., R22]' (single rollout, TCP pose) or "
        "'[[x, y, z, R00, ..., R22], [x, y, z, R00, ..., R22]]' (multiple rollouts).",
    )


def _is_numeric_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(
        isinstance(x, Real) and not isinstance(x, bool)
        for x in value
    )


def _as_state_vector(values: list[Any], expected_dim: int, context: str) -> np.ndarray:
    if len(values) != expected_dim:
        raise ValueError(
            f"{context} must have length {expected_dim}, got {len(values)}."
        )
    return np.asarray(values, dtype=float)


def _normalize_state_specs(
    specs: Any | None,
    total_dim: int,
    label: str,
) -> list[np.ndarray]:
    if specs is None:
        return []

    if not isinstance(specs, (list, tuple)):
        raise ValueError(f"{label} specs must evaluate to a list-like object.")

    values = list(specs)
    if len(values) == 0:
        return []

    # Case 1: one rollout vector, e.g. [q1, ..., q7, v1, ..., v7]
    if all(isinstance(x, Real) and not isinstance(x, bool) for x in values):
        return [_as_state_vector(values, expected_dim=total_dim, context=f"{label.capitalize()}")]

    if not all(isinstance(item, (list, tuple)) for item in values):
        raise ValueError(
            f"{label} specs must be a numeric vector or a list of numeric vectors."
        )

    row_values = [list(item) for item in values]

    # Case 2: list of rollout vectors,
    # e.g. [[...], [...]]
    if all(_is_numeric_sequence(row) for row in row_values):
        return [
            _as_state_vector(row, expected_dim=total_dim, context=f"{label.capitalize()} #{idx}")
            for idx, row in enumerate(row_values)
        ]

    raise ValueError(f"{label} specs must be a numeric vector or a list of numeric vectors.")


def normalize_initial_state_specs(
    simulator: DynamicsProtocol,
    initial_states: Any | None,
) -> list[np.ndarray]:
    return _normalize_state_specs(
        initial_states,
        total_dim=int(simulator.nx),
        label="initial state",
    )


def normalize_goal_state_specs(
    simulator: DynamicsProtocol,
    goal_states: Any | None,
) -> list[np.ndarray]:
    return _normalize_state_specs(
        goal_states,
        total_dim=int(simulator.goal_dim),
        label="goal state",
    )
