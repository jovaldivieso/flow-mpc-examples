"""The readers of the fields of a config mapping, which validate what they read and raise ``ConfigurationError``."""

from __future__ import annotations

from typing import Any, Mapping

# The keys of the config of every system (a system adds its own)
SYSTEM_CONFIG_KEYS = {
    "goal",
    "randomize_goal",
    "max_acceleration",
    "max_jerk",
    "min_start_goal_distance",
    "pos_tol",
    "vel_tol",
    "obstacles",
}


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is missing or malformed."""


def _not_numeric(key: str, idx: int, value: Any) -> ConfigurationError:
    hint = " (YAML reads 1e-3 as a string, write 1.0e-3)" if isinstance(value, str) else ""
    return ConfigurationError(f"'{key}[{idx}]' must be numeric, got {type(value).__name__} {value!r}{hint}.")


def read_float(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be numeric, got {type(value).__name__}.")
    return float(value)


def read_int(config: Mapping[str, Any], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"'{key}' must be int, got {type(value).__name__}.")
    return int(value)


def read_bool(config: Mapping[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{key}' must be bool, got {type(value).__name__}.")
    return value


def read_vector(config: Mapping[str, Any], key: str, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = config.get(key, list(default))
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list of length {size}.")
    if len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")
    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _not_numeric(key, idx, value)
        out.append(float(value))
    return tuple(out)


def read_positive_vector(raw: Any, key: str, size: int | None = None) -> tuple[float, ...]:
    """A list of positive numbers, of length ``size`` if given (else any, but not empty)."""
    if not isinstance(raw, list):
        raise ConfigurationError(f"'{key}' must be a list" + ("" if size is None else f" of length {size}") + ".")
    if size is None and not raw:
        raise ConfigurationError(f"'{key}' must not be empty.")
    if size is not None and len(raw) != size:
        raise ConfigurationError(f"'{key}' must have length {size}, got {len(raw)}.")

    out: list[float] = []
    for idx, value in enumerate(raw):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise _not_numeric(key, idx, value)
        value_f = float(value)
        if value_f <= 0:
            raise ConfigurationError(f"'{key}[{idx}]' must be positive.")
        out.append(value_f)
    return tuple(out)


def require_only_known_keys(raw: Mapping[str, Any], allowed_keys: set[str], *, context: str) -> None:
    unknown_keys = sorted(key for key in raw.keys() if key not in allowed_keys)
    if unknown_keys:
        formatted = ", ".join(f"'{key}'" for key in unknown_keys)
        raise ConfigurationError(f"Unknown keys in {context}: {formatted}.")


def read_limits(raw_config: Mapping[str, Any], key: str, default: tuple[float, ...] | float, size: int) -> tuple[float, ...]:
    """A positive limit per joint (or axis), given as one number for all of them or a list."""
    raw = raw_config.get(key, list(default) if isinstance(default, tuple) else default)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [float(raw)] * size
    return read_positive_vector(raw, key, size)


def read_tolerances(raw_config: Mapping[str, Any], pos_tol: float, vel_tol: float) -> dict[str, float]:
    tolerances = {"pos_tol": read_float(raw_config, "pos_tol", pos_tol), "vel_tol": read_float(raw_config, "vel_tol", vel_tol)}
    for key, value in tolerances.items():
        if value <= 0:
            raise ConfigurationError(f"'{key}' must be positive.")
    return tolerances


def read_obstacles(raw_config: Mapping[str, Any], dimension: int) -> list[dict[str, Any]]:
    """Spherical obstacles: each entry has ``position`` (length ``dimension``) and a positive ``radius``."""
    raw = raw_config.get("obstacles", [])
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ConfigurationError(f"'obstacles' must be a list of mappings, got {type(raw).__name__}.")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ConfigurationError(f"'obstacles[{idx}]' must be a mapping, got {type(item).__name__}.")
        require_only_known_keys(item, {"position", "radius"}, context=f"'obstacles[{idx}]'")
        if "position" not in item or "radius" not in item:
            raise ConfigurationError(f"'obstacles[{idx}]' must define 'position' and 'radius'.")
        pos = read_vector(item, "position", dimension, ())
        radius = read_float(item, "radius", 0.0)
        if radius <= 0:
            raise ConfigurationError(f"'obstacles[{idx}].radius' must be positive.")
        out.append({"position": list(pos), "radius": radius})
    return out
