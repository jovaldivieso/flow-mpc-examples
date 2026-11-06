"""How the Franka system draws a start and a goal: a TCP position in the workspace in front of the arm and a hand direction
(the tool z-axis) within ``HAND_TILT`` of ``HAND_AXIS``, at a random roll. A pose is accepted when inverse kinematics from
one of a few random configurations reaches it with a valid one, so a pose is reached in different configurations from one
draw to the next.

The workspace is also where the TCP stays: ``workspace_contains`` is the test the planners keep their paths by.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# A box of TCP positions in front of the base and the hand within HAND_TILT [rad] of HAND_AXIS, which spans pointing
# forward to pointing down
WORKSPACE_BOX = {"x": [0.2, 0.8], "y": [-0.6, 0.6], "z": [0.1, 0.8]}
HAND_AXIS = [1.0, 0.0, -1.0]
HAND_TILT = 0.9

POSE_ATTEMPTS = 100  # poses drawn per start or goal
IK_STARTS = 6  # random configurations inverse kinematics starts from for a pose, which is dropped when none gives a valid one

_LOW = np.array([WORKSPACE_BOX[axis][0] for axis in "xyz"])
_HIGH = np.array([WORKSPACE_BOX[axis][1] for axis in "xyz"])
_AXIS = np.array(HAND_AXIS) / np.linalg.norm(HAND_AXIS)


def workspace_contains(position: np.ndarray) -> bool:
    """Whether the TCP ``position`` lies in the workspace box, its faces included."""
    return bool(np.all(position >= _LOW) and np.all(position <= _HIGH))


def _rotation(rng: np.random.Generator) -> np.ndarray:
    """A rotation whose z-axis lies in the cone of the hand direction, at a random roll."""
    helper = np.array([0.0, 0.0, 1.0]) if abs(_AXIS[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    side = np.cross(_AXIS, helper)
    side /= np.linalg.norm(side)
    tilt, heading = HAND_TILT * rng.uniform(), rng.uniform(0.0, 2 * np.pi)
    z = np.cos(tilt) * _AXIS + np.sin(tilt) * (np.cos(heading) * side + np.sin(heading) * np.cross(_AXIS, side))
    a = np.cross(z, helper)
    a /= np.linalg.norm(a)
    roll = rng.uniform(0.0, 2 * np.pi)
    x = np.cos(roll) * a + np.sin(roll) * np.cross(z, a)
    return np.column_stack([x, np.cross(z, x), z])


class WorkspaceSampler:
    """Draws the start configuration and the goal pose in the workspace. ``inverse_kinematics(position, rotation, q_init)``
    returns the configuration of a pose that it finds from ``q_init`` (raising ``RuntimeError`` when it does not), which
    ``is_valid`` must accept; the starts of the solver are drawn uniformly between the joint limits ``lower`` and ``upper``."""

    def __init__(
        self,
        inverse_kinematics: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        is_valid: Callable[[np.ndarray], bool],
    ) -> None:
        self._inverse_kinematics = inverse_kinematics
        self._lower, self._upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self._is_valid = is_valid

    def configuration(self, rng: np.random.Generator) -> np.ndarray:
        """A valid configuration whose TCP pose is in the workspace."""
        return self._reach_in_workspace(rng)[1]

    def goal(self, rng: np.random.Generator) -> np.ndarray:
        """A goal pose ``[position, rotation matrix row-major]`` of the workspace that a valid configuration reaches."""
        position, _, rotation = self._reach_in_workspace(rng)
        return np.concatenate([position, rotation.ravel()])

    def _reach_in_workspace(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(POSE_ATTEMPTS):
            position, rotation = rng.uniform(_LOW, _HIGH), _rotation(rng)
            q = self._reach(position, rotation, rng)
            if q is not None:
                return position, q, rotation
        raise RuntimeError(f"Unable to sample a pose of the workspace in {POSE_ATTEMPTS} attempts.")

    def _reach(self, position: np.ndarray, rotation: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        """The first valid configuration of the pose that inverse kinematics finds from random starts, or None."""
        for _ in range(IK_STARTS):
            try:
                q = self._inverse_kinematics(position, rotation, rng.uniform(self._lower, self._upper))
            except RuntimeError:
                continue
            if self._is_valid(q):
                return q
        return None
