"""How the Franka system draws an episode: one of the modes of its config, by weight, each a region for the start and one
for the goal.

A region is a box of TCP positions and a cone of hand directions (the tool z-axis within a tilt of an axis, at a random
roll). A pose of the region is accepted when inverse kinematics from one of a few random configurations reaches it with
a valid one, so the regions can lie anywhere the arm reaches, however far its configuration is from the ready pose, and
a pose is reached in different configurations from one draw to the next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

POSE_ATTEMPTS = 100  # poses drawn per start or goal
IK_STARTS = 6  # random configurations inverse kinematics starts from for a pose, which is dropped when none gives a valid one


@dataclass(frozen=True)
class PoseRegion:
    """A box of TCP positions with the hand along ``axis`` within ``tilt`` [rad]."""

    low: np.ndarray
    high: np.ndarray
    axis: np.ndarray
    tilt: float

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> PoseRegion:
        box = np.array([fields[axis] for axis in "xyz"], dtype=float)
        axis = np.asarray(fields["tool_axis"], dtype=float)
        return cls(box[:, 0], box[:, 1], axis / np.linalg.norm(axis), float(fields["tool_tilt"]))

    def pose(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """A position of the box and a rotation whose z-axis lies in the cone."""
        return rng.uniform(self.low, self.high), self._rotation(rng)

    def _rotation(self, rng: np.random.Generator) -> np.ndarray:
        helper = np.array([0.0, 0.0, 1.0]) if abs(self.axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        side = np.cross(self.axis, helper)
        side /= np.linalg.norm(side)
        tilt, heading = self.tilt * rng.uniform(), rng.uniform(0.0, 2 * np.pi)
        z = np.cos(tilt) * self.axis + np.sin(tilt) * (np.cos(heading) * side + np.sin(heading) * np.cross(self.axis, side))
        a = np.cross(z, helper)
        a /= np.linalg.norm(a)
        roll = rng.uniform(0.0, 2 * np.pi)
        x = np.cos(roll) * a + np.sin(roll) * np.cross(z, a)
        return np.column_stack([x, np.cross(z, x), z])


@dataclass(frozen=True)
class SamplingMode:
    """What an episode of the mode looks like: where it starts, where it ends, and how fast the arm starts, as a
    fraction of the joint velocity limits (0: at rest)."""

    weight: float
    start: PoseRegion
    goal: PoseRegion
    start_speed: float

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> SamplingMode:
        return cls(
            float(fields["weight"]),
            PoseRegion.from_fields(fields["start"]),
            PoseRegion.from_fields(fields["goal"]),
            float(fields["start_speed"]),
        )


class EpisodeSampler:
    """Draws the mode of an episode among the selected ones by weight, and the start configuration and goal pose in its
    regions. ``inverse_kinematics(position, rotation, q_init)`` returns the configuration of a pose that it finds from
    ``q_init`` (raising ``RuntimeError`` when it does not), which ``is_valid`` must accept; the starts are drawn
    uniformly between the joint limits ``lower`` and ``upper``."""

    def __init__(
        self,
        modes: Mapping[str, SamplingMode],
        selected: Sequence[str],
        inverse_kinematics: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        lower: np.ndarray,
        upper: np.ndarray,
        is_valid: Callable[[np.ndarray], bool],
    ) -> None:
        self.modes = [modes[name] for name in selected]
        self._probabilities = np.array([mode.weight for mode in self.modes]) / sum(mode.weight for mode in self.modes)
        self._inverse_kinematics = inverse_kinematics
        self._lower, self._upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        self._is_valid = is_valid

    def draw_mode(self, rng: np.random.Generator) -> SamplingMode:
        return self.modes[rng.choice(len(self.modes), p=self._probabilities)]

    def configuration(self, region: PoseRegion, rng: np.random.Generator) -> np.ndarray:
        """A valid configuration whose TCP pose is in the region."""
        return self._reach_in(region, rng)[1]

    def goal(self, region: PoseRegion, rng: np.random.Generator) -> np.ndarray:
        """A goal pose ``[position, rotation matrix row-major]`` of the region that a valid configuration reaches."""
        position, _, rotation = self._reach_in(region, rng)
        return np.concatenate([position, rotation.ravel()])

    def _reach_in(self, region: PoseRegion, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(POSE_ATTEMPTS):
            position, rotation = region.pose(rng)
            q = self._reach(position, rotation, rng)
            if q is not None:
                return position, q, rotation
        raise RuntimeError(f"Unable to sample a pose of the region in {POSE_ATTEMPTS} attempts.")

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
