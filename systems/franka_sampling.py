"""How the Franka system draws an episode: one of the modes of its config, by weight, each a region for the start and one
for the goal.

A region is a box of TCP positions and a cone of hand directions (the tool z-axis within a tilt of an axis, at a random
roll). A pose of the region is accepted when inverse kinematics reaches it with a valid configuration, so the regions
can lie anywhere the arm reaches, however far its configuration is from the ready pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

POSE_ATTEMPTS = 100  # poses drawn per start or goal (each is a multi-start inverse kinematics)


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
    regions. ``reach`` returns a valid configuration for a pose (raising ``RuntimeError`` when there is none)."""

    def __init__(
        self,
        modes: Mapping[str, SamplingMode],
        selected: Sequence[str],
        reach: Callable[[np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        self.modes = [modes[name] for name in selected]
        self._probabilities = np.array([mode.weight for mode in self.modes]) / sum(mode.weight for mode in self.modes)
        self._reach = reach

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
            try:
                return position, self._reach(position, rotation), rotation
            except RuntimeError:
                continue
        raise RuntimeError(f"Unable to sample a pose of the region in {POSE_ATTEMPTS} attempts.")
