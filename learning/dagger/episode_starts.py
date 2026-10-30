from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systems.initial_state_utils import normalize_goal_state_specs, normalize_initial_state_specs
from systems.system import SystemSimulator

from .utils import rng_for_seed_spec, sample_initial_state


def _setup_episode_start(
    simulator: SystemSimulator,
    index: int,
    usable_count: int,
    provided_initial_states: list[np.ndarray],
    provided_goal_states: list[np.ndarray],
    baseline_goal: np.ndarray,
    seed_spec: int,
) -> np.ndarray:
    """Set the goal of episode ``index`` (counted from 0) and return its initial state.

    The first ``usable_count`` episodes take their initial state from ``provided_initial_states`` and their goal from
    ``provided_goal_states``; with no goals provided the goal is drawn as ``randomize_goal_for_reset`` does, seeded with
    ``seed_spec``, so these episodes still vary per the config instead of freezing at the baseline goal. The others
    restore ``baseline_goal`` first (an earlier episode's explicit goal mutated the simulator, and
    ``randomize_goal_for_reset`` is a no-op under ``randomize_goal: false``, which would leak that goal into this
    episode) and sample a start (and goal) from ``seed_spec``.
    """
    if index < usable_count:
        if provided_goal_states:
            simulator.set_goal(provided_goal_states[index])
        else:
            simulator.randomize_goal_for_reset(rng_for_seed_spec(seed_spec))
        return provided_initial_states[index]
    simulator.set_goal(baseline_goal)
    return sample_initial_state(simulator, seed_spec)


@dataclass(frozen=True)
class EpisodeStarts:
    """Where the episodes of a collection or an evaluation start: the first ones from the initial states (and goals)
    that were provided, the others sampled (``_setup_episode_start``)."""

    initial_states: list[np.ndarray]
    goal_states: list[np.ndarray]
    baseline_goal: np.ndarray

    @classmethod
    def from_specs(
        cls,
        simulator: SystemSimulator,
        initial_states: list[np.ndarray] | None,
        goal_states: list[np.ndarray] | None,
    ) -> EpisodeStarts:
        return cls(
            normalize_initial_state_specs(simulator, initial_states),
            normalize_goal_state_specs(simulator, goal_states),
            simulator.goal.copy(),
        )

    @property
    def usable_count(self) -> int:
        return min(len(self.initial_states), len(self.goal_states)) if self.goal_states else len(self.initial_states)

    def is_provided(self, index: int) -> bool:
        return index < self.usable_count

    def begin(self, simulator: SystemSimulator, index: int, seed_spec: int) -> np.ndarray:
        """Set the goal of episode ``index`` (counted from 0) and return its initial state."""
        return _setup_episode_start(
            simulator,
            index,
            self.usable_count,
            self.initial_states,
            self.goal_states,
            self.baseline_goal,
            seed_spec,
        )
