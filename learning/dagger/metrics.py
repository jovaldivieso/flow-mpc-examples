from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DaggerEvalMetrics:
    success_rate: float
    mean_steps: float
    min_steps: int
    max_steps: int
    num_episodes: int
    # Split of the same episodes by initial-state/goal source: the leading
    # `usable_count` episodes drawn from the curriculum's explicit
    # initial_states/goal_states lists ("config") versus the remainder
    # sampled via simulator RNG fallback ("random") -- see
    # evaluate_policy_rollouts. Raw counts (not a rate) so "X/Y solved"
    # displays exactly, with no division until a caller wants a percentage.
    # Left at the defaults (0 episodes each) by callers that don't
    # distinguish a source split, e.g. aggregation's own success accounting.
    config_successes: int = 0
    config_num_episodes: int = 0
    random_successes: int = 0
    random_num_episodes: int = 0
    # Why a failed episode failed -- timeout (ran out of steps without reaching
    # tolerance) or solve_failure (the policy/projector itself raised, e.g.
    # SafeFlowMPC's CasADi projector) -- see rollout_policy_with_action_fn.
    timeout_failures: int = 0
    solve_failures: int = 0
    collision_failures: int = 0  # the system touched an obstacle (ground truth, ``simulator.collides``)
    # Collection bookkeeping, only populated by collect_dagger_rollouts: how many episode attempts it took,
    # how many were dropped (expert solve failure), how many expert solves failed, and how many times the
    # policy's own solve failed and the expert action was used instead.
    attempted_episodes: int = 0
    discarded_episodes: int = 0
    planner_failures: int = 0
    policy_solve_fallbacks: int = 0
    # What the expert planner did during the collection (``Planner.statistics``, e.g. the OMPL paths it planned and
    # replanned), as the change over the collection
    expert_statistics: Mapping[str, int] = field(default_factory=dict)

    # --- the structure of a round's entry in ``training_summary.yaml`` ---------

    @staticmethod
    def _percent(successes: int, episodes: int) -> float | None:
        return 100.0 * successes / episodes if episodes > 0 else None

    def to_dict(self) -> dict[str, Any]:
        """The evaluation part of a round's summary entry: the success rate, the steps, the success by source of the
        episodes and why the others failed."""
        return {
            "eval_success_rate": self.success_rate,
            "eval_mean_steps": self.mean_steps,
            "eval_min_steps": self.min_steps,
            "eval_max_steps": self.max_steps,
            "eval_num_episodes": self.num_episodes,
            "eval_success_by_source": {
                "config": {
                    "successes": self.config_successes,
                    "total": self.config_num_episodes,
                    "rate": self._percent(self.config_successes, self.config_num_episodes),
                },
                "random": {
                    "successes": self.random_successes,
                    "total": self.random_num_episodes,
                    "rate": self._percent(self.random_successes, self.random_num_episodes),
                },
            },
            "eval_failure_breakdown": {
                "timeout": self.timeout_failures,
                "solve_failure": self.solve_failures,
                "collision": self.collision_failures,
            },
        }

    @classmethod
    def not_evaluated(cls) -> dict[str, Any]:
        """The same structure for a round that was not evaluated: no rates, no steps, zero counts."""
        empty = cls(success_rate=None, mean_steps=None, min_steps=None, max_steps=None, num_episodes=None)
        return empty.to_dict()

    def collection_summary(self) -> dict[str, Any]:
        """The collection part of a round's summary entry: what it took to collect the episodes, and what the expert
        planner did (when it kept statistics)."""
        summary: dict[str, Any] = {
            "aggregation_failures": {
                "attempted_episodes": self.attempted_episodes,
                "discarded_episodes": self.discarded_episodes,
                "planner_failures": self.planner_failures,
                "policy_solve_fallbacks": self.policy_solve_fallbacks,
            }
        }
        if self.expert_statistics:
            summary["expert_statistics"] = dict(self.expert_statistics)
        return summary
