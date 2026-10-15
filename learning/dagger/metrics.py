from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


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
    # Why a failed episode failed -- collision (self-collision of the arm),
    # timeout (ran out of steps without colliding or reaching tolerance), or
    # solve_failure (the policy/projector itself raised, e.g. SafeFlowMPC's
    # CasADi projector) -- see rollout_policy_with_action_fn. Distinguishing
    # these matters: a policy that times out while actively avoiding is a
    # different (and better) failure mode than one that collides outright,
    # which a bare success_rate can't tell apart. Left at the defaults (0)
    # by callers that don't populate this breakdown, e.g. collect_dagger_
    # rollouts' own aggregation accounting (discarded episodes are a
    # different concept, already instrumented separately).
    collision_failures: int = 0
    timeout_failures: int = 0
    solve_failures: int = 0
    # Collection bookkeeping, only populated by collect_dagger_rollouts: how many episode attempts it took,
    # how many were dropped (expert solve failure, unsafe expert label or collision), how many expert
    # solves failed, and how many times the policy's own solve failed and the expert action was used instead.
    attempted_episodes: int = 0
    discarded_episodes: int = 0
    planner_failures: int = 0
    policy_solve_fallbacks: int = 0
    # What the expert planner did during the collection (``Planner.statistics``, e.g. the OMPL paths it planned and
    # replanned), as the change over the collection
    expert_statistics: Mapping[str, int] = field(default_factory=dict)
