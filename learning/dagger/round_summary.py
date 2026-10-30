from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .metrics import DaggerEvalMetrics


def append_round_summary(
    summary_path: Path,
    round_index: int,
    train_loss: float | None,
    eval_metrics: DaggerEvalMetrics | None,
    aggregation_metrics: DaggerEvalMetrics | None = None,
    safeflow_eval_metrics: DaggerEvalMetrics | None = None,
    latest_checkpoint: str | None = None,
    iteration_checkpoint: str | None = None,
) -> None:
    """Add the round to the YAML summary at ``summary_path`` (created if missing): its success and failure stats, the
    best round so far and the totals of the collection."""
    summary: dict[str, Any]
    if summary_path.exists():
        with summary_path.open("r") as f:
            summary = yaml.safe_load(f) or {}
    else:
        summary = {}

    rounds = summary.get("rounds", [])
    entry: dict[str, Any] = {
        "round": int(round_index),
        "train_loss": train_loss,
        "aggregation_success_rate": aggregation_metrics.success_rate if aggregation_metrics is not None else None,
        **(eval_metrics.to_dict() if eval_metrics is not None else DaggerEvalMetrics.not_evaluated()),
    }
    if safeflow_eval_metrics is not None:
        entry["safeflow_eval"] = safeflow_eval_metrics.to_dict()
    if aggregation_metrics is not None:
        entry.update(aggregation_metrics.collection_summary())
    if latest_checkpoint is not None or iteration_checkpoint is not None:
        entry["checkpoints"] = {
            "latest": str(latest_checkpoint) if latest_checkpoint is not None else None,
            "iteration": str(iteration_checkpoint) if iteration_checkpoint is not None else None,
        }
    rounds.append(entry)
    summary["rounds"] = rounds
    valid_rates = [
        round_entry["eval_success_rate"]
        for round_entry in rounds
        if round_entry["eval_success_rate"] is not None
    ]
    if valid_rates:
        best_round = max(
            rounds,
            key=lambda round_entry: round_entry["eval_success_rate"]
            if round_entry["eval_success_rate"] is not None
            else float("-inf"),
        )
        summary["best_round"] = best_round["round"]
        summary["best_success_rate"] = best_round["eval_success_rate"]
    else:
        summary["best_round"] = None
        summary["best_success_rate"] = None
    summary["latest_round"] = int(round_index)
    collection_rounds = [r["aggregation_failures"] for r in rounds if "aggregation_failures" in r]
    summary["total_aggregation_failures"] = {
        key: sum(r[key] for r in collection_rounds)
        for key in ("attempted_episodes", "discarded_episodes", "planner_failures", "policy_solve_fallbacks")
    }
    total_expert_statistics = Counter()
    for round_entry in rounds:
        total_expert_statistics.update(round_entry.get("expert_statistics", {}))
    if total_expert_statistics:
        summary["total_expert_statistics"] = dict(total_expert_statistics)
    with summary_path.open("w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
