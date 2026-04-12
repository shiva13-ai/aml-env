"""
Core AML environment — reset / step / state / grader logic.
"""
from __future__ import annotations
import uuid
from typing import Optional

from .models import (
    AMLAction,
    AMLObservation,
    AMLState,
    StepResult,
    Transaction,
    TransactionGroundTruth,
)
from .data import TASKS

# Chain groups: identifying the full chain earns a bonus
CHAIN_GROUPS = {
    "triage_chain": [
        {"CHN001", "CHN002", "CHN003", "CHN005"},  # first wave full chain
        {"CHN007", "CHN009"},                        # second wave
    ]
}

REASONING_KEYWORDS = [
    "structur", "smurf", "layer", "integrat", "shell", "pep", "sancti",
    "mixer", "mirror", "front", "placement", "velocity", "threshold",
    "ofac", "fatf", "round", "nominee", "crypto",
]


class AMLEnvironment:
    def __init__(self) -> None:
        self._state: Optional[AMLState] = None
        self._task_name: Optional[str] = None

    # ── Public API

    def reset(self, task_name: str) -> AMLObservation:
        if task_name not in TASKS:
            raise ValueError(
                f"Unknown task '{task_name}'. "
                f"Available tasks: {list(TASKS.keys())}"
            )

        task = TASKS[task_name]
        episode_id = str(uuid.uuid4())

        transactions = [Transaction(**t) for t in task["transactions"]]
        observation = AMLObservation(
            task_name=task_name,
            episode_id=episode_id,
            transactions=transactions,
            investigation_budget=task["investigation_budget"],
        )
        ground_truth = [
            TransactionGroundTruth(**g) for g in task["ground_truth"]
        ]

        self._task_name = task_name
        self._state = AMLState(
            task_name=task_name,
            episode_id=episode_id,
            done=False,
            observation=observation,
            ground_truth=ground_truth,
        )
        return observation

    def step(self, action: AMLAction) -> StepResult:
        if self._state is None or self._state.done:
            raise RuntimeError(
                "No active episode. Call /reset before /step."
            )

        obs = self._state.observation
        gt_map = {g.transaction_id: g for g in self._state.ground_truth}
        valid_ids = {t.id for t in obs.transactions}

        # Validate submitted IDs
        unknown_ids = [
            d.transaction_id for d in action.decisions
            if d.transaction_id not in valid_ids
        ]
        if unknown_ids:
            raise ValueError(
                f"Unknown transaction_id(s): {unknown_ids}. "
                f"Valid IDs are: {sorted(valid_ids)}"
            )

        # Score each decision
        raw_score = 0.0
        max_score = 0.0
        reasoning_bonus = 0.0
        investigate_count = 0
        decision_results = []

        for decision in action.decisions:
            tid = decision.transaction_id
            gt = gt_map[tid]
            dec = decision.decision
            reasoning = (decision.reasoning or "").lower()
            is_suspicious = gt.true_label == "suspicious"

            if is_suspicious:
                max_score += 3
            else:
                max_score += 1

            points = 0.0
            if dec == "block":
                points = 3.0 if is_suspicious else -1.0
            elif dec == "investigate":
                investigate_count += 1
                points = 2.0 if is_suspicious else -1.0
            elif dec == "clear":
                points = 1.0 if not is_suspicious else -2.0

            r_bonus = 0.0
            if is_suspicious and reasoning:
                hits = sum(1 for kw in REASONING_KEYWORDS if kw in reasoning)
                if hits >= 1:
                    r_bonus = 0.02

            raw_score += points
            reasoning_bonus += r_bonus

            decision_results.append({
                "transaction_id": tid,
                "decision": dec,
                "correct": (
                    (dec in ["block", "investigate"] and is_suspicious) or
                    (dec == "clear" and not is_suspicious)
                ),
                "true_label": gt.true_label,
                "points_earned": points,
                "reasoning_bonus": r_bonus,
            })

        # Budget penalty
        budget = obs.investigation_budget
        over_budget = max(0, investigate_count - budget)
        budget_penalty = over_budget * 0.5

        # Chain bonus
        chain_bonus = 0.0
        if self._task_name in CHAIN_GROUPS:
            flagged = {
                d.transaction_id for d in action.decisions
                if d.decision in ["block", "investigate"]
            }
            for chain in CHAIN_GROUPS[self._task_name]:
                if chain.issubset(flagged):
                    chain_bonus += 0.08

        # Cap reasoning bonus
        reasoning_bonus = min(reasoning_bonus, 0.10)

        # Normalise to [0, 1]
        if max_score <= 0:
            reward = 0.0
        else:
            reward = (raw_score - budget_penalty) / max_score
            reward = reward + reasoning_bonus + chain_bonus
            epsilon = 0.0001
            reward = max(epsilon, min(1.0 - epsilon, reward))

        self._state.done = True

        submitted_ids = {d.transaction_id for d in action.decisions}
        missed = [
            tid for tid, gt in gt_map.items()
            if gt.true_label == "suspicious" and tid not in submitted_ids
        ]

        return StepResult(
            observation=obs,
            reward=round(reward, 4),
            done=True,
            info={
                "raw_score": raw_score,
                "max_possible_raw": max_score,
                "reasoning_bonus": round(reasoning_bonus, 4),
                "chain_bonus": round(chain_bonus, 4),
                "budget_penalty": round(budget_penalty, 4),
                "investigate_count": investigate_count,
                "investigation_budget": budget,
                "over_budget_by": over_budget,
                "decision_breakdown": decision_results,
                "missed_suspicious_transactions": missed,
                "total_transactions": len(valid_ids),
                "submitted_decisions": len(action.decisions),
            },
        )

    def get_state(self) -> AMLState:
        if self._state is None:
            raise RuntimeError("No active episode. Call /reset first.")
        return self._state