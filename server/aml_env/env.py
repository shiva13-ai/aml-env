"""
AML Compliance Environment - Core Logic
Implements the OpenEnv step()/reset()/state() interface.
"""
from __future__ import annotations
import uuid
from typing import Any, Dict, List, Optional

from .models import (
    AMLAction, AMLObservation, AMLState, StepResult, TransactionDecision
)
from .data import get_task_data

VALID_TASKS = ["triage_basic", "triage_network", "triage_adversarial", "correspondent_banking", "sanctions_screening", "crypto_defi_aml"]


class AMLEnv:
    """
    Anti-Money Laundering Compliance Officer Environment.

    The agent acts as a Compliance Officer reviewing batches of flagged
    transactions and must decide: investigate / block / clear.

    Reward shaping:
    - Partial credit at decision level (not just binary end-of-episode)
    - Reasoning quality bonus
    - Budget constraint penalty
    - False positive / false negative asymmetric penalties
    - Score always strictly in (0.001, 0.999) per spec
    """

    def __init__(self) -> None:
        self._episode_id: str = ""
        self._task_name: str = "triage_basic"
        self._step: int = 0
        self._done: bool = False
        self._ground_truth: Dict[str, str] = {}
        self._transactions = []
        self._instructions: str = ""
        self._investigation_budget: int = 0
        self._decisions_made: List[TransactionDecision] = []
        self._last_reward: float = 0.0

    def reset(self, task_name: Optional[str] = None) -> StepResult:
        if task_name and task_name not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {VALID_TASKS}")

        self._task_name = task_name or "triage_basic"
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._done = False
        self._decisions_made = []
        self._last_reward = 0.0

        self._transactions, self._ground_truth, self._instructions, self._investigation_budget = \
            get_task_data(self._task_name)

        obs = AMLObservation(
            task_name=self._task_name,
            step=self._step,
            transactions=self._transactions,
            budget_remaining=self._investigation_budget,
            total_transactions=len(self._transactions),
            instructions=self._instructions,
        )
        return StepResult(observation=obs, reward=0.0, done=False, info={"episode_id": self._episode_id})

    def step(self, action: AMLAction) -> StepResult:
        if not self._episode_id:
            raise RuntimeError("Must call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        self._decisions_made = action.decisions

        reward, info = self._grade(action.decisions)
        self._last_reward = reward
        self._done = True

        obs = AMLObservation(
            task_name=self._task_name,
            step=self._step,
            transactions=self._transactions,
            budget_remaining=0,
            total_transactions=len(self._transactions),
            instructions=self._instructions,
        )
        return StepResult(observation=obs, reward=reward, done=True, info=info)

    def state(self) -> AMLState:
        return AMLState(
            episode_id=self._episode_id,
            task_name=self._task_name,
            step=self._step,
            done=self._done,
            total_reward=self._last_reward,
            decisions_made=self._decisions_made,
            ground_truth=self._ground_truth if self._done else None,
        )

    def _grade(self, decisions: List[TransactionDecision]) -> tuple[float, Dict[str, Any]]:
        """
        Deterministic, shaped reward grader. Returns score strictly in (0.001, 0.999).

        Scoring weights (designed for meaningful partial-progress signal):
        ┌─────────────────────────────────────────────┬────────┐
        │ Outcome                                     │ Points │
        ├─────────────────────────────────────────────┼────────┤
        │ Correct BLOCK on block-labeled txn          │  +4    │
        │ INVESTIGATE on block-labeled txn (cautious) │  +2    │
        │ Correct INVESTIGATE on investigate-labeled  │  +3    │
        │ BLOCK on investigate-labeled (over-cautious)│  +2    │
        │ Correct CLEAR on clear-labeled txn          │  +1    │
        │ FALSE NEGATIVE: clear on suspicious txn     │  -3    │
        │ FALSE POSITIVE: flag/block on legit txn     │  -1    │
        │ Over-budget investigate (per extra)         │  -1    │
        │ Reasoning bonus (per suspicious txn)        │  +0.03 │
        └─────────────────────────────────────────────┴────────┘
        """
        decision_map: Dict[str, str] = {d.transaction_id: d.decision for d in decisions}
        reasoning_map: Dict[str, str] = {d.transaction_id: d.reasoning for d in decisions}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        raw_score = 0.0
        investigate_count = 0
        per_txn_results = {}

        # Max possible score for normalization
        max_score = (
            4 * sum(1 for v in self._ground_truth.values() if v == "block") +
            3 * sum(1 for v in self._ground_truth.values() if v == "investigate") +
            1 * sum(1 for v in self._ground_truth.values() if v == "clear")
        )
        if max_score == 0:
            max_score = 1

        for txn_id, correct in self._ground_truth.items():
            agent = decision_map.get(txn_id, "clear")
            if agent == "investigate":
                investigate_count += 1

            if correct == "block":
                if agent == "block":
                    raw_score += 4
                    true_positives += 1
                    per_txn_results[txn_id] = "correct_block"
                elif agent == "investigate":
                    raw_score += 2   # partial: flagged but didn't stop
                    true_positives += 1
                    per_txn_results[txn_id] = "partial_investigate_vs_block"
                else:
                    raw_score -= 3   # missed a block — severe
                    false_negatives += 1
                    per_txn_results[txn_id] = "missed_block"

            elif correct == "investigate":
                if agent == "investigate":
                    raw_score += 3
                    true_positives += 1
                    per_txn_results[txn_id] = "correct_investigate"
                elif agent == "block":
                    raw_score += 2   # over-cautious but still flagged
                    true_positives += 1
                    per_txn_results[txn_id] = "overblocked_investigate"
                else:
                    raw_score -= 3   # missed suspicious activity
                    false_negatives += 1
                    per_txn_results[txn_id] = "missed_investigate"

            else:  # correct == "clear"
                if agent == "clear":
                    raw_score += 1
                    per_txn_results[txn_id] = "correct_clear"
                else:
                    raw_score -= 1   # false positive
                    false_positives += 1
                    per_txn_results[txn_id] = "false_positive"

        # Budget penalty
        over_budget = max(0, investigate_count - self._investigation_budget)
        budget_penalty = over_budget * 1.0
        raw_score -= budget_penalty

        # Reasoning quality bonus (rewards agents that explain their decisions)
        reasoning_bonus = 0.0
        for txn_id, correct in self._ground_truth.items():
            if correct in ("block", "investigate"):
                agent = decision_map.get(txn_id, "clear")
                reasoning = reasoning_map.get(txn_id, "")
                if agent in ("block", "investigate") and len(reasoning) > 25:
                    reasoning_bonus += 0.03
        reasoning_bonus = min(reasoning_bonus, 0.12)

        # Normalize and clamp strictly between 0.001 and 0.999
        normalized = raw_score / max_score
        normalized = normalized + reasoning_bonus
        normalized = max(0.001, min(0.999, normalized))

        info = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "investigate_count": investigate_count,
            "budget_allowed": self._investigation_budget,
            "over_budget": over_budget,
            "budget_penalty": budget_penalty,
            "reasoning_bonus": round(reasoning_bonus, 4),
            "raw_score": round(raw_score, 4),
            "max_score": max_score,
            "precision": round(true_positives / max(true_positives + false_positives, 1), 3),
            "recall": round(true_positives / max(true_positives + false_negatives, 1), 3),
            "per_transaction": per_txn_results,
            "ground_truth": self._ground_truth,
        }
        return normalized, info