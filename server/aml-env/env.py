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

VALID_TASKS = ["triage_basic", "triage_network", "triage_adversarial"]


class AMLEnv:
    """
    Anti-Money Laundering Compliance Officer Environment.

    The agent acts as a Compliance Officer. It receives a batch of flagged
    transactions and must decide: investigate / block / clear.

    Episode structure (single-step per batch):
      1. reset(task) → returns observation with all transactions
      2. step(action) → agent submits decisions → env grades → returns reward
      3. done=True after first step (batch triage episode)
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

    # ── reset ─────────────────────────────────────────────────────────────────

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

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action: AMLAction) -> StepResult:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self._step += 1
        self._decisions_made = action.decisions

        reward, info = self._grade(action.decisions)
        self._last_reward = reward
        self._done = True  # single-step episode

        obs = AMLObservation(
            task_name=self._task_name,
            step=self._step,
            transactions=self._transactions,
            budget_remaining=0,
            total_transactions=len(self._transactions),
            instructions=self._instructions,
        )
        return StepResult(observation=obs, reward=reward, done=True, info=info)

    # ── state ─────────────────────────────────────────────────────────────────

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

    # ── grader ────────────────────────────────────────────────────────────────

    def _grade(self, decisions: List[TransactionDecision]) -> tuple[float, Dict[str, Any]]:
        """
        Deterministic grader. Returns normalized score in [0, 1].

        Scoring logic:
        - Correct block of truly-suspicious tx: +3 pts
        - Correct investigate of truly-suspicious tx: +2 pts (partial credit)
        - Correct clear of truly-legit tx: +1 pt
        - False positive (clear→blocked/investigated): -1 pt each
        - False negative (suspicious→cleared): -2 pts each
        - Exceeding investigation budget: -0.5 per over-budget investigate
        - Budget left-over: no penalty (conservative approach is fine)

        Max possible = 3 * n_block_gt + 2 * n_investigate_gt + 1 * n_clear_gt
        """
        decision_map: Dict[str, str] = {d.transaction_id: d.decision for d in decisions}

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        reasoning_bonus = 0.0
        raw_score = 0.0

        # Count ground-truth categories for normalization
        n_suspicious = sum(1 for v in self._ground_truth.values() if v in ("block", "investigate"))
        n_legit = sum(1 for v in self._ground_truth.values() if v == "clear")
        max_score = (3 * sum(1 for v in self._ground_truth.values() if v == "block") +
                     2 * sum(1 for v in self._ground_truth.values() if v == "investigate") +
                     1 * n_legit)
        if max_score == 0:
            max_score = 1

        investigate_count = 0

        per_txn_results = {}
        for txn_id, correct in self._ground_truth.items():
            agent_decision = decision_map.get(txn_id, "clear")  # default: clear if not mentioned

            if agent_decision == "investigate":
                investigate_count += 1

            if correct == "block":
                if agent_decision == "block":
                    raw_score += 3
                    true_positives += 1
                    per_txn_results[txn_id] = "correct_block"
                elif agent_decision == "investigate":
                    raw_score += 1  # partial: caught it but didn't stop it
                    true_positives += 1
                    per_txn_results[txn_id] = "partial_investigate_vs_block"
                else:  # clear
                    raw_score -= 2
                    false_negatives += 1
                    per_txn_results[txn_id] = "missed_block"

            elif correct == "investigate":
                if agent_decision == "investigate":
                    raw_score += 2
                    true_positives += 1
                    per_txn_results[txn_id] = "correct_investigate"
                elif agent_decision == "block":
                    raw_score += 1  # over-cautious but still flagged
                    true_positives += 1
                    per_txn_results[txn_id] = "overblocked_investigate"
                else:  # clear
                    raw_score -= 2
                    false_negatives += 1
                    per_txn_results[txn_id] = "missed_investigate"

            else:  # correct == "clear"
                if agent_decision == "clear":
                    raw_score += 1
                    per_txn_results[txn_id] = "correct_clear"
                else:
                    raw_score -= 1
                    false_positives += 1
                    per_txn_results[txn_id] = "false_positive"

        # Budget penalty: each investigate over budget costs 0.5
        budget_penalty = 0.0
        over_budget = max(0, investigate_count - self._investigation_budget)
        budget_penalty = over_budget * 0.5
        raw_score -= budget_penalty

        # Reasoning bonus: check decisions for high-quality reasoning strings
        reasoning_bonus = 0.0
        for d in decisions:
            if len(d.reasoning) > 30 and self._ground_truth.get(d.transaction_id) in ("block", "investigate"):
                reasoning_bonus += 0.02  # small bonus for providing reasoning on suspicious txns
        reasoning_bonus = min(reasoning_bonus, 0.1)

        # Normalize to [0, 1]
        normalized = raw_score / max_score
        normalized = max(0.0, min(1.0, normalized + reasoning_bonus))

        info = {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "investigate_count": investigate_count,
            "budget_allowed": self._investigation_budget,
            "over_budget": over_budget,
            "budget_penalty": budget_penalty,
            "reasoning_bonus": reasoning_bonus,
            "raw_score": raw_score,
            "max_score": max_score,
            "per_transaction": per_txn_results,
            "ground_truth": self._ground_truth,
        }
        return normalized, info