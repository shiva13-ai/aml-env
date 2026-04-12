import uuid
from typing import Optional
from dataclasses import dataclass, field

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import AMLAction, AMLObservation
from data import TASKS


@dataclass
class StepResult:
    observation: object
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class AMLEnvironment(Environment):
    """
    AML Compliance Officer environment.
    Each episode: agent reviews transactions one-by-one, deciding block/investigate/clear.
    Multi-turn: one step = one transaction decision.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True
    TASK_NAMES = ["triage_basic", "triage_network", "triage_adversarial"]

    def __init__(self):
        self._state = State(episode_id=str(uuid.uuid4()), step_count=0)
        self._task_name: str = "triage_basic"
        self._transactions: list = []
        self._ground_truth: dict = {}
        self._budget: int = 2
        self._decisions: dict = {}
        self._step_index: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self.reset()

    def reset(self, task_name: Optional[str] = None, episode_id: Optional[str] = None, **kwargs) -> AMLObservation:
        """Start a new episode. Optionally select a task."""
        if task_name is None:
            task_name = "triage_basic"
        if task_name not in TASKS:
            task_name = "triage_basic"

        task = TASKS[task_name]
        self._task_name = task_name
        self._transactions = task["transactions"]
        self._ground_truth = task["ground_truth"]
        self._budget = task["investigation_budget"]
        self._decisions = {}
        self._step_index = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._state = State(episode_id=episode_id or str(uuid.uuid4()), step_count=0)

        return self._make_observation("Episode started. Review each transaction carefully.")

    def step(self, action: AMLAction, **kwargs) -> StepResult:
        """Process one transaction decision."""

        # ✅ Guard: episode already finished
        if self._done or self._step_index >= len(self._transactions):
            obs = self._make_observation("Episode already finished. Call reset() to start again.")
            return StepResult(
                observation=obs,
                reward=0.0,
                done=True,
                info={"error": "episode_done"}
            )

        # Force correct transaction_id to match current step
        current_txn = self._transactions[self._step_index]
        action.transaction_id = current_txn["id"]

        # Record decision
        self._decisions[current_txn["id"]] = {
            "decision": action.decision,
            "reasoning": action.reasoning
        }

        # Compute per-step reward
        reward = self._score_decision(
            txn_id=current_txn["id"],
            decision=action.decision,
            reasoning=action.reasoning
        )
        self._cumulative_reward += reward
        self._step_index += 1
        self._state.step_count += 1

        # Check if episode is done
        if self._step_index >= len(self._transactions):
            self._done = True
            final_score = self._normalize_score()
            obs = self._make_observation(
                f"Episode complete! Final score: {final_score:.3f}",
                override_done=True
            )
            return StepResult(
                observation=obs,
                reward=final_score,
                done=True,
                info={
                    "final_score": final_score,
                    "decisions": self._decisions,
                    "task": self._task_name
                }
            )

        obs = self._make_observation(
            f"Decision recorded: {action.decision}. Next transaction below."
        )
        return StepResult(
            observation=obs,
            reward=reward,
            done=False,
            info={
                "step_reward": reward,
                "cumulative": self._cumulative_reward
            }
        )

    @property
    def state(self) -> State:
        return self._state

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self, message: str = "", override_done: bool = False) -> AMLObservation:
        """Build observation from current transaction. Safe — never goes out of range."""
        if not self._transactions:
            return AMLObservation(
                transaction_id="N/A",
                amount=0.0,
                sender_country="XX",
                receiver_country="XX",
                transaction_type="unknown",
                velocity_24h=0,
                is_round_number=False,
                prior_flags=0,
                amount_vs_avg_ratio=0.0,
                high_risk_country=False,
                structuring_indicator=False,
                shell_company_indicator=False,
                pep_involved=False,
                notes="Episode not started.",
                step_number=0,
                total_transactions=0,
                investigation_budget=0,
                investigations_used=0,
                cumulative_reward=self._cumulative_reward,
                task_name=self._task_name,
                episode_done=self._done or override_done,
                message=message,
            )

        # ✅ Clamp index so it never causes IndexError
        idx = min(self._step_index, len(self._transactions) - 1)
        txn = self._transactions[idx]

        investigations_used = sum(
            1 for d in self._decisions.values() if d["decision"] == "investigate"
        )

        return AMLObservation(
            transaction_id=txn["id"],
            amount=txn["amount"],
            sender_country=txn["sender_country"],
            receiver_country=txn["receiver_country"],
            transaction_type=txn["transaction_type"],
            velocity_24h=txn["velocity_24h"],
            is_round_number=txn["is_round_number"],
            prior_flags=txn["prior_flags"],
            amount_vs_avg_ratio=txn["amount_vs_avg_ratio"],
            high_risk_country=txn["high_risk_country"],
            structuring_indicator=txn["structuring_indicator"],
            shell_company_indicator=txn["shell_company_indicator"],
            pep_involved=txn["pep_involved"],
            notes=txn.get("notes", ""),
            step_number=self._step_index + 1,
            total_transactions=len(self._transactions),
            investigation_budget=self._budget,
            investigations_used=investigations_used,
            cumulative_reward=self._cumulative_reward,
            task_name=self._task_name,
            episode_done=self._done or override_done,
            message=message,
        )

    def _score_decision(self, txn_id: str, decision: str, reasoning: str) -> float:
        """Score a single decision. Returns raw reward (normalized at episode end)."""
        truth = self._ground_truth.get(txn_id, "legitimate")
        investigations_used = sum(
            1 for d in self._decisions.values() if d["decision"] == "investigate"
        )

        if truth == "suspicious":
            if decision == "block":
                reward = 3.0
            elif decision == "investigate":
                # Penalize going over budget
                if investigations_used > self._budget:
                    reward = 0.5
                else:
                    reward = 2.0
            else:  # clear — false negative, worst outcome
                reward = -2.0
        else:  # legitimate
            if decision == "clear":
                reward = 1.0
            elif decision == "investigate":
                if investigations_used > self._budget:
                    reward = -1.0
                else:
                    reward = -0.5
            else:  # block — false positive
                reward = -1.0

        # Reasoning bonus (up to +0.2 per suspicious txn with good explanation)
        if truth == "suspicious" and reasoning and len(reasoning.strip()) > 10:
            reward += 0.2

        return reward

    def _normalize_score(self) -> float:
        """Normalize cumulative reward to [0, 1]."""
        n_suspicious = sum(1 for v in self._ground_truth.values() if v == "suspicious")
        n_legit = len(self._ground_truth) - n_suspicious

        # Best possible: all suspicious blocked with reasoning, all legit cleared
        max_possible = (n_suspicious * 3.2) + (n_legit * 1.0)
        # Worst possible: all suspicious cleared, all legit blocked
        min_possible = (n_suspicious * -2.0) + (n_legit * -1.0)

        if max_possible == min_possible:
            return 0.0

        shifted = self._cumulative_reward - min_possible
        range_val = max_possible - min_possible
        score = shifted / range_val

        return max(0.0, min(1.0, score))