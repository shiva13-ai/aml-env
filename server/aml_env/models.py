"""
AML Compliance Environment - Pydantic Models
Typed Action, Observation, State, and StepResult for OpenEnv spec compliance.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Transaction ───────────────────────────────────────────────────────────────

class Transaction(BaseModel):
    id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., description="Transaction amount in USD")
    sender_id: str = Field(..., description="Sender account identifier")
    receiver_id: str = Field(..., description="Receiver account identifier")
    sender_country: str = Field(..., description="Sender's country (ISO-2)")
    receiver_country: str = Field(..., description="Receiver's country (ISO-2)")
    transaction_type: str = Field(..., description="wire/cash/crypto/internal")
    velocity_24h: int = Field(..., description="Number of transactions by sender in last 24h")
    is_round_number: bool = Field(..., description="Amount is a suspiciously round number")
    prior_flags: int = Field(..., description="Number of prior AML flags on this account")
    amount_vs_avg_ratio: float = Field(..., description="Amount / historical avg for this account")
    high_risk_country: bool = Field(..., description="Involves a FATF high-risk jurisdiction")
    structuring_indicator: bool = Field(..., description="Multiple txns just below reporting threshold")
    shell_company_indicator: bool = Field(..., description="Counterparty flagged as possible shell company")
    pep_involved: bool = Field(..., description="Politically Exposed Person involved")
    notes: str = Field(default="", description="Analyst notes / free-text context")


# ── Action ────────────────────────────────────────────────────────────────────

DecisionLiteral = Literal["investigate", "block", "clear"]

class TransactionDecision(BaseModel):
    transaction_id: str = Field(..., description="Which transaction this decision applies to")
    decision: DecisionLiteral = Field(
        ...,
        description="'investigate' = flag for review, 'block' = freeze funds, 'clear' = mark safe"
    )
    reasoning: str = Field(default="", description="Optional analyst reasoning (improves partial score)")


class AMLAction(BaseModel):
    """Agent action: a list of decisions, one per flagged transaction."""
    decisions: List[TransactionDecision] = Field(
        ..., description="One decision per transaction in the current observation"
    )


# ── Observation ───────────────────────────────────────────────────────────────

class AMLObservation(BaseModel):
    task_name: str = Field(..., description="Current task identifier")
    step: int = Field(..., description="Current step number")
    transactions: List[Transaction] = Field(..., description="Transactions awaiting review")
    budget_remaining: int = Field(..., description="Remaining investigation budget (blocks are free)")
    total_transactions: int = Field(..., description="Total transactions in this episode")
    instructions: str = Field(..., description="Task instructions for the agent")


# ── Reward ────────────────────────────────────────────────────────────────────

class AMLReward(BaseModel):
    value: float = Field(..., ge=0.0, le=1.0, description="Normalized reward [0, 1]")
    true_positives: int = Field(default=0, description="Correctly actioned suspicious transactions")
    false_positives: int = Field(default=0, description="Legit transactions wrongly blocked/investigated")
    false_negatives: int = Field(default=0, description="Missed suspicious transactions")
    budget_penalty: float = Field(default=0.0, description="Penalty for exceeding investigation budget")
    reasoning_bonus: float = Field(default=0.0, description="Bonus for correct reasoning on hard cases")


# ── State ─────────────────────────────────────────────────────────────────────

class AMLState(BaseModel):
    episode_id: str
    task_name: str
    step: int
    done: bool
    total_reward: float
    decisions_made: List[TransactionDecision]
    ground_truth: Optional[Dict[str, str]] = None  # hidden until episode ends


# ── StepResult ────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: AMLObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)