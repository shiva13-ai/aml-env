from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


# ── Action models

class TransactionDecision(BaseModel):
    transaction_id: str = Field(..., description="ID of the transaction to decide on")
    decision: Literal["block", "investigate", "clear"]
    reasoning: Optional[str] = Field(default="", description="Optional reasoning — earns a small bonus")


class AMLAction(BaseModel):
    decisions: List[TransactionDecision] = Field(..., min_length=1)

    @model_validator(mode="after")
    def no_duplicate_ids(self) -> "AMLAction":
        ids = [d.transaction_id for d in self.decisions]
        if len(ids) != len(set(ids)):
            dupes = list({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"Duplicate transaction_ids in decisions: {dupes}")
        return self

 

class Transaction(BaseModel):
    id: str
    amount: float
    sender_country: str
    receiver_country: str
    transaction_type: Literal["wire", "cash", "crypto", "internal"]
    velocity_24h: int
    is_round_number: bool
    prior_flags: int
    amount_vs_avg_ratio: float
    high_risk_country: bool
    structuring_indicator: bool
    shell_company_indicator: bool
    pep_involved: bool
    notes: str


class AMLObservation(BaseModel):
    task_name: str
    episode_id: str
    transactions: List[Transaction]
    investigation_budget: int
    instructions: str = (
        "Review each transaction and decide: block / investigate / clear. "
        "You may use at most `investigation_budget` investigate actions. "
        "Provide reasoning for suspicious transactions to earn a bonus."
    )


class TransactionGroundTruth(BaseModel):
    transaction_id: str
    true_label: Literal["suspicious", "legitimate"]
    risk_level: Literal["high", "medium", "low"]
    explanation: str


class AMLState(BaseModel):
    task_name: str
    episode_id: str
    done: bool
    observation: Optional[AMLObservation] = None
    ground_truth: Optional[List[TransactionGroundTruth]] = None


class StepResult(BaseModel):
    observation: AMLObservation
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: dict


# ── Task info

class TaskInfo(BaseModel):
    name: str
    difficulty: Literal["easy", "medium", "hard", "expert"]
    description: str
    num_transactions: int
    investigation_budget: int
    expected_baseline_score: float


class TaskList(BaseModel):
    tasks: List[TaskInfo]



class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    environment: str = "AML Compliance Officer"