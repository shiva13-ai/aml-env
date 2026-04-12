from typing import Literal
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class AMLAction(Action):
    """One compliance decision on a single transaction."""
    transaction_id: str = Field(..., description="ID of the transaction being decided")
    decision: Literal["block", "investigate", "clear"] = Field(
        ..., description="block=freeze funds, investigate=flag for review, clear=legitimate"
    )
    reasoning: str = Field(default="", description="Optional explanation (earns small bonus)")


class AMLObservation(Observation):
    """Current transaction the agent must decide on."""
    transaction_id: str = Field(..., description="Transaction identifier")
    amount: float = Field(..., description="USD transaction amount")
    sender_country: str = Field(..., description="Sender ISO-2 country code")
    receiver_country: str = Field(..., description="Receiver ISO-2 country code")
    transaction_type: str = Field(..., description="wire/cash/crypto/internal")
    velocity_24h: int = Field(..., description="Transactions by sender in last 24h")
    is_round_number: bool = Field(..., description="Suspiciously round amount flag")
    prior_flags: int = Field(..., description="Historical AML flags on this account")
    amount_vs_avg_ratio: float = Field(..., description="Amount / account historical average")
    high_risk_country: bool = Field(..., description="FATF high-risk jurisdiction involved")
    structuring_indicator: bool = Field(..., description="Multiple txns just below reporting threshold")
    shell_company_indicator: bool = Field(..., description="Counterparty flagged as possible shell")
    pep_involved: bool = Field(..., description="Politically Exposed Person involved")
    notes: str = Field(default="", description="Analyst free-text context")
    # Episode context
    step_number: int = Field(..., description="Current step (1-indexed)")
    total_transactions: int = Field(..., description="Total transactions in this episode")
    investigation_budget: int = Field(..., description="Max investigations allowed")
    investigations_used: int = Field(..., description="Investigations used so far")
    cumulative_reward: float = Field(default=0.0, description="Reward accumulated so far")
    task_name: str = Field(default="", description="Current task name")
    episode_done: bool = Field(default=False, description="Whether episode is complete")
    message: str = Field(default="", description="Status or feedback message")