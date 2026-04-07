"""
AML Environment - FastAPI Server
Implements OpenEnv spec endpoints: POST /reset, POST /step, GET /state
"""
from __future__ import annotations
from typing import Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .env import AMLEnv, VALID_TASKS
from .models import AMLAction, StepResult, AMLState

app = FastAPI(
    title="AML Compliance Officer Environment",
    description=(
        "An OpenEnv-compatible Anti-Money Laundering environment. "
        "The agent acts as a Compliance Officer triaging suspicious transactions."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instance (single-session server)
_env = AMLEnv()


class ResetRequest(BaseModel):
    task_name: Optional[str] = "triage_basic"


@app.post("/reset", response_model=StepResult)
async def reset(request: ResetRequest = ResetRequest()) -> StepResult:
    """
    Reset the environment and return the initial observation.
    task_name options: triage_basic | triage_network | triage_adversarial
    """
    try:
        return _env.reset(task_name=request.task_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/step", response_model=StepResult)
async def step(action: AMLAction) -> StepResult:
    """
    Submit compliance decisions for the current transaction batch.
    Returns reward and grading breakdown.
    """
    try:
        return _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", response_model=AMLState)
async def state() -> AMLState:
    """Return the current environment state."""
    return _env.state()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks."""
    return {
        "tasks": [
            {
                "name": "triage_basic",
                "difficulty": "easy",
                "description": "5 transactions with clear AML signals",
                "n_transactions": 5,
                "investigation_budget": 2,
            },
            {
                "name": "triage_network",
                "difficulty": "medium",
                "description": "10 transactions forming a layering network",
                "n_transactions": 10,
                "investigation_budget": 3,
            },
            {
                "name": "triage_adversarial",
                "difficulty": "hard",
                "description": "15 adversarially-crafted transactions to evade detection",
                "n_transactions": 15,
                "investigation_budget": 4,
            },
        ]
    }


@app.get("/health")
async def health():
    return {"status": "ok", "environment": "aml-compliance-env", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)