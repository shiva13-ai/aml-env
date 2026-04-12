from typing import Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .aml_environment import AMLEnvironment
from .models import (
    AMLAction,
    AMLObservation,
    AMLState,
    HealthResponse,
    StepResult,
    TaskInfo,
    TaskList,
)
from .data import TASKS

app = FastAPI(
    title="AML Compliance Officer — OpenEnv",
    description=(
        "RL environment where an AI agent plays a bank Compliance Officer "
        "reviewing flagged transactions. Decide: block / investigate / clear."
    ),
    version="1.0.0",
)

env = AMLEnvironment()

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "detail": exc.errors(),
            "hint": "Check that your JSON matches the AMLAction schema.",
        },
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": "Bad request", "detail": str(exc)},
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    return JSONResponse(
        status_code=409,
        content={"error": "Environment state error", "detail": str(exc)},
    )

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    return HealthResponse()

@app.get("/tasks", response_model=TaskList, tags=["meta"])
async def list_tasks():
    return TaskList(tasks=[
        TaskInfo(
            name=name,
            difficulty=data["difficulty"],
            description=data["description"],
            num_transactions=len(data["transactions"]),
            investigation_budget=data["investigation_budget"],
            expected_baseline_score=data["expected_baseline_score"],
        )
        for name, data in TASKS.items()
    ])

@app.post("/reset", response_model=AMLObservation, tags=["env"])
async def reset(body: Optional[dict] = None):
    if body is None:
        task_name = "triage_basic"
    else:
        task_name = body.get("task_name", "triage_basic")

    if task_name not in TASKS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{task_name}' not found. Available: {list(TASKS.keys())}",
        )
    try:
        return env.reset(task_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/step", response_model=StepResult, tags=["env"])
async def step(action: AMLAction):
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/state", response_model=AMLState, tags=["env"])
async def get_state():
    try:
        return env.get_state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

def main():
    """Entry point required by openenv validation."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()