"""
AML Environment - Async Python Client
Compatible with OpenEnv client interface conventions.
"""
from __future__ import annotations
import asyncio
from typing import Optional
import httpx

# Re-export models so users can import from the client package
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../server"))
from aml_env.models import (   # noqa: E402
    AMLAction, AMLObservation, AMLState, StepResult, Transaction, TransactionDecision
)


class AMLEnvClient:
    """
    Async HTTP client for the AML Compliance Officer environment.

    Usage:
        async with AMLEnvClient(base_url="http://localhost:7860") as client:
            result = await client.reset(task_name="triage_basic")
            result = await client.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None
        self._timeout = timeout

    async def __aenter__(self) -> "AMLEnvClient":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    async def reset(self, task_name: str = "triage_basic") -> StepResult:
        assert self._client, "Use as async context manager"
        r = await self._client.post("/reset", json={"task_name": task_name})
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def step(self, action: AMLAction) -> StepResult:
        assert self._client, "Use as async context manager"
        r = await self._client.post("/step", content=action.model_dump_json(), headers={"Content-Type": "application/json"})
        r.raise_for_status()
        return StepResult.model_validate(r.json())

    async def state(self) -> AMLState:
        assert self._client, "Use as async context manager"
        r = await self._client.get("/state")
        r.raise_for_status()
        return AMLState.model_validate(r.json())

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @classmethod
    async def from_url(cls, base_url: str) -> "AMLEnvClient":
        client = cls(base_url=base_url)
        client._client = httpx.AsyncClient(base_url=base_url, timeout=client._timeout)
        return client


__all__ = [
    "AMLEnvClient",
    "AMLAction",
    "AMLObservation",
    "AMLState",
    "StepResult",
    "Transaction",
    "TransactionDecision",
]