"""
Async HTTP client for the AML Compliance Officer environment.

Usage:
    async with AMLEnvClient("http://localhost:7860") as client:
        obs = await client.reset("triage_basic")
        result = await client.step([
            {"transaction_id": "TXN001", "decision": "block", "reasoning": "OFAC sanctions + PEP"},
            {"transaction_id": "TXN002", "decision": "clear", "reasoning": ""},
        ])
        print(f"Score: {result['reward']:.3f}")
"""
from __future__ import annotations
from typing import Any, Optional
import httpx


class AMLEnvClient:
    VALID_TASKS = ["triage_basic", "triage_network", "triage_adversarial", "triage_chain"]

    def __init__(self, base_url: str = "http://localhost:7860", timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self) -> "AMLEnvClient":
        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Use 'async with AMLEnvClient(...) as client:'")
        return self._client

    async def health(self) -> dict[str, Any]:
        r = await self._http().get("/health")
        _check(r)
        return r.json()

    async def list_tasks(self) -> list[dict[str, Any]]:
        r = await self._http().get("/tasks")
        _check(r)
        return r.json()["tasks"]

    async def reset(self, task_name: str) -> dict[str, Any]:
        if task_name not in self.VALID_TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Valid: {self.VALID_TASKS}")
        r = await self._http().post("/reset", json={"task_name": task_name})
        _check(r)
        return r.json()

    async def step(self, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        r = await self._http().post("/step", json={"decisions": decisions})
        _check(r)
        return r.json()

    async def get_state(self) -> dict[str, Any]:
        r = await self._http().get("/state")
        _check(r)
        return r.json()


def _check(response: httpx.Response) -> None:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except Exception:
            detail = response.text
        raise httpx.HTTPStatusError(
            f"HTTP {response.status_code}: {detail}",
            request=response.request,
            response=response,
        )