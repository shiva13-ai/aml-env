"""
Baseline inference script — AML Compliance Officer Environment
==============================================================
MANDATORY env variables:
    HF_TOKEN       Your HuggingFace / API key
    MODEL_NAME     Model identifier     (default: Qwen/Qwen2.5-72B-Instruct)
    API_BASE_URL   LLM API endpoint     (default: https://router.huggingface.co/v1)
    ENV_BASE_URL   AML env server       (default: http://localhost:7860)

STDOUT FORMAT (strict — required by OpenEnv evaluator):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import AsyncOpenAI

# ── Config 
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

BENCHMARK = "aml-compliance-env"
TASKS: List[str] = ["triage_basic", "triage_network", "triage_adversarial", "triage_chain"]
SUCCESS_THRESHOLD = 0.5

# ── Logging helpers (must match OpenEnv spec exactly)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # spec: reward=<0.00>  done=<true|false>  error=<msg|null>
    reward_safe = _clamp(reward)
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward_safe:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # spec: score and each reward formatted to 2 decimal places, strictly in (0, 1)
    score_safe = _clamp(score)
    rewards_str = ",".join(f"{_clamp(r):.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score_safe:.2f} rewards={rewards_str}",
        flush=True,
    )


def _clamp(v: float) -> float:
    """Clamp to strictly open interval (0.01, 0.99) so :.2f never prints 0.00 or 1.00."""
    return max(0.01, min(0.99, float(v)))


# ── System prompt 
SYSTEM_PROMPT = """You are a senior AML (Anti-Money Laundering) Compliance Officer at a global bank.

Review each transaction and decide:
- "block": Freeze immediately — OFAC/SDN match, PEP + shell company, confirmed mixer output, sanctions country.
- "investigate": Flag for review — suspicious but not conclusive, structuring pattern, high-risk corridor.
- "clear": Mark as legitimate — only when genuinely confident.

RULES:
1. BUDGET: Never exceed the stated investigation budget with "investigate" decisions. If over budget, use "block" instead.
2. RISK: Clearing a suspicious transaction = -2 penalty. When uncertain and budget is gone, BLOCK not clear.
3. OUTPUT: Valid JSON only, no markdown, no text outside the JSON.

Format:
{
  "decisions": [
    {"transaction_id": "TXN001", "decision": "block", "reasoning": "specific reason"},
    {"transaction_id": "TXN002", "decision": "clear", "reasoning": ""}
  ]
}

Include EVERY transaction_id. Do not skip any."""


# ── LLM call 
async def call_llm(client: AsyncOpenAI, user_message: str) -> str:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=3000,
    )
    return response.choices[0].message.content or ""


def parse_action(raw: str) -> Dict[str, Any]:
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output:\n{raw[:300]}")
    return json.loads(match.group())


def format_observation(obs: Dict) -> str:
    txns   = obs["transactions"]
    budget = obs["investigation_budget"]
    task   = obs["task_name"]
    lines  = [
        f"TASK: {task}",
        f"INVESTIGATION BUDGET: {budget} (max {budget} 'investigate' decisions allowed)",
        f"TRANSACTIONS TO REVIEW ({len(txns)} total):",
        "",
    ]
    for txn in txns:
        lines += [
            f"--- Transaction {txn['id']} ---",
            f"  Amount:            ${txn['amount']:,.2f}",
            f"  Type:              {txn['transaction_type']}",
            f"  Sender Country:    {txn['sender_country']}  |  Receiver Country: {txn['receiver_country']}",
            f"  Velocity (24h):    {txn['velocity_24h']} transactions by sender",
            f"  Amount vs Avg:     {txn['amount_vs_avg_ratio']:.1f}x account historical average",
            f"  Prior AML Flags:   {txn['prior_flags']}",
            f"  Risk Signals:",
            f"    High-risk country:     {txn['high_risk_country']}",
            f"    Round number:          {txn['is_round_number']}",
            f"    Structuring indicator: {txn['structuring_indicator']}",
            f"    Shell company:         {txn['shell_company_indicator']}",
            f"    PEP involved:          {txn['pep_involved']}",
            f"  Analyst Notes: {txn['notes']}",
            "",
        ]
    return "\n".join(lines)


# ── Per-task episode runner 
async def run_task(http: httpx.AsyncClient, llm: AsyncOpenAI, task_name: str) -> float:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.01   # safe fallback — never exact 0.0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── Reset
        r = await http.post("/reset", json={"task_name": task_name})
        r.raise_for_status()
        obs = r.json()

        # ── Get LLM decision
        raw = await call_llm(llm, format_observation(obs))
        action = parse_action(raw)

        # ── Step
        r = await http.post("/step", json=action)
        r.raise_for_status()
        result = r.json()

        reward      = float(result.get("reward", 0.01))
        done        = bool(result.get("done", True))
        info        = result.get("info", {})
        steps_taken = 1
        rewards.append(reward)

        log_step(
            step=1,
            action=json.dumps(action),
            reward=reward,
            done=done,
            error=None,
        )

        # Debug breakdown (not parsed by evaluator)
        print(f"  Raw: {info.get('raw_score','?')} / {info.get('max_possible_raw','?')} "
              f"| bonus: +{info.get('reasoning_bonus',0):.4f} "
              f"| chain: +{info.get('chain_bonus',0):.4f} "
              f"| budget_penalty: -{info.get('budget_penalty',0):.4f}", flush=True)

        score   = reward
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] {task_name} error: {exc}", flush=True)
        score   = 0.01
        rewards = [0.01]

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ── Main
async def main() -> None:
    if not HF_TOKEN:
        print("⚠  HF_TOKEN not set", flush=True)

    llm    = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    scores: Dict[str, float] = {}

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=120.0) as http:
        # Health check
        r = await http.get("/health")
        r.raise_for_status()
        print(f"Health: {r.json()}", flush=True)

        for task in TASKS:
            try:
                scores[task] = await run_task(http, llm, task)
            except Exception as e:
                print(f"[DEBUG] {task} outer error: {e}", flush=True)
                scores[task] = 0.01   # safe fallback — never exact 0.0

    # Summary (not parsed by evaluator)
    print(f"\n{'='*60}", flush=True)
    print("FINAL SCORES", flush=True)
    print(f"{'='*60}", flush=True)
    for task, s in scores.items():
        bar = "█" * int(s * 20)
        print(f"  {task:<25} {s:.4f}  {bar}", flush=True)
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':<25} {avg:.4f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())