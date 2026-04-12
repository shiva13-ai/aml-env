"""
Baseline inference script — runs all 4 tasks using an LLM.

Usage:
    export HF_TOKEN=your_token_here
    export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct        # optional
    export API_BASE_URL=https://router.huggingface.co/v1 # optional
    export ENV_BASE_URL=http://localhost:7860             # optional
    python inference.py
"""
import asyncio
import json
import os
import re
from typing import Any

import httpx
from openai import AsyncOpenAI

# ── Config 

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.1-70b-versatile")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")

TASKS = ["triage_basic", "triage_network", "triage_adversarial", "triage_chain"]

# ── System prompt

SYSTEM_PROMPT = """You are a senior AML (Anti-Money Laundering) Compliance Officer at a global bank with 20 years of experience detecting financial crime.

You are reviewing a batch of flagged transactions. For each transaction, you must decide:
- **block**: Freeze funds immediately. Use when there is strong, clear evidence (OFAC sanctions, PEP + shell company, confirmed crypto mixer, placement stage with no plausible explanation).
- **investigate**: Flag for deeper review. Use when suspicious but not conclusive (anomalous pattern, high-risk country without other flags, structuring indicator needing verification).
- **clear**: Mark as legitimate. Only when you are genuinely confident the transaction is not suspicious.

CRITICAL ANALYSIS RULES:
1. Standard AML Procedures: Evaluate each transaction based on standard global banking regulations (FATF/FinCEN guidelines).
2. Holistic Review: Consider the relationship between different transactions in the batch to identify complex patterns.
3. Accuracy Matters: Your performance is based on correctly identifying suspicious activity while maintaining a low false-positive rate for legitimate business.
4. Professional Justification: Provide a brief, professional justification for any 'block' or 'investigate' decisions using industry-standard terminology.

OUTPUT: Respond ONLY with a valid JSON object. No markdown, no explanation outside the JSON:
{
  "decisions": [
    {"transaction_id": "TXN001", "decision": "block", "reasoning": "Structuring: cash deposits just below $10k CTR threshold on consecutive days."},
    {"transaction_id": "TXN002", "decision": "clear", "reasoning": ""}
  ]
}

Include EVERY transaction_id from the observation. Do not skip any.
"""


# ── LLM call 

async def call_llm(user_message: str) -> str:
    client = AsyncOpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=3000,
    )
    return response.choices[0].message.content


def parse_action(raw: str) -> dict[str, Any]:
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{raw}")
    return json.loads(match.group())


def format_observation(obs: dict) -> str:
    txns = obs["transactions"]
    budget = obs["investigation_budget"]
    task = obs["task_name"]

    lines = [
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


# ── Task runner 

async def run_task(client: httpx.AsyncClient, task_name: str) -> float:
    print(f"\n{'='*60}")
    print(f"  TASK: {task_name}")
    print(f"{'='*60}")

    r = await client.post("/reset", json={"task_name": task_name})
    r.raise_for_status()
    obs = r.json()
    await asyncio.sleep(0.5)
    print(f"  Transactions: {len(obs['transactions'])}  |  Budget: {obs['investigation_budget']}")

    print(f"  Calling {MODEL_NAME}...")
    raw = await call_llm(format_observation(obs))

    try:
        action = parse_action(raw)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  ❌ Parse failed: {e}\n  Raw: {raw[:300]}")
        return 0.0

    r = await client.post("/step", json=action)
    if r.status_code != 200:
        print(f"  ❌ /step error {r.status_code}: {r.text}")
        return 0.0

    result = r.json()
    reward = result["reward"]
    info   = result["info"]

    print(f"\n  Score:            {reward:.4f}")
    print(f"  Raw:              {info['raw_score']:.1f} / {info['max_possible_raw']:.1f}")
    print(f"  Reasoning bonus:  +{info['reasoning_bonus']:.4f}")
    if info.get("chain_bonus", 0) > 0:
        print(f"  Chain bonus:      +{info['chain_bonus']:.4f}")
    print(f"  Budget penalty:   -{info['budget_penalty']:.4f}")
    print(f"  Investigate used: {info['investigate_count']} / {info['investigation_budget']}")

    if info.get("missed_suspicious_transactions"):
        print(f"  ⚠  Missed: {info['missed_suspicious_transactions']}")

    print(f"\n  Decision breakdown:")
    for d in info["decision_breakdown"]:
        mark  = "✅" if d["correct"] else "❌"
        bonus = f"  (+{d['reasoning_bonus']:.2f})" if d["reasoning_bonus"] > 0 else ""
        print(f"    {mark} {d['transaction_id']}: {d['decision']} "
              f"(true={d['true_label']}, pts={d['points_earned']:+.1f}){bonus}")

    return reward


async def main():
    if not HF_TOKEN:
        print("⚠  HF_TOKEN not set — export HF_TOKEN=your_token_here")

    print(f"\nAML Compliance Officer — Baseline Inference")
    print(f"Model : {MODEL_NAME}")
    print(f"API   : {API_BASE_URL}")
    print(f"Env   : {ENV_BASE_URL}")

    scores: dict[str, float] = {}

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=120.0) as client:
        r = await client.get("/health")
        r.raise_for_status()
        print(f"\nHealth: {r.json()}")

        for task in TASKS:
            try:
                scores[task] = await run_task(client, task)
            except Exception as e:
                print(f"  ❌ {task} failed: {e}")
                scores[task] = 0.0

    print(f"\n{'='*60}")
    print("  FINAL SCORES")
    print(f"{'='*60}")
    for task, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"  {task:<25} {score:.4f}  {bar}")
    avg = sum(scores.values()) / len(scores) if scores else 0
    print(f"\n  Overall average:  {avg:.4f}")


if __name__ == "__main__":
    asyncio.run(main())