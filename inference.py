"""
Inference Script - AML Compliance Officer Environment
=======================================================
MANDATORY variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Defaults:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

STDOUT FORMAT (strict):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""
import asyncio
import json
import os
import textwrap
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY: str = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL: str = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK: str = "aml-compliance-env"
MAX_STEPS: int = 1
TEMPERATURE: float = 0.2
MAX_TOKENS: int = 2000
SUCCESS_SCORE_THRESHOLD: float = 0.5

TASKS = ["triage_basic", "triage_network", "triage_adversarial", "correspondent_banking"]

# ── Logging helpers ────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_compact = action.replace("\n", " ")[:200]
    print(
        f"[STEP] step={step} action={action_compact} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Prompt builder ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert AML (Anti-Money Laundering) Compliance Officer at a major international bank.
You will receive a batch of flagged transactions and must classify each one as:
  - "block": Immediately freeze funds. Use for clear sanctions violations, known bad actors,
             crypto mixers, shell company integration, hawala round-trips, nested correspondents
             with no AML program, TBML, and state-owned enterprise sanctions evasion.
  - "investigate": Flag for deeper manual review. Use for suspicious patterns needing
                   more information: structuring, unusual amounts, PEP involvement,
                   payable-through account concerns, sanctions-adjacent routes.
  - "clear": Mark as legitimate. Use when you are confident the transaction is normal
             (payroll, documented trade finance, treasury operations, pension disbursements).

You must respond with a valid JSON object ONLY — no markdown, no explanation outside JSON.
Format:
{
  "decisions": [
    {"transaction_id": "TXN001", "decision": "block", "reasoning": "...specific reason citing risk signals..."},
    ...
  ]
}

Key AML typologies to detect:
STRUCTURING: Multiple transactions just below $10,000 reporting threshold from same sender
SANCTIONS: IR=Iran, KP=North Korea, YE=Yemen, SY=Syria, MM=Myanmar, AF=Afghanistan
SHELL COMPANIES: shell_company_indicator=true — especially with PEP + offshore jurisdiction
LAYERING NETWORKS: Same account appears as sender/receiver in multiple transactions
CORRESPONDENT BANKING: Payable-through accounts, nested correspondents, hawala round-trips
TRADE-BASED ML (TBML): Over/under-invoiced goods, art/cultural goods with inflated values
CRYPTO: Mixer outputs, unregistered VASPs feeding correspondent accounts
PEP: Politically Exposed Person — amounts inconsistent with known salary/role
REAL ESTATE: Anonymous LLC all-cash purchases with no beneficial owner disclosure
FALSE POSITIVES TO AVOID: Payroll, documented LC settlements, nostro reconciliation,
                          World Bank/MDB disbursements, regulated escrow, FX settlements
""").strip()


def build_user_prompt(observation: Dict[str, Any]) -> str:
    txns = observation["transactions"]
    lines = [
        f"Task: {observation['task_name']}",
        f"Investigation Budget: {observation['budget_remaining']} (max 'investigate' decisions allowed)",
        f"Instructions: {observation['instructions']}",
        "",
        "=== TRANSACTIONS TO REVIEW ===",
    ]
    for t in txns:
        lines.append(
            f"\nID: {t['id']}"
            f"\n  Amount: ${t['amount']:,.2f}"
            f"\n  Sender: {t['sender_id']} ({t['sender_country']}) → "
            f"Receiver: {t['receiver_id']} ({t['receiver_country']})"
            f"\n  Type: {t['transaction_type']} | Velocity(24h): {t['velocity_24h']}"
            f"\n  Round Number: {t['is_round_number']} | Prior Flags: {t['prior_flags']}"
            f"\n  Amount vs Avg: {t['amount_vs_avg_ratio']:.1f}x | "
            f"High-Risk Country: {t['high_risk_country']}"
            f"\n  Structuring: {t['structuring_indicator']} | "
            f"Shell Company: {t['shell_company_indicator']} | PEP: {t['pep_involved']}"
            f"\n  Notes: {t['notes']}"
        )
    lines.append("\nProvide your decisions as JSON only. Include reasoning for every decision.")
    return "\n".join(lines)


def get_model_decisions(client: OpenAI, observation: Dict[str, Any]) -> tuple[str, Optional[str]]:
    user_prompt = build_user_prompt(observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip(), None
    except Exception as exc:
        return '{"decisions":[]}', str(exc)


def parse_decisions(raw: str, txn_ids: List[str]) -> tuple[Dict[str, Any], Optional[str]]:
    try:
        data = json.loads(raw)
        covered = {d["transaction_id"] for d in data.get("decisions", [])}
        for tid in txn_ids:
            if tid not in covered:
                data["decisions"].append({
                    "transaction_id": tid,
                    "decision": "clear",
                    "reasoning": "Not mentioned by model — defaulting to clear"
                })
        return data, None
    except json.JSONDecodeError as e:
        fallback = {
            "decisions": [
                {"transaction_id": tid, "decision": "clear", "reasoning": "parse error"}
                for tid in txn_ids
            ]
        }
        return fallback, f"JSON parse error: {e}"


async def run_episode(task_name: str, client: OpenAI) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    error_msg = None

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=120.0) as http:
        try:
            r = await http.post("/reset", json={"task_name": task_name})
            r.raise_for_status()
            result = r.json()
            obs = result["observation"]
            txn_ids = [t["id"] for t in obs["transactions"]]

            for step in range(1, MAX_STEPS + 1):
                if result.get("done", False):
                    break

                raw_action, llm_error = get_model_decisions(client, obs)
                action_data, parse_error = parse_decisions(raw_action, txn_ids)
                error_msg = llm_error or parse_error

                r = await http.post(
                    "/step",
                    json=action_data,
                    headers={"Content-Type": "application/json"}
                )
                r.raise_for_status()
                result = r.json()

                reward = float(result.get("reward", 0.0))
                done = bool(result.get("done", True))
                rewards.append(reward)
                steps_taken = step

                log_step(
                    step=step,
                    action=json.dumps(action_data),
                    reward=reward,
                    done=done,
                    error=error_msg,
                )

                if done:
                    break

            score = rewards[-1] if rewards else 0.0
            score = min(max(score, 0.0), 1.0)
            success = score >= SUCCESS_SCORE_THRESHOLD

        except Exception as exc:
            error_msg = str(exc)
            print(f"[DEBUG] Episode error: {exc}", flush=True)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    all_scores = []
    for task in TASKS:
        score = await run_episode(task_name=task, client=client)
        all_scores.append(score)
        print(f"[DEBUG] Task={task} score={score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"[DEBUG] Average score across all tasks: {avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())