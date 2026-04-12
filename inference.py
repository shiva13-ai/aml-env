"""
AML Compliance Officer — Inference Script
[START] / [STEP] / [END] mandatory format
"""
import asyncio
import os
import textwrap
import json
from typing import List

import httpx
from openai import OpenAI

API_KEY       = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL  = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
ENV_BASE_URL  = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK     = "aml-env"
MAX_STEPS     = 20
SUCCESS_SCORE_THRESHOLD = 0.5
TASKS         = ["triage_basic", "triage_network", "triage_adversarial"]

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior AML Compliance Officer at a major bank.
You review flagged transactions one at a time and make ONE decision per transaction:
  - block: Freeze funds immediately (clear money laundering / sanctions violation)
  - investigate: Flag for enhanced due diligence (suspicious but not certain)
  - clear: Mark as legitimate (evidence supports normal business activity)

You have a limited investigation budget per episode — use it wisely.

Respond ONLY with valid JSON and nothing else, no markdown, no explanation:
{"decision": "block|investigate|clear", "reasoning": "one sentence explanation"}
""").strip()


def log_start(task, model):
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def build_prompt(obs: dict) -> str:
    return textwrap.dedent(f"""
    Transaction ID: {obs.get('transaction_id')}
    Amount: ${obs.get('amount', 0):,.2f}
    Type: {obs.get('transaction_type')}
    Sender: {obs.get('sender_country')} -> Receiver: {obs.get('receiver_country')}
    Velocity (24h): {obs.get('velocity_24h')} transactions
    Round number: {obs.get('is_round_number')}
    Prior flags: {obs.get('prior_flags')}
    Amount vs avg: {obs.get('amount_vs_avg_ratio', 1.0):.1f}x
    High-risk country: {obs.get('high_risk_country')}
    Structuring indicator: {obs.get('structuring_indicator')}
    Shell company: {obs.get('shell_company_indicator')}
    PEP involved: {obs.get('pep_involved')}
    Notes: {obs.get('notes', '')}

    Budget remaining: {obs.get('investigation_budget', 0) - obs.get('investigations_used', 0)} investigations left
    Progress: {obs.get('step_number')}/{obs.get('total_transactions')} transactions
    """).strip()


def get_decision(client: OpenAI, obs: dict) -> dict:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(obs)},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        text = (completion.choices[0].message.content or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        if parsed.get("decision") not in ("block", "investigate", "clear"):
            parsed["decision"] = "investigate"
        return parsed
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"decision": "investigate", "reasoning": "fallback"}


async def run_task(client: OpenAI, http: httpx.AsyncClient, task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # Reset
        r = await http.post("/reset", json={"task_name": task_name})

        if r.status_code != 200:
            raise Exception(f"Server /reset returned {r.status_code}: {r.text}")

        data = r.json()
        obs = data.get("observation", data)
        state = data.get("state", {})

        for step in range(1, MAX_STEPS + 1):
            if data.get("done", False):
                break

            parsed = get_decision(client, obs)
            decision = parsed.get("decision", "investigate")
            reasoning = parsed.get("reasoning", "")

            action = {
                "transaction_id": obs.get("transaction_id", ""),
                "decision": decision,
                "reasoning": reasoning,
            }

            # openenv server requires action wrapped in {"action": ...} along with the state
            r = await http.post("/step", json={"action": action, "state": state})

            if r.status_code != 200:
                raise Exception(f"Server /step returned {r.status_code}: {r.text}")

            data = r.json()
            obs = data.get("observation", data)
            reward = float(data.get("reward", 0.0))
            done = bool(data.get("done", False))
            info = data.get("info", {})
            state = data.get("state", state)

            rewards.append(reward)
            steps_taken = step
            action_str = f"{decision}({action['transaction_id']})"
            log_step(step=step, action=action_str, reward=reward, done=done)

            if done:
                score = float(info.get("final_score", 0.0))
                break

        if score == 0.0 and rewards:
            score = max(0.0, min(1.0, sum(x for x in rewards if x > 0) / (len(rewards) * 3.0)))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task '{task_name}' error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    async with httpx.AsyncClient(base_url=ENV_BASE_URL, timeout=30.0) as http:
        for task in TASKS:
            await run_task(llm_client, http, task)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())