"""
AML Compliance Officer — Inference Script
Mandatory format: [START] / [STEP] / [END]
"""
import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from aml_env import AMLEnv, AMLAction

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
MAX_STEPS = 20
TEMPERATURE = 0.3
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.5

TASKS = ["triage_basic", "triage_network", "triage_adversarial"]

SYSTEM_PROMPT = textwrap.dedent("""
You are a senior AML Compliance Officer at a major bank.
You are reviewing flagged transactions one at a time.
For each transaction, you must make ONE of these decisions:
  - block: Freeze funds immediately (use for clear money laundering / sanctions)
  - investigate: Flag for enhanced due diligence (use when suspicious but not certain)  
  - clear: Mark as legitimate (use when evidence supports normal business activity)

You have a limited investigation budget per episode — use it wisely.

Respond with EXACTLY this JSON format (no other text):
{"decision": "block|investigate|clear", "reasoning": "one sentence explanation"}
""").strip()


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error=None):
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(obs) -> str:
    return textwrap.dedent(f"""
    Transaction ID: {obs.transaction_id}
    Amount: ${obs.amount:,.2f}
    Type: {obs.transaction_type}
    Sender: {obs.sender_country} → Receiver: {obs.receiver_country}
    Velocity (24h): {obs.velocity_24h} transactions
    Round number: {obs.is_round_number}
    Prior flags: {obs.prior_flags}
    Amount vs avg: {obs.amount_vs_avg_ratio:.1f}x
    High-risk country: {obs.high_risk_country}
    Structuring indicator: {obs.structuring_indicator}
    Shell company: {obs.shell_company_indicator}
    PEP involved: {obs.pep_involved}
    Notes: {obs.notes}

    Budget remaining: {obs.investigation_budget - obs.investigations_used} investigations left
    Progress: {obs.step_number}/{obs.total_transactions} transactions

    Decide: block, investigate, or clear?
    """).strip()


def get_decision(client: OpenAI, obs) -> dict:
    import json
    prompt = build_prompt(obs)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return parsed
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", flush=True)
        return {"decision": "investigate", "reasoning": "fallback"}


async def run_task(client: OpenAI, env: AMLEnv, task_name: str):
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    benchmark = "aml-env"

    log_start(task=task_name, env=benchmark, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        obs = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            parsed = get_decision(client, obs)
            decision = parsed.get("decision", "investigate")
            reasoning = parsed.get("reasoning", "")

            if decision not in ("block", "investigate", "clear"):
                decision = "investigate"

            action = AMLAction(
                transaction_id=obs.transaction_id,
                decision=decision,
                reasoning=reasoning,
            )

            result = await env.step(action)
            reward = result.reward or 0.0
            done = result.done
            obs = result.observation

            rewards.append(reward)
            steps_taken = step
            action_str = f"{decision}({obs.transaction_id})"
            log_step(step=step, action=action_str, reward=reward, done=done)

            if done:
                score = result.info.get("final_score", 0.0)
                break

        if not rewards:
            score = 0.0
        elif score == 0.0:
            # Fallback normalization
            score = max(0.0, min(1.0, sum(rewards) / max(len(rewards), 1) / 3.0))

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env_url = os.getenv("ENV_BASE_URL", "http://localhost:7860")

    async with AMLEnv(base_url=env_url) as env:
        for task in TASKS:
            await run_task(client, env, task)


if __name__ == "__main__":
    asyncio.run(main())