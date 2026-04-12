---
title: AML Compliance Officer Environment
emoji: 🏦
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: apache-2.0
tags:
  - openenv
  - aml
  - compliance
  - finance
---

# 🏦 AML Compliance Officer Environment

> **OpenEnv-compatible Anti-Money Laundering environment for training and evaluating AI compliance agents.**

An AI agent plays the role of a bank **Compliance Officer** reviewing a batch of flagged transactions. For each transaction, the agent must decide: **`block`** (freeze funds), **`investigate`** (flag for review), or **`clear`** (mark as legitimate) — maximizing detection accuracy while staying within an investigation budget.

---

## 🎯 Why This Matters

Money laundering costs the global economy **$800B–$2T per year** (UNODC). Banks employ thousands of compliance analysts to manually review flagged transactions — a slow, expensive, and error-prone process. This environment trains and evaluates AI agents that can assist or automate AML triage — a genuine, high-value real-world task.

---

## 🌍 Environment Description

| Property | Value |
|---|---|
| **Domain** | Financial Crime / Regulatory Compliance |
| **Task type** | Batch transaction classification with budget constraint |
| **Episode structure** | Single-step batch (all decisions submitted at once) |
| **Action space** | Discrete: `block` / `investigate` / `clear` per transaction |
| **Observation** | 13 risk features per transaction + free-text analyst notes |
| **Reward** | Continuous [0, 1] — partial credit for partial detection |

---

## 📋 Action Space

```python
class AMLAction(Action):
    decisions: list[AMLDecision]   # one entry per transaction

class AMLDecision:
    transaction_id: str            # which transaction to decide on
    decision: Literal["block", "investigate", "clear"]
    reasoning: str                 # optional — earns a small reward bonus
```

## 👁️ Observation Space

Each episode exposes all transactions at once with these features per transaction:

| Feature | Type | Description |
|---|---|---|
| `id` | str | Transaction identifier |
| `amount` | float | USD amount |
| `sender_country` / `receiver_country` | str | ISO-2 country codes |
| `transaction_type` | str | wire / cash / crypto / internal |
| `velocity_24h` | int | Transactions by sender in last 24h |
| `is_round_number` | bool | Suspiciously round amount |
| `prior_flags` | int | Historical AML flags on account |
| `amount_vs_avg_ratio` | float | Amount / account historical average |
| `high_risk_country` | bool | FATF high-risk jurisdiction involved |
| `structuring_indicator` | bool | Multiple txns just below reporting threshold |
| `shell_company_indicator` | bool | Counterparty flagged as possible shell |
| `pep_involved` | bool | Politically Exposed Person |
| `notes` | str | Analyst free-text context |
| `investigation_budget` | int | Max `investigate` decisions allowed for the batch |

---

## 🎮 Tasks

### Task 1: `triage_basic` — Easy
- **5 transactions** with clear, single-signal AML indicators
- Signals: sanctions country, structuring, PEP, round amounts
- Investigation budget: **2**
- Expected baseline score: ~0.85

### Task 2: `triage_network` — Medium
- **10 transactions** forming a structuring → layering → integration network
- Requires correlating multiple transactions across the same sender/receiver
- Investigation budget: **3**
- Expected baseline score: ~0.65

### Task 3: `triage_adversarial` — Hard
- **15 adversarially-crafted transactions** including:
  - Smurfing with delayed aggregation
  - Trade-based money laundering (over-invoiced goods)
  - Mirror trades
  - Charity front abuse
  - Crypto mixer outputs (Hydra darknet)
  - Anonymous LLC real estate
- Mix of obvious and subtle signals; false positives penalized
- Investigation budget: **4**
- Expected baseline score: ~0.45

### Task 4: `triage_chain` — Hard (Chain Detection)
- **10 transactions** forming interconnected money-laundering chains
- Bonus reward for identifying complete chains (e.g. CHN001→CHN003→CHN005)
- Requires end-to-end chain detection across two waves of transactions
- Investigation budget: **4**
- Expected baseline score: ~0.85

---

## 🏆 Reward Function

```
score = normalized_cumulative_reward  ∈ [0, 1]

Per-decision raw reward:
  +3.0  correct BLOCK of suspicious transaction
  +2.0  correct INVESTIGATE of suspicious transaction
  +1.0  correct CLEAR of legitimate transaction
  -2.0  FALSE NEGATIVE (suspicious transaction cleared)
  -1.0  FALSE POSITIVE (legit transaction blocked or investigated)

Bonuses & Penalties:
  +0.02  per suspicious transaction with quality reasoning keywords
  +0.10  reasoning bonus cap (across all decisions in episode)
  +0.08  per fully identified chain (triage_chain only), up to +0.16 total
  -0.50  per investigation over budget

Final score = (raw_score - budget_penalty) / max_possible_raw
            + reasoning_bonus + chain_bonus
            clamped to [0.0001, 0.9999]
```

The reward provides **continuous partial-progress signal** — not just binary success/failure.

---

## 🚀 Setup & Usage

### Prerequisites
```bash
pip install openenv-core fastapi uvicorn pydantic httpx openai
```

### Local Development

```bash
# Clone the repo
git clone https://github.com/shiva13-ai/aml-env
cd aml-env/server

# Run the server
python -m uvicorn app:app --host 0.0.0.0 --port 7860
```

### Test the server (new terminal)

```bash
# Health check
curl http://localhost:7860/health

# Reset environment
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d "{\"task_name\": \"triage_basic\"}"

# Submit batch decisions
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d "{\"decisions\": [{\"transaction_id\": \"TXN001\", \"decision\": \"investigate\", \"reasoning\": \"structuring pattern\"}, {\"transaction_id\": \"TXN002\", \"decision\": \"clear\", \"reasoning\": \"\"}]}"
```

### Docker

```bash
# Build
docker build -t aml-env .

# Run
docker run -p 7860:7860 aml-env

# Health check
curl http://localhost:7860/health
```

### Run Inference Script

```bash
# Set environment variables (Groq — default)
set HF_TOKEN=your_groq_api_key
set MODEL_NAME=llama-3.3-70b-versatile
set API_BASE_URL=https://api.groq.com/openai/v1
set ENV_BASE_URL=http://localhost:7860

# Or use HuggingFace Inference Router
set HF_TOKEN=your_huggingface_token
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set API_BASE_URL=https://router.huggingface.co/v1

# Run inference
python inference.py
```

---

## 📊 Baseline Scores

Run with `llama-3.3-70b-versatile` via Groq API:

| Task | Score | Raw Score | Notes |
|---|---|---|---|
| `triage_basic` | **0.9691** | 10.0 / 11.0 | Handles clear single signals well |
| `triage_network` | **0.9636** | 19.0 / 22.0 | Full budget used efficiently |
| `triage_adversarial` | **0.7216** | 23.0 / 37.0 | TBML / mirror trades can fool the model |
| `triage_chain` | **0.9999** | 19.0 / 22.0 | Both chains fully identified (+0.16 chain bonus) |
| **Overall average** | **0.9135** | — | — |

> Scores above reflect actual inference runs. `triage_adversarial` is the primary challenge — ADV001/ADV002 (subtle smurfing) were the only missed suspicious transactions.

---

## 🗂️ Project Structure

```
aml-env/
├── models.py                ← Pydantic Action + Observation models
├── client.py                ← AMLEnvClient (async HTTP client)
├── __init__.py              ← Package exports
├── openenv.yaml             ← OpenEnv spec metadata
├── pyproject.toml           ← Package config
├── inference.py             ← Baseline inference script (ROOT)
├── README.md                ← This file
└── server/
    ├── app.py               ← FastAPI app (create_fastapi_app)
    ├── aml_environment.py   ← Core AMLEnvironment logic
    ├── data.py              ← Synthetic transaction datasets
    ├── models.py            ← Copy of root models.py (for Docker)
    ├── requirements.txt     ← Server dependencies
    └── Dockerfile           ← Container image definition
```

---

## 🔗 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_name": "triage_basic"}` |
| `/step` | POST | Submit all decisions. Body: `{"decisions": [...]}` |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all available tasks |
| `/health` | GET | Health check — returns `{"status": "ok"}` |

---

## 📜 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | Groq or HuggingFace API key |
| `API_BASE_URL` | No | `https://api.groq.com/openai/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `llama-3.3-70b-versatile` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:7860` | AML env server URL |

---

## ⚖️ License

Apache 2.0