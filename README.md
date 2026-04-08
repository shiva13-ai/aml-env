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

> **OpenEnv-compatible Anti-Money Laundering environment (v1.1.0) for training and evaluating AI compliance agents.**

An AI agent plays the role of a bank **Compliance Officer** reviewing batches of flagged transactions. For each transaction, the agent must decide: **`block`** (freeze funds), **`investigate`** (flag for review), or **`clear`** (mark as legitimate) — maximizing detection accuracy while staying within an investigation budget.

---

## 🎯 Why This Matters

Money laundering costs the global economy **$800B–$2T per year** (UNODC). Banks employ thousands of compliance analysts to manually review flagged transactions — a slow, expensive, and error-prone process. This environment trains and evaluates AI agents that can assist or automate AML triage across **6 task levels**, from basic transaction screening to expert-level correspondent banking and crypto/DeFi money laundering — genuine daily workflows at major financial institutions worldwide.

---

## 🌍 Environment Description

| Property | Value |
|---|---|
| **Version** | 1.1.0 |
| **Domain** | Financial Crime / Regulatory Compliance |
| **Task type** | Batch classification with budget constraint |
| **Episode structure** | Single-step (review full batch, submit decisions) |
| **Action space** | Discrete: `block` / `investigate` / `clear` per transaction |
| **Observation** | 15 risk features per transaction + free-text notes |
| **Reward** | Continuous (0.001, 0.999) — partial credit for partial detection |
| **Tasks** | 6 — Easy → Expert |

---

## 📋 Action Space

```python
class AMLAction(BaseModel):
    decisions: List[TransactionDecision]

class TransactionDecision(BaseModel):
    transaction_id: str        # which transaction
    decision: Literal["block", "investigate", "clear"]
    reasoning: str             # optional — earns a small bonus
```

---

## 👁️ Observation Space

Each transaction exposes **15 features**:

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

---

## 🎮 Tasks

### Task 1: `triage_basic` — Easy

- **5 transactions** with clear, single-signal AML indicators
- Signals: sanctions country, structuring, PEP, round amount
- Investigation budget: **2**
- Expected baseline score: ~0.85

### Task 2: `triage_network` — Medium

- **10 transactions** forming a structuring → layering → integration network
- Requires correlating multiple transactions (same sender/receiver)
- Investigation budget: **3**
- Expected baseline score: ~0.65

### Task 3: `triage_adversarial` — Hard

- **15 adversarially-crafted transactions** including:
  - Smurfing with delayed aggregation
  - Trade-based money laundering (over-invoiced goods)
  - Mirror trades
  - Charity front abuse
  - Crypto mixer outputs
  - Anonymous LLC real estate
- Mix of obvious and subtle signals; false positives penalized
- Investigation budget: **4**
- Expected baseline score: ~0.45

### Task 4: `correspondent_banking` — Expert 🆕

- **20 transactions** through nested correspondent banking relationships across 3 respondent banks
- Requires identifying:
  - Payable-through account abuse
  - Nested correspondents with no AML program
  - TBML via cultural goods (over-invoiced art)
  - Hawala round-trips
  - Sanctions evasion via State-Owned Enterprises (SOEs)
- Investigation budget: **5**
- Expected baseline score: ~0.40

### Task 5: `sanctions_screening` — Medium-Hard 🆕

- **12 transactions** requiring OFAC SDN / UN / EU sanctions list screening
- Covers:
  - Name disambiguation (partial matches, aliases)
  - General license exceptions
  - Subsidiary liability
  - FinCEN advisory matches
- Investigation budget: **4**
- Expected baseline score: ~0.55

### Task 6: `crypto_defi_aml` — Expert 🆕

- **18 on-chain and off-chain crypto transactions** covering DeFi-specific AML typologies:
  - Tornado Cash / mixer outputs
  - Lazarus Group wallet interactions
  - Rug pulls and exit scams
  - Darknet peel chains
  - Cross-chain bridge hopping
  - NFT wash trading
  - Ransomware payments
  - Sanctioned exchange withdrawals
- Investigation budget: **5**
- Expected baseline score: ~0.38

---

## 🏆 Reward Function

```
score = (raw_score + reasoning_bonus) / max_score  ∈ (0.001, 0.999)

raw_score:
  +4  correct BLOCK of suspicious transaction
  +3  correct INVESTIGATE of suspicious transaction  (or +2 if labeled block)
  +1  correct CLEAR of legitimate transaction
  -3  FALSE NEGATIVE (suspicious transaction cleared)
  -1  FALSE POSITIVE (legit transaction blocked/investigated)
  -1  per investigate over budget

reasoning_bonus:
  +0.03 per suspicious transaction with quality reasoning (capped at 0.12)
```

> Score is strictly bounded between **0.001 and 0.999** — never exactly 0 or 1.

The reward provides **continuous partial-progress signal** — not just binary success/failure.

---

## 🔍 AML Typologies Covered

The inference agent is prompted to detect all of the following typologies:

| Typology | Description |
|---|---|
| **Structuring** | Multiple transactions just below $10,000 reporting threshold |
| **Sanctions** | IR, KP, YE, SY, MM, AF country codes (OFAC/UN/EU lists) |
| **Shell Companies** | `shell_company_indicator=true` + offshore jurisdiction + PEP |
| **Layering Networks** | Same account appearing as both sender and receiver |
| **Correspondent Banking** | Payable-through accounts, nested correspondents, hawala round-trips |
| **TBML** | Over/under-invoiced goods, inflated art/cultural goods |
| **Crypto** | Mixer outputs, unregistered VASPs, cross-chain bridges |
| **PEP** | Amounts inconsistent with known salary/role |
| **Real Estate** | Anonymous LLC all-cash purchases, no beneficial owner |
| **False Positives** | Payroll, LC settlements, nostro reconciliation, MDB disbursements |

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone the repo
git clone https://github.com/shiva13-ai/aml-env
cd aml-env

# Install server deps
cd server && pip install -e .

# Run the server
python -m uvicorn aml_env.server:app --host 0.0.0.0 --port 7860

# In another terminal, run inference across all 6 tasks
cd ..
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

### Docker

```bash
# Build
docker build -t aml-env .

# Run
docker run -p 7860:7860 aml-env

# Test
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "triage_basic"}'
```

### Python Client

```python
import asyncio
import httpx

async def demo():
    async with httpx.AsyncClient(base_url="http://localhost:7860") as client:
        # Reset with any task name
        r = await client.post("/reset", json={"task_name": "correspondent_banking"})
        obs = r.json()["observation"]
        print(f"Got {len(obs['transactions'])} transactions")

        # Submit decisions
        action = {
            "decisions": [
                {"transaction_id": "TXN001", "decision": "block",       "reasoning": "North Korea SOE + sanctions evasion"},
                {"transaction_id": "TXN002", "decision": "investigate", "reasoning": "nested correspondent, no AML program"},
                {"transaction_id": "TXN003", "decision": "clear",       "reasoning": "documented World Bank disbursement"},
            ]
        }
        r = await client.post("/step", json=action)
        result = r.json()
        print(f"Score: {result['reward']:.3f}")
        print(f"Info: {result['info']}")

asyncio.run(demo())
```

---

## 📊 Baseline Scores

Run with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Inference Router:

| Task | Difficulty | Transactions | Score | Notes |
|---|---|---|---|---|
| `triage_basic` | Easy | 5 | ~0.85 | Model handles clear single signals well |
| `triage_network` | Medium | 10 | ~0.65 | Network correlations sometimes missed |
| `triage_adversarial` | Hard | 15 | ~0.45 | Adversarial patterns challenge frontier models |
| `sanctions_screening` | Medium-Hard | 12 | ~0.55 | Name disambiguation is tricky |
| `correspondent_banking` | Expert | 20 | ~0.40 | Nested relationships require deep reasoning |
| `crypto_defi_aml` | Expert | 18 | ~0.38 | DeFi typologies at the frontier of LLM knowledge |

---

## 🗂️ Project Structure

```
aml-env/
├── server/
│   ├── aml_env/
│   │   ├── __init__.py      # Package exports
│   │   ├── env.py           # Core AMLEnv logic (reset/step/state/grader)
│   │   ├── models.py        # Pydantic models (Action/Observation/State/StepResult)
│   │   ├── data.py          # Synthetic transaction datasets for all 6 tasks
│   │   └── server.py        # FastAPI server (OpenEnv HTTP endpoints)
│   ├── requirements.txt
│   └── pyproject.toml
├── client/
│   └── aml_env_client/
│       └── __init__.py      # Async HTTP client
├── Dockerfile               # HuggingFace Spaces deployment
├── openenv.yaml             # OpenEnv spec metadata (v1.1.0)
├── inference.py             # Baseline inference script (runs all 6 tasks)
└── README.md
```

---

## 🔗 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_name": "<task>"}` |
| `/step` | POST | Submit decisions. Body: `AMLAction` JSON |
| `/state` | GET | Get current environment state |
| `/tasks` | GET | List all 6 available tasks |
| `/health` | GET | Health check |

---

## 📜 Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes (inference) | — | HuggingFace API key |
| `API_BASE_URL` | No | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | No | `http://localhost:7860` | AML env server URL |

---

## ⚖️ License

Apache 2.0