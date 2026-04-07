"""
AML Environment - Synthetic Transaction Dataset
Each task has a fixed seed for reproducibility.
Ground truth labels: "block" | "investigate" | "clear"
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from .models import Transaction

# FATF high-risk jurisdictions (simplified)
HIGH_RISK_COUNTRIES = {"IR", "KP", "MM", "YE", "SY", "AF", "SO", "SS"}


def get_task_data(task_name: str) -> Tuple[List[Transaction], Dict[str, str], str, int]:
    """
    Returns (transactions, ground_truth, instructions, investigation_budget).
    ground_truth maps transaction_id -> correct decision.
    """
    if task_name == "triage_basic":
        return _task_basic()
    elif task_name == "triage_network":
        return _task_network()
    elif task_name == "triage_adversarial":
        return _task_adversarial()
    else:
        raise ValueError(f"Unknown task: {task_name}")


# ── TASK 1: Basic (Easy) ──────────────────────────────────────────────────────
def _task_basic() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "You are a Compliance Officer reviewing flagged transactions. "
        "For each transaction decide: 'block' (freeze suspicious funds immediately), "
        "'investigate' (flag for deeper manual review), or 'clear' (mark as legitimate). "
        "This batch has 5 transactions with clear signals. "
        "Investigation budget: 2 (you may only 'investigate' up to 2 transactions). "
        "Blocking is always available. Maximize AML detection accuracy."
    )
    txns = [
        Transaction(
            id="TXN001",
            amount=9_900.0,
            sender_id="ACC_1001",
            receiver_id="ACC_2001",
            sender_country="US",
            receiver_country="US",
            transaction_type="cash",
            velocity_24h=3,
            is_round_number=False,
            prior_flags=0,
            amount_vs_avg_ratio=1.1,
            high_risk_country=False,
            structuring_indicator=True,   # Just below $10k reporting threshold
            shell_company_indicator=False,
            pep_involved=False,
            notes="3 cash deposits of $9,900 in 48 hours by same account"
        ),
        Transaction(
            id="TXN002",
            amount=1_000_000.0,
            sender_id="ACC_1002",
            receiver_id="ACC_2002",
            sender_country="IR",           # Iran - FATF blacklist
            receiver_country="US",
            transaction_type="wire",
            velocity_24h=1,
            is_round_number=True,
            prior_flags=2,
            amount_vs_avg_ratio=85.0,
            high_risk_country=True,
            structuring_indicator=False,
            shell_company_indicator=False,
            pep_involved=True,
            notes="Large wire from sanctioned country, PEP sender, round amount"
        ),
        Transaction(
            id="TXN003",
            amount=250.0,
            sender_id="ACC_1003",
            receiver_id="ACC_2003",
            sender_country="CA",
            receiver_country="US",
            transaction_type="internal",
            velocity_24h=1,
            is_round_number=False,
            prior_flags=0,
            amount_vs_avg_ratio=0.9,
            high_risk_country=False,
            structuring_indicator=False,
            shell_company_indicator=False,
            pep_involved=False,
            notes="Regular payroll disbursement, consistent with history"
        ),
        Transaction(
            id="TXN004",
            amount=5_000.0,
            sender_id="ACC_1004",
            receiver_id="ACC_2004",
            sender_country="GB",
            receiver_country="GB",
            transaction_type="wire",
            velocity_24h=1,
            is_round_number=True,
            prior_flags=0,
            amount_vs_avg_ratio=1.05,
            high_risk_country=False,
            structuring_indicator=False,
            shell_company_indicator=False,
            pep_involved=False,
            notes="Regular supplier payment, invoice matched"
        ),
        Transaction(
            id="TXN005",
            amount=87_500.0,
            sender_id="ACC_1005",
            receiver_id="ACC_2005",
            sender_country="US",
            receiver_country="KP",          # North Korea - sanctioned
            transaction_type="crypto",
            velocity_24h=5,
            is_round_number=False,
            prior_flags=1,
            amount_vs_avg_ratio=22.0,
            high_risk_country=True,
            structuring_indicator=False,
            shell_company_indicator=True,
            pep_involved=False,
            notes="Crypto transfer to sanctioned jurisdiction, shell company receiver"
        ),
    ]
    ground_truth = {
        "TXN001": "investigate",  # structuring — needs deeper review
        "TXN002": "block",        # sanctioned country + PEP + huge deviation
        "TXN003": "clear",        # normal payroll
        "TXN004": "clear",        # normal supplier
        "TXN005": "block",        # sanctioned country + shell company
    }
    return txns, ground_truth, instructions, 2


# ── TASK 2: Network (Medium) ──────────────────────────────────────────────────
def _task_network() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "You are reviewing 10 transactions that may form a layering/smurfing network. "
        "Multiple transactions may involve the same accounts — look for patterns across the batch. "
        "Decisions: 'block', 'investigate', or 'clear'. "
        "Investigation budget: 3. "
        "Consider: structuring patterns, round-trip flows, and network-level risk."
    )
    txns = [
        Transaction(
            id="NET001", amount=9800.0, sender_id="ACC_A", receiver_id="ACC_B",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=4, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=2.5, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Part of 4-part structuring series"
        ),
        Transaction(
            id="NET002", amount=9750.0, sender_id="ACC_A", receiver_id="ACC_C",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=4, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=2.4, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Same sender as NET001, same day"
        ),
        Transaction(
            id="NET003", amount=9850.0, sender_id="ACC_A", receiver_id="ACC_D",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=4, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=2.6, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Third leg of structuring from ACC_A"
        ),
        Transaction(
            id="NET004", amount=29000.0, sender_id="ACC_B", receiver_id="ACC_SHELL1",
            sender_country="US", receiver_country="PA",  # Panama
            transaction_type="wire", velocity_24h=2, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=8.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="ACC_B aggregates then wires to Panama shell"
        ),
        Transaction(
            id="NET005", amount=450.0, sender_id="ACC_E", receiver_id="ACC_F",
            sender_country="DE", receiver_country="FR", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=0.95, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Intra-EU payroll, fully documented"
        ),
        Transaction(
            id="NET006", amount=12000.0, sender_id="ACC_G", receiver_id="ACC_H",
            sender_country="AE", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=1,
            amount_vs_avg_ratio=3.2, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=True, notes="UAE PEP wiring to US, shell company receiver"
        ),
        Transaction(
            id="NET007", amount=180.0, sender_id="ACC_I", receiver_id="ACC_J",
            sender_country="US", receiver_country="US", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Utility bill payment"
        ),
        Transaction(
            id="NET008", amount=29500.0, sender_id="ACC_C", receiver_id="ACC_SHELL1",
            sender_country="US", receiver_country="PA",
            transaction_type="wire", velocity_24h=2, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=7.8, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="ACC_C (received from NET001) also wires to same shell"
        ),
        Transaction(
            id="NET009", amount=500_000.0, sender_id="ACC_SHELL1", receiver_id="ACC_K",
            sender_country="PA", receiver_country="CH",  # Switzerland
            transaction_type="wire", velocity_24h=1, is_round_number=True,
            prior_flags=3, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="Shell aggregates all received funds, moves to Swiss account"
        ),
        Transaction(
            id="NET010", amount=3200.0, sender_id="ACC_L", receiver_id="ACC_M",
            sender_country="AU", receiver_country="NZ", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Business invoice payment, supporting docs present"
        ),
    ]
    ground_truth = {
        "NET001": "investigate",  # structuring leg 1
        "NET002": "investigate",  # structuring leg 2
        "NET003": "investigate",  # structuring leg 3
        "NET004": "block",        # layering to shell
        "NET005": "clear",        # legit payroll
        "NET006": "block",        # PEP + shell company
        "NET007": "clear",        # utility payment
        "NET008": "block",        # connected to layering network
        "NET009": "block",        # final integration stage
        "NET010": "clear",        # documented business payment
    }
    return txns, ground_truth, instructions, 3


# ── TASK 3: Adversarial (Hard) ────────────────────────────────────────────────
def _task_adversarial() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "HARD MODE: This batch contains 15 adversarially-constructed transactions. "
        "Sophisticated actors have deliberately designed transactions to appear legitimate "
        "while hiding illicit flows. Look for: smurfing with delayed aggregation, "
        "trade-based money laundering, mirror trades, and charity abuse. "
        "Decisions: 'block', 'investigate', or 'clear'. "
        "Investigation budget: 4. Choose wisely — budget exhaustion is penalized."
    )
    txns = [
        # Smurfing with plausible cover stories
        Transaction(
            id="ADV001", amount=8500.0, sender_id="SMRF_A", receiver_id="SMRF_HUB",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.8, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Freelance payment — but amount is 1.8x sender average"
        ),
        Transaction(
            id="ADV002", amount=8200.0, sender_id="SMRF_B", receiver_id="SMRF_HUB",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.7, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Different sender but same destination as ADV001"
        ),
        Transaction(
            id="ADV003", amount=8900.0, sender_id="SMRF_C", receiver_id="SMRF_HUB",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=2.0, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="Third unique sender to same hub account"
        ),
        # Trade-based ML: over-invoiced goods
        Transaction(
            id="ADV004", amount=450_000.0, sender_id="IMPORT_CO", receiver_id="EXPORT_CO",
            sender_country="US", receiver_country="CN", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=3.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Import payment for electronics — invoice provided but price 3x market rate"
        ),
        # Legitimate large wire with all docs
        Transaction(
            id="ADV005", amount=2_500_000.0, sender_id="HEDGE_FUND_1", receiver_id="PRIME_BROKER",
            sender_country="US", receiver_country="GB", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=0.9, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Margin call payment — all docs filed, consistent with fund activity"
        ),
        # Charity abuse
        Transaction(
            id="ADV006", amount=75_000.0, sender_id="CHARITY_FRONT", receiver_id="OVERSEAS_ACCT",
            sender_country="US", receiver_country="YE",  # Yemen
            transaction_type="wire", velocity_24h=3, is_round_number=True,
            prior_flags=1, amount_vs_avg_ratio=5.5, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="NGO claims humanitarian aid — but Yemen + shell receiver + elevated frequency"
        ),
        # Mirror trade: buy in RU, sell in USD
        Transaction(
            id="ADV007", amount=320_000.0, sender_id="BROKER_RU", receiver_id="BROKER_US",
            sender_country="RU", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=2,
            amount_vs_avg_ratio=6.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=True, notes="Equity purchase paired with identical RU sell order — mirror trade signature"
        ),
        # Perfectly normal transactions to test false-positive discipline
        Transaction(
            id="ADV008", amount=1_200.0, sender_id="RETAIL_CUST_1", receiver_id="MERCHANT_1",
            sender_country="US", receiver_country="US", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Credit card payment for electronics purchase"
        ),
        Transaction(
            id="ADV009", amount=3_500.0, sender_id="RETAIL_CUST_2", receiver_id="LANDLORD_1",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Monthly rent payment, consistent every month"
        ),
        # Crypto mixing
        Transaction(
            id="ADV010", amount=42_000.0, sender_id="CRYPTO_MIXER_OUT", receiver_id="EXCHANGE_1",
            sender_country="--", receiver_country="US", transaction_type="crypto",
            velocity_24h=8, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=15.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Funds exiting known mixer address — blockchain analysis flagged"
        ),
        # Real estate layering
        Transaction(
            id="ADV011", amount=890_000.0, sender_id="LLC_ANON", receiver_id="TITLE_CO",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="All-cash real estate purchase by anonymous LLC — no beneficial owner disclosed"
        ),
        # Payroll (clear)
        Transaction(
            id="ADV012", amount=4_800.0, sender_id="EMPLOYER_CORP", receiver_id="EMP_4521",
            sender_country="US", receiver_country="US", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Bi-weekly payroll, matches employee records"
        ),
        # Loan repayment (clear)
        Transaction(
            id="ADV013", amount=22_000.0, sender_id="BORROWER_1", receiver_id="BANK_LOAN",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Quarterly loan installment — matches amortization schedule"
        ),
        # Delayed smurfing aggregation (hard to catch)
        Transaction(
            id="ADV014", amount=26_000.0, sender_id="SMRF_HUB", receiver_id="OFFSHORE_1",
            sender_country="US", receiver_country="VG",  # British Virgin Islands
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=4.5, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="SMRF_HUB (received ADV001+ADV002+ADV003) now wires aggregated funds offshore"
        ),
        # High-value legitimate M&A escrow
        Transaction(
            id="ADV015", amount=5_000_000.0, sender_id="ACQ_CORP", receiver_id="ESCROW_AGENT",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=0,
            amount_vs_avg_ratio=12.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="M&A acquisition deposit to regulated escrow agent — legal docs on file"
        ),
    ]
    ground_truth = {
        "ADV001": "investigate",  # structuring leg
        "ADV002": "investigate",  # structuring leg
        "ADV003": "investigate",  # structuring leg
        "ADV004": "investigate",  # trade-based ML — needs expert review
        "ADV005": "clear",        # legit hedge fund margin call
        "ADV006": "block",        # charity front + Yemen + shell
        "ADV007": "block",        # mirror trade + PEP
        "ADV008": "clear",        # normal retail
        "ADV009": "clear",        # normal rent
        "ADV010": "block",        # crypto mixer output
        "ADV011": "investigate",  # anonymous LLC real estate
        "ADV012": "clear",        # payroll
        "ADV013": "clear",        # loan repayment
        "ADV014": "block",        # smurfing integration leg
        "ADV015": "clear",        # legit M&A escrow
    }
    return txns, ground_truth, instructions, 4