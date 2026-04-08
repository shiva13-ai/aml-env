"""
AML Environment - Synthetic Transaction Dataset
Each task has a fixed seed for reproducibility.
Ground truth labels: "block" | "investigate" | "clear"
"""
from __future__ import annotations
from typing import Dict, List, Tuple
from .models import Transaction

HIGH_RISK_COUNTRIES = {"IR", "KP", "MM", "YE", "SY", "AF", "SO", "SS"}


def get_task_data(task_name: str) -> Tuple[List[Transaction], Dict[str, str], str, int]:
    if task_name == "triage_basic":
        return _task_basic()
    elif task_name == "triage_network":
        return _task_network()
    elif task_name == "triage_adversarial":
        return _task_adversarial()
    elif task_name == "correspondent_banking":
        return _task_correspondent()
    elif task_name == "sanctions_screening":
        return _task_sanctions()
    elif task_name == "crypto_defi_aml":
        return _task_crypto_defi()
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
            id="TXN001", amount=9_900.0, sender_id="ACC_1001", receiver_id="ACC_2001",
            sender_country="US", receiver_country="US", transaction_type="cash",
            velocity_24h=3, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False, notes="3 cash deposits of $9,900 in 48 hours by same account"
        ),
        Transaction(
            id="TXN002", amount=1_000_000.0, sender_id="ACC_1002", receiver_id="ACC_2002",
            sender_country="IR", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=2,
            amount_vs_avg_ratio=85.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=True, notes="Large wire from sanctioned country, PEP sender, round amount"
        ),
        Transaction(
            id="TXN003", amount=250.0, sender_id="ACC_1003", receiver_id="ACC_2003",
            sender_country="CA", receiver_country="US", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=0.9, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Regular payroll disbursement, consistent with history"
        ),
        Transaction(
            id="TXN004", amount=5_000.0, sender_id="ACC_1004", receiver_id="ACC_2004",
            sender_country="GB", receiver_country="GB", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=0,
            amount_vs_avg_ratio=1.05, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Regular supplier payment, invoice matched"
        ),
        Transaction(
            id="TXN005", amount=87_500.0, sender_id="ACC_1005", receiver_id="ACC_2005",
            sender_country="US", receiver_country="KP", transaction_type="crypto",
            velocity_24h=5, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=22.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="Crypto transfer to sanctioned jurisdiction, shell company receiver"
        ),
    ]
    ground_truth = {
        "TXN001": "investigate",
        "TXN002": "block",
        "TXN003": "clear",
        "TXN004": "clear",
        "TXN005": "block",
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
            sender_country="US", receiver_country="PA", transaction_type="wire",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=8.0, high_risk_country=False,
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
            sender_country="US", receiver_country="PA", transaction_type="wire",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=7.8, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="ACC_C (received from NET001) also wires to same shell"
        ),
        Transaction(
            id="NET009", amount=500_000.0, sender_id="ACC_SHELL1", receiver_id="ACC_K",
            sender_country="PA", receiver_country="CH", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=3,
            amount_vs_avg_ratio=999.0, high_risk_country=False,
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
        "NET001": "investigate",
        "NET002": "investigate",
        "NET003": "investigate",
        "NET004": "block",
        "NET005": "clear",
        "NET006": "block",
        "NET007": "clear",
        "NET008": "block",
        "NET009": "block",
        "NET010": "clear",
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
        Transaction(
            id="ADV004", amount=450_000.0, sender_id="IMPORT_CO", receiver_id="EXPORT_CO",
            sender_country="US", receiver_country="CN", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=3.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Import payment for electronics — invoice provided but price 3x market rate"
        ),
        Transaction(
            id="ADV005", amount=2_500_000.0, sender_id="HEDGE_FUND_1", receiver_id="PRIME_BROKER",
            sender_country="US", receiver_country="GB", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=0.9, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Margin call payment — all docs filed, consistent with fund activity"
        ),
        Transaction(
            id="ADV006", amount=75_000.0, sender_id="CHARITY_FRONT", receiver_id="OVERSEAS_ACCT",
            sender_country="US", receiver_country="YE", transaction_type="wire",
            velocity_24h=3, is_round_number=True, prior_flags=1,
            amount_vs_avg_ratio=5.5, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="NGO claims humanitarian aid — but Yemen + shell receiver + elevated frequency"
        ),
        Transaction(
            id="ADV007", amount=320_000.0, sender_id="BROKER_RU", receiver_id="BROKER_US",
            sender_country="RU", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=2,
            amount_vs_avg_ratio=6.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=True, notes="Equity purchase paired with identical RU sell order — mirror trade signature"
        ),
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
        Transaction(
            id="ADV010", amount=42_000.0, sender_id="CRYPTO_MIXER_OUT", receiver_id="EXCHANGE_1",
            sender_country="--", receiver_country="US", transaction_type="crypto",
            velocity_24h=8, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=15.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Funds exiting known mixer address — blockchain analysis flagged"
        ),
        Transaction(
            id="ADV011", amount=890_000.0, sender_id="LLC_ANON", receiver_id="TITLE_CO",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="All-cash real estate purchase by anonymous LLC — no beneficial owner disclosed"
        ),
        Transaction(
            id="ADV012", amount=4_800.0, sender_id="EMPLOYER_CORP", receiver_id="EMP_4521",
            sender_country="US", receiver_country="US", transaction_type="internal",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Bi-weekly payroll, matches employee records"
        ),
        Transaction(
            id="ADV013", amount=22_000.0, sender_id="BORROWER_1", receiver_id="BANK_LOAN",
            sender_country="US", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False, notes="Quarterly loan installment — matches amortization schedule"
        ),
        Transaction(
            id="ADV014", amount=26_000.0, sender_id="SMRF_HUB", receiver_id="OFFSHORE_1",
            sender_country="US", receiver_country="VG", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=4.5, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False, notes="SMRF_HUB (received ADV001+ADV002+ADV003) now wires aggregated funds offshore"
        ),
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
        "ADV001": "investigate",
        "ADV002": "investigate",
        "ADV003": "investigate",
        "ADV004": "investigate",
        "ADV005": "clear",
        "ADV006": "block",
        "ADV007": "block",
        "ADV008": "clear",
        "ADV009": "clear",
        "ADV010": "block",
        "ADV011": "investigate",
        "ADV012": "clear",
        "ADV013": "clear",
        "ADV014": "block",
        "ADV015": "clear",
    }
    return txns, ground_truth, instructions, 4


# ── TASK 4: Correspondent Banking (Expert) ────────────────────────────────────
def _task_correspondent() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "EXPERT MODE — Correspondent Banking Review. "
        "Your bank acts as a correspondent for 3 foreign respondent banks. "
        "You are reviewing 20 transactions flowing through nested correspondent relationships. "
        "Key risk: you cannot see the ultimate beneficial owners — only the respondent bank. "
        "You must infer risk from: originating jurisdiction, transaction patterns, "
        "respondent bank risk rating, and known typologies (payable-through accounts, "
        "nested correspondents, bulk cash repatriation). "
        "Decisions: 'block', 'investigate', or 'clear'. "
        "Investigation budget: 5. Precision matters — false positives disrupt legitimate trade finance. "
        "This scenario reflects real correspondent banking de-risking decisions made daily at major banks."
    )
    txns = [
        # Payable-through account abuse (respondent bank passes through unknown customers)
        Transaction(
            id="CB001", amount=47_000.0, sender_id="RESP_BANK_A", receiver_id="CORP_US_1",
            sender_country="MM", receiver_country="US", transaction_type="wire",
            velocity_24h=6, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=4.2, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Myanmar respondent bank — 6 wires today via payable-through account, ultimate beneficiaries unknown"
        ),
        Transaction(
            id="CB002", amount=51_000.0, sender_id="RESP_BANK_A", receiver_id="CORP_US_2",
            sender_country="MM", receiver_country="US", transaction_type="wire",
            velocity_24h=6, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=4.5, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Same Myanmar bank, same day — second payable-through wire, different US recipient"
        ),
        # Legitimate correspondent for trade finance
        Transaction(
            id="CB003", amount=220_000.0, sender_id="RESP_BANK_B", receiver_id="IMPORT_US",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Singapore respondent — Letter of Credit settlement for electronics shipment, full docs"
        ),
        Transaction(
            id="CB004", amount=185_000.0, sender_id="RESP_BANK_B", receiver_id="EXPORT_US",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=0.95, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Singapore respondent — trade finance reimbursement, consistent pattern"
        ),
        # Nested correspondent (bank within a bank) — highest risk
        Transaction(
            id="CB005", amount=380_000.0, sender_id="RESP_BANK_C", receiver_id="NESTED_BANK_X",
            sender_country="AE", receiver_country="AF", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=2,
            amount_vs_avg_ratio=7.8, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="UAE respondent forwarding to Afghan nested correspondent — respondent has no AML program on file"
        ),
        # Bulk cash repatriation disguised as trade
        Transaction(
            id="CB006", amount=995_000.0, sender_id="RESP_BANK_C", receiver_id="IMPORT_SHELL",
            sender_country="AE", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=2,
            amount_vs_avg_ratio=18.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=True,
            notes="UAE respondent + PEP-linked shell importer + near $1M + 18x avg — bulk cash repatriation pattern"
        ),
        # Clean international payroll
        Transaction(
            id="CB007", amount=8_200.0, sender_id="RESP_BANK_B", receiver_id="EMP_OVERSEAS",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Singapore respondent — monthly expat payroll, consistent, HR-verified"
        ),
        # Hawala-style round-trip
        Transaction(
            id="CB008", amount=125_000.0, sender_id="RESP_BANK_C", receiver_id="RESP_BANK_C_RETURN",
            sender_country="AE", receiver_country="AE", transaction_type="wire",
            velocity_24h=2, is_round_number=True, prior_flags=2,
            amount_vs_avg_ratio=9.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Round-trip: funds leave AE respondent and return to same bank same day — hawala indicator"
        ),
        # Sanctions evasion via front company
        Transaction(
            id="CB009", amount=62_000.0, sender_id="FRONT_CO_TR", receiver_id="RESP_BANK_A",
            sender_country="TR", receiver_country="MM", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=6.5, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False,
            notes="Turkish front company → Myanmar bank — Turkey-Myanmar route used for Iran sanctions evasion"
        ),
        # Legitimate multilateral development bank disbursement
        Transaction(
            id="CB010", amount=5_000_000.0, sender_id="WORLD_BANK_DISBURSE", receiver_id="GOV_AGENCY",
            sender_country="US", receiver_country="KE", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="World Bank project disbursement to Kenyan government ministry — fully documented grant"
        ),
        # Crypto-to-correspondent integration
        Transaction(
            id="CB011", amount=290_000.0, sender_id="CRYPTO_EXCHANGE_UNREG", receiver_id="RESP_BANK_C",
            sender_country="--", receiver_country="AE", transaction_type="crypto",
            velocity_24h=3, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=12.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Unregistered crypto exchange → UAE respondent — unlicensed VASP feeding correspondent account"
        ),
        # Normal trade settlement
        Transaction(
            id="CB012", amount=44_500.0, sender_id="RESP_BANK_B", receiver_id="PHARMA_IMPORT",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.05, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Pharmaceutical import payment — FDA-registered importer, consistent monthly pattern"
        ),
        # Over-invoiced art (TBML via cultural goods)
        Transaction(
            id="CB013", amount=780_000.0, sender_id="ART_DEALER_SHELL", receiver_id="RESP_BANK_C",
            sender_country="CH", receiver_country="AE", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=True,
            notes="Swiss art dealer shell → UAE: artwork valued at $50k sold for $780k — classic TBML via cultural goods"
        ),
        # Pension fund disbursement (clear)
        Transaction(
            id="CB014", amount=2_100.0, sender_id="RESP_BANK_B", receiver_id="RETIREE_US",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="Monthly pension disbursement to US retiree from Singapore fund — 8 year history"
        ),
        # Multiple structuring via correspondent
        Transaction(
            id="CB015", amount=9600.0, sender_id="RESP_BANK_A", receiver_id="CASH_INTENSIVE_BIZ",
            sender_country="MM", receiver_country="US", transaction_type="cash",
            velocity_24h=5, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=3.0, high_risk_country=True,
            structuring_indicator=True, shell_company_indicator=False,
            pep_involved=False,
            notes="5th sub-$10k cash transaction via Myanmar respondent to same US cash business today"
        ),
        # Diplomatic pouch abuse (PEP + high risk)
        Transaction(
            id="CB016", amount=450_000.0, sender_id="DIPLOMAT_ACCT", receiver_id="REAL_ESTATE_LLC",
            sender_country="AF", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=3,
            amount_vs_avg_ratio=35.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=True,
            notes="Afghan diplomat account → US real estate LLC — diplomat salary inconsistent with amount"
        ),
        # Routine treasury operation (clear)
        Transaction(
            id="CB017", amount=10_000_000.0, sender_id="RESP_BANK_B", receiver_id="NOSTRO_ACCOUNT",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=True, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="End-of-day nostro reconciliation by Singapore respondent — standard treasury operation"
        ),
        # Misuse of development aid corridor
        Transaction(
            id="CB018", amount=340_000.0, sender_id="RESP_BANK_A", receiver_id="NGO_SHELL",
            sender_country="MM", receiver_country="US", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=1,
            amount_vs_avg_ratio=28.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True,
            pep_involved=False,
            notes="Myanmar respondent → US NGO shell: NGO has no web presence, incorporated 2 weeks ago"
        ),
        # Clean interbank FX settlement
        Transaction(
            id="CB019", amount=1_250_000.0, sender_id="RESP_BANK_B", receiver_id="FX_DESK_US",
            sender_country="SG", receiver_country="US", transaction_type="wire",
            velocity_24h=2, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=False,
            notes="FX spot settlement — SWIFT MT202 confirmed, matches confirmed trade ticket"
        ),
        # State-owned enterprise used for sanctions evasion
        Transaction(
            id="CB020", amount=2_800_000.0, sender_id="SOE_FRONT", receiver_id="RESP_BANK_C",
            sender_country="RU", receiver_country="AE", transaction_type="wire",
            velocity_24h=1, is_round_number=False, prior_flags=0,
            amount_vs_avg_ratio=55.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False,
            pep_involved=True,
            notes="Russian SOE (state-owned enterprise) → UAE respondent: 55x avg, post-sanctions activity, PEP board members"
        ),
    ]
    ground_truth = {
        "CB001": "investigate",   # payable-through, Myanmar
        "CB002": "block",         # repeated payable-through same day
        "CB003": "clear",         # legit LC settlement
        "CB004": "clear",         # legit trade finance
        "CB005": "block",         # nested correspondent, no AML program
        "CB006": "block",         # bulk cash + PEP + shell
        "CB007": "clear",         # legit payroll
        "CB008": "block",         # hawala round-trip
        "CB009": "investigate",   # sanctions evasion route
        "CB010": "clear",         # World Bank disbursement
        "CB011": "block",         # unlicensed VASP
        "CB012": "clear",         # normal pharma trade
        "CB013": "block",         # TBML via art
        "CB014": "clear",         # pension
        "CB015": "block",         # structuring via correspondent
        "CB016": "block",         # diplomat + real estate + PEP
        "CB017": "clear",         # nostro reconciliation
        "CB018": "block",         # new shell NGO
        "CB019": "clear",         # FX settlement
        "CB020": "block",         # Russian SOE sanctions evasion
    }
    return txns, ground_truth, instructions, 5


# ── TASK 5: Sanctions Screening (Medium-Hard) ─────────────────────────────────
def _task_sanctions() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "SANCTIONS SCREENING: You are screening 12 transactions against OFAC SDN list, "
        "UN Security Council sanctions, and EU consolidated sanctions list. "
        "Some transactions involve entities that are close to — but not exactly on — the sanctions list "
        "(name variations, subsidiaries, front companies). "
        "Decisions: 'block' (sanctions match confirmed), 'investigate' (possible match, needs human review), "
        "'clear' (no sanctions concern). "
        "Investigation budget: 4. "
        "False positives in sanctions screening have serious diplomatic consequences — be precise."
    )
    txns = [
        Transaction(
            id="SAN001", amount=2_500_000.0, sender_id="ROSNEFT_SUBSIDIARY", receiver_id="TRADING_CO",
            sender_country="RU", receiver_country="AE",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=True,
            notes="Rosneft Trading S.A. — direct subsidiary of OFAC-designated Rosneft. Wire to UAE trading company."
        ),
        Transaction(
            id="SAN002", amount=45_000.0, sender_id="JOHN_SMITH_COMMON", receiver_id="RETAIL_MERCHANT",
            sender_country="US", receiver_country="US",
            transaction_type="internal", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="John Smith — common name with 3,400 SDN matches. This individual is a US citizen, DOB 1985, SSN verified."
        ),
        Transaction(
            id="SAN003", amount=180_000.0, sender_id="BANK_MELLI_IRAN", receiver_id="EURO_CORP",
            sender_country="IR", receiver_country="DE",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=3.5, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Bank Melli Iran — directly on OFAC SDN list since 2007. Payment for 'consulting services'."
        ),
        Transaction(
            id="SAN004", amount=12_000.0, sender_id="CUBA_DIPLOMAT_ACCT", receiver_id="HOTEL_PAYMENT",
            sender_country="CU", receiver_country="US",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=True,
            notes="Cuban diplomatic mission — OFAC general license GL-3 covers Cuban diplomat transactions for official duties."
        ),
        Transaction(
            id="SAN005", amount=890_000.0, sender_id="NOVATEK_GAS", receiver_id="LNG_BUYER",
            sender_country="RU", receiver_country="JP",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=1, amount_vs_avg_ratio=2.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Novatek — on EU sanctions list but has OFAC wind-down license for LNG energy transactions until Q4."
        ),
        Transaction(
            id="SAN006", amount=320_000.0, sender_id="KIM_CHOL_CO", receiver_id="MACHINERY_IMPORT",
            sender_country="KP", receiver_country="CN",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=8.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=True, pep_involved=True,
            notes="Kim Chol — on UN DPRK sanctions list. Front company purchasing dual-use machinery via China."
        ),
        Transaction(
            id="SAN007", amount=7_500.0, sender_id="HUMANITARIAN_NGO", receiver_id="SYRIA_FIELD_OPS",
            sender_country="CH", receiver_country="SY",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="ICRC — International Committee of Red Cross. OFAC General License covers humanitarian NGO ops in Syria."
        ),
        Transaction(
            id="SAN008", amount=55_000.0, sender_id="WAGNER_GROUP_SHELL", receiver_id="ARMS_BROKER",
            sender_country="ZA", receiver_country="LY",
            transaction_type="wire", velocity_24h=2, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=7.5, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True, pep_involved=False,
            notes="South Africa shell co linked to Wagner Group (OFAC-designated). Payment to Libya arms broker."
        ),
        Transaction(
            id="SAN009", amount=3_200.0, sender_id="IRAN_STUDENT_ASSOC", receiver_id="UNIVERSITY_FEE",
            sender_country="IR", receiver_country="US",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Iranian student — OFAC GL covers tuition payments by Iranian nationals studying in the US."
        ),
        Transaction(
            id="SAN010", amount=4_200_000.0, sender_id="GAZPROM_NEFT", receiver_id="ENERGY_TRADER",
            sender_country="RU", receiver_country="TR",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=True,
            notes="Gazprom Neft — OFAC SDN designated. No active license. Turkey energy trade does not exempt SDN entities."
        ),
        Transaction(
            id="SAN011", amount=28_000.0, sender_id="SIMILAR_NAME_CORP", receiver_id="SUPPLIER_EU",
            sender_country="AE", receiver_country="FR",
            transaction_type="wire", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.2, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="'Al Quds Trading LLC' — similar name to SDN-listed 'Al Quds Brigades' but different entity. UAE trade reg verified."
        ),
        Transaction(
            id="SAN012", amount=650_000.0, sender_id="HAMAS_FRONT_CHARITY", receiver_id="WEST_BANK_ACCT",
            sender_country="QA", receiver_country="PS",
            transaction_type="wire", velocity_24h=3, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=12.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True, pep_involved=False,
            notes="Qatar charity — FinCEN advisory links this specific org to Hamas financing. 3 prior SAR filings."
        ),
    ]
    ground_truth = {
        "SAN001": "block",        # Rosneft subsidiary = designated
        "SAN002": "clear",        # common name, verified US citizen
        "SAN003": "block",        # Bank Melli Iran = directly on SDN
        "SAN004": "clear",        # Cuban diplomat + valid OFAC GL
        "SAN005": "investigate",  # EU-listed but has wind-down license — needs review
        "SAN006": "block",        # UN DPRK + dual-use machinery
        "SAN007": "clear",        # ICRC + valid humanitarian GL
        "SAN008": "block",        # Wagner Group shell
        "SAN009": "clear",        # Iranian student + valid GL for tuition
        "SAN010": "block",        # Gazprom Neft SDN, no license
        "SAN011": "investigate",  # similar name — needs human OFAC review
        "SAN012": "block",        # Hamas-linked charity + FinCEN advisory
    }
    return txns, ground_truth, instructions, 4


# ── TASK 6: Crypto DeFi AML (Expert) ─────────────────────────────────────────
def _task_crypto_defi() -> Tuple[List[Transaction], Dict[str, str], str, int]:
    instructions = (
        "CRYPTO/DEFI AML EXPERT MODE: You are reviewing 18 on-chain and off-chain transactions "
        "involving cryptocurrency, DeFi protocols, NFTs, and crypto exchanges. "
        "This task requires knowledge of blockchain-specific typologies: "
        "mixer/tumbler usage, cross-chain bridge hopping, peel chains, NFT wash trading, "
        "rug pulls, darknet market off-ramps, and ransomware payment flows. "
        "Decisions: 'block', 'investigate', or 'clear'. "
        "Investigation budget: 5. "
        "Blockchain analytics (Chainalysis/Elliptic) flags are included in notes where available."
    )
    txns = [
        Transaction(
            id="DFI001", amount=85_000.0, sender_id="TORNADO_CASH_OUTPUT", receiver_id="CEX_DEPOSIT",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Chainalysis: funds exiting Tornado Cash (OFAC-designated mixer). Direct deposit to centralized exchange."
        ),
        Transaction(
            id="DFI002", amount=12_500.0, sender_id="DEFI_WALLET_LEGIT", receiver_id="CEX_WITHDRAW",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=2, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Uniswap LP withdrawal — on-chain history shows legitimate DeFi yield farming for 14 months. KYC verified."
        ),
        Transaction(
            id="DFI003", amount=340_000.0, sender_id="RANSOMWARE_WALLET", receiver_id="MIXING_SERVICE",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=5, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="FBI alert: wallet linked to Lazarus Group ransomware campaign. Funds moving to unlicensed mixer."
        ),
        Transaction(
            id="DFI004", amount=2_800.0, sender_id="NFT_TRADER_A", receiver_id="NFT_TRADER_B",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=8, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="NFT trades between same two wallets 8 times today at incrementing prices — classic wash trading pattern."
        ),
        Transaction(
            id="DFI005", amount=920_000.0, sender_id="BRIDGE_HOP_WALLET", receiver_id="CLEAN_WALLET",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=3, is_round_number=False,
            prior_flags=1, amount_vs_avg_ratio=15.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Elliptic: funds bridged ETH→BSC→Polygon→Arbitrum in 2 hours — cross-chain layering to obscure trail."
        ),
        Transaction(
            id="DFI006", amount=45_000.0, sender_id="STAKING_REWARDS", receiver_id="PERSONAL_WALLET",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Ethereum staking rewards withdrawal — Lido Finance, 18 month staking history, tax docs filed."
        ),
        Transaction(
            id="DFI007", amount=1_200_000.0, sender_id="RUGPULL_DEPLOYER", receiver_id="MIXER_WALLET",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Smart contract deployer drained $1.2M from liquidity pool (rug pull) and immediately routing to mixer."
        ),
        Transaction(
            id="DFI008", amount=8_900.0, sender_id="PEEL_CHAIN_END", receiver_id="EXCHANGE_ACCT",
            sender_country="--", receiver_country="GB",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=1, amount_vs_avg_ratio=3.0, high_risk_country=False,
            structuring_indicator=True, shell_company_indicator=False, pep_involved=False,
            notes="Chainalysis: end of 47-hop peel chain originating from darknet market wallet. Small amount to avoid detection."
        ),
        Transaction(
            id="DFI009", amount=5_500.0, sender_id="GAMING_WALLET", receiver_id="PLAYER_ACCT",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Axie Infinity play-to-earn withdrawal — legitimate gaming income, consistent weekly pattern for 8 months."
        ),
        Transaction(
            id="DFI010", amount=2_100_000.0, sender_id="LAZARUS_BRIDGE_OUT", receiver_id="OTC_DESK",
            sender_country="KP", receiver_country="--",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=999.0, high_risk_country=True,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="UN Panel: DPRK Lazarus Group wallet. Funds from Harmony Bridge hack. Moving to unlicensed OTC desk."
        ),
        Transaction(
            id="DFI011", amount=33_000.0, sender_id="PRIVACY_COIN_MONERO", receiver_id="CEX_XMR",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=2, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=2.5, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Monero (XMR) deposit to exchange — privacy coin, untraceable. No prior flags but 2.5x account average."
        ),
        Transaction(
            id="DFI012", amount=15_000.0, sender_id="CRYPTO_SALARY_ACCT", receiver_id="PERSONAL_BANK",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Monthly crypto salary from verified tech employer (Coinbase payroll) — W2 on file, consistent pattern."
        ),
        Transaction(
            id="DFI013", amount=780_000.0, sender_id="DARKNET_MARKET_WALLET", receiver_id="CHAIN_HOP",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=4, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Chainalysis High Risk: wallet directly linked to Hydra darknet marketplace. Multi-chain hop initiated."
        ),
        Transaction(
            id="DFI014", amount=4_200.0, sender_id="AIRDROP_RECIPIENT", receiver_id="DEX_SWAP",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Uniswap airdrop token claim and immediate swap to ETH — standard DeFi user behavior, no risk flags."
        ),
        Transaction(
            id="DFI015", amount=95_000.0, sender_id="SANCTIONED_EXCHANGE", receiver_id="PERSONAL_WALLET",
            sender_country="--", receiver_country="RU",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=2, amount_vs_avg_ratio=8.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Withdrawal from Garantex — Russian crypto exchange directly on OFAC SDN list since April 2022."
        ),
        Transaction(
            id="DFI016", amount=220_000.0, sender_id="NFT_WASH_ORCHESTRATOR", receiver_id="ART_PLATFORM",
            sender_country="--", receiver_country="US",
            transaction_type="crypto", velocity_24h=6, is_round_number=False,
            prior_flags=1, amount_vs_avg_ratio=20.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=True, pep_involved=False,
            notes="6 NFT sales today between controlled wallets — blockchain analysis shows same entity controls buyer and seller wallets."
        ),
        Transaction(
            id="DFI017", amount=3_400.0, sender_id="P2P_TRADER_VERIFIED", receiver_id="USER_WALLET",
            sender_country="NG", receiver_country="US",
            transaction_type="crypto", velocity_24h=1, is_round_number=False,
            prior_flags=0, amount_vs_avg_ratio=1.1, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Binance P2P verified trader — remittance to family in US. KYC Level 2 completed, consistent pattern."
        ),
        Transaction(
            id="DFI018", amount=560_000.0, sender_id="DEFI_PROTOCOL_HACK", receiver_id="TUMBLER_NETWORK",
            sender_country="--", receiver_country="--",
            transaction_type="crypto", velocity_24h=2, is_round_number=False,
            prior_flags=3, amount_vs_avg_ratio=999.0, high_risk_country=False,
            structuring_indicator=False, shell_company_indicator=False, pep_involved=False,
            notes="Rekt.news confirmed: $560k from Euler Finance exploit. Hacker moving funds through tumbler network."
        ),
    ]
    ground_truth = {
        "DFI001": "block",        # Tornado Cash = OFAC designated
        "DFI002": "clear",        # legit DeFi yield farming
        "DFI003": "block",        # Lazarus ransomware + mixer
        "DFI004": "investigate",  # NFT wash trading — needs review
        "DFI005": "investigate",  # cross-chain layering — suspicious
        "DFI006": "clear",        # legit staking rewards
        "DFI007": "block",        # rug pull + mixer
        "DFI008": "block",        # darknet peel chain
        "DFI009": "clear",        # legit gaming income
        "DFI010": "block",        # Lazarus Group DPRK
        "DFI011": "investigate",  # privacy coin — elevated risk
        "DFI012": "clear",        # legit crypto salary
        "DFI013": "block",        # Hydra darknet market
        "DFI014": "clear",        # normal airdrop claim
        "DFI015": "block",        # Garantex = OFAC SDN
        "DFI016": "investigate",  # NFT wash trading orchestrator
        "DFI017": "clear",        # verified P2P remittance
        "DFI018": "block",        # DeFi exploit funds
    }
    return txns, ground_truth, instructions, 5