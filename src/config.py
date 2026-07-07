"""Central configuration: paths and column definitions.

The column lists below encode the most important modeling decision in this
project: what a lender actually knows at origination time. Everything in
LEAKAGE_COLS describes what happened *after* the loan was issued (payments,
recoveries, hardship plans...). Including any of them makes default trivially
predictable and the model useless for underwriting.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"
PROCESSED_PARQUET = PROJECT_ROOT / "data" / "processed" / "loans.parquet"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TARGET = "is_default"

# Loan outcomes we can label. Loans still "Current" or "Late" are unresolved
# and excluded — we don't yet know how they end.
RESOLVED_STATUSES = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Columns known at origination — the only ones the model may see.
# Note: int_rate/grade/sub_grade ARE available at decision time (LendingClub
# assigns them at listing), but they embed LC's own risk model. We keep them
# and discuss the implications in the README.
ORIGINATION_COLS = [
    "loan_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "fico_range_low",
    "fico_range_high",
    "inq_last_6mths",
    "mths_since_last_delinq",
    "mths_since_last_record",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "application_type",
    "mort_acc",
    "pub_rec_bankruptcies",
]

# Needed for labeling and the out-of-time split, never used as features.
META_COLS = ["issue_d", "loan_status"]

# Post-origination columns present in the raw file. We load none of them
# (we whitelist via ORIGINATION_COLS), but the list is kept explicit because
# it documents *why* the whitelist exists and is checked in tests.
LEAKAGE_COLS = [
    "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
    "recoveries", "collection_recovery_fee", "last_pymnt_d",
    "last_pymnt_amnt", "next_pymnt_d", "last_credit_pull_d",
    "last_fico_range_high", "last_fico_range_low",
    "debt_settlement_flag", "debt_settlement_flag_date",
    "settlement_status", "settlement_date", "settlement_amount",
    "settlement_percentage", "settlement_term",
    "hardship_flag", "hardship_type", "hardship_reason",
    "hardship_status", "hardship_amount", "hardship_start_date",
    "hardship_end_date", "payment_plan_start_date", "hardship_length",
    "hardship_dpd", "hardship_loan_status",
    "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "pymnt_plan",
]

# emp_length is NOT here: it's mapped to an ordinal numeric scale (0-10)
# in features.py, so the model treats it as a number.
CATEGORICAL_COLS = [
    "term",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
    "application_type",
]

# LendingClub's own pricing outputs. The "bureau" model variant excludes
# them to answer: how well can we underwrite from raw applicant/bureau
# attributes alone, without leaning on LC's risk model?
# (installment is included: it's a function of loan_amnt, term and int_rate,
# so it leaks the assigned rate.)
PRICING_COLS = ["grade", "sub_grade", "int_rate", "installment"]

# Engineered in features.py
ENGINEERED_COLS = ["fico", "credit_history_years"]
DROPPED_AFTER_ENGINEERING = ["fico_range_low", "fico_range_high", "earliest_cr_line"]

# Out-of-time split: train on loans issued before this date, validate on the rest.
# 2016+ holds ~30% of resolved loans and simulates deploying on future vintages.
SPLIT_DATE = "2016-01-01"
