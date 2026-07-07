"""Feature engineering shared by training and the scoring API.

Every transform here must be computable for a single applicant at request
time — no target statistics, no dataset-level aggregates.
"""

import numpy as np
import pandas as pd

from src import config

EMP_LENGTH_MAP = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}


def _to_numeric_pct(series: pd.Series) -> pd.Series:
    """Handle both numeric and '13.5%'-style string encodings."""
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype("str").str.rstrip("%")
    return pd.to_numeric(series, errors="coerce")


def _parse_term(series: pd.Series) -> pd.Series:
    """' 36 months' or '36' -> '36'. Kept categorical: only two values exist."""
    return series.astype("str").str.extract(r"(\d+)")[0]


def engineer(df: pd.DataFrame, issue_date: pd.Series | None = None) -> pd.DataFrame:
    """Apply origination-time feature transforms.

    issue_date is the loan issue date (or application date when scoring live),
    used only to compute credit history length.
    """
    df = df.copy()

    df["int_rate"] = _to_numeric_pct(df["int_rate"])
    df["revol_util"] = _to_numeric_pct(df["revol_util"])
    df["term"] = _parse_term(df["term"])
    df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP).astype("float")

    df["fico"] = (df["fico_range_low"] + df["fico_range_high"]) / 2

    earliest = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    if issue_date is None:
        issue_date = pd.Timestamp.now()
    df["credit_history_years"] = (
        (pd.to_datetime(issue_date) - earliest).dt.days / 365.25
    )

    df = df.drop(columns=config.DROPPED_AFTER_ENGINEERING)

    for col in config.CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    return df


def feature_columns() -> list[str]:
    """Final model feature list, in a stable order."""
    cols = [c for c in config.ORIGINATION_COLS if c not in config.DROPPED_AFTER_ENGINEERING]
    return cols + config.ENGINEERED_COLS
