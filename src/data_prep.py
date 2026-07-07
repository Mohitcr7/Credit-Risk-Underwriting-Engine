"""Raw CSV -> clean modeling table.

Reads only origination-time columns (see config.ORIGINATION_COLS — this is
the leakage firewall), labels resolved loans, applies feature engineering,
and writes a parquet ready for training.

Run:  python -m src.data_prep
"""

import pandas as pd

from src import config, features


def load_raw() -> pd.DataFrame:
    usecols = config.ORIGINATION_COLS + config.META_COLS
    print(f"Reading {config.RAW_CSV.name} ({len(usecols)} of ~151 columns)...")
    df = pd.read_csv(config.RAW_CSV, usecols=usecols, low_memory=False)
    print(f"  {len(df):,} rows loaded")
    return df


def label_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["loan_status"].isin(config.RESOLVED_STATUSES)].copy()
    df[config.TARGET] = df["loan_status"].map(config.RESOLVED_STATUSES).astype("int8")
    print(f"  {len(df):,} resolved loans "
          f"(default rate {df[config.TARGET].mean():.1%})")

    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df = df.dropna(subset=["issue_d"])

    # Basic sanity filters — documented data-quality decisions.
    df = df[df["annual_inc"].between(0, 5_000_000)]  # drop absurd incomes
    df = df[(df["dti"] >= 0) | df["dti"].isna()]
    return df


def main() -> None:
    df = load_raw()
    df = label_and_filter(df)
    df = features.engineer(df, issue_date=df["issue_d"])

    keep = features.feature_columns() + [config.TARGET, "issue_d"]
    df = df[keep]

    config.PROCESSED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.PROCESSED_PARQUET, index=False)
    print(f"Wrote {config.PROCESSED_PARQUET} "
          f"({len(df):,} rows x {len(df.columns)} cols)")


if __name__ == "__main__":
    main()
