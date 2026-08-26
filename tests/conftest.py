"""Hermetic fixtures for the MCP server tests.

`models/` and `data/processed/` are gitignored, so a fresh clone has no trained
model and no loan book. These fixtures build a miniature but *real* version of
both — a genuine LightGBM booster, a genuine isotonic calibrator, a genuine
parquet loan book — into a temp directory, then point `src.config` at it.

That keeps the suite deterministic (fixed seeds, no dependence on which vintage
of the 1.35M file happens to be on disk), fast, and runnable with no network and
no Databricks/Spark warehouse. Every code path under test is the production
path; only the artifacts are small.
"""

from __future__ import annotations

import asyncio
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from mcp.client import Client
from sklearn.isotonic import IsotonicRegression

from src import config, features

# Category pools. Test applicants must draw from these: LightGBM stores the
# pandas category levels it saw at training time and remaps at predict time,
# so a level absent from the fixture would arrive as missing.
GRADES = list("ABCDEFG")
SUB_GRADES = [f"{g}{i}" for g in GRADES for i in range(1, 6)]
PURPOSES = ["debt_consolidation", "credit_card", "home_improvement", "small_business", "other"]
STATES = ["CA", "NY", "TX", "FL", "IL"]
HOME = ["RENT", "MORTGAGE", "OWN"]
VERIFICATION = ["Verified", "Source Verified", "Not Verified"]
APP_TYPE = ["Individual", "Joint App"]
EMP_LENGTH = ["< 1 year", "1 year", "3 years", "5 years", "10+ years"]

N_ROWS = 4000
SEED = 20260828

# Matches the production policy in models/policy.json. `test_production_policy_
# threshold_is_still_0475` asserts the real file still agrees when it exists.
POLICY = {
    "approve_below_pd": 0.475,
    "lgd_assumption": 0.65,
    "expected_approval_rate": 0.9307822519385673,
    "pd_is_calibrated": True,
}


def _synthetic_raw(n: int = N_ROWS, seed: int = SEED) -> pd.DataFrame:
    """Raw-shaped loans: every origination column, plus issue_d and the target."""
    rng = np.random.default_rng(seed)
    grade_idx = rng.integers(0, len(GRADES), n)
    fico_low = np.clip(760 - grade_idx * 22 + rng.normal(0, 18, n), 620, 845).round(0)
    term = rng.choice(["36", "60"], n, p=[0.72, 0.28])

    raw = pd.DataFrame({
        "loan_amnt": rng.integers(1000, 40000, n).astype(float),
        "term": term,
        "int_rate": (6.0 + grade_idx * 3.4 + rng.normal(0, 1.1, n)).round(2),
        "installment": rng.uniform(30, 1400, n).round(2),
        "grade": [GRADES[i] for i in grade_idx],
        "sub_grade": [f"{GRADES[i]}{rng.integers(1, 6)}" for i in grade_idx],
        "emp_length": rng.choice(EMP_LENGTH, n),
        "home_ownership": rng.choice(HOME, n),
        "annual_inc": rng.lognormal(11.0, 0.55, n).round(0),
        "verification_status": rng.choice(VERIFICATION, n),
        "purpose": rng.choice(PURPOSES, n),
        "addr_state": rng.choice(STATES, n),
        "dti": np.clip(rng.normal(18, 8, n), 0, 60).round(2),
        "delinq_2yrs": rng.poisson(0.3, n).astype(float),
        "earliest_cr_line": [
            f"{m}-{y}" for m, y in zip(
                rng.choice(["Jan", "Apr", "Aug", "Nov"], n),
                rng.integers(1990, 2012, n),
            )
        ],
        "fico_range_low": fico_low,
        "fico_range_high": fico_low + 4,
        "inq_last_6mths": rng.poisson(0.7, n).astype(float),
        "mths_since_last_delinq": rng.choice([np.nan, 12.0, 30.0, 60.0], n),
        "mths_since_last_record": rng.choice([np.nan, 24.0, 80.0], n),
        "open_acc": rng.integers(2, 30, n).astype(float),
        "pub_rec": rng.poisson(0.1, n).astype(float),
        "revol_bal": rng.uniform(0, 60000, n).round(0),
        "revol_util": np.clip(rng.normal(50, 24, n), 0, 130).round(1),
        "total_acc": rng.integers(4, 60, n).astype(float),
        "application_type": rng.choice(APP_TYPE, n, p=[0.9, 0.1]),
        "mort_acc": rng.integers(0, 5, n).astype(float),
        "pub_rec_bankruptcies": rng.poisson(0.05, n).astype(float),
        "issue_d": pd.to_datetime("2014-01-01") + pd.to_timedelta(
            rng.integers(0, 4 * 365, n), unit="D"
        ),
    })

    # A learnable signal, so the miniature booster ranks better than chance and
    # the reason codes point somewhere sensible.
    logit = (
        -2.6
        + 0.30 * grade_idx
        + 0.55 * (term == "60")
        + 0.020 * (raw["revol_util"].to_numpy() - 50)
        + 0.020 * (raw["dti"].to_numpy() - 18)
        - 0.011 * (fico_low - 700)
    )
    p = 1 / (1 + np.exp(-logit))
    raw[config.TARGET] = (np.random.default_rng(seed + 1).random(n) < p).astype(int)
    return raw


def _build_artifacts(model_dir, parquet_path) -> None:
    raw = _synthetic_raw()
    # The production feature transform, not a test-only copy.
    engineered = features.engineer(raw.drop(columns=[config.TARGET]), issue_date=raw["issue_d"])
    engineered[config.TARGET] = raw[config.TARGET].to_numpy()
    engineered["issue_d"] = raw["issue_d"].to_numpy()

    feat_cols = features.feature_columns()
    split = pd.Timestamp("2016-01-01")
    train = engineered[engineered["issue_d"] < split]
    calib = engineered[engineered["issue_d"] >= split]

    booster = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_data_in_leaf": 40,
            "verbosity": -1,
            "seed": 42,
            "deterministic": True,
            "num_threads": 1,
        },
        lgb.Dataset(train[feat_cols], label=train[config.TARGET]),
        num_boost_round=60,
    )

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(booster.predict(calib[feat_cols]), calib[config.TARGET])

    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_dir / "pd_model.txt"))
    joblib.dump(iso, model_dir / "calibrator.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(feat_cols, indent=2))
    (model_dir / "policy.json").write_text(json.dumps(POLICY, indent=2))

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    engineered.to_parquet(parquet_path, index=False)


@pytest.fixture(scope="session", autouse=True)
def artifacts(tmp_path_factory):
    """Point src.config at a miniature model + loan book built in a temp dir."""
    root = tmp_path_factory.mktemp("credit_risk_artifacts")
    model_dir = root / "models"
    parquet_path = root / "data" / "processed" / "loans.parquet"
    _build_artifacts(model_dir, parquet_path)

    mp = pytest.MonkeyPatch()
    mp.setattr(config, "MODEL_DIR", model_dir)
    mp.setattr(config, "PROCESSED_PARQUET", parquet_path)
    yield {"model_dir": model_dir, "parquet": parquet_path, "policy": POLICY}
    mp.undo()


@pytest.fixture
def call():
    """Run a coroutine against an in-memory MCP client connected to our server.

    No transport, no subprocess, no socket: `Client` speaks to the `MCPServer`
    object directly, which is what makes the suite hermetic.
    """
    from mcp_server.server import mcp

    def _call(fn):
        async def _run():
            async with Client(mcp) as client:
                return await fn(client)

        return asyncio.run(_run())

    return _call
