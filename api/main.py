"""Credit-risk scoring API.

POST /score takes an applicant's origination-time attributes and returns:
  - pd: calibrated probability of default
  - decision: approve/decline against the profit-optimal policy threshold
  - reasons: top SHAP drivers (regulatory-style reason codes)

Run:  uvicorn api.main:app --reload
"""

import json
from contextlib import asynccontextmanager

import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src import config, explain, features

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = lgb.Booster(model_file=str(config.MODEL_DIR / "pd_model.txt"))
    state["policy"] = json.loads((config.MODEL_DIR / "policy.json").read_text())
    state["feat_cols"] = json.loads((config.MODEL_DIR / "feature_columns.json").read_text())
    calibrator_path = config.MODEL_DIR / "calibrator.pkl"
    state["calibrator"] = joblib.load(calibrator_path) if calibrator_path.exists() else None
    yield
    state.clear()


app = FastAPI(title="Credit Risk Scoring API", lifespan=lifespan)


class Applicant(BaseModel):
    """Origination-time attributes. Fields mirror the LendingClub schema."""

    loan_amnt: float = Field(..., gt=0, description="Requested loan amount, USD")
    term: str = Field("36", description="'36' or '60' (months)")
    int_rate: float = Field(..., description="Offered interest rate, e.g. 13.5")
    installment: float = Field(..., gt=0)
    grade: str = Field(..., description="LC grade A-G")
    sub_grade: str = Field(..., description="e.g. B3")
    emp_length: str | None = Field(None, description="e.g. '10+ years'")
    home_ownership: str = "RENT"
    annual_inc: float = Field(..., ge=0)
    verification_status: str = "Not Verified"
    purpose: str = "debt_consolidation"
    addr_state: str = "CA"
    dti: float | None = None
    delinq_2yrs: float | None = 0
    earliest_cr_line: str = Field(..., description="e.g. 'Aug-2005'")
    fico_range_low: float = Field(..., ge=300, le=850)
    fico_range_high: float = Field(..., ge=300, le=850)
    inq_last_6mths: float | None = 0
    mths_since_last_delinq: float | None = None
    mths_since_last_record: float | None = None
    open_acc: float | None = None
    pub_rec: float | None = 0
    revol_bal: float | None = None
    revol_util: float | None = None
    total_acc: float | None = None
    application_type: str = "Individual"
    mort_acc: float | None = None
    pub_rec_bankruptcies: float | None = 0


class ScoreResponse(BaseModel):
    pd: float
    decision: str
    threshold: float
    reasons: list[dict]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/policy")
def policy():
    return state["policy"]


@app.post("/score", response_model=ScoreResponse)
def score(applicant: Applicant):
    row = pd.DataFrame([applicant.model_dump()])
    row = features.engineer(row)  # credit history measured against today
    X = row[state["feat_cols"]].copy()
    # None-valued optional fields arrive as object columns; LightGBM needs numeric.
    for col in X.columns:
        if col not in config.CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    prob = float(state["model"].predict(X)[0])
    if state["calibrator"] is not None:
        prob = float(state["calibrator"].predict([prob])[0])
    threshold = state["policy"]["approve_below_pd"]
    reasons = explain.reason_codes(state["model"], X)[0]

    return ScoreResponse(
        pd=round(prob, 4),
        decision="approve" if prob < threshold else "decline",
        threshold=threshold,
        reasons=reasons,
    )
