# Credit Risk Engine — PD Model, Profit-Optimal Policy & AI Decision Explainer

An end-to-end consumer credit underwriting system built on 1.35M real LendingClub
loans (2007–2018): a leakage-safe probability-of-default model, a profit-driven
approval policy, regulatory-style reason codes, a FastAPI scoring service, and a
Claude-powered agent that explains individual credit decisions in plain language.

```
raw CSV (2.26M loans, 151 cols)
   └─ data_prep      leakage firewall + labeling        → 1.35M resolved loans
       └─ train      LightGBM, out-of-time validation   → PD model (AUC 0.718)
           ├─ business   profit-sweep over thresholds   → approval policy
           ├─ explain    SHAP global + per-loan reasons → reason codes
           └─ api        FastAPI /score                 → PD + decision + reasons
               └─ agent  Claude tool-use loop           → plain-language explanations
```

## Why this project is built the way it is

**1. Leakage is the whole game.** The raw file contains 151 columns, but ~40 of
them describe what happened *after* origination (`recoveries`, `last_pymnt_amnt`,
hardship and settlement fields...). Models trained on them score AUC 0.95+ and are
useless for underwriting. This project whitelists 28 origination-time columns
([src/config.py](src/config.py)) and documents every excluded leakage column.
The honest out-of-time AUC is **0.718** — which is what real credit models look like.

**2. Out-of-time validation, not random K-fold.** The model trains on loans issued
before 2016 (829k) and validates on 2016+ vintages (519k), because that's how the
model would actually be deployed: on future applicants. The validation default rate
(22.4%) is higher than training (18.5%) — a real vintage shift that random splits
would hide.

**3. Probabilities become decisions via economics, not F1.** A PD threshold is
chosen by sweeping approval cutoffs and computing realized portfolio profit
(interest earned on repaid loans vs. 65%-LGD losses on defaults). The profit curve
is strikingly flat above PD ≈ 0.3: risky borrowers pay higher rates that largely
offset their losses. The optimal policy (approve PD < 0.40) approves 93.7% —
very different from what accuracy- or F1-optimal thresholds would suggest.

**4. Explainability is a regulatory requirement, not a nice-to-have.** ECOA
requires lenders to give specific reasons for adverse decisions. The API returns
per-applicant SHAP reason codes, and the agent turns them into language an
applicant could actually understand.

## Results (out-of-time validation, 2016+ vintages, n=519k)

| Metric | Value |
|---|---|
| ROC-AUC | 0.718 |
| PR-AUC (default rate 22.4%) | 0.411 |
| Brier score | 0.158 |

Top global risk drivers (mean |SHAP|): `sub_grade`, `term`, `grade`, `fico`,
`dti`, `annual_inc`. See [reports/shap_summary.png](reports/shap_summary.png).

**Known limitation, deliberately surfaced:** the calibration table
([models/calibration.csv](models/calibration.csv)) shows predicted PDs run below
realized default rates in every decile — the vintage shift means a model trained
pre-2016 underestimates 2016+ risk. In production this is handled by recalibrating
on a rolling recent window; discussed further below.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# macOS: brew install libomp   (LightGBM's OpenMP runtime)

# 1. Get the data (1.67 GB, CC0)
curl -L -o data/raw/accepted_2007_to_2018Q4.csv \
  "https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv"

# 2. Run the pipeline
.venv/bin/python -m src.data_prep     # raw CSV -> clean parquet
.venv/bin/python -m src.train         # train + out-of-time eval
.venv/bin/python -m src.business      # profit sweep -> policy.json
.venv/bin/python -m src.explain       # SHAP summary + global importance

# 3. Serve
.venv/bin/uvicorn api.main:app --port 8000

# 4. Score an applicant
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "loan_amnt": 15000, "term": "36", "int_rate": 13.5, "installment": 509.5,
  "grade": "C", "sub_grade": "C1", "annual_inc": 65000, "dti": 18.2,
  "earliest_cr_line": "Aug-2005", "fico_range_low": 685, "fico_range_high": 689,
  "revol_util": 62.5, "home_ownership": "RENT", "purpose": "debt_consolidation"}'
# -> {"pd": 0.18, "decision": "approve", "threshold": 0.4, "reasons": [...]}

# 5. Talk to the decision-explainer agent (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m agent.explainer_agent
```

Example agent session:

```
you> Score this applicant: $35k over 60 months at 28.9%, grade G4, income $30k,
     FICO 662, 95% revolving utilization, 3 delinquencies in 2 years.
agent> [calls score_applicant] This application would be declined. The model
       estimates a 78% probability of default vs. the 40% policy cutoff. The
       biggest factors: the G4 sub-grade, the 60-month term, and the
       small-business purpose all pushed risk up...
you> What would need to change for an approval?
agent> [re-scores modified applicants to verify] Moving to a 36-month term and
       a smaller amount brings the estimate down to...
```

## Design decisions & honest caveats

- **`int_rate`/`grade`/`sub_grade` are kept as features.** They're known at
  decision time (LendingClub assigns them at listing), but they embed LC's own
  risk model — so this model partly learns to trust LC's pricing. Dropping them
  yields a "from raw bureau attributes only" model; that variant is a natural
  extension.
- **The profit function is simplified**: repaid loans earn full-term interest,
  defaulted loans lose 65% of principal with no credit for installments paid
  before default. Both assumptions are conservative in opposite directions.
- **Survivorship scope**: only *accepted* loans are observable, so the model
  can't correct LendingClub's original approval decisions (reject inference is
  the classic follow-up).
- **Calibration drift** (see above) is shown, not hidden.

## Repo layout

| Path | What it does |
|---|---|
| [src/config.py](src/config.py) | Paths + the origination-time column whitelist / leakage blacklist |
| [src/data_prep.py](src/data_prep.py) | Labeling, filtering, parquet output |
| [src/features.py](src/features.py) | Feature engineering shared by training and serving (no train/serve skew) |
| [src/train.py](src/train.py) | LightGBM + out-of-time eval + calibration table |
| [src/business.py](src/business.py) | Threshold sweep → profit-optimal approval policy |
| [src/explain.py](src/explain.py) | SHAP global importance + per-loan reason codes |
| [api/main.py](api/main.py) | FastAPI `/score`, `/policy`, `/health` |
| [agent/explainer_agent.py](agent/explainer_agent.py) | Claude tool-use agent over the scoring API |

Data: [LendingClub accepted loans 2007–2018Q4](https://huggingface.co/datasets/codesignal/lending-club-loan-accepted) (CC0).
