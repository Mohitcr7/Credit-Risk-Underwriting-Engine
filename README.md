![Credit Risk Underwriting Engine — architecture: raw loan tape into bronze Delta, through a Unity Catalog leakage firewall that blocks post-origination columns, into LightGBM training with out-of-time validation, isotonic calibration, expected-value pricing and SHAP explainability, then MLflow pyfunc packaging, Databricks Model Serving, Lakehouse drift monitoring and a guarded text2sql agent.](docs/architecture.svg)

# Credit Risk Engine — PD Model, Profit-Optimal Policy & AI Decision Explainer

An end-to-end consumer credit underwriting system built on 1.35M real LendingClub
loans (2007–2018): a leakage-safe probability-of-default model, a profit-driven
approval policy, regulatory-style reason codes, a FastAPI scoring service, and a
Claude-powered agent that explains individual credit decisions in plain language.

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

| Metric | Full model | Bureau-only* |
|---|---|---|
| ROC-AUC | 0.718 | 0.706 |
| PR-AUC (default rate 22.4%) | 0.411 | 0.402 |
| Brier score (raw) | 0.158 | 0.160 |

\* Bureau-only excludes LendingClub's pricing outputs (`grade`, `sub_grade`,
`int_rate`, `installment`). Losing all of LC's pricing signal costs only ~1.2
AUC points — most of it is recoverable from raw applicant/bureau attributes.

**Calibration:** raw PDs underestimate risk on newer vintages (the training
window has a lower default rate). An isotonic calibrator fit on the 2016
vintage and evaluated strictly out-of-time on 2017+ fixes this: decile-level
predicted PDs move from ~25% below actuals to within ~1-2 points, Brier
0.1533 → 0.1520 (`python -m src.calibrate`). The API serves calibrated PDs.

Top global risk drivers (mean |SHAP|): `sub_grade`, `term`, `grade`, `fico`,
`dti`, `annual_inc`. See [reports/shap_summary.png](reports/shap_summary.png).

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
.venv/bin/python -m src.calibrate     # isotonic drift correction
.venv/bin/python -m src.business      # profit sweep -> policy.json (calibrated)
.venv/bin/python -m src.explain       # SHAP summary + global importance
.venv/bin/python -m src.train --variant bureau   # optional: no-pricing variant

# 3. Serve (or: docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api)
.venv/bin/uvicorn api.main:app --port 8000

# 4. Score an applicant
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "loan_amnt": 15000, "term": "36", "int_rate": 13.5, "installment": 509.5,
  "grade": "C", "sub_grade": "C1", "annual_inc": 65000, "dti": 18.2,
  "earliest_cr_line": "Aug-2005", "fico_range_low": 685, "fico_range_high": 689,
  "revol_util": 62.5, "home_ownership": "RENT", "purpose": "debt_consolidation"}'
# -> {"pd": 0.25, "decision": "approve", "threshold": 0.475, "reasons": [...]}

# 5. Talk to the decision-explainer agent (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/python -m agent.explainer_agent

# 6. Text2SQL over the loan book
.venv/bin/python -m src.text2sql --self-test          # guardrail tests, no API key needed
.venv/bin/python -m src.text2sql "default rate by grade for 2017 vintages"

# 7. MCP server (stdio) — Claude Code picks it up from .mcp.json in this repo
.venv/bin/python -m mcp_server
.venv/bin/python -m pytest -q                         # 54 offline tests, no network
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

## Text2SQL with read-only guardrails

[src/text2sql.py](src/text2sql.py) turns portfolio questions ("default rate by grade for
2017 vintages") into SQL over the 1.35M-loan book, exposed to the agent as a
`query_loan_data` tool and mirrored in the notebook (§11b) against Unity Catalog.

An LLM writing SQL against a governed table is an injection surface, so generated SQL is
**never executed as-is**. `validate_sql()` enforces: single statement (blocks stacked
`; DROP TABLE`), `SELECT`/`WITH` only (blocks all DDL/DML), no comment markers (blocks
`--` smuggling), a table whitelist (blocks `system.information_schema`, arbitrary joins),
and a mandatory row cap. Rejections are returned to the model as a readable tool error so
it can rewrite the query rather than guess.

Guardrails are pure functions — unit-testable with no API key and no database:

```bash
.venv/bin/python -m src.text2sql --self-test
```

Sample output (`default rate by grade`, from the real loan book):

| grade | n | default_rate | avg_rate |
|---|---|---|---|
| A | 235,172 | 6.0% | 7.11% |
| D | 201,640 | 30.4% | 17.71% |
| G | 9,326 | 49.7% | 27.54% |

## MCP server

[mcp_server/server.py](mcp_server/server.py) exposes the engine over the **Model Context
Protocol**, so any MCP client (Claude Code, Claude Desktop, a custom agent) can underwrite
an applicant, get its reason codes, and query the loan book directly.

It is a *thin* layer, which is the point: scoring calls the same `api.main.score()` the
HTTP service serves — driven through the same lifespan, same booster, same calibrator, same
policy — and SQL goes through the same `validate_sql()` guardrail. No business logic is
duplicated, and the FastAPI service is untouched. A test asserts both tools return an
identical PD, so an MCP client and an HTTP client cannot disagree about a credit decision.

Targets spec revision **2026-07-28** — the stateless revision: no `initialize` handshake, no
`Mcp-Session-Id`, protocol version and client capabilities ride in `_meta` on every request —
on Python SDK **v2** (`mcp>=2,<3`; the v1 line does not implement this revision). Roots,
Sampling and Logging are deprecated in 2026-07-28, so the server implements none of them and
sends diagnostics to stderr, which is the migration the spec recommends for stdio servers.

| Tool | Returns |
|---|---|
| `score_applicant` | Calibrated PD, approve/decline against the 0.475 policy threshold, SHAP reason codes. Input schema is the *same Pydantic model* FastAPI validates against, so the 28-column firewall is inherited — no post-origination field is accepted. |
| `explain_decision` | The same decision, partitioned into ECOA principal reasons (risk-increasing) and mitigating factors, ranked by \|SHAP\|. |
| `query_loans` | Rows from the 1.35M-loan book. The client writes the SQL; the server validates it and never generates any. |

| Resource | Contents |
|---|---|
| `creditrisk://schema/origination-firewall` | The 28 origination columns, the engineered features, and every excluded leakage column — generated *from* [src/config.py](src/config.py), so it cannot drift from the model. |
| `creditrisk://schema/unity-catalog` | Schema and lineage provenance of the Databricks `loans_features` table. |
| `creditrisk://schema/loan-book` | Column dictionary for `loans` plus the guardrails any query must satisfy. Read before writing SQL. |

Reference material lives behind resource URIs rather than being stuffed into tool
descriptions or responses, which keeps `tools/list` small on every request.

**Prompt:** `adverse_action_notice` — turns `explain_decision` output into an ECOA-compliant
(Reg B, 12 CFR 1002.9) adverse action explanation: principal reasons in |SHAP| order, no
number the JSON does not contain, no allusion to protected characteristics, and the
applicant's 60-day rights paragraph.

### Connect it to Claude Code

The repo ships [.mcp.json](.mcp.json), so from the project root it is already wired:

```json
{
  "mcpServers": {
    "credit-risk": {
      "command": ".venv/bin/python",
      "args": ["-m", "mcp_server"]
    }
  }
}
```

Or register it explicitly:

```bash
claude mcp add credit-risk -- .venv/bin/python -m mcp_server
```

**stdio** is the transport: the client spawns the server as a subprocess, so there is no
port, no OAuth, and no `Mcp-Method`/`Mcp-Name` header plumbing. What the stateless rewrite
buys Streamable HTTP — any request landing on any replica behind a round-robin load balancer
with no shared store — is a multi-instance concern that does not exist locally. Switching is
one argument: `mcp.run(transport="streamable-http")`.

### Tests

```bash
.venv/bin/python -m pytest -q      # 54 tests, ~1s
```

Fully hermetic. `models/` and `data/processed/` are gitignored, so the fixtures build a
miniature but real LightGBM booster, isotonic calibrator and parquet loan book with fixed
seeds into a temp directory and point `src.config` at it — the code paths under test are the
production ones, only the artifacts are small. No network (DNS and outbound connects are
hard-failed for the duration of one test), no Spark, no Databricks, no API key.

Every write and DDL form is asserted to be refused through `query_loans` — `DROP`, `DELETE`,
`UPDATE`, `INSERT`, `TRUNCATE`, `ALTER`, `CREATE`, stacked `;` statements, `--` and `/* */`
comment smuggling, `COPY` exfiltration, `ATTACH`, and tables outside the whitelist — and a
further test re-counts the loan book afterwards to prove nothing executed.

## Design decisions & honest caveats

- **`int_rate`/`grade`/`sub_grade` are kept in the served model.** They're known
  at decision time (LendingClub assigns them at listing), but they embed LC's own
  risk model — so the full model partly learns to trust LC's pricing. The
  bureau-only variant quantifies exactly how much that's worth (~1.2 AUC points).
- **The profit function is simplified**: repaid loans earn full-term interest,
  defaulted loans lose 65% of principal with no credit for installments paid
  before default. Both assumptions are conservative in opposite directions.
- **Survivorship scope**: only *accepted* loans are observable, so the model
  can't correct LendingClub's original approval decisions (reject inference is
  the classic follow-up).
- **Later vintages are less mature.** 60-month loans issued in 2017 weren't all
  resolved by 2018Q4, so the resolved-only filter slightly biases late vintages
  toward early defaults. The calibrator partially absorbs this.
- **The Dockerfile is build-untested** (no Docker on the dev machine); it follows
  the standard python-slim + libgomp1 pattern for LightGBM.

## Databricks / Lakehouse deployment

[credit_risk_databricks.py](credit_risk_databricks.py) is a single notebook that
re-architects the whole pipeline onto Databricks (runs top-to-bottom on the free
**Free Edition** — serverless + Unity Catalog), turning the three things reviewers
probe into *provable platform facts*:

- **Leakage safety → provable via lineage.** The 28-column origination firewall is
  a Unity Catalog table built from raw bronze via a whitelist `SELECT`. UC
  **column lineage** shows no post-origination column (recoveries, total_pymnt,
  last_fico_*, …) feeds any feature; a programmatic assertion enforces it in-run.
- **Calibration drift → caught automatically.** The booster + isotonic calibrator +
  SHAP reason codes are packaged as **one custom MLflow `pyfunc`**, registered to
  **Unity Catalog**, and served from a **scale-to-zero Model Serving endpoint** with
  inference logging on. **Lakehouse Monitoring** on the inference log surfaces the
  vintage-driven calibration drift the isotonic step corrects.
- **Governance → one MLflow run** holds params, out-of-time ROC-AUC/PR-AUC, and the
  raw→calibrated **Brier improvement** as compared metrics.

Import it into a Databricks workspace and Run All (set the `catalog`/`schema`
widgets if your default catalog isn't `workspace`). Serving/Monitoring cells are
best-effort and degrade gracefully if those features are gated on your workspace.

**On AWS:** the same notebook runs unchanged on a Databricks-on-AWS workspace, where
Unity Catalog tables are physically **Delta files in S3** and compute is **EC2** — the
`§5b AWS footprint` cell prints the `s3://` locations as proof. Setup steps (UC
metastore = S3 bucket + IAM role, serverless, cost hygiene) are in
[AWS_DATABRICKS_SETUP.md](AWS_DATABRICKS_SETUP.md).

## Repo layout

| Path | What it does |
|---|---|
| [src/config.py](src/config.py) | Paths + the origination-time column whitelist / leakage blacklist |
| [src/data_prep.py](src/data_prep.py) | Labeling, filtering, parquet output |
| [src/features.py](src/features.py) | Feature engineering shared by training and serving (no train/serve skew) |
| [src/train.py](src/train.py) | LightGBM + out-of-time eval + calibration table (`--variant bureau` for the no-pricing model) |
| [src/calibrate.py](src/calibrate.py) | Isotonic recalibration: fit on 2016, evaluated on 2017+ |
| [src/business.py](src/business.py) | Threshold sweep → profit-optimal approval policy |
| [src/explain.py](src/explain.py) | SHAP global importance + per-loan reason codes |
| [api/main.py](api/main.py) | FastAPI `/score`, `/policy`, `/health` |
| [src/text2sql.py](src/text2sql.py) | Natural language → SQL with read-only guardrails (DuckDB over the loan book) |
| [agent/explainer_agent.py](agent/explainer_agent.py) | Claude tool-use agent: scoring API + guarded text2sql over the loan book |
| [credit_risk_databricks.py](credit_risk_databricks.py) | Databricks notebook: Delta + UC lineage firewall + MLflow + pyfunc + UC Model Serving + Lakehouse Monitoring |
| [mcp_server/server.py](mcp_server/server.py) | MCP server (spec 2026-07-28, SDK v2, stdio): scoring, ECOA reason codes and guarded SQL as tools; firewall/UC/loan-book schemas as resources |
| [tests/test_mcp_server.py](tests/test_mcp_server.py) | Hermetic offline tests for the MCP layer, including that every write/DDL form is refused |

Data: [LendingClub accepted loans 2007–2018Q4](https://huggingface.co/datasets/codesignal/lending-club-loan-accepted) (CC0).
