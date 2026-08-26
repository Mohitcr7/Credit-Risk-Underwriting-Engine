"""MCP server over the credit-risk underwriting engine.

A *thin* layer. There is no business logic in this file:

  - scoring calls `api.main.score()` — the exact function the FastAPI service
    serves, driven through the exact same lifespan, so an MCP client and an
    HTTP client cannot disagree about a PD;
  - reason codes are whatever `src.explain.reason_codes()` already returned
    inside that response — this module only reframes them for ECOA;
  - SQL goes through `src.text2sql.validate_sql()` before it goes anywhere
    near a database. The guardrail is a security boundary and is called, never
    re-implemented or relaxed.

Protocol notes (MCP revision 2026-07-28):
  - The protocol is stateless: there is no initialize handshake and no
    session. Nothing here keeps per-client state. The model artifacts held in
    `api.main.state` are process-level and immutable after startup — that is a
    server-side cache, not protocol session state.
  - Roots, Sampling and Logging are deprecated in this revision, so this
    server implements none of them. Diagnostics go to stderr, which is the
    migration path the spec recommends for stdio servers.
  - Tool failures raise `ToolError`, whose message reaches the client. A bare
    exception would be masked as "Error executing tool ...", which would hide
    the guardrail's reason and stop a model from rewriting a rejected query.

Run (stdio, the transport for a local dev server):
    python -m mcp_server
"""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from typing import Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import ValidationError

import api.main as scoring_api
from src import config, features, text2sql

SERVER_NAME = "credit-risk"
SERVER_VERSION = "1.0.0"

INSTRUCTIONS = """Tools for a consumer-credit underwriting model (LightGBM PD model
trained on 1.35M resolved LendingClub loans, 2007-2018, out-of-time validated).

- `score_applicant` returns a calibrated probability of default and the
  approve/decline decision against the profit-optimal policy threshold.
- `explain_decision` returns the SHAP reason codes behind that decision, framed
  for an ECOA adverse-action notice.
- `query_loans` runs read-only SQL against the historical loan book. You write
  the SQL; it is validated before execution and rejected if it is not a single
  SELECT/WITH over the allowed tables. Read the resource
  `creditrisk://schema/loan-book` first to get the column list.

Every number you report must come from a tool result. Do not estimate a PD and
do not state a portfolio statistic you did not obtain from `query_loans`."""


def _log(message: str) -> None:
    """Diagnostics to stderr — the 2026-07-28 migration path for Logging."""
    print(f"[{SERVER_NAME}] {message}", file=sys.stderr)


@asynccontextmanager
async def lifespan(server: MCPServer):
    """Load model artifacts by driving the FastAPI service's own lifespan.

    Reusing `api.main.lifespan` rather than re-loading the booster here is what
    guarantees the MCP server and the HTTP service score identically: same
    booster file, same calibrator, same policy, same feature order.
    """
    async with scoring_api.lifespan(scoring_api.app):
        _log(
            f"artifacts loaded — threshold={scoring_api.state['policy']['approve_below_pd']}, "
            f"calibrated={scoring_api.state['policy'].get('pd_is_calibrated')}, "
            f"features={len(scoring_api.state['feat_cols'])}"
        )
        yield {}


mcp = MCPServer(
    name=SERVER_NAME,
    title="Credit Risk Underwriting Engine",
    version=SERVER_VERSION,
    instructions=INSTRUCTIONS,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Shared helper — one scoring path, used by both scoring tools.
# --------------------------------------------------------------------------

def _score(applicant: scoring_api.Applicant) -> scoring_api.ScoreResponse:
    """Delegate to the FastAPI handler. No scoring logic lives in this module."""
    if "model" not in scoring_api.state:
        raise ToolError(
            "model artifacts are not loaded. Run the training pipeline first: "
            "python -m src.data_prep && python -m src.train && "
            "python -m src.calibrate && python -m src.business"
        )
    try:
        return scoring_api.score(applicant)
    except ValidationError as e:  # pragma: no cover - schema is enforced upstream
        raise ToolError(f"invalid applicant: {e}") from e


def _policy() -> dict[str, Any]:
    return scoring_api.state["policy"]


# --------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------

@mcp.tool(
    name="score_applicant",
    title="Score a loan applicant",
    description=(
        "Score a loan applicant with the probability-of-default model. Returns the "
        "calibrated PD, the approve/decline decision against the current policy "
        "threshold, and the top SHAP reason codes. The PD returned is the "
        "post-isotonic-calibration probability — the same number the production "
        "scoring API serves and the same one the policy threshold is applied to. "
        "Only origination-time attributes are accepted; the model never sees "
        "post-origination data."
    ),
    annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True),
)
def score_applicant(applicant: scoring_api.Applicant) -> dict[str, Any]:
    result = _score(applicant)
    policy = _policy()
    return {
        "pd": result.pd,
        "pd_is_calibrated": bool(policy.get("pd_is_calibrated", False)),
        "decision": result.decision,
        "threshold": result.threshold,
        "policy": {
            "approve_below_pd": policy["approve_below_pd"],
            "lgd_assumption": policy["lgd_assumption"],
            "expected_approval_rate": policy["expected_approval_rate"],
            "basis": "expected-value threshold sweep: interest earned vs LGD-weighted loss",
        },
        "reasons": result.reasons,
    }


@mcp.tool(
    name="explain_decision",
    title="Explain a credit decision (ECOA reason codes)",
    description=(
        "Score an applicant and return the decision drivers as ECOA-style adverse "
        "action reason codes: the features that moved the risk estimate most, each "
        "with its value, SHAP contribution and direction. Under ECOA a lender must "
        "give specific principal reasons for an adverse action, so risk-increasing "
        "drivers are listed separately. Use the `adverse_action_notice` prompt to "
        "turn this into applicant-facing language."
    ),
    annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True),
)
def explain_decision(applicant: scoring_api.Applicant) -> dict[str, Any]:
    result = _score(applicant)
    # Presentation only: `adverse_action_required` restates `decision == "decline"`,
    # and the two lists are a partition of `result.reasons` by the `direction`
    # field that `src.explain.reason_codes` already computed. No new logic.
    increasing = [r for r in result.reasons if r["direction"] == "increases risk"]
    decreasing = [r for r in result.reasons if r["direction"] != "increases risk"]
    return {
        "pd": result.pd,
        "decision": result.decision,
        "threshold": result.threshold,
        "adverse_action_required": result.decision == "decline",
        "principal_reasons": increasing,
        "mitigating_factors": decreasing,
        "reasons": result.reasons,
        "method": "TreeSHAP over the LightGBM booster; ranked by |SHAP| descending",
        "disclosure_note": (
            "Reason codes describe this model's drivers only. They are not a "
            "statement about protected characteristics, which the model never sees."
        ),
    }


@mcp.tool(
    name="query_loans",
    title="Query the historical loan book (read-only SQL)",
    description=(
        "Run a read-only SQL query against the historical loan book (1.35M resolved "
        "loans, table `loans`) to answer portfolio questions: default rates by "
        "segment, vintage trends, distributions. You write the SQL. It is validated "
        "before execution and rejected unless it is a single SELECT or WITH "
        "statement over the allowed tables, with no comment markers; results are "
        "row-capped. Read `creditrisk://schema/loan-book` for the columns. If a "
        "query is rejected, read the error and rewrite it — do not guess the answer."
    ),
    annotations=ToolAnnotations(read_only_hint=True, destructive_hint=False, idempotent_hint=True),
)
def query_loans(sql: str) -> dict[str, Any]:
    # The guardrail is a security boundary: validate before anything touches a
    # database, and surface the rejection reason so the model can rewrite.
    try:
        guarded = text2sql.validate_sql(sql)
    except text2sql.UnsafeSQLError as e:
        raise ToolError(f"query rejected by guardrails: {e}") from e

    if not config.PROCESSED_PARQUET.exists():
        raise ToolError(
            f"loan book not found at {config.PROCESSED_PARQUET}. "
            "Run: python -m src.data_prep"
        )
    try:
        rows = text2sql.run_sql(guarded)
    except Exception as e:
        raise ToolError(f"SQL execution failed: {type(e).__name__}: {e}") from e

    return {
        "sql": guarded,
        "row_count": len(rows),
        "row_cap": text2sql.MAX_ROWS,
        "truncated": len(rows) >= text2sql.MAX_ROWS,
        "rows": json.loads(json.dumps(rows, default=str)),
    }


# --------------------------------------------------------------------------
# Resources — large, stable reference material stays out of tool responses.
# Each is generated from src/config.py so it cannot drift from the model.
# --------------------------------------------------------------------------

@mcp.resource(
    "creditrisk://schema/origination-firewall",
    name="origination_firewall",
    title="Origination-time feature firewall (28 columns)",
    description=(
        "The leakage firewall: the 28 origination-time columns the model may see, "
        "the engineered features derived from them, and the post-origination "
        "columns that are excluded. Generated from src/config.py."
    ),
    mime_type="application/json",
)
def origination_firewall() -> str:
    model_features = features.feature_columns()
    payload = {
        "rationale": (
            "The raw LendingClub file has 151 columns; ~40 describe what happened "
            "AFTER origination (payments, recoveries, hardship, settlement). Training "
            "on them yields ROC-AUC 0.95+ and a model that cannot underwrite, because "
            "none of it is knowable at decision time. The firewall is a whitelist, not "
            "a blacklist: an unrecognised new column is excluded by default, so a "
            "schema change fails closed."
        ),
        "target": config.TARGET,
        "out_of_time_split_date": config.SPLIT_DATE,
        "origination_columns": config.ORIGINATION_COLS,
        "origination_column_count": len(config.ORIGINATION_COLS),
        "engineered_columns": {
            "fico": "mean of fico_range_low and fico_range_high",
            "credit_history_years": "(issue date - earliest_cr_line) / 365.25",
        },
        "dropped_after_engineering": config.DROPPED_AFTER_ENGINEERING,
        "model_feature_columns": model_features,
        "model_feature_count": len(model_features),
        "categorical_columns": config.CATEGORICAL_COLS,
        "pricing_columns": {
            "columns": config.PRICING_COLS,
            "note": (
                "Known at decision time (LendingClub assigns them at listing) but they "
                "embed LC's own risk model. The bureau-only model variant excludes them "
                "to quantify the dependency: it costs ~1.2 ROC-AUC points."
            ),
        },
        "meta_columns": {
            "columns": config.META_COLS,
            "note": "Used for labeling and the out-of-time split. Never features.",
        },
        "excluded_leakage_columns": config.LEAKAGE_COLS,
        "excluded_leakage_column_count": len(config.LEAKAGE_COLS),
    }
    return json.dumps(payload, indent=2)


@mcp.resource(
    "creditrisk://schema/unity-catalog",
    name="unity_catalog_schema",
    title="Unity Catalog table schema (loans_features)",
    description=(
        "Schema and provenance of the Databricks Unity Catalog feature table "
        "`<catalog>.<schema>.loans_features`, the lakehouse form of the firewall."
    ),
    mime_type="text/markdown",
)
def unity_catalog_schema() -> str:
    return f"""# Unity Catalog: `{{catalog}}.{{schema}}.loans_features`

Built by `credit_risk_databricks.py` §5. `catalog`/`schema` are notebook widgets
(defaults `workspace` / `credit_risk`; the notebook falls back to `main` on
Databricks-on-AWS). Physically a Delta table — S3 objects on an AWS workspace.

## Provenance

`loans_raw` (bronze, all 151 raw columns) -> `loans_features` via a single
whitelist `CREATE OR REPLACE TABLE ... AS SELECT`. Because every feature is
derived in Spark SQL from named bronze columns, Unity Catalog captures
**column-level lineage**: the Lineage tab shows that `recoveries`,
`total_pymnt`, `last_fico_range_high`, and the rest of the post-origination
columns have no downstream edge into any feature. That is the leakage proof.

The notebook also asserts it programmatically, so the run fails rather than
silently ships:

```python
leaked = set(spark.table(f"{{CATALOG}}.{{SCHEMA}}.loans_features").columns) & set(LEAKAGE_COLS)
assert not leaked, f"FIREWALL BREACH: leakage columns in feature table: {{leaked}}"
```

`TBLPROPERTIES (delta.enableChangeDataFeed = true)`.

## Columns

| Column | Type | Role |
|---|---|---|
| `loan_amnt` | DOUBLE | feature |
| `term` | STRING | feature (categorical, `'36'` / `'60'`) |
| `int_rate` | DOUBLE | feature (percent, `%` stripped) |
| `installment` | DOUBLE | feature |
| `grade` | STRING | feature (categorical, `A`..`G`) |
| `sub_grade` | STRING | feature (categorical, `A1`..`G5`) |
| `emp_length` | INT | feature (ordinal 0-10, `NULL` if unknown) |
| `home_ownership` | STRING | feature (categorical) |
| `annual_inc` | DOUBLE | feature |
| `verification_status` | STRING | feature (categorical) |
| `purpose` | STRING | feature (categorical) |
| `addr_state` | STRING | feature (categorical, 2-letter) |
| `dti` | DOUBLE | feature |
| `delinq_2yrs` | DOUBLE | feature |
| `inq_last_6mths` | DOUBLE | feature |
| `mths_since_last_delinq` | DOUBLE | feature |
| `mths_since_last_record` | DOUBLE | feature |
| `open_acc` | DOUBLE | feature |
| `pub_rec` | DOUBLE | feature |
| `revol_bal` | DOUBLE | feature |
| `revol_util` | DOUBLE | feature (percent, `%` stripped) |
| `total_acc` | DOUBLE | feature |
| `application_type` | STRING | feature (categorical) |
| `mort_acc` | DOUBLE | feature |
| `pub_rec_bankruptcies` | DOUBLE | feature |
| `fico` | DOUBLE | engineered feature (mean of the FICO range) |
| `credit_history_years` | DOUBLE | engineered feature |
| `issue_d` | DATE | metadata — out-of-time split key, never a feature |
| `is_default` | INT | target — 1 charged off, 0 fully paid |

Row filter: resolved loans only (`Fully Paid` / `Charged Off`, including the
"Does not meet the credit policy" variants), non-null `issue_d`, and
`annual_inc BETWEEN 0 AND 5000000`.

## Querying it

`query_loans` targets the local DuckDB view `loans`. The Spark variant in
notebook §11b applies the identical guardrails to
`{{catalog}}.{{schema}}.loans_features` before execution; both accept the same
table whitelist, so a query written against `loans` runs against
`loans_features` unchanged.
"""


@mcp.resource(
    "creditrisk://schema/loan-book",
    name="loan_book_schema",
    title="Loan book SQL schema (table `loans`)",
    description=(
        "Column dictionary for the `loans` table that `query_loans` queries, plus "
        "the guardrails any generated SQL must satisfy. Read this before writing SQL."
    ),
    mime_type="text/markdown",
)
def loan_book_schema() -> str:
    return f"""# Loan book SQL schema

{text2sql.SCHEMA_DOC}

## Guardrails enforced on every query

Generated SQL is never executed as-is. `validate_sql()` rejects anything that
is not:

1. a **single statement** — a `;` outside a string literal is refused, blocking
   stacked `; DROP TABLE ...`;
2. **`SELECT` or `WITH` only** — no DDL or DML;
3. **free of comment markers** — `--`, `/*` and `*/` are refused, blocking
   keyword smuggling past the parser;
4. restricted to the **table whitelist** {sorted(text2sql.ALLOWED_TABLES)} —
   qualified names are resolved to their base name, and CTE aliases are allowed;
5. **row-capped** — a `LIMIT {text2sql.MAX_ROWS}` is appended when absent.

Forbidden keywords: {", ".join(f"`{k}`" for k in sorted(text2sql.FORBIDDEN))}.

A rejection comes back as a tool error carrying the reason. Rewrite the query
from that reason rather than guessing at the answer.
"""


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

@mcp.prompt(
    name="adverse_action_notice",
    title="ECOA adverse action notice",
    description=(
        "Turn `explain_decision` output into an ECOA-compliant adverse action "
        "explanation written for the applicant. Pass the tool's JSON result."
    ),
)
def adverse_action_notice(
    decision_json: str,
    applicant_name: str = "the applicant",
    lender_name: str = "the lender",
) -> str:
    return f"""You are drafting an adverse action explanation for {applicant_name} on
behalf of {lender_name}.

Under the Equal Credit Opportunity Act (Regulation B, 12 CFR 1002.9), an
applicant who is declined must be told the **specific principal reasons** for
the decision. Generic statements ("you did not score well enough", "internal
policy") do not satisfy this.

Here is the model output:

```json
{decision_json}
```

Write the explanation to these rules:

1. State the decision plainly in the first sentence.
2. Give the principal reasons in `principal_reasons` order — that is descending
   |SHAP|, so it is the correct order of importance. Use the model's own
   `feature` and `value` fields; translate each into plain language a borrower
   would understand (`revol_util` of 92.0 becomes "your revolving credit
   utilization of 92%").
3. Use at most four reasons. Do not invent a reason that is not in the JSON.
4. State no number the JSON does not contain. Do not restate the probability of
   default as a percentage chance of the applicant personally defaulting — it is
   a model estimate for applicants with these attributes.
5. Do not speculate about, or allude to, any protected characteristic. The model
   does not receive race, colour, religion, national origin, sex, marital status,
   age, or receipt of public assistance.
6. If `mitigating_factors` is non-empty, note briefly which attributes worked in
   the applicant's favour — it makes the notice actionable.
7. Close with the applicant's right to request the specific reasons in writing
   within 60 days, and to obtain a free copy of any consumer report used.

If `adverse_action_required` is false the application was approved: write a
short approval note explaining the main drivers instead, and omit the ECOA
rights paragraph.

Keep it under 250 words, second person, no jargon, no bullet-point dump."""


def main() -> None:
    """stdio: the transport for a local dev server. Streamable HTTP is one arg away."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
