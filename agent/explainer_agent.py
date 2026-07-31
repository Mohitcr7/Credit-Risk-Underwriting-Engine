"""Claude-powered credit-decision explainer agent.

A tool-using agent that sits on top of the scoring API. Ask it things like:
  "Score this applicant: 690 FICO, $15k loan, 36 months, ..."
  "Why was that declined? What would need to change for an approval?"

It calls the model for scores and SHAP reason codes, then translates them
into plain-language, regulation-friendly explanations. It never invents
numbers — every figure comes from a tool call.

Setup:  export ANTHROPIC_API_KEY=...   (and start the API: uvicorn api.main:app)
Run:    python -m agent.explainer_agent
"""

import json
import os

import anthropic
import httpx
from dotenv import load_dotenv

from src import text2sql

load_dotenv()

API_URL = os.environ.get("SCORING_API_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-fable-5")

TOOLS = [
    {
        "name": "score_applicant",
        "description": (
            "Score a loan applicant with the credit risk model. Returns the "
            "probability of default (pd), an approve/decline decision against "
            "the current policy threshold, and the top SHAP reason codes. "
            "Required fields: loan_amnt, int_rate, installment, grade, "
            "sub_grade, annual_inc, earliest_cr_line (e.g. 'Aug-2005'), "
            "fico_range_low, fico_range_high. Everything else is optional."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "applicant": {
                    "type": "object",
                    "description": "Applicant fields matching the /score API schema.",
                }
            },
            "required": ["applicant"],
        },
    },
    {
        "name": "get_policy",
        "description": "Get the current approval policy: the PD threshold, LGD assumption, and expected approval rate.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_loan_data",
        "description": (
            "Run a read-only SQL query against the historical loan book (1.35M "
            "resolved loans, table `loans`) to answer portfolio questions — "
            "default rates by segment, vintage trends, distributions. "
            "You write the SQL; it is validated against read-only guardrails "
            "before execution and rejected if it is not a single SELECT/WITH "
            "statement over `loans`. Results are row-capped."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A single SELECT (or WITH ... SELECT) statement over `loans`.",
                }
            },
            "required": ["sql"],
        },
    },
]

SYSTEM = f"""You are a credit-decision assistant for a consumer lender.

You explain the decisions of a LightGBM probability-of-default model trained
on LendingClub data (2007-2018). You have tools to score applicants, read the
approval policy, and query the historical loan book with SQL.

Rules:
- Every number you state must come from a tool result. Never estimate a PD yourself,
  and never state a portfolio statistic you did not obtain from `query_loan_data`.
- Explain SHAP reason codes in plain language a loan applicant could understand
  (e.g. "your revolving credit utilization of 92% strongly increased the risk estimate").
- When asked "what would change the decision", reason from the SHAP directions,
  and where useful, re-score a modified applicant to verify your suggestion.
- For portfolio questions ("default rate by grade", "how did 2017 vintages perform"),
  write SQL and call `query_loan_data`. Write a single SELECT/WITH statement over
  `loans` only — anything else is rejected by the guardrails. If a query is
  rejected, read the error and rewrite it rather than guessing at the answer.
- Be precise about what the model does NOT know (it has no post-2018 data,
  no open-banking cash-flow features, etc.).

{text2sql.SCHEMA_DOC}
"""


def call_tool(name: str, tool_input: dict) -> str:
    try:
        if name == "query_loan_data":
            # Guardrails run BEFORE execution; a rejection is returned to the
            # model as a normal tool error so it can rewrite the query.
            try:
                sql = text2sql.validate_sql(tool_input["sql"])
            except text2sql.UnsafeSQLError as e:
                return json.dumps({"error": f"query rejected by guardrails: {e}"})
            rows = text2sql.run_sql(sql)
            return json.dumps({"sql": sql, "row_count": len(rows), "rows": rows}, default=str)
        if name == "score_applicant":
            r = httpx.post(f"{API_URL}/score", json=tool_input["applicant"], timeout=30)
        elif name == "get_policy":
            r = httpx.get(f"{API_URL}/policy", timeout=10)
        else:
            return json.dumps({"error": f"unknown tool {name}"})
        r.raise_for_status()
        return json.dumps(r.json())
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"API {e.response.status_code}: {e.response.text}"})
    except httpx.HTTPError as e:
        return json.dumps({"error": f"API unreachable ({e}). Is uvicorn running?"})


def chat() -> None:
    client = anthropic.Anthropic()
    messages: list[dict] = []
    print("Credit-decision explainer agent. Ctrl-C to exit.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        messages.append({"role": "user", "content": user})

        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                for block in response.content:
                    if block.type == "text":
                        print(f"\nagent> {block.text}\n")
                break

            results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [tool: {block.name}]")
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": call_tool(block.name, block.input),
                    })
            messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    chat()
