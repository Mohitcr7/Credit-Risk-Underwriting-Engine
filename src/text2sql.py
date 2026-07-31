"""Natural language -> SQL over the loan book, with read-only guardrails.

Two execution backends, one contract:
  - DuckDB over data/processed/loans.parquet   (local / API / agent)
  - Spark SQL over <catalog>.<schema>.loans_features   (Databricks notebook)

The interesting part is not the generation, it's the *validation*. An LLM writing
SQL against a production table is an injection surface, so generated SQL is never
executed as-is. `validate_sql()` enforces:

  1. single statement only        (blocks stacked "; DROP TABLE ...")
  2. SELECT / WITH only           (blocks DDL + DML: DROP, DELETE, UPDATE, INSERT,
                                   ALTER, CREATE, GRANT, COPY, ATTACH, ...)
  3. no comment markers           (blocks "-- " / "/* */" smuggling past the parser)
  4. whitelisted table names      (the model may only touch the loan table)
  5. mandatory row cap            (a LIMIT is injected when absent)

Guardrails are pure functions and unit-testable without an API key or a database:
    python -m src.text2sql --self-test

Run a question end-to-end (needs ANTHROPIC_API_KEY):
    python -m src.text2sql "default rate by grade for 2017 vintages"
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from src import config

MAX_ROWS = 200

# The only relation generated SQL may reference. `loans` is the DuckDB view name;
# `loans_features` is the Unity Catalog table name used on Databricks.
ALLOWED_TABLES = {"loans", "loans_features"}

FORBIDDEN = {
    "drop", "delete", "update", "insert", "alter", "create", "replace",
    "truncate", "grant", "revoke", "merge", "copy", "attach", "detach",
    "install", "load", "export", "import", "pragma", "set", "call", "vacuum",
}

# Column docs handed to the model. Keeping this explicit (rather than dumping the
# schema) keeps the prompt small and stops the model inventing columns.
SCHEMA_DOC = """Table: loans  (one row per resolved LendingClub loan, 1.35M rows)

Target / dates:
  is_default INT        1 = charged off, 0 = fully paid
  issue_d    DATE       loan issue date (vintage), 2007-06-01 .. 2018-12-01

Loan terms:
  loan_amnt DOUBLE, term VARCHAR ('36' or '60'), int_rate DOUBLE (percent, e.g. 13.5),
  installment DOUBLE, grade VARCHAR ('A'..'G'), sub_grade VARCHAR ('A1'..'G5'),
  purpose VARCHAR, application_type VARCHAR

Borrower:
  annual_inc DOUBLE, emp_length DOUBLE (years, 0..10), home_ownership VARCHAR,
  verification_status VARCHAR, addr_state VARCHAR (2-letter), dti DOUBLE,
  fico DOUBLE (avg of FICO range), credit_history_years DOUBLE

Credit bureau:
  delinq_2yrs, inq_last_6mths, mths_since_last_delinq, mths_since_last_record,
  open_acc, pub_rec, revol_bal, revol_util (percent), total_acc, mort_acc,
  pub_rec_bankruptcies  -- all DOUBLE

Notes:
  - Default rate = AVG(is_default). Express as a rate, not a count, unless asked.
  - Vintage/year = YEAR(issue_d).
  - Only resolved loans are present (no 'Current' loans), so rates are realized."""

SYSTEM_PROMPT = f"""You translate questions about a consumer loan book into a single SQL query.

{SCHEMA_DOC}

Rules:
- Return ONLY the SQL. No prose, no markdown fences, no explanation.
- Exactly one statement. SELECT (or WITH ... SELECT) only.
- Never write DDL or DML. Never reference any table other than `loans`.
- Prefer readable aggregates with explicit column aliases.
- Round rates to 4 decimals and order results sensibly.
- The dialect is ANSI SQL compatible with DuckDB and Spark SQL."""


class UnsafeSQLError(ValueError):
    """Generated SQL violated a guardrail and was not executed."""


def _strip_fences(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```[a-zA-Z]*\n?", "", sql)
        sql = re.sub(r"\n?```$", "", sql)
    return sql.strip().rstrip(";").strip()


def validate_sql(sql: str, allowed_tables: set[str] | None = None) -> str:
    """Return guarded SQL, or raise UnsafeSQLError. Never executes anything."""
    allowed = allowed_tables or ALLOWED_TABLES
    sql = _strip_fences(sql)
    if not sql:
        raise UnsafeSQLError("empty query")

    # 3. comment markers (checked before tokenizing so they can't hide keywords)
    if "--" in sql or "/*" in sql or "*/" in sql:
        raise UnsafeSQLError("SQL comments are not allowed")

    # 1. single statement (a ';' inside a string literal is not our concern:
    #    literals are stripped first, so only real separators remain)
    literal_free = re.sub(r"'[^']*'", "''", sql)
    if ";" in literal_free:
        raise UnsafeSQLError("only a single statement is allowed")

    lowered = literal_free.lower()
    tokens = set(re.findall(r"[a-z_]+", lowered))

    # 2. read-only
    if not re.match(r"^\s*(select|with)\b", lowered):
        raise UnsafeSQLError("query must start with SELECT or WITH")
    banned = tokens & FORBIDDEN
    if banned:
        raise UnsafeSQLError(f"forbidden keyword(s): {', '.join(sorted(banned))}")

    # 4. table whitelist — every FROM/JOIN target must be allowed or a CTE alias
    cte_names = set(re.findall(r"(?:with|,)\s+([a-z_][a-z0-9_]*)\s+as\s*\(", lowered))
    referenced = set(re.findall(r"(?:from|join)\s+([a-z_][a-z0-9_\.]*)", lowered))
    for ref in referenced:
        base = ref.split(".")[-1]  # tolerate catalog.schema.table
        if base not in allowed and base not in cte_names:
            raise UnsafeSQLError(f"table '{ref}' is not allowed")

    # 5. row cap
    if not re.search(r"\blimit\s+\d+", lowered):
        sql = f"{sql}\nLIMIT {MAX_ROWS}"
    return sql


def generate_sql(question: str, model: str | None = None) -> str:
    """Ask Claude for SQL. Returns validated SQL or raises UnsafeSQLError."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", "claude-opus-5"),
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    raw = "".join(b.text for b in response.content if b.type == "text")
    return validate_sql(raw)


def run_sql(sql: str) -> list[dict]:
    """Execute validated SQL against the parquet via DuckDB (read-only)."""
    import duckdb

    con = duckdb.connect(database=":memory:")
    try:
        # Relation API rather than an interpolated CREATE VIEW: the parquet path
        # never becomes part of a SQL string.
        con.read_parquet(str(config.PROCESSED_PARQUET)).create_view("loans")
        cur = con.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def ask(question: str) -> dict:
    """Full path: question -> SQL -> guardrails -> rows."""
    sql = generate_sql(question)
    return {"question": question, "sql": sql, "rows": run_sql(sql)}


# --------------------------------------------------------------------------
# Self-test: guardrails are pure functions, so this needs no API key and no DB.
# --------------------------------------------------------------------------

ATTACKS = [
    ("SELECT 1; DROP TABLE loans", "stacked statement"),
    ("DROP TABLE loans", "DDL"),
    ("DELETE FROM loans", "DML"),
    ("UPDATE loans SET is_default = 0", "DML"),
    ("SELECT * FROM loans -- ; DROP TABLE loans", "comment smuggling"),
    ("SELECT * FROM /* hidden */ loans", "block comment"),
    ("SELECT * FROM secrets", "table not whitelisted"),
    ("SELECT * FROM main.private.customers", "qualified table not whitelisted"),
    ("COPY loans TO 'out.csv'", "exfiltration"),
    ("ATTACH 'evil.db' AS evil", "attach"),
    ("", "empty"),
]

SAFE = [
    "SELECT grade, AVG(is_default) AS default_rate FROM loans GROUP BY grade",
    "WITH v AS (SELECT YEAR(issue_d) AS y, is_default FROM loans) "
    "SELECT y, AVG(is_default) FROM v GROUP BY y",
    "SELECT * FROM loans LIMIT 5",
]


def self_test() -> int:
    failures = 0
    print("Guardrail tests — attacks must be REJECTED:")
    for sql, label in ATTACKS:
        try:
            validate_sql(sql)
            print(f"  FAIL  ({label}) allowed: {sql!r}")
            failures += 1
        except UnsafeSQLError as e:
            print(f"  ok    ({label}) blocked: {e}")

    print("\nLegitimate queries must be ACCEPTED:")
    for sql in SAFE:
        try:
            out = validate_sql(sql)
            capped = "LIMIT" in out.upper()
            print(f"  ok    accepted (row cap present: {capped}): {sql[:58]}...")
        except UnsafeSQLError as e:
            print(f"  FAIL  rejected legitimate query: {e}")
            failures += 1

    print("\nFAILURES:" if failures else "\nAll guardrail tests passed.", failures or "")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Natural language -> SQL over the loan book")
    parser.add_argument("question", nargs="*", help="question in plain English")
    parser.add_argument("--self-test", action="store_true", help="run guardrail tests only")
    parser.add_argument("--sql", help="execute a literal SQL string (still guarded)")
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.sql:
        sql = validate_sql(args.sql)
        print(f"-- guarded SQL --\n{sql}\n")
        for row in run_sql(sql):
            print(row)
        return 0
    if not args.question:
        parser.print_help()
        return 1

    result = ask(" ".join(args.question))
    print(f"-- generated SQL --\n{result['sql']}\n")
    for row in result["rows"]:
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
