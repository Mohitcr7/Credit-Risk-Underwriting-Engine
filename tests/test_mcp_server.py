"""Offline tests for the MCP layer over the credit-risk engine.

Every test runs against an in-memory MCP client connected directly to the
server object — no subprocess, no socket, no HTTP, no Spark, no Databricks.
Artifacts come from `tests/conftest.py`, which builds a miniature model and
loan book with fixed seeds.

The security-relevant tests are `TestQueryLoansGuardrails`: the read-only SQL
guardrail is a boundary, so writes and DDL are asserted to be refused *and* to
have left the data untouched.
"""

from __future__ import annotations

import json
import socket

import duckdb
import pytest

from src import config, text2sql

# Category values below all appear in the fixture's training data.
LOW_RISK = {
    "loan_amnt": 8000, "term": "36", "int_rate": 7.2, "installment": 248.0,
    "grade": "A", "sub_grade": "A1", "emp_length": "10+ years",
    "home_ownership": "MORTGAGE", "annual_inc": 140000,
    "verification_status": "Verified", "purpose": "credit_card",
    "addr_state": "CA", "dti": 8.0, "delinq_2yrs": 0,
    "earliest_cr_line": "Jan-1995", "fico_range_low": 800, "fico_range_high": 804,
    "inq_last_6mths": 0, "open_acc": 12, "pub_rec": 0, "revol_bal": 3000,
    "revol_util": 9.0, "total_acc": 34, "application_type": "Individual",
    "mort_acc": 2, "pub_rec_bankruptcies": 0,
}

HIGH_RISK = {
    "loan_amnt": 35000, "term": "60", "int_rate": 28.9, "installment": 1100.0,
    "grade": "G", "sub_grade": "G4", "emp_length": "< 1 year",
    "home_ownership": "RENT", "annual_inc": 30000,
    "verification_status": "Not Verified", "purpose": "small_business",
    "addr_state": "NY", "dti": 35.0, "delinq_2yrs": 3,
    "earliest_cr_line": "Aug-2010", "fico_range_low": 660, "fico_range_high": 664,
    "inq_last_6mths": 4, "open_acc": 6, "pub_rec": 1, "revol_bal": 24000,
    "revol_util": 95.0, "total_acc": 9, "application_type": "Individual",
    "mort_acc": 0, "pub_rec_bankruptcies": 1,
}

# Statements that must never reach a database. Mirrors src.text2sql.ATTACKS and
# adds the ones an MCP client is most likely to be talked into emitting.
WRITE_AND_DDL = [
    ("DROP TABLE loans", "drop table"),
    ("DELETE FROM loans", "delete rows"),
    ("DELETE FROM loans WHERE is_default = 1", "conditional delete"),
    ("UPDATE loans SET is_default = 0", "update rows"),
    ("INSERT INTO loans (grade) VALUES ('A')", "insert rows"),
    ("TRUNCATE TABLE loans", "truncate"),
    ("ALTER TABLE loans DROP COLUMN is_default", "alter table"),
    ("CREATE TABLE evil AS SELECT * FROM loans", "create table"),
    ("SELECT 1; DROP TABLE loans", "stacked statement"),
    ("SELECT * FROM loans; DELETE FROM loans", "stacked write"),
    ("SELECT * FROM loans -- ; DROP TABLE loans", "line-comment smuggling"),
    ("SELECT * FROM /* hidden */ loans", "block-comment smuggling"),
    ("COPY loans TO 'out.csv'", "exfiltration via COPY"),
    ("ATTACH 'evil.db' AS evil", "attach a second database"),
    ("SELECT * FROM secrets", "table outside the whitelist"),
    ("SELECT * FROM main.private.customers", "qualified table outside the whitelist"),
    ("PRAGMA database_list", "pragma"),
    ("", "empty statement"),
]


def text_of(result) -> str:
    return "\n".join(b.text for b in result.content if getattr(b, "type", None) == "text")


def loan_book_row_count() -> int:
    con = duckdb.connect(database=":memory:")
    try:
        con.read_parquet(str(config.PROCESSED_PARQUET)).create_view("loans")
        return con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    finally:
        con.close()


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

class TestServerSurface:
    def test_exposes_the_three_tools(self, call):
        names = {t.name for t in call(lambda c: c.list_tools()).tools}
        assert names == {"score_applicant", "explain_decision", "query_loans"}

    def test_exposes_the_resources(self, call):
        uris = {str(r.uri) for r in call(lambda c: c.list_resources()).resources}
        assert uris == {
            "creditrisk://schema/origination-firewall",
            "creditrisk://schema/unity-catalog",
            "creditrisk://schema/loan-book",
        }

    def test_exposes_the_adverse_action_prompt(self, call):
        prompts = {p.name: p for p in call(lambda c: c.list_prompts()).prompts}
        assert "adverse_action_notice" in prompts
        required = {a.name for a in prompts["adverse_action_notice"].arguments if a.required}
        assert required == {"decision_json"}

    def test_every_tool_is_annotated_read_only(self, call):
        for tool in call(lambda c: c.list_tools()).tools:
            assert tool.annotations is not None, tool.name
            assert tool.annotations.read_only_hint is True, tool.name
            assert tool.annotations.destructive_hint is False, tool.name

    def test_applicant_schema_admits_no_post_origination_column(self, call):
        """The MCP surface inherits the firewall: no leakage field is accepted."""
        tools = {t.name: t for t in call(lambda c: c.list_tools()).tools}
        for name in ("score_applicant", "explain_decision"):
            blob = json.dumps(tools[name].input_schema)
            leaked = [c for c in config.LEAKAGE_COLS if f'"{c}"' in blob]
            assert leaked == [], f"{name} exposes leakage columns: {leaked}"


# --------------------------------------------------------------------------
# score_applicant
# --------------------------------------------------------------------------

class TestScoreApplicant:
    def test_returns_calibrated_pd_and_decision(self, call, artifacts):
        out = call(lambda c: c.call_tool("score_applicant", {"applicant": LOW_RISK}))
        assert out.is_error is False
        body = out.structured_content
        assert 0.0 <= body["pd"] <= 1.0
        assert body["pd_is_calibrated"] is True
        assert body["decision"] in {"approve", "decline"}
        assert body["threshold"] == artifacts["policy"]["approve_below_pd"] == 0.475
        assert body["policy"]["lgd_assumption"] == 0.65
        assert body["reasons"]

    @pytest.mark.parametrize("applicant", [LOW_RISK, HIGH_RISK], ids=["low_risk", "high_risk"])
    def test_decision_is_the_threshold_comparison(self, call, applicant):
        body = call(lambda c: c.call_tool("score_applicant", {"applicant": applicant})).structured_content
        expected = "approve" if body["pd"] < body["threshold"] else "decline"
        assert body["decision"] == expected

    def test_high_risk_applicant_is_declined_at_0475(self, call):
        low = call(lambda c: c.call_tool("score_applicant", {"applicant": LOW_RISK})).structured_content
        high = call(lambda c: c.call_tool("score_applicant", {"applicant": HIGH_RISK})).structured_content
        assert high["pd"] > low["pd"], "model must rank G4/60mo/95%-util above A1/36mo/9%-util"
        assert low["decision"] == "approve"
        assert high["decision"] == "decline"

    def test_is_deterministic(self, call):
        a = call(lambda c: c.call_tool("score_applicant", {"applicant": HIGH_RISK})).structured_content
        b = call(lambda c: c.call_tool("score_applicant", {"applicant": HIGH_RISK})).structured_content
        assert a["pd"] == b["pd"]
        assert a["reasons"] == b["reasons"]

    def test_rejects_a_malformed_applicant(self, call):
        broken = {k: v for k, v in LOW_RISK.items() if k != "annual_inc"}
        out = call(lambda c: c.call_tool("score_applicant", {"applicant": broken}))
        assert out.is_error is True
        assert "annual_inc" in text_of(out)

    def test_rejects_an_out_of_range_fico(self, call):
        out = call(lambda c: c.call_tool(
            "score_applicant", {"applicant": {**LOW_RISK, "fico_range_low": 9999}}))
        assert out.is_error is True


# --------------------------------------------------------------------------
# explain_decision
# --------------------------------------------------------------------------

class TestExplainDecision:
    def test_returns_ecoa_reason_codes(self, call):
        body = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).structured_content
        assert body["adverse_action_required"] == (body["decision"] == "decline")
        assert body["principal_reasons"], "a declined applicant needs principal reasons"
        for reason in body["reasons"]:
            assert set(reason) == {"feature", "value", "shap", "direction"}
            assert reason["feature"] in config.ORIGINATION_COLS + config.ENGINEERED_COLS

    def test_partitions_reasons_by_direction(self, call):
        body = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).structured_content
        assert all(r["direction"] == "increases risk" for r in body["principal_reasons"])
        assert all(r["direction"] != "increases risk" for r in body["mitigating_factors"])
        assert len(body["principal_reasons"]) + len(body["mitigating_factors"]) == len(body["reasons"])

    def test_reasons_are_ranked_by_absolute_shap(self, call):
        body = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).structured_content
        magnitudes = [abs(r["shap"]) for r in body["reasons"]]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_agrees_with_score_applicant(self, call):
        """Both tools must go through one scoring path — no second opinion."""
        scored = call(lambda c: c.call_tool(
            "score_applicant", {"applicant": HIGH_RISK})).structured_content
        explained = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).structured_content
        assert scored["pd"] == explained["pd"]
        assert scored["decision"] == explained["decision"]
        assert scored["threshold"] == explained["threshold"]
        assert scored["reasons"] == explained["reasons"]

    def test_approved_applicant_needs_no_adverse_action(self, call):
        body = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": LOW_RISK})).structured_content
        assert body["decision"] == "approve"
        assert body["adverse_action_required"] is False


# --------------------------------------------------------------------------
# query_loans — the security boundary
# --------------------------------------------------------------------------

class TestQueryLoans:
    def test_runs_a_legitimate_aggregate(self, call):
        out = call(lambda c: c.call_tool("query_loans", {
            "sql": "SELECT grade, AVG(is_default) AS default_rate "
                   "FROM loans GROUP BY grade ORDER BY grade"}))
        assert out.is_error is False
        body = out.structured_content
        assert body["row_count"] == 7
        assert [r["grade"] for r in body["rows"]] == list("ABCDEFG")
        assert all(0.0 <= r["default_rate"] <= 1.0 for r in body["rows"])

    def test_accepts_a_cte(self, call):
        out = call(lambda c: c.call_tool("query_loans", {
            "sql": "WITH v AS (SELECT YEAR(issue_d) AS y, is_default FROM loans) "
                   "SELECT y, AVG(is_default) AS dr FROM v GROUP BY y ORDER BY y"}))
        assert out.is_error is False
        assert out.structured_content["row_count"] >= 1

    def test_injects_the_row_cap_when_absent(self, call):
        body = call(lambda c: c.call_tool(
            "query_loans", {"sql": "SELECT grade FROM loans"})).structured_content
        assert f"LIMIT {text2sql.MAX_ROWS}" in body["sql"]
        assert body["row_count"] == text2sql.MAX_ROWS
        assert body["truncated"] is True

    def test_respects_an_explicit_limit(self, call):
        body = call(lambda c: c.call_tool(
            "query_loans", {"sql": "SELECT grade FROM loans LIMIT 3"})).structured_content
        assert body["row_count"] == 3
        assert body["truncated"] is False

    def test_reports_a_sql_error_without_crashing(self, call):
        out = call(lambda c: c.call_tool(
            "query_loans", {"sql": "SELECT no_such_column FROM loans"}))
        assert out.is_error is True
        assert "SQL execution failed" in text_of(out)


class TestQueryLoansGuardrails:
    """A write or DDL statement must be refused before it reaches a database."""

    @pytest.mark.parametrize("sql,label", WRITE_AND_DDL, ids=[l for _, l in WRITE_AND_DDL])
    def test_refuses_writes_and_ddl(self, call, sql, label):
        out = call(lambda c: c.call_tool("query_loans", {"sql": sql}))
        assert out.is_error is True, f"{label} was NOT refused: {sql!r}"
        assert "rejected by guardrails" in text_of(out)

    def test_refusal_explains_itself_so_the_model_can_rewrite(self, call):
        out = call(lambda c: c.call_tool("query_loans", {"sql": "SELECT * FROM secrets"}))
        assert "table 'secrets' is not allowed" in text_of(out)

    def test_a_refused_write_leaves_the_loan_book_untouched(self, call):
        before = loan_book_row_count()
        for sql in ("DELETE FROM loans", "UPDATE loans SET is_default = 0",
                    "DROP TABLE loans", "SELECT 1; DELETE FROM loans"):
            assert call(lambda c: c.call_tool("query_loans", {"sql": sql})).is_error is True
        assert loan_book_row_count() == before
        assert config.PROCESSED_PARQUET.exists()

    def test_the_server_calls_the_existing_guardrail(self, monkeypatch, call):
        """Not a re-implementation: neutering validate_sql must break the tool."""
        calls: list[str] = []

        def spy(sql, allowed_tables=None):
            calls.append(sql)
            raise text2sql.UnsafeSQLError("sentinel")

        monkeypatch.setattr(text2sql, "validate_sql", spy)
        out = call(lambda c: c.call_tool("query_loans", {"sql": "SELECT 1 FROM loans"}))
        assert calls == ["SELECT 1 FROM loans"]
        assert out.is_error is True
        assert "sentinel" in text_of(out)

    def test_the_server_never_generates_sql_itself(self, monkeypatch, call):
        """The client writes the SQL. No LLM call happens inside this server."""
        def boom(*a, **k):
            raise AssertionError("mcp_server must not call text2sql.generate_sql")

        monkeypatch.setattr(text2sql, "generate_sql", boom)
        monkeypatch.setattr(text2sql, "ask", boom)
        out = call(lambda c: c.call_tool("query_loans", {"sql": "SELECT grade FROM loans LIMIT 1"}))
        assert out.is_error is False


# --------------------------------------------------------------------------
# Resources
# --------------------------------------------------------------------------

class TestResources:
    def test_firewall_resource_lists_the_28_origination_columns(self, call):
        res = call(lambda c: c.read_resource("creditrisk://schema/origination-firewall"))
        assert res.contents[0].mime_type == "application/json"
        doc = json.loads(res.contents[0].text)
        assert doc["origination_column_count"] == 28
        assert doc["origination_columns"] == config.ORIGINATION_COLS
        assert doc["out_of_time_split_date"] == config.SPLIT_DATE

    def test_firewall_resource_keeps_leakage_out_of_the_feature_list(self, call):
        doc = json.loads(call(lambda c: c.read_resource(
            "creditrisk://schema/origination-firewall")).contents[0].text)
        assert set(doc["model_feature_columns"]).isdisjoint(config.LEAKAGE_COLS)
        assert set(doc["origination_columns"]).isdisjoint(config.LEAKAGE_COLS)
        assert set(doc["excluded_leakage_columns"]) == set(config.LEAKAGE_COLS)

    def test_unity_catalog_resource_documents_the_feature_table(self, call):
        text = call(lambda c: c.read_resource(
            "creditrisk://schema/unity-catalog")).contents[0].text
        assert "loans_features" in text
        assert "is_default" in text and "issue_d" in text
        assert "lineage" in text.lower()

    def test_loan_book_resource_documents_the_guardrails(self, call):
        text = call(lambda c: c.read_resource(
            "creditrisk://schema/loan-book")).contents[0].text
        assert text2sql.SCHEMA_DOC in text
        for keyword in ("drop", "delete", "update", "insert"):
            assert f"`{keyword}`" in text
        assert str(text2sql.MAX_ROWS) in text

    def test_resources_stay_well_under_the_50kb_tool_response_budget(self, call):
        for uri in ("creditrisk://schema/origination-firewall",
                    "creditrisk://schema/unity-catalog",
                    "creditrisk://schema/loan-book"):
            size = len(call(lambda c: c.read_resource(uri)).contents[0].text.encode())
            assert 0 < size < 50_000, f"{uri} is {size} bytes"


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

class TestAdverseActionPrompt:
    def test_renders_with_the_decision_payload_embedded(self, call):
        decision = call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).structured_content
        payload = json.dumps(decision, indent=2, default=str)
        result = call(lambda c: c.get_prompt("adverse_action_notice", {
            "decision_json": payload, "applicant_name": "Jordan Reyes"}))
        assert len(result.messages) == 1
        text = result.messages[0].content.text
        assert payload in text
        assert "Jordan Reyes" in text

    def test_carries_the_ecoa_constraints(self, call):
        text = call(lambda c: c.get_prompt(
            "adverse_action_notice", {"decision_json": "{}"})).messages[0].content.text
        assert "Equal Credit Opportunity Act" in text
        assert "1002.9" in text
        assert "60 days" in text
        assert "protected characteristic" in text


# --------------------------------------------------------------------------
# Offline guarantees
# --------------------------------------------------------------------------

class TestRunsOffline:
    def test_no_tool_opens_a_network_connection(self, monkeypatch, call):
        """DNS and outbound connects are hard-failed for the duration."""
        def no_network(*a, **k):
            raise AssertionError("the MCP server attempted a network call")

        monkeypatch.setattr(socket, "getaddrinfo", no_network)
        monkeypatch.setattr(socket, "create_connection", no_network)

        assert call(lambda c: c.call_tool(
            "score_applicant", {"applicant": LOW_RISK})).is_error is False
        assert call(lambda c: c.call_tool(
            "explain_decision", {"applicant": HIGH_RISK})).is_error is False
        assert call(lambda c: c.call_tool(
            "query_loans", {"sql": "SELECT COUNT(*) AS n FROM loans"})).is_error is False
        assert call(lambda c: c.read_resource(
            "creditrisk://schema/origination-firewall")).contents
        assert call(lambda c: c.get_prompt(
            "adverse_action_notice", {"decision_json": "{}"})).messages

    def test_needs_no_warehouse(self):
        """Nothing in the served path imports Spark or a Databricks client."""
        import sys

        import mcp_server.server  # noqa: F401

        assert not [m for m in sys.modules if m.startswith(("pyspark", "databricks"))]


class TestProductionArtifacts:
    """Run against the real repo artifacts when they exist; skip on a fresh clone."""

    def test_production_policy_threshold_is_still_0475(self):
        real = config.PROJECT_ROOT / "models" / "policy.json"
        if not real.exists():
            pytest.skip("no trained policy on this machine — run python -m src.business")
        policy = json.loads(real.read_text())
        assert policy["approve_below_pd"] == 0.475
        assert policy["lgd_assumption"] == 0.65
        assert policy["pd_is_calibrated"] is True
