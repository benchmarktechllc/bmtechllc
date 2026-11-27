"""Read tools, the deny-by-default policy, migration 002 and the synthetic seed."""

from __future__ import annotations

import json
from typing import Any

import psycopg
import pytest

from triage_agent.db.seed import apply_seed
from triage_agent.tools.definitions import TOOLS
from triage_agent.tools.executors import execute_tool
from triage_agent.tools.policy import POLICY, decide

Db = psycopg.Connection[tuple[Any, ...]]

NEW_TABLES = ["messages", "traces", "tickets", "kb_articles", "teammates"]


# --- policy ----------------------------------------------------------------------


def test_policy_allows_only_the_read_tools() -> None:
    assert POLICY == {"get_ticket": "allow", "search_kb": "allow"}
    assert decide("get_ticket") == "allow"
    assert decide("search_kb") == "allow"


@pytest.mark.parametrize("name", ["close_ticket", "delete_ticket", "", "GET_TICKET"])
def test_any_other_tool_is_denied(name: str) -> None:
    assert decide(name) == "deny"


def test_tool_definitions_are_strict_and_match_the_policy() -> None:
    assert sorted(t["name"] for t in TOOLS) == sorted(POLICY)
    for tool in TOOLS:
        assert tool.get("strict") is True
        schema = tool["input_schema"]
        assert schema.get("additionalProperties") is False
        assert schema.get("required")


# --- schema and seed ---------------------------------------------------------------


@pytest.mark.parametrize("table", NEW_TABLES)
def test_new_tables_have_rls_enabled_with_no_policies(db: Db, table: str) -> None:
    assert db.execute(
        "SELECT relrowsecurity FROM pg_class WHERE relname = %s", (table,)
    ).fetchone() == (True,)
    assert db.execute(
        "SELECT count(*) FROM pg_policies WHERE tablename = %s", (table,)
    ).fetchone() == (0,)


def test_traces_columns(db: Db) -> None:
    cols = {
        r[0]
        for r in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'traces'"
        )
    }
    assert {
        "id",
        "session_id",
        "iteration",
        "model",
        "stop_reason",
        "input_tokens",
        "output_tokens",
        "tool_use_id",
        "tool_name",
        "tool_input",
        "tool_result",
        "hook_decision",
        "latency_ms",
        "error",
        "created_at",
    } <= cols


def test_hook_decision_is_constrained(db: Db) -> None:
    sid = db.execute(
        "INSERT INTO sessions (interaction_id, guild_id, channel_id, asker_user_id, question)"
        " VALUES ('x', 'g', 'c', 'u', 'q') RETURNING id"
    ).fetchone()
    assert sid is not None
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO traces (session_id, iteration, model, hook_decision)"
            " VALUES (%s, 1, 'm', 'maybe')",
            (sid[0],),
        )


def test_seed_loads_the_synthetic_data_and_is_idempotent(seeded_database_url: str) -> None:
    with psycopg.connect(seeded_database_url) as conn:
        apply_seed(conn)  # a second run changes nothing
        counts = {
            t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()  # noqa: S608 - fixed names
            for t in ("tickets", "kb_articles", "teammates")
        }
    assert counts == {"tickets": (10,), "kb_articles": (8,), "teammates": (4,)}


# --- executors ----------------------------------------------------------------------


@pytest.mark.anyio
async def test_get_ticket_returns_the_ticket(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "get_ticket", {"ticket_id": "T-1002"})
    assert out.is_error is False
    ticket = json.loads(out.content)
    assert ticket["id"] == "T-1002"
    assert {"subject", "status", "customer", "history"} <= ticket.keys()


@pytest.mark.anyio
async def test_get_ticket_is_case_and_space_tolerant(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "get_ticket", {"ticket_id": " t-1002 "})
    assert json.loads(out.content)["id"] == "T-1002"


@pytest.mark.anyio
async def test_missing_ticket_is_an_error_result(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "get_ticket", {"ticket_id": "T-9999"})
    assert out.is_error is True
    assert "not found" in out.content


@pytest.mark.anyio
async def test_search_kb_finds_matching_articles(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "search_kb", {"query": "how do I reset my password"})
    assert out.is_error is False
    articles = json.loads(out.content)
    assert 1 <= len(articles) <= 3
    assert "password" in articles[0]["title"].lower()


@pytest.mark.anyio
async def test_search_kb_with_no_match_returns_an_empty_list(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "search_kb", {"query": "zxqv flibbertigibbet"})
    assert out.is_error is False
    assert json.loads(out.content) == []


@pytest.mark.anyio
async def test_search_kb_treats_sql_and_tsquery_syntax_as_text(seeded_database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, "search_kb", {"query": "'); DROP TABLE tickets; -- & | !"})
        still_there = await (await conn.execute("SELECT count(*) FROM tickets")).fetchone()
    assert out.is_error is False
    assert still_there == (10,)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("name", "tool_input"),
    [
        ("get_ticket", {}),
        ("get_ticket", {"ticket_id": 1002}),
        ("search_kb", {"query": ""}),
        ("search_kb", {"q": "x"}),
        ("close_ticket", {"ticket_id": "T-1002"}),
    ],
)
async def test_bad_input_or_unknown_tool_is_an_error_result(
    seeded_database_url: str, name: str, tool_input: dict[str, Any]
) -> None:
    async with await psycopg.AsyncConnection.connect(seeded_database_url) as conn:
        out = await execute_tool(conn, name, tool_input)
    assert out.is_error is True
