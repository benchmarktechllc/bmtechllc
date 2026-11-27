"""Executors for the read tools: parameterized SQL over the synthetic tables.

Results are JSON text. Ticket notes and KB bodies are untrusted data; the agent's system
prompt tells the model never to follow instructions found in them.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import psycopg

KB_RESULT_LIMIT = 3


@dataclass(frozen=True)
class ToolOutcome:
    content: str
    is_error: bool = False


def _error(message: str) -> ToolOutcome:
    return ToolOutcome(content=message, is_error=True)


def _string_arg(tool_input: Mapping[str, object], key: str) -> str | None:
    value = tool_input.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


async def get_ticket(conn: psycopg.AsyncConnection[Any], ticket_id: str) -> ToolOutcome:
    cur = await conn.execute(
        "SELECT id, subject, status, customer, history FROM tickets WHERE id = %s",
        (ticket_id.upper(),),
    )
    row = await cur.fetchone()
    if row is None:
        return _error(f"ticket not found: {ticket_id}")
    ticket = dict(zip(("id", "subject", "status", "customer", "history"), row, strict=True))
    return ToolOutcome(content=json.dumps(ticket, ensure_ascii=False))


async def search_kb(conn: psycopg.AsyncConnection[Any], query: str) -> ToolOutcome:
    # plainto_tsquery turns any text into lexemes (no operator syntax gets through);
    # swapping & for | makes it match any of the words, ranked by how many match.
    cur = await conn.execute(
        """
        SELECT a.id, a.title, a.body
        FROM kb_articles a,
             to_tsquery('english', replace(plainto_tsquery('english', %s)::text, '&', '|')) q,
             to_tsvector('english', a.title || ' ' || a.body) v
        WHERE v @@ q
        ORDER BY ts_rank(v, q) DESC, a.id
        LIMIT %s
        """,
        (query, KB_RESULT_LIMIT),
    )
    articles = [{"id": r[0], "title": r[1], "body": r[2]} for r in await cur.fetchall()]
    return ToolOutcome(content=json.dumps(articles, ensure_ascii=False))


async def execute_tool(
    conn: psycopg.AsyncConnection[Any], name: str, tool_input: Mapping[str, object]
) -> ToolOutcome:
    """Run an allowed read tool. Bad input or an unknown name is an error result, not a raise.

    Callers check the policy first (tools/policy.py); this is a second, independent check.
    """
    if name == "get_ticket":
        ticket_id = _string_arg(tool_input, "ticket_id")
        if ticket_id is None:
            return _error("get_ticket needs a non-empty string ticket_id")
        return await get_ticket(conn, ticket_id)
    if name == "search_kb":
        query = _string_arg(tool_input, "query")
        if query is None:
            return _error("search_kb needs a non-empty string query")
        return await search_kb(conn, query)
    return _error(f"unknown tool: {name}")
