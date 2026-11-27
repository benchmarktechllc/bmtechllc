"""The close_ticket executor (SPEC.md section 5): the one irreversible action.

One transaction: it refuses unless an `approved` request exists, returns the stored receipt
if the close already ran, and otherwise changes the ticket and inserts the receipt together.
This check is independent of the gate in agent/loop.py (defense in depth).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from triage_agent.tools.executors import ToolOutcome

CLOSE_TICKET = "close_ticket"


def _stored(result: object) -> ToolOutcome:
    if not isinstance(result, dict) or not isinstance(result.get("content"), str):
        raise ValueError("malformed receipt result")
    return ToolOutcome(content=result["content"], is_error=bool(result.get("is_error")))


async def execute_close(conn: psycopg.AsyncConnection[Any], request_id: UUID) -> ToolOutcome:
    async with conn.transaction():
        cur = await conn.execute(
            "SELECT session_id, tool_name, tool_input, status FROM approval_requests"
            " WHERE id = %s FOR UPDATE",
            (request_id,),
        )
        req = await cur.fetchone()
        if req is None or req[1] != CLOSE_TICKET or req[3] != "approved":
            return ToolOutcome("close_ticket refused: no approved request", is_error=True)
        session_id, tool_name, tool_input, _ = req

        cur = await conn.execute(
            "SELECT result FROM receipts WHERE approval_request_id = %s", (request_id,)
        )
        receipt = await cur.fetchone()
        if receipt is not None:
            return _stored(receipt[0])

        raw_id = tool_input.get("ticket_id") if isinstance(tool_input, dict) else None
        ticket_id = raw_id.strip().upper() if isinstance(raw_id, str) else ""
        cur = await conn.execute(
            "SELECT status FROM tickets WHERE id = %s FOR UPDATE", (ticket_id,)
        )
        row = await cur.fetchone()
        if row is None:
            outcome = ToolOutcome(f"ticket not found: {raw_id}", is_error=True)
        elif row[0] == "closed":
            outcome = ToolOutcome(
                json.dumps({"ticket_id": ticket_id, "status": "closed", "already_closed": True})
            )
        else:
            note = {
                "at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "by": "triage-agent",
                "note": f"Closed after approval (request {request_id}).",
            }
            await conn.execute(
                "UPDATE tickets SET status = 'closed', history = history || %s WHERE id = %s",
                (Jsonb([note]), ticket_id),
            )
            outcome = ToolOutcome(
                json.dumps({"ticket_id": ticket_id, "status": "closed", "already_closed": False})
            )

        await conn.execute(
            "INSERT INTO receipts (approval_request_id, session_id, tool_name, tool_input,"
            " outcome, result) VALUES (%s, %s, %s, %s, %s, %s)",
            (
                request_id,
                session_id,
                tool_name,
                Jsonb(tool_input),
                "failed" if outcome.is_error else "succeeded",
                Jsonb({"content": outcome.content, "is_error": outcome.is_error}),
            ),
        )
        return outcome
