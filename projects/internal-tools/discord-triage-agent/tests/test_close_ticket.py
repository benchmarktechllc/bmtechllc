"""The close_ticket executor (SPEC.md section 5): one transaction, idempotent through its
receipt, and refusing to run without an approved request."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

import psycopg
import pytest

from tests.conftest import AVERY, insert_test_session
from tests.flows import receipts, ticket
from triage_agent.tools.close_ticket import execute_close

Db = psycopg.Connection[tuple[Any, ...]]


def make_request(
    db: Db,
    session_id: UUID,
    ticket_id: str,
    status: str = "approved",
    tool_use_id: str = "tu_close",
) -> UUID:
    decided = status != "pending"
    row = db.execute(
        "INSERT INTO approval_requests (session_id, turn, tool_use_id, tool_name, tool_input,"
        " status, requested_by, decided_by, decided_at, expires_at)"
        " VALUES (%s, 1, %s, 'close_ticket', %s, %s, 'u', %s,"
        " CASE WHEN %s THEN now() END, now() + interval '1 h')"
        " RETURNING id",
        (
            session_id,
            tool_use_id,
            json.dumps({"ticket_id": ticket_id}),
            status,
            AVERY if decided and status != "expired" else None,
            decided,
        ),
    ).fetchone()
    assert row is not None
    request_id: UUID = row[0]
    return request_id


async def close(url: str, request_id: UUID) -> Any:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as conn:
        return await execute_close(conn, request_id)


@pytest.mark.anyio
async def test_approved_close_updates_the_ticket_and_writes_one_receipt(
    seeded_database_url: str, db: Db
) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1002")
    _, notes_before = ticket(db, "T-1002")

    out = await close(seeded_database_url, rid)

    assert out.is_error is False
    assert json.loads(out.content) == {
        "ticket_id": "T-1002",
        "status": "closed",
        "already_closed": False,
    }
    assert ticket(db, "T-1002") == ("closed", notes_before + 1)
    ((req_id, tool_name, tool_input, outcome, result),) = receipts(db, sid)
    assert (req_id, tool_name, tool_input, outcome) == (
        rid,
        "close_ticket",
        {"ticket_id": "T-1002"},
        "succeeded",
    )
    assert result == {"content": out.content, "is_error": False}


@pytest.mark.anyio
async def test_repeated_call_returns_the_stored_result_and_leaves_the_ticket_alone(
    seeded_database_url: str, db: Db
) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1002")
    first = await close(seeded_database_url, rid)
    # Reopen it by hand: a second run that touched the ticket would close it again.
    db.execute("UPDATE tickets SET status = 'open' WHERE id = 'T-1002'")
    notes = ticket(db, "T-1002")[1]

    second = await close(seeded_database_url, rid)

    assert second == first
    assert ticket(db, "T-1002") == ("open", notes)
    assert len(receipts(db, sid)) == 1


@pytest.mark.anyio
async def test_concurrent_runs_for_one_request_close_once(seeded_database_url: str, db: Db) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1002")
    _, notes_before = ticket(db, "T-1002")

    a, b = await asyncio.gather(close(seeded_database_url, rid), close(seeded_database_url, rid))

    assert a == b
    assert ticket(db, "T-1002") == ("closed", notes_before + 1)
    assert len(receipts(db, sid)) == 1


@pytest.mark.anyio
async def test_already_closed_ticket_succeeds_with_already_closed(
    seeded_database_url: str, db: Db
) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1004")
    before = ticket(db, "T-1004")
    assert before[0] == "closed"

    out = await close(seeded_database_url, rid)

    assert out.is_error is False
    assert json.loads(out.content)["already_closed"] is True
    assert ticket(db, "T-1004") == before
    ((_, _, _, outcome, _),) = receipts(db, sid)
    assert outcome == "succeeded"


@pytest.mark.anyio
async def test_missing_ticket_gives_a_failed_receipt_and_an_error_result(
    seeded_database_url: str, db: Db
) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-9999")

    out = await close(seeded_database_url, rid)

    assert out.is_error is True
    assert "ticket not found" in out.content
    ((_, _, _, outcome, result),) = receipts(db, sid)
    assert outcome == "failed"
    assert result == {"content": out.content, "is_error": True}


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["pending", "denied", "expired"])
async def test_refuses_without_an_approved_request(
    status: str, seeded_database_url: str, db: Db
) -> None:
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1002", status=status)
    before = ticket(db, "T-1002")

    out = await close(seeded_database_url, rid)

    assert out.is_error is True
    assert ticket(db, "T-1002") == before
    assert receipts(db, sid) == []


@pytest.mark.anyio
async def test_unknown_request_is_refused(seeded_database_url: str, db: Db) -> None:
    out = await close(seeded_database_url, UUID(int=1))
    assert out.is_error is True
    assert db.execute("SELECT count(*) FROM receipts").fetchone() == (0,)


@pytest.mark.anyio
async def test_ticket_change_rolls_back_if_the_receipt_cannot_be_written(
    seeded_database_url: str, db: Db
) -> None:
    db.execute(
        "CREATE FUNCTION fail_receipt() RETURNS trigger LANGUAGE plpgsql AS"
        " $$ BEGIN RAISE EXCEPTION 'receipt write failed'; END $$"
    )
    db.execute(
        "CREATE TRIGGER fail_receipt BEFORE INSERT ON receipts"
        " FOR EACH ROW EXECUTE FUNCTION fail_receipt()"
    )
    sid = insert_test_session(db)
    rid = make_request(db, sid, "T-1002")
    before = ticket(db, "T-1002")

    with pytest.raises(psycopg.Error):
        await close(seeded_database_url, rid)

    assert ticket(db, "T-1002") == before
