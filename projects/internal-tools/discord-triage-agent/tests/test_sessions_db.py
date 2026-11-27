"""migrations/001_sessions.sql and the migration runner, against a real local Postgres."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
import pytest

from triage_agent.db.migrate import MIGRATIONS_DIR, apply_migrations
from triage_agent.db.sessions import NewSession, insert_session

EXPECTED_COLUMNS = {
    "id",
    "interaction_id",
    "guild_id",
    "channel_id",
    "asker_user_id",
    "question",
    "concerns",
    "more_concerns",
    "primary_category",
    "categories",
    "confidence",
    "status",
    "iterations",
    "heartbeat_at",
    "reply_error",
    "created_at",
    "updated_at",
}


def new_session(interaction_id: str = "i-1") -> NewSession:
    return NewSession(
        interaction_id=interaction_id,
        guild_id="g",
        channel_id="c",
        asker_user_id="u",
        question="q?",
    )


def test_sessions_table_has_spec_columns(db: psycopg.Connection[tuple[Any, ...]]) -> None:
    cols = {
        r[0]
        for r in db.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'sessions'"
        )
    }
    assert cols == EXPECTED_COLUMNS


def test_sessions_has_rls_enabled_with_no_policies(
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    assert db.execute(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'sessions'"
    ).fetchone() == (True,)
    assert db.execute(
        "SELECT count(*) FROM pg_policies WHERE tablename = 'sessions'"
    ).fetchone() == (0,)


def test_interaction_id_is_unique(db: psycopg.Connection[tuple[Any, ...]]) -> None:
    insert = (
        "INSERT INTO sessions (interaction_id, guild_id, channel_id, asker_user_id, question)"
        " VALUES ('dup', 'g', 'c', 'u', 'q')"
    )
    db.execute(insert)
    with pytest.raises(psycopg.errors.UniqueViolation):
        db.execute(insert)


def test_status_is_constrained(db: psycopg.Connection[tuple[Any, ...]]) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        db.execute(
            "INSERT INTO sessions"
            " (interaction_id, guild_id, channel_id, asker_user_id, question, status)"
            " VALUES ('x', 'g', 'c', 'u', 'q', 'bogus')"
        )


def test_new_session_defaults(db: psycopg.Connection[tuple[Any, ...]]) -> None:
    db.execute(
        "INSERT INTO sessions (interaction_id, guild_id, channel_id, asker_user_id, question)"
        " VALUES ('d', 'g', 'c', 'u', 'q')"
    )
    row = db.execute(
        "SELECT status, iterations, heartbeat_at IS NOT NULL, created_at IS NOT NULL FROM sessions"
    ).fetchone()
    assert row == ("running", 0, True, True)


@pytest.mark.anyio
async def test_insert_session_is_idempotent_on_interaction_id(database_url: str) -> None:
    async with await psycopg.AsyncConnection.connect(database_url, autocommit=True) as conn:
        first_id, first_created = await insert_session(conn, new_session())
        again_id, again_created = await insert_session(conn, new_session())
        other_id, other_created = await insert_session(conn, new_session("i-2"))
        count = await (await conn.execute("SELECT count(*) FROM sessions")).fetchone()

    assert first_created is True
    assert (again_id, again_created) == (first_id, False)
    assert other_created is True and other_id != first_id
    assert count == (2,)


def test_runner_skips_already_applied_migrations(
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    # The fixture already applied everything; a second run applies nothing.
    assert apply_migrations(db) == []
    applied = [r[0] for r in db.execute("SELECT filename FROM schema_migrations ORDER BY 1")]
    assert applied == sorted(p.name for p in MIGRATIONS_DIR.glob("*.sql"))
    assert "001_sessions.sql" in applied


def test_runner_applies_new_files_in_order(
    db: psycopg.Connection[tuple[Any, ...]], tmp_path: Path
) -> None:
    (tmp_path / "002_b.sql").write_text("CREATE TABLE b_marker (n int);")
    (tmp_path / "001_a.sql").write_text("CREATE TABLE a_marker (n int);")
    assert apply_migrations(db, tmp_path) == ["001_a.sql", "002_b.sql"]
    assert apply_migrations(db, tmp_path) == []


def test_failed_migration_is_not_recorded(
    db: psycopg.Connection[tuple[Any, ...]], tmp_path: Path
) -> None:
    (tmp_path / "001_bad.sql").write_text("CREATE TABLE ok_part (n int); SELECT broken(;")
    with pytest.raises(psycopg.Error):
        apply_migrations(db, tmp_path)
    assert db.execute(
        "SELECT count(*) FROM schema_migrations WHERE filename = '001_bad.sql'"
    ).fetchone() == (0,)
    assert db.execute("SELECT to_regclass('ok_part')").fetchone() == (None,)


def test_schema_migrations_has_rls_enabled_with_no_policies(
    db: psycopg.Connection[tuple[Any, ...]],
) -> None:
    # Must hold on plain Postgres too, not only where Supabase auto-enables RLS.
    assert db.execute(
        "SELECT relrowsecurity FROM pg_class WHERE relname = 'schema_migrations'"
    ).fetchone() == (True,)
    assert db.execute(
        "SELECT count(*) FROM pg_policies WHERE tablename = 'schema_migrations'"
    ).fetchone() == (0,)
