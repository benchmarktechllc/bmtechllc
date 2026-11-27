"""Apply migrations/*.sql in filename order, each once, each in its own transaction.

Usage: python -m triage_agent.db.migrate   (reads DATABASE_URL from the environment / .env)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

# src/triage_agent/db/migrate.py -> project root. Works for the editable install used here.
MIGRATIONS_DIR = Path(__file__).resolve().parents[3] / "migrations"

_TRACKING_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    filename   text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
)
"""


def apply_migrations(conn: psycopg.Connection[Any], directory: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply every not-yet-applied migration in `directory`. Returns the files applied.

    A migration that fails is rolled back and not recorded, and later files are not run.
    """
    with conn.transaction():
        conn.execute(_TRACKING_TABLE)
        # Like every table here: no PostgREST/anon access. Idempotent.
        conn.execute("ALTER TABLE schema_migrations ENABLE ROW LEVEL SECURITY")
    applied = {row[0] for row in conn.execute("SELECT filename FROM schema_migrations")}

    done: list[str] = []
    for path in sorted(directory.glob("*.sql")):
        if path.name in applied:
            continue
        with conn.transaction():
            # bytes, not str: a migration file is trusted repo SQL, run as one script.
            conn.execute(path.read_bytes())
            conn.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
        done.append(path.name)
    return done


def main() -> None:
    from triage_agent.config import require_env

    load_dotenv()
    with psycopg.connect(require_env("DATABASE_URL")) as conn:
        applied = apply_migrations(conn)
    print("applied:", ", ".join(applied) if applied else "nothing (up to date)")


if __name__ == "__main__":
    main()
