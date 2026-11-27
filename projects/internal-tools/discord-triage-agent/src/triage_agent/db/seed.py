"""Load seed/synthetic.sql (synthetic tickets, KB articles, teammates). Idempotent.

Usage: python -m triage_agent.db.seed   (reads DATABASE_URL from the environment / .env)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

# src/triage_agent/db/seed.py -> project root. Like MIGRATIONS_DIR, needs the editable install.
SEED_FILE = Path(__file__).resolve().parents[3] / "seed" / "synthetic.sql"


def apply_seed(conn: psycopg.Connection[Any], path: Path = SEED_FILE) -> None:
    """Run the seed file in one transaction. Rows that already exist are left alone."""
    with conn.transaction():
        # bytes, not str: the seed is trusted repo SQL, run as one script.
        conn.execute(path.read_bytes())


def main() -> None:
    from triage_agent.config import require_env

    load_dotenv()
    with psycopg.connect(require_env("DATABASE_URL")) as conn:
        apply_seed(conn)
    print("seeded:", SEED_FILE.name)


if __name__ == "__main__":
    main()
