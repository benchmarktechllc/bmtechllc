# Notes

## 2026-11-27: Slice 1 done

State:
- Slice 1 is complete and verified live: Portal accepted `<tunnel-url>/interactions`, `/ask` in the test guild replied "Received — session <id>", one row in the Supabase `sessions` table.
- Supabase has `001_sessions.sql` applied (tracked in `schema_migrations`, RLS on both tables).
- `/ask` is registered for the test guild. The bot is invited with `bot` + `applications.commands`.
- Local test Postgres: Docker container `triage-agent-test-pg` on 127.0.0.1:54329.
- `cloudflared` 2026.11.2 is installed at `~/.local/bin/cloudflared` (no auto-update).

Open items:
- README *Backup* section: record the Supabase plan, whether idle projects pause, and included backups (still TBD).
- `.env` still needs `ANTHROPIC_API_KEY` for Slice 2, and `APPROVER_CHANNEL_ID`, `APPROVER_ROLE_ID`, `APPROVAL_TTL` for Slice 3.

Next: Slice 2 (classify and answer with read tools), via `/new-slice 2`.
- Will need new dependencies (ask first): at least `anthropic`. Check whether a DB pool (`psycopg_pool`) is worth it once the background loop holds connections.
- The background task in `app.py` (`send_received`) is where the classifier and loop get kicked off; the "Received" edit gets replaced by the real reply.
- `migrations/002_*.sql`: `messages`, `traces`, `tickets`, `kb_articles`, `teammates`, each with RLS enabled. Add `seed/synthetic.sql`.
- Add the scripted fake Anthropic client to `tests/conftest.py`, next to `FakeDiscordRest`.
