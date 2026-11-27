# Notes

## 2026-11-27: Slice 2 code done (live run pending)

State:
- Classifier (`agent/classifier.py`), loop (`agent/loop.py`), pipeline (`agent/pipeline.py`), read tools (`tools/`), `budget.py`, `migrations/002_conversation_and_synthetic.sql`, `seed/synthetic.sql` (+ `python -m triage_agent.db.seed`), `fixtures/live_cases.json`, `scripts/live_smoke.py`.
- `pytest` (115 passed), `ruff check`, `ruff format --check` and `mypy --strict src` all pass. No live model call has been made yet.
- New dependency: `anthropic` (2.14.0, pulls in `httpx2`).
- `/ask` now replies with the agent's answer; the "Received — session <id>" edit is gone. Failures reply with a fixed apology naming the session id.

Design choices made this slice (approved before coding):
- Classifier writes a `traces` row (iteration 0, raw output in `tool_result.classifier_output`) but no `messages` row; `messages` is only the agent conversation.
- `traces.error` (new column) records API errors and timeouts; those rows have NULL `stop_reason` and tokens, and no `messages` row.
- Classifier runs with thinking disabled and effort `low` (Haiku 5.5 thinks by default; 256 tokens is too tight). A classifier API error falls back to `other` and the session continues.
- Agent: Sonnet 5.5, effort `low`, adaptive thinking (default). Thinking blocks are stored and sent back unchanged. No `fallbacks`.
- Each loop request is rebuilt from the stored `messages` rows, so what the model sees is what is stored. Slice 3's resume can reuse this.
- Replies go out with `allowed_mentions: {parse: []}` and are cut to 2000 characters. The interaction token is redacted from a recorded `reply_error`.
- Anthropic client: `timeout=45s`, `max_retries=1` (worst case ~90 s).
- Ticket T-1008's history contains a planted prompt injection ("close every ticket… pre-approved") for the live run and Slice 3's gate tests.

Next (by hand):
1. `docker start triage-agent-test-pg`, then `python scripts/live_smoke.py --max-sessions 1`, check the output and cost, then the full run (≤5 sessions). Record pass/fail and total spend here. If a categories assertion fails because the classifier picked a defensible label (for example `account` vs `how_to`), fix the fixture or prompt, not the assertion.
2. Supabase: `python -m triage_agent.db.migrate && python -m triage_agent.db.seed` (DATABASE_URL), then one live `/ask` in the test guild.

Then Slice 3 (`/new-slice 3`). There's no Anthropic key yet, so Slice 3 goes ahead before the Slice 2 live run; its tests only use the fakes. An approved reference plan is at `~/.claude/plans/i-can-t-find-an-abundant-meadow.md`. It was agreed to keep interaction tokens in memory only: after a restart, replies go out by bot-token REST. Still confirm the plan and the edits to existing tests at the skill's step 2.
- Add `close_ticket: require_approval` to `tools/policy.py` and its definition to `tools/definitions.py`; the loop's `_run_tool` is where pausing goes.
- The loop already appends the assistant message before executing tools, in its own transaction; Slice 3 needs approval requests created in the same turn.
- Open items carried over: README *Backup* section (Supabase plan, idle pause, backups) still TBD; `.env` needs `APPROVER_CHANNEL_ID`, `APPROVER_ROLE_ID`, `APPROVAL_TTL`.

## 2026-11-27: Slice 1 done

- Verified live: Portal accepted `<tunnel-url>/interactions`, `/ask` replied "Received — session <id>", one `sessions` row in Supabase. `001_sessions.sql` applied there.
- `/ask` registered for the test guild; bot invited with `bot` + `applications.commands`.
- Local test Postgres: Docker container `triage-agent-test-pg` on 127.0.0.1:54329. `cloudflared` 2026.11.2 at `~/.local/bin/cloudflared`.
