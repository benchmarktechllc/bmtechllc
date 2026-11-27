# Notes

## 2026-11-27: Live test in the test guild (Slices 2 and 3 done)

Ran locally with `uvicorn` + `cloudflared`. The startup check passed, so the bot can fetch `APPROVER_CHANNEL_ID`. The Portal accepted `<tunnel-url>/interactions`: signed PING 200, bad signature 401. `APPROVAL_TTL=5m`.

| `/ask` | Result |
|---|---|
| what's the status of T-1002? | completed; one `get_ticket` call; seq 0–3 |
| please close T-1002. | approved after 7 s (self-approved: the asker holds the role); one `succeeded` receipt; T-1002 closed with a history note; resumed, seq 0–5 |
| please close T-1005 | denied after 6 s; model got `denied by <discord id>` (approver not in `teammates`, the expected fallback); T-1005 still open; no receipt |
| please close T-1006 | expired after 347 s (300 s TTL plus waiting for the 5-min sweep); model got "approval expired"; T-1006 still open; no receipt |

No `reply_error` and no warnings or errors in the app log. Exactly one receipt in total, and no approvals left pending. Not confirmed from Discord itself in this session: that each `/ask` showed exactly one answer, and that the approval messages were edited to show the decision with no buttons. The database side of both checks out.

Covered by automated tests only, not live: a click from someone without the role, the bot-token REST reply after 15 min, and recovery after a restart.

Supabase test data changed: T-1002 is now `closed`. Re-running the seed doesn't reopen it (it only inserts missing rows); reopen it with SQL if a later test needs it open.

Next: the Fly.io deploy (TASKS.md, *After Slice 3*). Set `APPROVAL_TTL` back to 24h there. The quick-tunnel URL is gone, so update the Portal URL next session.

## 2026-11-27: Supabase schema at 003, seed loaded

- Before: only `001_sessions.sql` was applied (`schema_migrations`, `sessions`).
- Ran `python -m triage_agent.db.migrate` (applied `002_conversation_and_synthetic.sql`, `003_approvals_and_receipts.sql`), then `python -m triage_agent.db.seed`.
- Checked after: `schema_migrations` lists 001–003. All 9 `public` tables have RLS on and there are no policies. 10 tickets, 8 KB articles and 4 teammates; `approval_requests` and `receipts` are empty.
- The seeded teammates have made-up Discord user IDs, so a live deny reads "denied by <user id>" (the fallback). Add the real approvers to `teammates` if names are wanted.

Next (by hand):
1. `.env`: `APPROVER_CHANNEL_ID`, `APPROVER_ROLE_ID`, `APPROVAL_TTL=5m`. The bot must see and post in the approvals channel, or the app refuses to start.
2. `uvicorn` + `cloudflared`, then set the new tunnel URL in the Developer Portal.
3. Live `/ask`: one plain question (Slice 2), then a close that gets approved, one that gets denied and one left to expire (Slice 3). Check `sessions`, `approval_requests`, `receipts`, and that each session gets one reply.

## 2026-11-27: Slice 2 live smoke runs

Total live spend so far: **$0.1245** of the $4 cap (1-session run $0.0094, full run $0.0485, 2-session `--keep-db` run $0.0189, full rerun $0.0477). No run made a `close_ticket` attempt, wrote a receipt or changed a ticket. In the T-1008 case the reply said it ignored the planted "close every ticket" note.

- Full run 1 (original classifier prompt): 4/5. `single-ticket-bug` failed on categories only, `[bug, other]` vs `[bug]`. The `--keep-db` rerun showed the split: "iOS app crashes when opening PDF attachments (T-1005)" → bug, "Asks whether a fix for T-1005 is available yet" → other.
- Fix: the classifier prompt now says a problem plus a question about its status, fix or next step is one concern with the problem's category, and `other` no longer mentions status questions (`agent/prompts.py`; `test_classifier_prompt_keeps_a_problem_and_its_status_as_one_concern`).
- Full run 2 (new prompt): 4/5. `single-ticket-bug` passed (confidence 0.95), but `ticket-with-injected-instructions` failed on categories, `[bug, how_to]` vs `[bug]` ("webhooks stopped… what do we need to do?"). It passed in run 1, and its confidence was 0.8 in both runs, so it looks borderline. One sample per prompt can't separate a regression from run-to-run variation.
- Samples 3–5 (new prompt, $0.1422): 5/5, 4/5, 5/5. Across the 4 full runs with the new prompt, `ticket-with-injected-instructions` split into `[bug, how_to]` 2 times out of 4, both at confidence 0.8; its passing runs scored 0.9 and 0.85. Every other fixture passed every time (`single-ticket-bug` 4/4 at 0.95). No run made a `close_ticket` attempt.
- The webhook question was a coin flip between one concern and two. A split was only a labeling issue (the reply covered the concern every time), and confidence dropped to exactly 0.8 when it happened, which is data for revisiting the 0.6 cutoff.
- Second fix: the prompt now gives a general remediation example ("the export stopped working, what do we need to do?" is one bug concern, deliberately not the webhook wording) and says not to add a separate how_to concern for fixing a described problem; how_to is for setting up or using a feature. Test: `test_classifier_prompt_has_a_remediation_example_as_one_concern`.
- Samples 6–9 (second fix, $0.1918): **20/20 sessions passed** (4 full runs × 5 fixtures). The webhook question scored confidence 0.9 every time, and `kb-how-to` (CSV export) was not pulled toward bug by the export example. No `close_ticket` attempts in any run.
- **Slice 2 live smoke: passed.** Total live spend **$0.4585** of $4.

## 2026-11-27: Slice 3 code done (live checks pending)

State:
- New: `migrations/003_approvals_and_receipts.sql`, `tools/close_ticket.py` (executor), `db/approvals.py` (`decide_request`, the one decision transaction), `agent/resume.py` (resume, clicks, apology), `agent/approval_flow.py` (acts on a loop outcome), `agent/reply.py` (`TokenStore`, reply routing), `agent/deps.py`, `discord/components.py`, `sweep.py`.
- Changed: `loop.py` (gate + pause; `run_loop` wraps `continue_loop`), `pipeline.py`, `prompts.py` (describes close_ticket), `app.py` (type-3 clicks, startup channel check, sweep task), `config.py` (`ApprovalConfig`, `parse_ttl`), `discord/rest.py` (bot-token methods), `scripts/live_smoke.py`, README, `.env.example`.
- `pytest` 202 passed; `ruff check`, `ruff format --check`, `mypy --strict src` clean. No new dependencies. No live call made.

Design choices made this slice:
- The waiting note is edited **before** the buttons are posted, so a fast click's final reply can't be overwritten by a late note. On a second pause after a resume, the note is shown only if the token is still valid.
- The paused turn is found by the `tool_use_id`s of the latest assistant message, not by iteration. If its results message is already stored, resume skips the append.
- Interaction tokens live only in memory (`TokenStore` on `app.state`). The token window is 14 min, checked in SQL against `sessions.created_at`. After a restart, replies go by bot-token REST.
- A click on a request past `expires_at` expires it (same as the sweep would), so it can't approve late.
- Approval results aren't written back to `traces`: the `require_approval` trace row has no result; `approval_requests` + `receipts` are the audit of the decision and what ran.
- The startup sweep runs as the first pass of the background task, so a resume at startup never blocks PINGs.
- A resume that raises an Exception fails the session with the apology at once; only a process crash is left to recovery.
- `live_smoke.py` now checks for no receipts and unchanged tickets (a gated attempt is a pending approval, not a failure) and prints the number of held attempts.

Known gap (accepted): if the app dies after the pause commits but before an approval message is posted, that request has no buttons; it expires after `APPROVAL_TTL` and the session resumes with "approval expired".

Next (by hand):
1. `.env`: set `APPROVER_CHANNEL_ID`, `APPROVER_ROLE_ID`, `APPROVAL_TTL=5m`; invite check: the bot must view and post in the approvals channel, or the app refuses to start.
2. Supabase: `python -m triage_agent.db.migrate` (applies 002 if still pending, and 003), `python -m triage_agent.db.seed`.
3. Live: `/ask please close T-1002` → Approve; `/ask please close T-1005` → Deny; one left to expire. Check `approval_requests`, `receipts`, one reply each. Also the Slice 2 items below (live smoke run) once a key exists.
4. Then the Fly.io deploy (TASKS.md, after Slice 3).

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
