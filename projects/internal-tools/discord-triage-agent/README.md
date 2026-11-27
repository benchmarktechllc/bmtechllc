# triage-agent

A Discord `/ask` command that classifies a support question, looks up synthetic tickets and KB articles, and replies. The one irreversible action (`close_ticket`) waits for a human approval in Discord. Everything is stored in Postgres for audit. The design lives in `docs/SPEC.md`, `docs/DECISIONS.md` and `docs/TASKS.md`.

- **Owner:** fanuel-mengistu
- **Status:** in development. Slices 1 (local skeleton on a tunnel URL), 2 (classify and answer with read tools) and 3 (gated `close_ticket` with approval and resume) are built; the live checks for slices 2 and 3 and the Fly.io deploy are next.
- **Data:** synthetic only. No customer or production data.

## Services and cost

| Service | Used for | Cost |
|---|---|---|
| Discord application (test guild only) | `/ask` slash command, interactions endpoint, bot token | $0 |
| Supabase (new, dedicated project) | Postgres: sessions, messages, traces, approvals, receipts | Plan to be confirmed (see *Backup*) |
| Anthropic API | Classifier (`claude-haiku-5-5`) and agent (`claude-sonnet-5-5`), from Slice 2 | About $5 in total, capped by `scripts/live_smoke.py` |
| Cloudflare quick tunnel (`cloudflared`) | Public HTTPS URL for the local app in v1 | $0, no account |
| Fly.io (after Slice 3) | One always-on `shared-cpu-1x` machine | About $2–5/mo |

## Setup

```bash
cd projects/internal-tools/discord-triage-agent
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env   # then fill in the values; .env is git-ignored
```

`.env` values:
- `DISCORD_PUBLIC_KEY`, `DISCORD_APPLICATION_ID`: Developer Portal → your app → General Information.
- `DISCORD_BOT_TOKEN`: Developer Portal → Bot → Reset Token. Never commit or paste it.
- `DISCORD_GUILD_ID`: the test server's ID (Discord developer mode → right-click the server → Copy Server ID).
- `DATABASE_URL`: the Supabase project's Postgres connection string.
- `TEST_DATABASE_URL`: the local test Postgres (see *Tests*), never the Supabase project.
- `ANTHROPIC_API_KEY`: required to start the app.
- `APPROVER_CHANNEL_ID`: the private approvals channel (right-click the channel → Copy Channel ID). Required: at startup the app fetches it with the bot token and refuses to start if it can't, so the bot must be able to view and post in it.
- `APPROVER_ROLE_ID`: the role allowed to approve (Server Settings → Roles → right-click → Copy Role ID). Required.
- `APPROVAL_TTL`: how long an approval waits before it expires. Optional, default `24h`; set `5m` locally to watch an expiry.

Apply the schema (only ever through `migrations/`; never edit tables in the Supabase dashboard):

```bash
python -m triage_agent.db.migrate
python -m triage_agent.db.seed      # synthetic tickets, KB articles, teammates; safe to re-run
```

Register `/ask` for the test guild (run again whenever the command definition changes):

```bash
python scripts/register_commands.py
```

## Local run with the tunnel

The bot only works while both of these are running.

```bash
uvicorn triage_agent.app:app --port 8000          # terminal 1
cloudflared tunnel --url http://localhost:8000    # terminal 2: prints https://<random>.trycloudflare.com
```

1. Check `https://<random>.trycloudflare.com/healthz` returns `{"status":"ok"}`.
2. In the Developer Portal → General Information, set **Interactions Endpoint URL** to `https://<random>.trycloudflare.com/interactions` and save. Discord sends a signed PING and a deliberately bad signature; the save only succeeds if the app answers both correctly.
3. In the test guild, run `/ask question: what's the status of T-1002?`. You should see "thinking…", then one reply grounded in the ticket (a few seconds; the classifier and the agent loop run after the ACK). If anything fails, the reply is a fixed apology naming the session id; look the session up in `sessions` and `traces`.

The quick-tunnel URL changes every time `cloudflared` starts, so step 2 is repeated each session.

## Approvals, the sweep and recovery

`close_ticket` is the only write tool, and it never runs on the model's say-so. The gate is plain code (`tools/policy.py`): `get_ticket` and `search_kb` run, `close_ticket` needs approval, and anything else is denied.

1. `/ask question: please close T-1002`. The asker's reply becomes "Waiting for approval on 1 action(s) — session <id>", and an Approve/Deny message appears in the approvals channel.
2. A member with `APPROVER_ROLE_ID` clicks. Anyone else gets a private "not authorized" and nothing changes. An asker who holds the role may approve their own request; it is stored with `self_approved = true`.
3. Once every request of that turn is decided, the same session resumes: the ticket closes (exactly once, with a row in `receipts`), and the asker gets one answer. Within 15 min it replaces the waiting note; later, or after a restart, it is posted in the asker's channel by the bot. If that post fails, `sessions.reply_error` is set and a notice mentioning the asker goes to the approvals channel.

A sweep runs at startup and every 5 min. It expires requests older than `APPROVAL_TTL` (the session resumes with "approval expired"), and it recovers sessions left `running` with no heartbeat for 10 min (a crash or restart): it finishes a session whose approvals were all decided, or marks it `failed` with an apology listing anything already done, such as "T-1002 was closed". While the app or tunnel is down, Discord shows "This interaction failed" for clicks; click again once it is back.

Audit with SQL: `approval_requests` (who decided what, when), `receipts` (what actually executed), `traces.hook_decision` (the gate's call for every tool use).

## Tests

Tests use a throwaway local Postgres, never the Supabase project, and make no live model or Discord calls.

```bash
docker run -d --name triage-agent-test-pg -e POSTGRES_PASSWORD=postgres -p 127.0.0.1:54329:5432 postgres:16
export TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54329/postgres   # or put it in .env
pytest
ruff check . && ruff format --check .
mypy --strict src
```

Each test gets its own freshly created database with `migrations/` applied, dropped afterwards. If `TEST_DATABASE_URL` isn't set, the database tests fail with these instructions rather than skipping. Later: `docker start triage-agent-test-pg`.

## Live smoke run (costs money)

Run by hand only, never in CI. It makes real Claude calls (Haiku classifier and grader, Sonnet agent) for up to 5 sessions from `fixtures/live_cases.json`, against a throwaway database on the **local** test Postgres, with Discord replaced by an in-memory capture. It refuses any database host other than localhost.

```bash
docker start triage-agent-test-pg
python scripts/live_smoke.py                 # all fixtures (max 5)
python scripts/live_smoke.py --max-sessions 1
```

It needs `ANTHROPIC_API_KEY` and `TEST_DATABASE_URL`. For each session it prints PASS/FAIL, the classifier confidence, the session cost and the running total. Each expected concern must be covered in the one reply, the session's categories must match the fixture, and no `close_ticket` may execute (no receipt, no ticket change). There are no clicks in a smoke run, so a close the model attempts stays a pending approval; the run prints how many there were. Before each session it compares a worst-case ceiling (about $0.40: 8 Sonnet calls at the token caps plus the Haiku calls) with what's left, and stops before the total could pass $4. Prices live in `src/triage_agent/budget.py` with the date they were checked.

## Backup

To confirm when the Supabase project is created, and record here:
- **Plan:** _TBD_
- **Does a free project pause when idle?** _TBD_ (a paused database would break the bot)
- **Backups included in the plan:** _TBD_

All data is synthetic and the schema plus seed data are recreated from `migrations/` and `seed/`, so losing the database loses only the audit history.
