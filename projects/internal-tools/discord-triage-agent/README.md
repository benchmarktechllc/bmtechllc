# triage-agent

A Discord `/ask` command that classifies a support question, looks up synthetic tickets and KB articles, and replies. The one irreversible action (`close_ticket`) waits for a human approval in Discord. Everything is stored in Postgres for audit. The design lives in `docs/SPEC.md`, `docs/DECISIONS.md` and `docs/TASKS.md`.

- **Owner:** fanuel-mengistu
- **Status:** in development. Slice 1 (local skeleton on a tunnel URL) is built; slices 2 and 3 are next.
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
- The rest (`ANTHROPIC_API_KEY`, `APPROVER_*`, `APPROVAL_TTL`) are used from Slice 2 and 3.

Apply the schema (only ever through `migrations/`; never edit tables in the Supabase dashboard):

```bash
python -m triage_agent.db.migrate
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
3. In the test guild, run `/ask question: what's the status of T-1002?`. You should see "thinking…" and then "Received — session <id>".

The quick-tunnel URL changes every time `cloudflared` starts, so step 2 is repeated each session.

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

## Backup

To confirm when the Supabase project is created, and record here:
- **Plan:** _TBD_
- **Does a free project pause when idle?** _TBD_ (a paused database would break the bot)
- **Backups included in the plan:** _TBD_

All data is synthetic and the schema plus seed data are recreated from `migrations/` and `seed/`, so losing the database loses only the audit history.
