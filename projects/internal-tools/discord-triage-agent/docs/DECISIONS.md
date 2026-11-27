# Decisions

## Selected stack: thin harness on the `anthropic` SDK + FastAPI interactions endpoint, run locally first, then Fly.io

| Area | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Matches `article-aggregator` and `benchmark-cli-tool`. |
| Packaging | `pyproject.toml` (hatchling), `src/triage_agent/{discord,agent,tools,db}/` | Same build backend and layout as sibling projects. |
| Web framework | FastAPI + uvicorn | Async webhook that can ACK within Discord's 3 s deadline and hand the loop to a background task. |
| Discord ingress | Slash-command interactions endpoint (no gateway), Ed25519 verification with PyNaCl | Needs no always-on socket and no privileged Message Content intent. Button clicks arrive at the same endpoint. |
| Discord egress | `httpx` REST: edit the original response for replies within 15 min, bot token `POST /channels/{id}/messages` after that | Interaction tokens expire in 15 min, and approvals can take hours. |
| Agent harness | `anthropic` SDK with a hand-written `while stop_reason == "tool_use"` loop | We own the gate, the trace per iteration and save/resume as plain, testable code. |
| PreToolUse gate | A plain-code policy table checked before every tool execution, deny by default | Blocking must be deterministic and must not depend on prompts. |
| Session resume | Reload the append-only `messages` rows from Supabase and append one user message holding every `tool_result` from the paused turn | The database *is* the session, so resume survives restarts, a closed laptop and later redeploys. |
| Agent model | `claude-sonnet-5-5`, effort `low`, `max_tokens` 1024, 8-iteration cap, `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`) | Good tool use at low cost within the ~$5 budget. The fallback retries refused requests on another model. |
| Classifier and fixture checks | `claude-haiku-5-5` with structured output (`output_config.format`), `max_tokens` 256 | A fixed-enum label at a tiny fraction of the agent's cost. |
| Database | Supabase Postgres through `psycopg` 3, with plain SQL migrations in `migrations/` | Same driver as the sibling projects. A session-row lock plus a conditional `UPDATE … RETURNING` makes approvals race-safe and resumes each turn once, and RLS blocks the public API. |
| Expiry and recovery | An asyncio sweep inside the app every 5 min and once at startup | A single process, with no separate scheduler. The startup pass catches approvals that expired while the app was stopped. |
| Hosting, v1 | Local `uvicorn` plus a Cloudflare quick tunnel (`cloudflared tunnel --url http://localhost:8000`) | $0, no account needed, and a public HTTPS URL Discord can call. The URL changes on every start, so the Developer Portal endpoint is updated each session. |
| Hosting, later | Fly.io, one `shared-cpu-1x` machine, `min_machines_running = 1` | A public HTTPS URL that doesn't sleep, so the 3 s PING never cold-starts. About $2–5/mo: Fly.io has no free tier for new accounts (only a short trial), so check first whether an older BMTech Fly organization still has the legacy free allowance and could run it at about $0. |
| Test database | A local Postgres through the Supabase CLI (`supabase start`) or a Postgres container | Real transactions for the race tests, and tests never write to the hosted project. |
| Tests | `pytest` with a scripted fake Anthropic client and mocked Discord; `scripts/live_smoke.py` for manual, capped live runs | CI costs $0, and live spend is bounded. |
| Lint / types | `ruff`, `mypy --strict src` | Same quality gates as `article-aggregator`. |
| Secrets | A local `.env` that Git ignores in v1, then `fly secrets`, for `ANTHROPIC_API_KEY`, `DISCORD_PUBLIC_KEY`, `DISCORD_BOT_TOKEN` and `DATABASE_URL`, plus a committed `.env.example` | Keeps secrets out of Git at both stages. |

## Rejected

- **Claude Agent SDK + discord.py + Railway.**
  - It has native `PreToolUse` hooks and `resume`, but it bundles the Claude Code harness and its built-in Bash/Read/Write tools, which would all have to be disabled and audited.
  - Its session transcripts live in local files that an ephemeral container loses, so they would need to be mirrored into Supabase anyway.
  - discord.py needs a gateway connection that is always on.
- **Anthropic SDK Tool Runner (beta) + discord.py + Render.**
  - The Tool Runner's loop hides the `stop_reason` steps the spec wants traced, and pausing a runner for hours still means reloading the session by hand.
  - Render's free tier sleeps and misses the 3 s deadline; paid Starter is $7/mo.
- **Railway** as host: works well, but costs about the same as Fly.io and has no stronger fit.
- **GitHub Actions runner** as host:
  - A job has no inbound public URL, so Discord can't reach it.
  - A job starts only from a GitHub event, tens of seconds to minutes later, so it can't answer an interaction within 3 s.
  - Jobs stop after 6 h, so a gateway bot would mean fragile job chaining that Actions isn't meant for.
  - Actions stays the choice for CI.
- **Oracle Cloud Always Free VM** as host: the only $0 option that is always on and meets the 3 s deadline without a rewrite, but we would run the server ourselves (HTTPS, OS updates, restarts) and signup needs a card. Fly.io's ~$2–5/mo is worth skipping that work.
- **supabase-py / PostgREST**: it would put a service-role key in the app and lacks the transactions the approval race needs.
- **Opus for the agent**: costs more than this budget can absorb for a two-tool lookup task.
