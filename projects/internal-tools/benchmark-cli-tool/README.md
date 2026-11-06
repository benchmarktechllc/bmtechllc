# Benchmark Retrieval Pipeline

A grounded RAG benchmark: document ingestion, OpenAI embeddings, PostgreSQL
full-text/vector search, RRF, calibrated abstention, and a citation-verified
answer agent.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
docker compose up -d
```

The application loads the existing `.env` first, then the optional `.env.local`
without overriding values already set in `.env`. Ensure they provide
`OPENAI_API_KEY` and `DATABASE_URL`. Place source exports in `data/documents/`;
that directory is ignored by Git.

## Ingest and retrieve

```bash
benchmark-ingest --source-dir data/documents --dry-run
benchmark-ingest --source-dir data/documents --create-indexes
```

## Discord sync

`benchmark-sync-discord` reads selected channels from Discord's REST API; it
does not create transcript files. It normalizes and redacts each retained
message before storing it in PostgreSQL, then rebuilds a channel's chunks when
new or recently edited messages are found.

Configure these runtime values only in your secret manager or local `.env`
files, never in Git:

```bash
DISCORD_INGEST_BOT_TOKEN=...
DISCORD_INGEST_CHANNEL_IDS=123456789012345678,234567890123456789
DATABASE_URL=...
OPENAI_API_KEY=...
```

The Discord application needs a bot token with `VIEW_CHANNEL` and
`READ_MESSAGE_HISTORY` for each configured channel, plus the privileged Message
Content intent enabled. Run a one-time backfill (or manually trigger the
workflow) with:

```bash
benchmark-sync-discord
```

The repository-level workflow at `.github/workflows/discord-ingestion.yml`
runs at 03:17 UTC daily and can be run manually. It runs on GitHub-hosted
Ubuntu and works inside this project directory. Configure an environment named
`discord-ingestion` with `DATABASE_URL`, `OPENAI_API_KEY`, and
`DISCORD_INGEST_BOT_TOKEN` secrets, plus a non-secret
`DISCORD_INGEST_CHANNEL_IDS` variable. If Supabase network restrictions are
enabled, GitHub-hosted runners will need an allowed network path.

The scheduled sync detects additions and edits in the range since its stored
cursor. A deleted older Discord message is not yet removed from the corpus;
that needs a future reconciliation pass or Discord gateway delete events.

## Discord ask bot

`benchmark-discord-ask` is a separate always-on gateway process. It calls
`run_agent` and posts the reply. Ingestion stays on `benchmark-sync-discord`;
this bot does not store Discord messages or rebuild the corpus.

People can ask in two ways:

- In `DISCORD_ASK_CHANNEL_ID` (the `#ask-bmtech` channel), every non-bot
  message that contains text is answered. Threads under that channel are
  included. The bot opens a thread on the message when Discord allows it, and
  replies on the message when it does not. Messages that are already in a
  thread get a thread reply. The bot ignores its own messages and other bots.
- `/ask question:<text>` works in every channel of the guild. The answer is
  posted in the channel where the command was used. Discord's first response
  window is shorter than retrieval, so the bot defers with a public thinking
  state and then sends the answer. A blank question gets an ephemeral reply.

Set `DISCORD_ASK_GUILD_ID` to register `/ask` on that guild immediately. Leave
it unset to sync a global command, which can take up to an hour to appear.
Use one of those modes. After switching from a global command to a guild
command, remove the leftover global `/ask` in the Discord developer portal so
the command is listed once. Leave `/ask` enabled for every channel in the
server's integration settings.

The answer bot needs the privileged Message Content Intent. Channel messages
arrive without text when that intent is off, and those messages are skipped.
`/ask` still works, because the question is a slash-command option. Invite the
application with the `bot` and `applications.commands` scopes. In the ask
channel, grant View Channel, Send Messages, Read Message History, Create
Public Threads, and Send Messages in Threads. In other channels where people
use `/ask`, grant View Channel and Send Messages.

Keep these values in a secret manager, the gitignored `.env` / `.env.local`
files, or a `chmod 600` env file on the VM:

```bash
DISCORD_ANSWER_BOT_TOKEN=...     # required gateway bot token
DISCORD_ASK_CHANNEL_ID=...       # required snowflake of #ask-bmtech
DATABASE_URL=...                 # required by run_agent
OPENAI_API_KEY=...               # required by run_agent
DISCORD_ASK_GUILD_ID=...         # optional; guild command sync
```

`DISCORD_ANSWER_BOT_TOKEN` is independent of `DISCORD_INGEST_BOT_TOKEN`.

Run it locally from this directory after Setup:

```bash
pip install -e '.[dev]'
benchmark-discord-ask
```

The process stays connected until you stop it. `.env` in this directory is
loaded on startup.

On the Hetzner VM, run it as a systemd service or a container. The scheduled
GitHub Actions ingest workflow stays as it is. Install the project into a
virtualenv, put the variables in `/etc/bmtech/discord-ask.env`, and use a unit
like:

```ini
[Unit]
Description=BMTech Discord ask bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=bmtech
WorkingDirectory=/opt/benchmark-cli-tool
EnvironmentFile=/etc/bmtech/discord-ask.env
ExecStart=/opt/benchmark-cli-tool/.venv/bin/benchmark-discord-ask
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now benchmark-discord-ask
sudo systemctl status benchmark-discord-ask
```

A local image can run the same process. Build from this directory and pass
the env file at runtime:

```bash
docker build -t benchmark-discord-ask .
docker run -d --name benchmark-discord-ask --restart unless-stopped \
  --env-file /etc/bmtech/discord-ask.env benchmark-discord-ask
```

Replies are capped at Discord's 2000-character limit. Citations from the
agent are included when they fit.

`search_docs(query, top_k=5)` embeds the query once, concurrently retrieves 20
pgvector candidates and 20 PostgreSQL FTS candidates, then merges them with
RRF (`k = 60`). It returns chunks and fused search scores only. The optional
`speaker`, `source` (source filename), and ISO-8601 `date` filters apply to
both retrieval arms.

Inspect retrieval manually with:

```bash
benchmark-search --query "home office equipment reimbursement"
benchmark-search --mode fts --query "equipment reimbursement"
benchmark-search --mode vector --query "remote work policy"
benchmark-search --query "architecture decision" --speaker Ada --date 2026-08-14
```

## Retrieval evaluation

Production golden datasets are intentionally not versioned because their
questions and labels can reveal company context. Store a reviewed dataset in a
secure location (or locally at the ignored default
`data/evaluation/golden_queries.json`). Its expected chunks should be anchored
by source filename and content hash, then resolved to the current database IDs
when evaluation runs. Review any reference that becomes unavailable after
source-content changes, then run:

```bash
benchmark-evaluate --dataset /path/to/golden_queries.json
benchmark-evaluate --dataset /path/to/golden_queries.json --compare-vector \
  --calibrate-abstention --answers --answer-report artifacts/answer-evaluation.json
pytest
```

The comparison report is a before/after table for vector-only and hybrid RRF
Recall@5/MRR. It records the deltas and whether both measures improved. The
evaluation log includes retrieval mode, latency, and quality columns.
`benchmark-answer --question "What was the HNSW p99 latency?"` returns a
structured grounded answer or `no_relevant_information`. Every returned quote
is verified against the cited stored chunk.

## Real integration check

The default suite is offline. To verify the complete pgvector and embedding
path with the supplied Docker database, run this opt-in test after `docker
compose up -d` and configuring `OPENAI_API_KEY`:

```bash
pytest -m integration_live tests/integration/test_hybrid_pgvector.py
```

It indexes two small fixtures with real OpenAI embeddings, verifies FTS, vector,
and fused retrieval, then clears only the local Compose database's chunk table.

## Layout

```text
src/benchmark_cli/
  ingestion/       Parse, normalize, chunk, embed, index
  providers/       Embedding provider boundary
  storage/         PostgreSQL/pgvector + FTS schema and queries
  retrieval.py     Concurrent hybrid retrieval and pure RRF fusion
  evaluation/      Golden dataset loading, metrics, CLI
  discord_sync.py  Scheduled channel ingestion
  discord_ask.py   Gateway bot for #ask-bmtech and /ask
tests/unit/        Chunking, embeddings, RRF, metric, and Discord reply tests
data/evaluation/   30-query golden-dataset template
```
