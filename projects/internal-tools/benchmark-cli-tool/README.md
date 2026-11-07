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

## Portal knowledge-base sync

`benchmark-sync-portal` reads `portal/content/**/*.md` from this repo (the
files behind the portal `/index.json` manifest). It does not crawl the built
site. A page is indexed only when frontmatter `status` is `published` and the
page is not stale. Stale uses the portal rule: whole calendar months since
`updated` are at least `reviewCycleMonths` (default 6). Draft, archived, and
stale pages are skipped. A page that was indexed earlier and is no longer
eligible is deleted from `document_chunks` on the next real run.

Each page is one source, `portal/<section>/<slug>.md`, with `source_type`
`kb_page`. Edits replace that source, the same way a Discord channel is
replaced. Chunk metadata includes the title, section, heading, tags, owner,
visibility, and URL.

Preview without embeddings or database writes:

```bash
benchmark-sync-portal --dry-run
```

From this directory the command finds `portal/content` by walking up from the
working directory. Override it with `--content-dir` or `PORTAL_CONTENT_DIR`.
`--index-json` (or `PORTAL_INDEX_JSON`) can further restrict the sync to a
built `/index.json`: an entry must be `published` and not `isStale`.
`PORTAL_BASE_URL` is the current GitHub Pages origin plus `/bmtechllc/portal`.
The portal ingestion workflow sets it from `github.repository_owner`, so an
organization transfer changes the host without pinning a previous owner.
Chunk metadata stores that absolute page URL and the portal home. Relative
`/guide/...` paths are only what older rows contain before the next sync.

```bash
DATABASE_URL=...
OPENAI_API_KEY=...
PORTAL_BASE_URL=https://<owner>.github.io/bmtechllc/portal
```

Keep those values in the secret manager or the gitignored `.env` / `.env.local`
files. The workflow `.github/workflows/portal-ingestion.yml` runs at 04:17 UTC
daily, on manual dispatch, and when `portal/content` changes on `main`. It
uses the existing GitHub environment `discord-ingestion` (the same one as
Discord sync) for `DATABASE_URL` and `OPENAI_API_KEY`. No new environment is
required.

## Trello sync

`benchmark-sync-trello` reads the BMTech General Board
(`https://trello.com/b/tIvVZqio/bmtech-general-board`, short link `tIvVZqio`).
Pass another board with `--board` or `TRELLO_BOARD_ID` (id or short link).

The sync pulls open lists and cards, plus descriptions, checklists, and
comments. Each card is one source, `trello/<shortLink>/<cardId>.md`, with
`source_type` `trello`. Metadata includes the board, list/column (`list` and
`column`), card id, labels, due date, and URL. Cards that leave the open
board, including cards on archived lists, are removed from the corpus. A card
whose `dateLastActivity` is unchanged skips the comment fetch and the
embedding call.

```bash
TRELLO_API_KEY=...
TRELLO_TOKEN=...
TRELLO_BOARD_ID=tIvVZqio   # optional; this short link is the default
DATABASE_URL=...
OPENAI_API_KEY=...
```

Do not commit those values. The token needs read access to the board. Run a
backfill with:

```bash
benchmark-sync-trello
```

`.github/workflows/trello-ingestion.yml` runs at 04:47 UTC daily and on
manual dispatch. It uses the `discord-ingestion` environment, alongside the
Discord and portal jobs. Add `TRELLO_API_KEY` and `TRELLO_TOKEN` as secrets
there. `TRELLO_BOARD_ID` is an optional Actions variable; leave it empty to
keep `tIvVZqio`. After the secrets are saved, run the workflow once by hand
for the first backfill. The schedule then keeps the board current.

## Fathom meeting sync

`benchmark-sync-fathom` reads meetings visible to a Fathom API key: recordings
that user made, or that were shared with them or their team. Generate the key
under Fathom user settings, API Access. The key never sees another user's
private meetings.

A meeting is indexed once Fathom has a transcript. The summary is included when
Fathom returns one. Each meeting is one source, `fathom/<recording_id>.md`,
with `source_type` `transcript` and `kind` `meeting_transcript`. Chunk metadata
includes the title, meeting date, attendees, recording id, and Fathom URL. The
calendar join link (Google Meet, Zoom, or Teams) is in the chunk text. Meetings
that leave the key's accessible set are removed from `document_chunks`. A listed
meeting with no transcript yet is skipped and retried next run.

Fathom's list payload has no `updated_at`. The sync skips the transcript fetch
and the embedding call when the recording end time (or `created_at` if the
recording has no end time), start time, title, URLs, and attendees still match
the stored row. When those change, a content hash still skips re-embedding if
the text is the same. A transcript edit that moves none of those fields is
picked up on the next metadata change.

Preview chunk counts without embeddings or database writes. This still calls
the Fathom API:

```bash
FATHOM_API_KEY=... benchmark-sync-fathom --dry-run
```

Backfill into the corpus:

```bash
FATHOM_API_KEY=...
DATABASE_URL=...
OPENAI_API_KEY=...
benchmark-sync-fathom
```

Do not commit those values. `.github/workflows/fathom-ingestion.yml` runs at
05:17 UTC daily and on manual dispatch. It uses the `discord-ingestion`
environment, alongside the Discord, portal, and Trello jobs. Add
`FATHOM_API_KEY` as a secret there. `DATABASE_URL` and `OPENAI_API_KEY` are
already on that environment. After the secret is saved, run the workflow once
by hand (Actions → Fathom ingestion → Run workflow) for the first backfill.
Transcript calls are rate limited, so a large backfill can take a while and
may need a second manual run; meetings already indexed are skipped. The
schedule then keeps the corpus current.

## Fireflies meeting sync

`benchmark-sync-fireflies` reads transcripts visible to a Fireflies API key.
Create the key under Fireflies integrations, Fireflies API. The sync lists
meeting metadata from `transcripts` (50 at a time, without sentences) and
downloads the transcript and summary only for new or changed meetings.

A meeting is indexed once it has a finalized transcript. Live meetings
(`is_live`), silent meetings (`silent_meeting`), and transcripts with no
sentences are skipped and retried next run. The summary is included when
Fireflies returns an overview, short summary, or action items. Each meeting is
one source, `fireflies/<transcript_id>.md`, with `source_type` `transcript` and
`kind` `meeting_transcript`. Chunk metadata reuses `recording_id` for the
Fireflies transcript id and includes the title, meeting date, attendees, and
transcript URL. The conferencing link (Google Meet, Zoom, or Teams) is in the
chunk text. Meetings that leave the key's accessible set are removed from
`document_chunks`.

The list payload has no separate updated time. The sync skips the detail fetch
and the embedding call when the title, transcript URL, meeting link, host,
attendees, date, duration, and summary status still match the stored row. When
those change, a content hash still skips re-embedding if the text is the same.

Free-plan keys allow about 30 requests a minute and 500 a day. The client
honors HTTP 429 and `Retry-After`. A large backfill can take more than one run;
meetings already indexed are skipped, and a failed run leaves stored meetings
in place. Fathom sync continues on its own schedule.

Preview chunk counts without embeddings or database writes. This still calls
the Fireflies API:

```bash
FIREFLIES_API_KEY=... benchmark-sync-fireflies --dry-run
```

Backfill into the corpus:

```bash
FIREFLIES_API_KEY=...
DATABASE_URL=...
OPENAI_API_KEY=...
benchmark-sync-fireflies
```

Do not commit those values. `.github/workflows/fireflies-ingestion.yml` runs at
05:47 UTC daily and on manual dispatch. It uses the `discord-ingestion`
environment, alongside the Discord, portal, Trello, and Fathom jobs. Add
`FIREFLIES_API_KEY` as a secret there. `DATABASE_URL` and `OPENAI_API_KEY` are
already on that environment. After the secret is saved, run the workflow once
by hand (Actions → Fireflies ingestion → Run workflow) for the first backfill.
The schedule then keeps the corpus current.

## Discord ask bot

`benchmark-discord-ask` is a separate always-on gateway process. It calls
`run_agent` and posts the reply. Ingestion stays on the sync commands
(`benchmark-sync-discord`, `benchmark-sync-portal`, `benchmark-sync-trello`,
`benchmark-sync-fathom`, and `benchmark-sync-fireflies`); this bot does not
store source text or rebuild the corpus. Once those syncs write chunks, the
bot can answer from them with no gateway change. Retrieval reads
`document_chunks` and does not filter on source type, so Fathom and Fireflies
transcripts are included as soon as they are indexed.

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
PORTAL_BASE_URL=https://<owner>.github.io/bmtechllc/portal
```

`PORTAL_BASE_URL` is required for a clickable portal answer until portal sync
has rewritten knowledge-base chunk metadata. Use the current repository
owner. Do not set a previous GitHub Pages host.

The production ask process is the Docker container `benchmark-discord-ask`.
It keeps using `/etc/bmtech/discord-ask.env`. Do not print that file. Acceptance
runs in a second container so the live bot stays up. Install the wheel; the
host does not need a git checkout.

```bash
docker ps --filter name=^benchmark-discord-ask$ --format '{{.Names}} {{.Status}} {{.Image}}'
image=$(docker inspect -f '{{.Config.Image}}' benchmark-discord-ask)
docker run --rm --name benchmark-ask-acceptance --user root \
  --env-file /etc/bmtech/discord-ask.env \
  -e PORTAL_BASE_URL=https://benchmarktechllc.github.io/bmtechllc/portal \
  -v "$PWD/benchmark_cli_tool-0.1.6-py3-none-any.whl":/tmp/benchmark_cli_tool.whl:ro \
  --entrypoint bash \
  "$image" \
  -lc 'pip install --no-cache-dir /tmp/benchmark_cli_tool.whl && benchmark-acceptance'
```

`-e PORTAL_BASE_URL` overrides a stale host still stored in the env file.
`--env-file` does not print secrets. If the new container cannot reach the
database, rerun the same command with
`--network container:benchmark-discord-ask` added. That shares the running
container's network and does not stop it.

`benchmark-acceptance` asks the original four questions plus paraphrases
("Where do I open the knowledge base?", "What is our current headcount?",
"Who is on the core team?", "What Discord channels do we ingest?"). It fails
if the portal answer is not the configured URL, if an employee answer states
a digit headcount or omits count uncertainty, if a citation comes from a
synthetic canary, or if the answer repeats bot no-information or
article-update boilerplate. A core-team question may abstain when no indexed
chunk uses that label. The JSON includes answer text, so do not paste it into
a public ticket when a citation quote is private Discord text.

After that report passes, replace the bot without deleting the old container:

```bash
docker rename benchmark-discord-ask benchmark-discord-ask-previous
docker run -d --name benchmark-discord-ask --restart unless-stopped \
  --env-file /etc/bmtech/discord-ask.env \
  -e PORTAL_BASE_URL=https://benchmarktechllc.github.io/bmtechllc/portal \
  -v /var/bmtech/benchmark_cli_tool-0.1.6-py3-none-any.whl:/opt/benchmark_cli_tool.whl:ro \
  --user root \
  --entrypoint bash \
  "$image" \
  -lc 'pip install --no-cache-dir /opt/benchmark_cli_tool.whl && exec su -s /bin/bash bmtech -c "exec benchmark-discord-ask"'
```

Rollback while the previous container still exists:

```bash
docker stop benchmark-discord-ask
docker rm benchmark-discord-ask
docker rename benchmark-discord-ask-previous benchmark-discord-ask
docker start benchmark-discord-ask
```

Runtime logs record retrieval status, chunk ids, source file names, cosine,
and keyword overlap. They do not record chunk text. `benchmark-search` prints
redacted chunk text; do not paste that output into a ticket if a backfill has
not yet rewritten stored Discord rows.

After deploying this revision, re-run the portal ingestion workflow and the
Discord ingestion workflow once. Portal sync stores the current absolute URL.
Discord sync redacts recognized tokens in stored messages, drops indexed bot
abstentions and notification templates, and rebuilds those channel chunks.
The next portal or Discord sync applies the full-text migration that indexes
URL path words such as `portal`. The ask process does not alter the schema on
startup. Revoke any exposed credential and delete the original Discord
message separately; sync rewrites the indexed copy and does not delete Discord
history.

`DISCORD_ANSWER_BOT_TOKEN` is independent of `DISCORD_INGEST_BOT_TOKEN`.

Run it locally from this directory after Setup:

```bash
pip install -e '.[dev]'
benchmark-discord-ask
```

The process stays connected until you stop it. `.env` in this directory is
loaded on startup.

The scheduled GitHub Actions ingest workflow stays as it is. A host that is
not already running the Docker container can use a virtualenv and systemd.
The current production host is the Docker container above, not this unit:

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

On a systemd host, roll back with the previous revision and
`sudo systemctl restart benchmark-discord-ask`. On the Docker host, use the
rename rollback in the acceptance section. A local image can still be built
from this directory when a checkout exists:

```bash
docker build -t benchmark-discord-ask:ask-retrieval .
docker run --rm --env-file /etc/bmtech/discord-ask.env benchmark-discord-ask:ask-retrieval
```

Replies are capped at Discord's 2000-character limit. Citations from the
agent are included when they fit.

`search_docs(query, top_k=5)` embeds the query once, concurrently retrieves 40
pgvector candidates and 40 PostgreSQL FTS candidates, then merges them with
RRF (`k = 60`). Ordinary answers omit synthetic canary transcripts and indexed
bot abstentions. It returns chunks and fused search scores only. The optional
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
  discord_sync.py  Scheduled Discord channel ingestion
  portal_sync.py   Published portal knowledge-base ingestion
  trello_sync.py   Trello board ingestion
  fathom_sync.py   Fathom meeting transcript ingestion
  fireflies_sync.py  Fireflies meeting transcript ingestion
  discord_ask.py   Gateway bot for #ask-bmtech and /ask
tests/unit/        Chunking, embeddings, RRF, metric, and Discord reply tests
data/evaluation/   30-query golden-dataset template
```
