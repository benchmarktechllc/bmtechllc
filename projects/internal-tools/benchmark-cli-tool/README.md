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
tests/unit/        Chunking, embeddings, RRF, and metric tests
data/evaluation/   30-query golden-dataset template
```
