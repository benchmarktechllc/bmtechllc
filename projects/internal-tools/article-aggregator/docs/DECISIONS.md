# Decisions

## Selected stack: Python 3.12 + GitHub Actions + free-tier Postgres

| Area | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Matches `benchmark-cli-tool`; the best RSS and extraction libraries are in Python. |
| Packaging | `pyproject.toml` (hatchling), `src/article_aggregator/{ingestion,discord,db,summarizer}/`, CLI `article-aggregator` | Same build backend as sibling projects; CLAUDE.md's folder split inside one importable package. |
| Feed parsing | `feedparser` | Most forgiving RSS/Atom parser available; normalizes dates, guids and summaries. |
| HTTP | `httpx` | Timeouts, size caps, conditional GET, and the same client for feeds and Discord. |
| Keyword filter, dedup, summary | Plain Python (`re`, `unicodedata`, `hashlib`, `urllib.parse`) | Deterministic and fully unit-testable; no model in the selection path. |
| Discord | Bot account + REST `POST /channels/{id}/messages` via `httpx`; `GET /channels/{id}/messages` to reconcile an interrupted digest | Works from short-lived cron jobs; `discord.py` needs an always-on gateway connection. |
| Scheduler/host | GitHub Actions cron (poll `23 */2 * * *`, falling back to `23 */3 * * *` if over the minutes budget; digest `53 5 * * *` UTC) | $0, already used in this monorepo; delay of up to ~30 min is fine for a daily digest. |
| Run exclusion | Postgres advisory lock shared by poll and digest; a separate Actions `concurrency` group per workflow | A shared Actions group would let a queued poll cancel the pending digest; the DB lock serializes them without that risk. |
| Database | Postgres via `psycopg` 3, plain SQL migrations | Serverless runs keep no local files; same driver as sibling tools. |
| DB hosting | Neon free tier | $0, scales to zero, and separate from the monorepo's existing database. |
| Config | YAML in `config/`, validated with `pydantic` | Changes go through PR review; schema errors fail CI, not the cron job. |
| Article text (later slice) | `trafilatura` + `urllib.robotparser` | Best main-text extractor; respects `robots.txt`. |
| Summaries (v1) | Plain-code truncation of the RSS summary | $0 budget; the `Summarizer` interface leaves room for an LLM later. |
| LLM (deferred) | Provider-agnostic `Summarizer` adapter | Blocked until a budget is approved; pick a model then. |
| Tests | `pytest` (unit + Postgres integration via a CI service container) | Matches the repo; deterministic logic is covered by fixture feeds. |
| Lint / types | `ruff`, `mypy --strict` on `src/` | Ruff matches the repo; mypy adds the typecheck gate this project needs. |
| Secrets | GitHub environment `article-aggregator`: `DATABASE_URL`, `DISCORD_DIGEST_BOT_TOKEN`; vars `DISCORD_DIGEST_CHANNEL_ID`, `DISCORD_DIGEST_TEST_CHANNEL_ID` (manual runs default to the test channel) | Same pattern as `discord-ingestion.yml`; nothing in the repo. |

## Rejected

- **TypeScript/Node on Actions.** Stronger Discord SDK, but weaker RSS parsing and article extraction, and it adds a second toolchain to a Python monorepo.
- **Cloudflare Workers + D1.** Free-plan CPU and outbound-request limits are too tight for parsing about 20 feeds plus article fetching. Worth revisiting only to host a future slash-command endpoint.
- **News APIs (NewsAPI, GNews).** Free tiers are development-only or delayed, and republishing terms are restrictive.
- **LLM in v1.** No budget. Free tiers may train on prompts and can change or end without notice.
