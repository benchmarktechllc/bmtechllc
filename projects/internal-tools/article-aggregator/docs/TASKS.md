# Tasks

Ordered vertical slices. Each one ships to production on its own. `docs/RUNBOOK.md` grows with each slice: whatever a slice puts into production, the runbook explains how to operate it.

**Standard checks** (run from `projects/internal-tools/article-aggregator/`, and in CI via `.github/workflows/article-aggregator.yml`):

```bash
python -m pip install -e '.[dev]'
pytest                # unit tests; integration tests need DATABASE_URL (Postgres service in CI)
ruff check .
mypy --strict src
```

---

## Slice 1: Walking skeleton (fetch → filter → post, deployed)

**Status (2026-11-18):** code, tests, CI and digest workflows, and RUNBOOK done; standard checks pass locally. **Not done:** the two live runs (needs the Discord bot, the `article-aggregator` environment and the PR merged to `main`). Real feeds (10) and keywords (17) are in `config/`.

**Goal:** A real digest appears in the Discord channel every morning, built only from plain code.

**Context:** No database yet. The job posts matching entries **published in the last 24 h** (entries without a date are skipped in this slice). Re-runs on the same day can double-post; that is accepted until slice 2b. Sets up the package, config, CI and the scheduled workflow. This slice is broad because it is the end-to-end skeleton; keep each module minimal.

**Acceptance criteria:**
- `config/feeds.yaml` and `config/keywords.yaml` load through pydantic models; invalid config fails with a clear message.
- `fetch.py`: `httpx` with a 15 s timeout, 5 MB cap, identifiable User-Agent, at most 5 concurrent requests. A failing feed is logged and skipped, and the run continues.
- `parse.py`: `feedparser` output becomes `Entry(title, url, guid, summary_html, published_at, feed_name)`.
- `match.py`: case-insensitive whole-word/phrase match on title + stripped summary. Unit tests cover `AI`≠`said`, `zero day`=`Zero  Day`, `C++`, accents/NFKC, and HTML in summaries.
- `discord.py` (REST client): builds embeds (title ≤ 256, url, description, footer `feed · keywords`, timestamp), posts a header message containing `Digest <date>` and the article count, then the **top 10** matches (ranked by distinct keywords matched, then newest, then URL; per-feed `max_per_digest` caps respected) as **separate single-embed messages**. It paces itself from Discord's rate-limit headers, retries once on 429 using `retry_after`, and a failure partway reports how many messages were already posted. Unit tests check the ranking ties, the 10-article cap, the per-feed caps with backfill, the header, the pacing, and the partial-failure report.
- CLI `article-aggregator digest [--dry-run] [--target test|production]`. Dry run prints the payloads and does not call Discord.
- Fixture feeds (RSS 2.0, Atom, malformed XML) are in `tests/fixtures/`. No live network in unit tests.
- Workflow `article-aggregator-digest.yml`:
  - cron `53 5 * * *`, `workflow_dispatch`, environment `article-aggregator`, its own `concurrency` group, `timeout-minutes: 5`;
  - installs runtime dependencies only, with the pip cache;
  - scheduled runs post to `DISCORD_DIGEST_CHANNEL_ID`; `workflow_dispatch` has a `target` input that defaults to `test` (`DISCORD_DIGEST_TEST_CHANNEL_ID`).
- CI workflow runs the standard checks on PRs that touch the project.
- `docs/RUNBOOK.md` covers rotating the bot token, disabling the scheduled workflow, and doing a manual run to the test channel.
- **Live:** one `workflow_dispatch` run with `target=test` posts to the test channel, and the first scheduled run posts to the production channel. Record both run IDs, and the billed minutes of the scheduled run, in the PR.
- Standard checks pass.

**Out of scope:** database, dedup, polling between digests, summaries beyond the raw truncated RSS text, retries beyond a single 429 retry.

---

## Slice 2a: Store and dedup matches

**Goal:** Every matched entry is stored once in Postgres, with dedup keys, between digests. The digest itself is unchanged from slice 1.

**Context:** Adds Postgres and the poll job. Nothing reader-facing changes, so this slice can ship and collect real data (and real Actions-minute usage) before the digest depends on it.

**Acceptance criteria:**
- SQL migrations in `migrations/` create `feeds`, `articles`, `article_keys` and `digests` exactly as in SPEC.md. `article-aggregator migrate` applies them idempotently.
- `normalize.py`: URL and title normalization per SPEC.md, plus sha256 keys (the `guid` key scoped by `feed_url`). Table-driven unit tests cover:
  - tracking params, `www.`, trailing slash, http vs https;
  - the publisher-suffix strip and the ≥ 5-word title guard.
- Dedup per SPEC.md *Dedup keys*. The insert of an article and its keys happens in one transaction (`ON CONFLICT DO NOTHING`). Integration tests cover:
  - the same story from two feeds, and re-polling the same feed;
  - guid-only matches, and **the same guid string in two different feeds stored as two articles**;
  - a ≥ 5-word title seen again after 8 days being stored as a new article (key reassigned);
  - a key hit updating `last_seen_at`.
- Feeds upsert per SPEC.md: a feed removed from the config gets `removed_at`, is not polled, and its articles stay; re-adding it clears `removed_at`. Integration test covers both.
- CLI `article-aggregator poll [--seed]` stores new matches. `--seed` stores them as `skipped` (used to rebuild dedup history after DB loss without re-posting). Poll uses `pg_try_advisory_lock` and exits 0 when the lock is held.
- Workflow `article-aggregator-poll.yml`:
  - cron `23 */2 * * *`, its own `concurrency` group (not shared with the digest), `timeout-minutes: 5`;
  - installs runtime dependencies only, with the pip cache.
- **Budget check:** after 7 days of scheduled polls, record the median and maximum billed minutes per poll and the projected monthly total (polls + digests + CI) in the PR or a follow-up issue. If the projection is over 350 minutes, switch poll to `23 */3 * * *` before starting slice 2b.
- `docs/RUNBOOK.md` adds running migrations and restoring after DB loss (`migrate`, then `poll --seed`).
- Standard checks pass, with integration tests running against the Postgres service container.

**Out of scope:** changing the digest, summaries, pruning.

---

## Slice 2b: Digest from the database, exactly once

**Goal:** Each matched article appears in at most one digest: either posted, or retired as `skipped`. No digest is ever posted twice, and none is silently lost.

**Context:** Switches `digest` to the database and implements SPEC.md *Digest procedure* and *Concurrency*. Summaries are still the raw truncated RSS text from slice 1.

**Acceptance criteria:**
- `digest` follows SPEC.md *Digest procedure* step by step: date resolution, lock, existing-row handling, snapshot, post, finalize, prune.
- `digest --date YYYY-MM-DD` and a matching `workflow_dispatch` `date` input. Unit tests:
  - a future date is refused;
  - a run at 00:30 EAT for today's date is refused, with a message suggesting yesterday's date;
  - a run at 08:53 EAT for today is accepted.
- Integration tests (Discord client faked):
  - running `digest` twice on the same date posts once;
  - with 13 snapshot articles, 10 are `posted`, 3 are `skipped`, and none of the 13 appear in the next day's digest;
  - with a capped feed holding 5 of the snapshot articles at `max_per_digest: 2`, its other 3 are `skipped` even when fewer than 10 are posted;
  - an article stored after the snapshot (simulated concurrent poll) stays `pending` with `digest_id IS NULL`, is not marked `skipped`, and appears in the next day's digest;
  - crash after Discord accepts the header and 4 article messages but before their IDs are committed (row left `posting`), with the faked channel history containing those 5 messages: the rerun records their IDs from history and posts only the remaining 6;
  - the same crash with nothing in channel history: the rerun posts the header and all 10 once;
  - Discord returns a non-429 error on the 7th article: the row is `failed`, the command exits non-zero, 6 articles keep their message IDs, and a rerun posts only articles 7–10 (no second header);
  - quiet day: no post, row `empty`, and a rerun does not post;
  - poll called while digest holds the lock exits 0 without writing.
- Prune per SPEC.md runs at the end of every digest. Integration tests:
  - a key last seen 91 days ago is deleted, and one first seen 200 days ago but last seen yesterday is kept;
  - a 91-day-old `pending` article is not deleted;
  - a removed feed with no remaining articles is deleted;
  - a prune error leaves the digest `posted`.
- `docs/RUNBOOK.md` adds re-running a missed digest (`--date`), what a `failed` or stuck `posting` row means, and how to recover from each.
- Standard checks pass.

**Out of scope:** summaries, article-page fetching, LLM, editing config from Discord.

---

## Slice 2c: Plain-code summaries

**Goal:** Every posted article has a clean summary, or a clean title-and-link fallback.

**Context:** Adds the `Summarizer` interface. The original plan had an AI summarizer here; with no LLM budget, the **plain-code summarizer is the only implementation** and the LLM moves to slice 6.

**Acceptance criteria:**
- `Summarizer` protocol with `PlainSummarizer`, run during `poll`:
  - strips HTML and drops the summary if it equals the title;
  - truncates to ≤ 300 chars at a sentence boundary (if one falls at ≥ 150 chars), otherwise at a word boundary plus `…`;
  - sets `summary_status` to `rss_summary` or `none`.
- A summarizer exception never blocks storing or posting; the article falls back to `none` with title and link only. Unit test with a summarizer that raises.
- `digest` posts `summary` when present. Articles stored before this slice (`summary_status = pending`) fall back to the slice 1 truncation; integration test covers one.
- Standard checks pass.

**Out of scope:** article-page fetching, LLM.

---

## Slice 3: Feed resilience and operator visibility

**Goal:** Broken or slow feeds are cheap and visible instead of silent.

**Context:** Feeds rot quietly. Conditional GET also cuts bandwidth and runner minutes.

**Acceptance criteria:**
- Conditional GET using stored `etag`/`last_modified`; a 304 response skips parsing. Unit-tested with `httpx.MockTransport`.
- `feeds.consecutive_failures` is updated on every run. When a feed reaches 6 consecutive failures (about 12 h at the 2 h cadence), the next digest adds a single "Feed health" embed listing it.
- A structured run summary goes to the Actions log: feeds ok/failed/304, entries seen, matched, new, posted, and run duration.
- `article-aggregator validate-config` checks YAML schema, duplicate URLs, and an empty keyword list, and runs in CI.
- `docs/RUNBOOK.md` adds adding, disabling and removing a feed or keyword, and reading the run summary and "Feed health" embed.
- Standard checks pass.

**Out of scope:** paging/alerting outside Discord, automatic feed disabling.

---

## Slice 4: Article-text summaries for non-paywalled pages

**Goal:** Better summaries when the full article is freely readable.

**Context:** Deterministic rule from the interview: use the page text only if it is clearly a real article; otherwise fall back to the RSS summary. This adds outbound requests to `poll`, so it must stay inside the minutes budget.

**Acceptance criteria:**
- For each new matched article (during `poll`, not `digest`), at most 10 per poll run and within the shared 5-concurrent-request limit: check `robots.txt`, then fetch with a 10 s timeout and 2 MB cap, and extract with `trafilatura`. Articles over the cap keep the slice 2c summary.
- Use the extracted text only if it has **≥ 150 words and is longer than the RSS summary**; set `summary_status = article_text`. Otherwise use the slice 2c path.
- Only the ≤ 300-char summary is stored and posted; the full text is never persisted.
- Fixture tests: a full article, a paywall teaser, a cookie wall, an empty JS shell, a `robots.txt` disallow, a timeout, and the 10-per-run cap.
- **Budget check:** after 7 days, re-record billed minutes per poll and the monthly projection; it must stay ≤ 350.
- Standard checks pass.

**Out of scope:** headless browsers, login cookies, paywall bypassing.

---

## Slice 5: Handoff

**Goal:** Meet the `internal-tools/README.md` requirements so someone else can run the tool.

**Acceptance criteria:**
- `README.md` lists:
  - the owner;
  - operating cost: $0, the limits it depends on, and the latest measured Actions minutes;
  - external services (Discord, Neon, GitHub Actions) and the deployment target;
  - backup strategy: data is rebuildable, and after DB loss run `migrate`, then `poll --seed` to rebuild dedup history without re-posting;
  - maintenance expectations.
- `docs/RUNBOOK.md` is reviewed end to end by someone other than the author, who follows it on the test channel.
- `internal-tools/README.md` lists the project.

**Out of scope:** new features.

---

## Slice 6 (blocked: needs LLM budget approval): LLM summaries

**Goal:** 1–2 sentence AI summaries per article, with the plain-code summary as fallback.

**Acceptance criteria (draft):** an `LLMSummarizer` implements the `Summarizer` protocol; per-call timeout; a daily cap on calls; any error or timeout falls back to the slice 2c or slice 4 summary; `summary_status = llm` on success; tests use a fake client; a cost estimate goes in the README.

**Out of scope:** digest-level "themes" summary; LLM-based filtering or dedup.

---

## Slice 7 (future, not v1): Slash commands

**Goal:** `/keywords add|remove|list` and `/feeds …` from Discord.

**Context:** Needs an HTTP interactions endpoint (Ed25519 signature check) on a free serverless host, and moves keywords and feeds from YAML into DB tables. Re-evaluate Python vs a small TS Worker at that point.
