# Article Aggregator — Spec

A scheduled job that reads RSS/Atom feeds, keeps articles matching a target keyword list, drops duplicates, and posts one daily digest to a Discord channel. All selection logic (parsing, matching, dedup, summarizing) is plain, deterministic code. v1 uses no LLM.

## Users

- **Readers:** BMTech team members in one Discord channel who want a once-a-day scan of relevant news.
- **Operator:** one maintainer who edits `config/feeds.yaml` and `config/keywords.yaml` through pull requests.

## Core loop

1. **Poll (every 2 h, GitHub Actions cron `23 */2 * * *` UTC).** Try to take the run lock (see *Concurrency*); if the digest holds it, exit 0 because the digest runs its own poll. For each enabled feed: fetch (15 s timeout, 5 MB cap), parse, normalize each entry. One failing feed never stops the others.
2. **Filter.** Keep an entry if any keyword matches the title or RSS summary as a **case-insensitive whole word or phrase** (HTML stripped, Unicode NFKC, whitespace collapsed; `zero day` matches `Zero  Day`, `AI` does not match `said`).
3. **Dedup.** Compute keys (see *Dedup keys*). If any key is a hit, skip the entry and update `last_seen_at` on its existing keys. Otherwise store the article and its keys in one transaction.
4. **Summarize.** Plain-code summary from the RSS summary: strip HTML, collapse whitespace, cut to ≤ 300 chars at a sentence (else word) boundary. Record `summary_status`.
5. **Digest (daily, cron `53 5 * * *` UTC ≈ 09:00 EAT; GitHub may start it up to ~30 min late).** See *Digest procedure*. It posts a **header message** (`Digest <date>` · article count; matches beyond the top 10 are not mentioned) followed by the **top 10** pending articles as **10 separate messages**, one embed each. It retires the rest as `skipped` (they are **not** carried over; most of them are arXiv overflow). A feed with `max_per_digest` (arXiv cs.AI 2, TechCrunch AI 2, LangChain Blog 3) never takes more slots than that; lower-ranked articles from other feeds fill in, so a day can have fewer than 10. **Quiet day: post nothing.** A digest is posted at most once per date.
6. **Prune** (end of every digest run; a prune failure never changes the digest's status):
   - delete keys whose `last_seen_at` is older than 90 days;
   - delete non-`pending` articles whose `first_seen_at` is older than 90 days and that no key references;
   - delete `feeds` rows that have `removed_at` set and no articles.

### Dedup keys

| Key | Value hashed (sha256 hex) | Counts as a hit when |
|---|---|---|
| `guid` | `feed_url` + `\n` + raw guid (guids are only unique within a feed) | the key exists |
| `url` | normalized URL | the key exists |
| `title` | normalized title; only when it has ≥ 5 words | the key exists **and** its `first_seen_at` is within the last 7 days |

A `title` key older than 7 days is not a hit. If the entry is otherwise new, that key row is reassigned to the new article (`article_id`, `first_seen_at` and `last_seen_at` reset). This lets recurring headlines like "Microsoft Patch Tuesday security updates released" through each month. An old item that stays in its feed is still caught by its `guid`/`url` keys. Those keys keep getting their `last_seen_at` refreshed, so they are never pruned while the item is still in the feed.

### Digest procedure

1. **Resolve `digest_date`** (Africa/Nairobi). Scheduled runs use today. Manual runs take `--date` (default today). The command refuses, exiting non-zero and writing nothing, if `digest_date` is in the future or if it is `digest_date` before 08:30 EAT. This stops a late-evening manual rerun from taking the next morning's slot.
2. **Take the run lock**, waiting up to 10 minutes; if it is still held after that, exit non-zero.
3. **If a `digests` row exists for the date:**
   - `posted` or `empty`: exit 0.
   - `posting` (a previous run died partway) or `failed`: reconcile, then resume at step 5. To reconcile, read the channel's last 100 messages from the bot posted since the start of `digest_date`:
     - a message whose content contains `Digest <digest_date>` is the header, so record its ID as `header_message_id`;
     - a message whose embed URL equals a shown article's URL is that article's message, so record its ID on the article.
     Step 5 then posts only what is still missing, so a rerun never duplicates a message.
4. **If no row exists:** run one final poll under the lock. Then snapshot every `pending` article with `digest_id IS NULL`, ranked by most distinct keywords matched, then newest (`published_at`, else `first_seen_at`), then normalized URL. In one transaction:
   - insert the `digests` row (`status = posting`, `ranked_article_ids` = the snapshot in order);
   - set `digest_id` on the snapshot articles;
   - commit.
   If the snapshot is empty, set `status = empty` and stop without posting.
5. **Post** the top 10 of the snapshot after per-feed caps (always computed from `ranked_article_ids`, so reruns pick the same articles) as separate messages, in order:
   - the **header** message (`Digest <digest_date>` and the article count), unless `header_message_id` is set. Commit its ID as soon as Discord confirms;
   - one **article** message per shown article that has no `discord_message_id`, in rank order. After each one Discord confirms, commit that article's `discord_message_id`, `posted_at` and `delivery_status = posted` in one transaction.
   The client waits out Discord's per-channel rate limit (from the `X-RateLimit-Remaining` / `X-RateLimit-Reset-After` headers) instead of tripping a 429.
6. **Finalize** in one transaction: mark the rest of the snapshot `skipped`, set the row to `posted` with `finished_at`.
   If Discord still fails after the single 429 retry, or returns any other error, set `status = failed` and exit non-zero, keeping the IDs already committed. The failed Actions run is the alert; a rerun resumes where it stopped.

Articles stored after the snapshot keep `digest_id IS NULL` and go into the next day's digest. Only snapshot articles (by ID) are ever marked `skipped`.

### Concurrency

Poll and digest share one Postgres advisory lock (`pg_try_advisory_lock` for poll, `pg_advisory_lock` with a 10-minute timeout for digest). Each workflow also has its own Actions `concurrency` group so that two runs of the same job never overlap. The two workflows deliberately do **not** share a group, because a newer queued run in a shared group cancels the older pending one: a queued poll could cancel the digest.

## Data model (Postgres)

Feeds and keywords are defined in config files (source of truth). The database holds only runtime state.

| Table | Columns (key ones) | Notes |
|---|---|---|
| `feeds` | `url` PK, `name`, `enabled`, `removed_at` null, `etag`, `last_modified`, `last_fetched_at`, `last_status`, `consecutive_failures` | Upserted from `config/feeds.yaml` each run. A feed missing from the config gets `removed_at` and is never polled; the row stays while articles reference it and is pruned after that. Re-adding a feed clears `removed_at`. |
| `articles` | `id` PK, `feed_url` FK, `url`, `title`, `rss_summary`, `published_at`, `first_seen_at`, `matched_keywords text[]`, `summary`, `summary_status`, `delivery_status` (`pending`/`posted`/`skipped`), `digest_id` FK null, `discord_message_id` null, `posted_at` null | Only **matched** articles are stored; `skipped` articles stay as dedup history. `digest_id` is set when the article enters a digest snapshot. |
| `article_keys` | `key_hash` PK (sha256 hex), `key_type` (`guid`/`url`/`title`), `article_id` FK, `first_seen_at`, `last_seen_at` | The "processed article hashes"; see *Dedup keys* |
| `digests` | `id` PK, `digest_date` UNIQUE, `status` (`posting`/`posted`/`failed`/`empty`), `ranked_article_ids bigint[]`, `started_at`, `finished_at`, `article_count`, `header_message_id` null | The unique date, the stored snapshot and the per-message IDs make reruns post each message once |

`summary_status` ∈ `pending`, `rss_summary`, `article_text`, `llm`, `none` (`llm` is reserved for a later slice, so adding it needs no schema change).

Config:
- `config/feeds.yaml`: list of `{name, url, enabled, max_per_digest}`; names and URLs are unique.
- `config/keywords.yaml`: flat list of 10–20 strings.

**URL normalization:** drop scheme and fragment, lowercase the host, strip `www.` and default ports, remove `utm_*`, `fbclid`, `gclid`, `mc_cid`, `mc_eid`, `ref`, sort remaining query params, strip the trailing `/`.
**Title normalization:** NFKC, lowercase, strip a trailing ` - Publisher` / ` | Publisher` segment of ≤ 4 words, remove punctuation, collapse whitespace.

## Non-goals (v1)

- Web UI or dashboard
- Manual article editing, approval, or curation
- Slash commands / bot interactions (config is edited via PR)
- LLM summaries (blocked until a budget is approved)
- Paid news APIs, social media, non-RSS scraping
- Reposting full article text (summary + link only)
- More than one reader-facing Discord channel; per-user subscriptions or DMs (a private test channel for manual runs is allowed)
- Semantic / near-duplicate dedup; re-posting edited articles
- Analytics or read tracking

## Constraints

- **$0/month.** GitHub Actions minutes plus a free-tier hosted Postgres. No paid APIs.
- **Actions minutes budget.** Projected monthly usage (polls + digests + CI) stays ≤ 350 minutes, leaving headroom under the ~400 available. GitHub bills each job rounded up to a whole minute, so 12 polls a day is already ≥ 360 billed minutes a month if each poll takes even one minute.
  - Scheduled workflows install runtime dependencies only (`pip install .`, not `.[dev]`), use the `actions/setup-python` pip cache, and set `timeout-minutes: 5`.
  - If measured usage projects over 350, poll drops to every 3 h (`23 */3 * * *`).
- Discord: bot account and REST API only (no gateway connection); permissions View Channel, Send Messages, Embed Links, Read Message History (for digest reconciliation). Limits: title ≤ 256 chars, 6,000 embed chars per message (one embed per message stays far below it); posting 11 messages in a row hits the per-channel rate limit, so the client paces itself from the rate-limit headers.
- Polite fetching: an identifiable User-Agent, at most 5 concurrent requests, and conditional GET (ETag/Last-Modified).

## Assumptions (confirmed 2026-11-18)

1. About 10–20 feeds (10 at launch). Measured 2026-11-18: ~160 matches per day, ~150 of them from arXiv cs.AI (the original 5–30 estimate was wrong).
2. Edited articles are ignored; hashes are kept 90 days after they were last seen, so items that stay in a feed are not re-posted after a prune.
3. Hosted DB: a new free-tier Neon Postgres.
4. The org's GitHub plan covers about 400 extra Actions minutes/month (12 polls/day plus the digest plus CI).
5. One title-only collision guard (≥ 5 words) is enough to keep generic titles like "Weekly roundup" from being treated as duplicates, and title keys only dedup within 7 days.
6. Articles with no published date are still included; they are ordered by `first_seen_at`.
7. "09:00 EAT" means about 09:00 (08:53 nominal start, plus GitHub cron delay).
8. Retiring matches outside the top 10 is intended: most days that is 100+ articles, almost all from arXiv.
9. The bot is granted Read Message History so an interrupted digest can be reconciled.
10. Poll falls back to every 3 h if the measured Actions minutes run over budget.
11. Manual runs default to a private test channel (`DISCORD_DIGEST_TEST_CHANNEL_ID`).
12. "Anthropic News" comes from an unofficial community mirror, because Anthropic publishes no RSS feed.
