# Runbook

Operating notes for what is in production. Each slice adds to this file.

## What runs

| Workflow | When | What it does |
|---|---|---|
| `.github/workflows/article-aggregator-digest.yml` | Daily 05:53 UTC (≈ 08:53 EAT, may start up to ~30 min late) and manual | Fetches feeds, keeps last-24 h keyword matches, posts a header message plus up to 10 article messages (arXiv and TechCrunch capped at 2, LangChain at 3) |
| `.github/workflows/article-aggregator.yml` | PRs and pushes to `main` touching the project | pytest, ruff, mypy |

Slice 1 has no database: a second run on the same day posts the same articles again, and a run that fails partway (`posted K of N digest messages, then failed`) leaves its first K messages in the channel; a rerun posts all N again. Delete the partial set by hand first if that matters. Use manual runs on the **test** channel.

## One-time setup

1. **Discord bot.** In the Discord Developer Portal, create an application, add a bot, and copy its token. Invite it with the `bot` scope and these permissions on the digest and test channels: View Channel, Send Messages, Embed Links, Read Message History. The bot needs no privileged intents.
2. **Channel IDs.** Turn on Developer Mode in Discord, then right-click each channel → Copy Channel ID.
3. **GitHub environment.** In repo Settings → Environments, create `article-aggregator` and add:
   - secret `DISCORD_DIGEST_BOT_TOKEN`;
   - variables `DISCORD_DIGEST_CHANNEL_ID` (readers) and `DISCORD_DIGEST_TEST_CHANNEL_ID` (private test channel).

`workflow_dispatch` only appears in the Actions tab once the workflow file is on `main`.

## Manual run to the test channel

Actions → **Article aggregator digest** → Run workflow → `target: test`.

Locally:

```bash
cd projects/internal-tools/article-aggregator
python -m pip install -e '.[dev]'
article-aggregator digest --dry-run          # prints the payload; needs no secrets
cp .env.example .env                         # fill in values; .env is gitignored
set -a; . ./.env; set +a
article-aggregator digest --target test
```

Exit codes: `0` ok (including a quiet day with no post), `1` all feeds failed or Discord rejected the post, `2` bad config or missing environment variable.

## Rotate the bot token

1. Developer Portal → the application → Bot → **Reset Token**. The old token stops working immediately.
2. Update the `DISCORD_DIGEST_BOT_TOKEN` secret in the `article-aggregator` environment.
3. Run the workflow manually with `target: test` to confirm.
4. If the token may have leaked, also check the server audit log for actions by the bot.

## Disable or re-enable the daily digest

Actions → **Article aggregator digest** → `⋯` → **Disable workflow** (or **Enable workflow**). This stops scheduled and manual runs; nothing else needs changing. GitHub also disables scheduled workflows automatically after 60 days without repository activity in public repos.

## Change feeds or keywords

Edit `config/feeds.yaml` or `config/keywords.yaml` in a PR. CI loads the repository config, so a schema mistake fails the PR, not the morning run.

## When a run fails

Open the failed run's log. The last `INFO article_aggregator.digest` line has the counts (feeds ok/failed, entries, matched).
- `feed '…' failed`: that feed was skipped; the digest still posted. Persistent failures: fix or disable the feed.
- `all N enabled feeds failed`: usually a runner network problem; re-run the job.
- `Discord API returned HTTP 401`: bad token (rotate it). `403`: the bot lacks channel permissions. `404`: wrong channel ID.
