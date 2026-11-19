# Notes

## 2026-11-18: Slice 1 built, not yet live

Next steps, in order:
1. Done: real feeds and 20 keywords are in `config/`, arXiv capped at 2 per digest (`max_per_digest`). Live dry run 2026-11-18: 7/10 feeds ok, 159 matches in 24 h, of which 154 from arXiv. Volume is far above the 5–30/day assumption, almost all arXiv.
   Later the same day: digest raised to 10 articles; arXiv and TechCrunch capped at 2; dead feeds replaced (LangChain `www.langchain.com/blog/rss.xml`, InfoQ `feed.infoq.com/ai-ml-data-eng/news/`, Anthropic via an unofficial GitHub mirror). Dry run: 10/10 feeds ok, 167 matches, LangChain took 6 of 10 slots, so it may need a cap too.
   Then: LangChain capped at 3; digest posts a header message plus one message per article, with the client pacing itself from Discord's rate-limit headers. Dry run: 9 articles (every feed with matches hit its cap). Slice 2b's crash recovery in SPEC.md was redesigned for per-message IDs. The "+N more" line was dropped: it was mostly arXiv overflow and gave readers nothing to act on.
2. Owner setup per `docs/RUNBOOK.md` → *One-time setup*: Discord bot, test + production channels, GitHub environment `article-aggregator`.
3. Open the PR (CI runs `article-aggregator.yml`), merge to `main`.
4. Actions → *Article aggregator digest* → Run workflow, `target: test`. Record the run ID in the PR.
5. Wait for the first scheduled run (05:53 UTC). Record its run ID and billed minutes. Then tick slice 1 done in TASKS.md.
6. Start slice 2a (Postgres on Neon, poll job, dedup).

Decisions made this session:
- Layout: CLAUDE.md's folder split inside one package, `src/article_aggregator/`.
- Dependencies approved: feedparser, httpx, pydantic, PyYAML; dev: pytest, ruff, mypy, types-PyYAML.
- CLI `--target` defaults to `test`; only the scheduled workflow posts to production.
- Exact-URL repeats within one run are dropped so the same item from two feeds isn't shown twice. Real dedup is still slice 2a.
- Feed text is escaped for Discord markdown and every message sets `allowed_mentions: {parse: []}`.
