---
title: How Ask BMTech Answers Code Questions
description: How the Ask BMTech Discord bot indexes the public bmtechllc repository and answers questions about its code, workflows, and recent pull requests, including what it skips and when it declines to answer.
tags: [reference, tooling, rag, llm]
status: published
visibility: internal
owner: team
updated: 2026-11-13
reviewCycleMonths: 6
order: 50
related: [hybrid-search-rrf-bm25-rag-evaluation-conceptual-cheatsheet]
---

Ask BMTech (the bot in `#ask-bmtech` and behind `/ask`) can now answer
questions about the code in the public `benchmarktechllc/bmtechllc`
repository: where something lives, how it is configured, which workflow runs
it, and what changed in recent pull requests. It shipped in
[PR #64](https://github.com/benchmarktechllc/bmtechllc/pull/64) and
[PR #65](https://github.com/benchmarktechllc/bmtechllc/pull/65).

This page explains what gets indexed, how a question ends up in the code lane,
and when the bot declines to answer.

## The constraint everything follows from

Code questions must never change how the bot answers anything else. Person,
board, headcount, portal, and saved-answer questions already had answers the
team relied on, and a code index sitting in the same table would compete with
them in every search.

So code lives apart. Repository chunks go into their own `github_chunks`
table, **never into `document_chunks`**. No existing query, index, or ranking
reads it. The only way in is a separate lane that a question has to qualify
for.

## What gets indexed

`benchmark-sync-github` reads a local checkout of the repository and stores
each file as citation-sized chunks with line numbers and the commit it came
from.

| Indexed | Skipped |
|---|---|
| Source and docs: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`, `.sql`, `.css`, `.html`, `.sh`, `.md` | Anything under `portal/content/` — already indexed as KB pages |
| Config: `.yml`, `.yaml`, `.toml`, Dockerfiles | Tests, fixtures, `archive/`, `project_memory/`, build output, `node_modules` |
| GitHub Actions workflows | `.env*`, `.pem`, `.key`, and any file named like a secret or credential |
| Pull requests: title, state, merge time, link, changed paths, and body | Lockfiles, and any file over 200 KB |

Pull requests are stored as one record each. **Diffs and individual commit
messages are not stored.** Every chunk passes through the same secret
redaction as the other syncs before it is saved.

> [!NOTE]
> Only what is on `main` is searchable. Code that lives only on an open
> branch is invisible to the bot until it merges.

## Staying current

The workflow `.github/workflows/github-code-ingestion.yml` runs on every push
to `main`, daily at 06:17 UTC, and on manual dispatch. Each run:

1. Compares each file's digest with the stored one and re-embeds **only the
   files that changed**.
2. Deletes chunks for files that left the tree.
3. Refreshes the pull-request records from the GitHub API.

It uses the existing `discord-ingestion` environment for `DATABASE_URL` and
`OPENAI_API_KEY`, and the workflow's built-in `GITHUB_TOKEN`. No new secret
is needed.

To run it by hand from `projects/internal-tools/benchmark-cli-tool`:

```bash
benchmark-sync-github --root <path-to-checkout>
```

| Flag | Effect |
|---|---|
| `--sha` | Commit to record. Defaults to `HEAD` of `--root` |
| `--full` | Reindex every file, ignoring stored digests |
| `--skip-pulls` | Leave indexed pull requests unchanged |
| `--pulls-json` / `--write-pulls-json` | Read or save the pull-request list instead of fetching it |

Without a token, the GitHub API allows 60 unauthenticated requests an hour.

## How a question reaches the code lane

A question goes to the code lane only when two things are true.

**It looks like a code question.** It mentions code words (repo, function,
workflow, cron, migration, table, env var, config, PR, commit, merged), names
an identifier (`something_like_this`, a file name, a backticked term), or asks
where or how the bot does something.

**Nothing vetoes it.** Words like ticket, card, board, backlog, assigned,
due, blocked, status, or "working on" send it to the board path instead. The
check errs toward *not* a code question.

The lane also runs **after** every existing shortcut: person and roster
questions, portal and Trello links, Trello lists and board questions,
headcount, and core team. A saved answer that matches the question exactly
still wins, but saved answers are never matched by paraphrase for a code
question.

## What happens inside the lane

**Recent pull requests.** "What changed in the last few PRs?" is answered
from the newest indexed pull requests, ordered by merge time, with links. No
search is involved, because embeddings do not order by date.

**Everything else** goes through three searches over code, config, workflow,
and doc chunks — vector, full-text, and a path/symbol header match — fused
with RRF. The results are re-ranked so that a match on a file's path or symbol
counts for more than a match on body text, and rarer words count for more than
common ones. Chunks from the project the question is about rank higher; files
from other projects count at half. A question that names no project, or says
"the bot", "Ask bot", or "BMTech bot", is scoped to
`projects/internal-tools/benchmark-cli-tool`.

At most eight chunks reach the model: no more than two per file and two
documentation chunks in total. Docs are weighted at half, so a source file
beats a README that describes it.

The model is the same cheap chat model the bot uses for everything else. It
sees only the lane chunks, marked as untrusted data, and is told to name the
file path, quote it verbatim, and prefer the source, workflow, or config file
over a document that restates it. **It does not write links.** GitHub links
to the exact lines are added afterwards from the chunks it cited.

## When it declines to answer

The answer is checked before it is posted:

- Every citation must point at a chunk the lane supplied, and its quote must
  appear in that chunk word for word.
- Any URL that is not in the supplied chunks is removed.
- If no citation survives, the model gets **one retry** with the reasons.

If the retry also fails, a "where is…" or "which file…" question gets a
plain pointer instead:

```text
Closest match: `<path>` lines <start>–<end> (<symbol>)
```

Anything else gets "No relevant information was found in the code
repository." The lane also declines when retrieval is weak — the best chunk
must clear the minimum similarity (`MIN_COSINE_SIMILARITY`, 0.30 by default)
and share at least one meaningful word with the question.

If `github_chunks` is empty or unreachable, the lane steps aside and the
question takes the normal answer path.

## What is deliberately not here

| Not built | Why |
|---|---|
| Code in `document_chunks` | It would compete with every non-code answer |
| Diffs and commit messages | Pull-request title, body, and paths answer "what changed" without the noise |
| Indexing open branches | The bot answers about what is actually on `main` |
| Model-written links | Links come from the cited chunk, so they cannot be invented |

## Where the code lives

All paths are under `projects/internal-tools/benchmark-cli-tool/`.

| File | Responsibility |
|---|---|
| `src/benchmark_cli/github_sync.py` | `benchmark-sync-github`: change detection, pull-request fetch, embedding |
| `src/benchmark_cli/github_files.py` | Which files are indexed, file kinds, chunking, line ranges, blob URLs |
| `src/benchmark_cli/github_answers.py` | The code-question gate, retrieval, re-ranking, prompt, and validator |
| `src/benchmark_cli/agent.py` | Where the code lane sits relative to the other answer paths |
| `src/benchmark_cli/storage/sql/009_github_sync.sql` | The `github_chunks` table and sync state |
| `.github/workflows/github-code-ingestion.yml` (repo root) | Scheduled and on-push sync |
