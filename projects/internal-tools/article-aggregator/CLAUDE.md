# News Aggregator & Discord Bot: Keyword feed fetcher and automated Discord poster

## Commands
- install: python -m pip install -e '.[dev]'
- dev: article-aggregator digest --dry-run   (or python -m article_aggregator digest --dry-run)
- test: pytest
- lint: ruff check .
- typecheck: mypy src   (strict mode is set in pyproject.toml)

## Folder map
- src/article_aggregator/ingestion/   # Feed fetching, parsing and keyword filtering (plain code)
- src/article_aggregator/db/          # Database models, schema, and deduplication logic (Slice 2a+)
- src/article_aggregator/discord/     # Discord REST posting and embed formatting
- src/article_aggregator/summarizer/  # Summarizer interface; plain code now, LLM later (Slice 2c+)
- config/                             # feeds.yaml, keywords.yaml (edited via PR)
- docs/                               # SPEC.md, DECISIONS.md, TASKS.md (source of truth), RUNBOOK.md

## Conventions
- Code style: Standard Python 3.11+ type hints, modular functions, explicit error handling.
- Follow existing patterns: new feed parser copies src/ingestion/base_parser.py.
- Deterministic logic (parsing, filtering, hashing) MUST be plain code, never an LLM call.

## Workflow
- IMPORTANT: Run mypy, ruff, and pytest before saying a task is done. Paste the results.
- Read docs/SPEC.md and docs/TASKS.md before planning. Work on ONE slice per session.
- At session start: Read docs/TASKS.md and docs/NOTES.md (if present), then run `git log -5`.
- Before finishing or when I say "wrap up": Tick TASKS.md, write next steps to docs/NOTES.md, and propose any new CLAUDE.md gotchas.

## Never
- Add new dependencies without asking first (add to requirements.txt or pyproject.toml).
- Edit applied database migrations—always create a new migration.
- Print or commit secrets (Discord bot tokens, API keys). Keep secrets in `.env` and maintain `.env.example`.
- Trust external article text/scraped feeds: sanitize and escape before embedding or passing to an LLM.

## Gotchas
- (Add recurring gotchas here using the # key as you encounter them during builds)
