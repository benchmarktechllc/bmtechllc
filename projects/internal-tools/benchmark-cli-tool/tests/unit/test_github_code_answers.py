"""Code questions get a separate GitHub lane; every other question keeps the baseline path."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import types
from pathlib import Path
from typing import Any

import pytest

from benchmark_cli import agent, github_answers, github_sync, retrieval
from benchmark_cli.discord_ask import format_discord_answer
from benchmark_cli.github_answers import (
    CodeRetrieval,
    answer_from_retrieval,
    append_github_links,
    asks_for_recent_pulls,
    is_code_question,
    pointer_answer,
    recent_pulls_answer,
    rerank,
    validate_code_answer,
)
from benchmark_cli.github_files import (
    blob_url,
    chunk_python,
    chunk_text,
    header_words,
    kind_for,
    render_chunk_text,
    should_index,
    source_file_for,
)
from benchmark_cli.github_sync import code_rows, plan_file_changes
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import RETRIEVAL_SOURCES, HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

_REPO = "benchmarktechllc/bmtechllc"
_SHA = "abc123def456abc123def456abc123def456abcd"
_ROOT = Path(__file__).resolve().parents[5]
_PKG = "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli"
_TRELLO_SYNC = f"{_PKG}/trello_sync.py"
# sha256 of agent.SYSTEM_PROMPT at the 0caab09 baseline. Non-code prompts must not change.
_BASELINE_SYSTEM_PROMPT_SHA256 = "620b1e2303770bebd85b98dcb55774886f71601b2ff36659fe591f006b4a6530"


def _chunk(chunk_id: int, source: str, text: str, metadata: dict[str, object], *, digest: str) -> SearchChunk:
    return SearchChunk(chunk_id, source, 0, text, metadata, 0.4, content_sha256=digest)


def _github_chunk(
    chunk_id: int = 1_000_000_050,
    path: str = _TRELLO_SYNC,
    body: str = "def sync_board() -> None:\n    return None\n",
    *,
    symbol: str = "sync_board",
    kind: str = "code",
    names: tuple[str, ...] = (),
) -> SearchChunk:
    lines = body.rstrip("\n").count("\n") + 1
    rendered = render_chunk_text(_REPO, _SHA, path, 1, lines, symbol, body)
    return _chunk(
        chunk_id,
        source_file_for(_REPO, path),
        rendered,
        {
            "source_type": "github",
            "kind": kind,
            "repo": _REPO,
            "path": path,
            "url": blob_url(_REPO, _SHA, path, 1, lines),
            "symbol": symbol,
            "header_words": header_words(path, symbol, names),
            "start_line": 1,
            "end_line": lines,
        },
        digest=hashlib.sha256(rendered.encode()).hexdigest(),
    )


def _cite(chunk: SearchChunk, quote: str, chunk_id: int | None = None) -> Citation:
    return Citation(
        source_file=chunk.source_file,
        chunk_id=chunk.id if chunk_id is None else chunk_id,
        content_sha256=chunk.content_sha256 or "0" * 64,
        quote=quote,
    )


def _answer(*citations: Citation, text: str = "Trello ingest is sync_board in trello_sync.py.") -> AnswerResponse:
    return AnswerResponse(status="answered", answer=text, confidence=0.8, citations=list(citations))


_NO_INFO = AnswerResponse(status="no_relevant_information", answer="No.", confidence=0.0, citations=[])


# --------------------------------------------------------------------------- indexing


@pytest.mark.unit
def test_index_keeps_code_docs_and_workflows_and_drops_secrets() -> None:
    assert should_index("src/app.py", 100)
    assert should_index("README.md", 100)
    assert should_index(".github/workflows/trello-ingestion.yml", 100)
    assert should_index("Dockerfile", 100)
    assert not should_index(".env", 20)
    assert not should_index("config/secrets.env", 20)
    assert not should_index("node_modules/lib/index.js", 20)
    assert not should_index("package-lock.json", 20)
    assert not should_index("assets/logo.png", 20)
    assert not should_index("big.py", 200_001)


@pytest.mark.unit
@pytest.mark.parametrize(
    "path",
    (
        "tests/unit/test_x.py",
        "projects/a/tests/unit/test_trello_board_questions.py",
        "test/helpers.py",
        "web/__tests__/a.ts",
        "projects/a/tests/fixtures/replay/paraphrases.json",
        "fixtures/board.md",
        "src/test_helpers.py",
        "src/thing_test.py",
        "web/src/a.test.tsx",
        "web/src/a.spec.ts",
        "conftest.py",
        "portal/content/a.md",
        "portal/content/playbooks/new-project-guide.md",
        "archive/old.py",
        "project_memory/notes.md",
        f"{_PKG}/ask_regress.py",
    ),
)
def test_index_drops_tests_fixtures_kb_duplicates_and_the_harness(path: str) -> None:
    assert not should_index(path, 100)


@pytest.mark.unit
def test_kind_for() -> None:
    assert kind_for(".github/workflows/trello-ingestion.yml") == "workflow"
    assert kind_for("README.md") == "doc"
    assert kind_for("pyproject.toml") == "config"
    assert kind_for(f"{_PKG}/storage/sql/001_init.sql") == "config"
    assert kind_for("Dockerfile") == "config"
    assert kind_for(f"{_PKG}/config.py") == "code"


@pytest.mark.unit
def test_python_chunks_keep_function_line_ranges_and_blob_urls() -> None:
    source = "import os\n\n\ndef sync_board() -> None:\n    value = 1\n    return value\n"
    pieces = chunk_python(source)
    symbols = {piece.symbol: piece for piece in pieces}
    assert "sync_board" in symbols
    piece = symbols["sync_board"]
    assert piece.start_line == 4
    rendered = render_chunk_text(_REPO, _SHA, "src/trello_sync.py", piece.start_line, piece.end_line, piece.symbol, piece.text)
    assert "def sync_board" in rendered
    assert blob_url(_REPO, _SHA, "src/trello_sync.py", piece.start_line, piece.end_line) in rendered
    # The repo@sha prefix is gone; the first line is the path, range, and symbol.
    assert rendered.splitlines()[0] == "src/trello_sync.py:L4-L6 sync_board"
    assert f"{_REPO}@" not in rendered


@pytest.mark.unit
def test_real_trello_sync_and_add_answer_chunk_to_those_files() -> None:
    trello = (_ROOT / _TRELLO_SYNC).read_text()
    ask = (_ROOT / f"{_PKG}/discord_ask.py").read_text()
    assert any(piece.symbol == "sync_board" for piece in chunk_python(trello))
    assert "class AddAnswerModal" in ask
    pieces = chunk_text(ask, "discord_ask.py")
    assert any("AddAnswerModal" in piece.symbol or "AddAnswerModal" in piece.text for piece in pieces)


@pytest.mark.unit
def test_oversized_function_repeats_its_signature() -> None:
    body = "\n\n".join(f"    value_{index} = {index}" for index in range(400))
    source = f"def giant() -> None:\n{body}\n"
    pieces = [piece for piece in chunk_python(source) if piece.symbol == "giant"]
    assert len(pieces) > 1
    assert all(piece.text.startswith("def giant") for piece in pieces)


@pytest.mark.unit
def test_method_slices_parse_on_their_own() -> None:
    source = "class AddAnswerModal:\n    def on_submit(self) -> None:\n        return None\n"
    method = next(piece for piece in chunk_python(source) if piece.symbol.endswith("on_submit"))
    assert method.start_line == 2
    ast.parse(method.text)


@pytest.mark.unit
def test_code_windows_do_not_repeat_lines() -> None:
    config = (_ROOT / f"{_PKG}/config.py").read_text()
    pieces = chunk_text(config, f"{_PKG}/config.py")
    seen: set[int] = set()
    for piece in pieces:
        lines = set(range(piece.start_line, piece.end_line + 1))
        assert not lines & seen
        seen |= lines
    long_yaml = "\n".join(f"key_{index}: value number {index} with padding words" for index in range(300))
    windows = chunk_text(long_yaml, "big.yml")
    assert len(windows) > 1
    assert all(left.end_line < right.start_line for left, right in zip(windows, windows[1:], strict=False))


@pytest.mark.unit
def test_module_constants_carry_their_names() -> None:
    path = f"{_PKG}/config.py"
    pieces = chunk_text((_ROOT / path).read_text(), path)
    tiers = next(piece for piece in pieces if "CHEAP_MODEL" in piece.names)
    assert {"CHEAP_MODEL", "FLAGSHIP_MODEL", "MODEL_TIERS"} <= set(tiers.names)
    assert tiers.symbol == ""
    words = header_words(path, tiers.symbol, tiers.names).split()
    assert words[:9] == ["projects", "internal", "tools", "benchmark", "cli", "tool", "src", "config", "py"]
    assert {"cheap", "model", "flagship", "tiers"} <= set(words)
    embedding = next(piece for piece in pieces if "EMBEDDING_MODEL" in piece.names)
    assert {"embedding", "model"} <= set(header_words(path, "", embedding.names).split())
    assert all(len(piece.names) <= 12 for piece in pieces)


@pytest.mark.unit
def test_header_words_split_camel_snake_and_paths_without_repeats() -> None:
    assert header_words("src/discord_ask.py", "AddAnswerModal.on_submit", ["QUESTION_MATCH_SIMILARITY"]) == (
        "src discord ask py add answer modal on submit question match similarity"
    )


@pytest.mark.unit
def test_workflow_header_and_summary_name_the_schedule_and_command() -> None:
    path = ".github/workflows/trello-ingestion.yml"
    text = (_ROOT / path).read_text()
    rows = code_rows(_REPO, _SHA, path, text)
    assert rows and all(row["kind"] == "workflow" for row in rows)
    first = rows[0]
    assert {"trello", "ingestion", "sync"} <= set(first["header_words"].split())
    assert "schedule " in first["embed_text"] and "benchmark-sync-trello" in first["embed_text"]
    assert "working-directory" not in first["embed_text"].split("Summary:")[1].splitlines()[0]


@pytest.mark.unit
def test_rows_have_two_texts_and_github_chunks_columns() -> None:
    rows = code_rows(_REPO, _SHA, "src/app.py", '"""App entry."""\n\n\ndef main() -> None:\n    """Run it."""\n    return None\n')
    assert rows
    row = next(item for item in rows if item["symbol"] == "main")
    assert set(row) == {
        "kind",
        "chunk_index",
        "symbol",
        "header_words",
        "start_line",
        "end_line",
        "chunk_text",
        "embed_text",
        "content_sha256",
        "token_count",
        "embedding",
        "metadata",
    }
    assert row["chunk_text"].startswith("src/app.py:L4-L6 main\nhttps://github.com/")
    assert row["embed_text"].startswith("File: src/app.py\nKind: source code\nSymbol: main\n")
    assert "Summary: Run it." in row["embed_text"]
    assert _SHA not in row["embed_text"] and "https://" not in row["embed_text"]
    assert row["content_sha256"] == hashlib.sha256((row["embed_text"] + row["chunk_text"]).encode()).hexdigest()
    assert row["metadata"]["url"].startswith(f"https://github.com/{_REPO}/blob/{_SHA}/src/app.py#L4-L6")


@pytest.mark.unit
def test_secrets_are_redacted_in_both_texts() -> None:
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0"
    rows = code_rows(_REPO, _SHA, "src/settings.py", f'API_KEY = "{secret}"\n')
    assert rows
    for row in rows:
        assert secret not in row["chunk_text"]
        assert secret not in row["embed_text"]
        assert "[REDACTED]" in row["embed_text"]


@pytest.mark.unit
def test_plan_reindexes_changed_files_and_deletes_missing_ones() -> None:
    plan = plan_file_changes(
        {"keep.py": "aaa", "old.py": "bbb", "edit.py": "ccc"},
        {"keep.py": "aaa", "edit.py": "ddd", "new.py": "eee"},
    )
    assert plan.changed == ("edit.py", "new.py")
    assert plan.removed == ("old.py",)


@pytest.mark.unit
def test_sync_writes_only_github_chunks_and_embeds_embed_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("def main() -> None:\n    return None\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_main() -> None:\n    assert True\n")
    calls: dict[str, list[Any]] = {"replace": [], "embedded": [], "file": []}
    monkeypatch.setattr(github_sync, "initialize_database", lambda: None)
    monkeypatch.setattr(github_sync, "github_file_digests", lambda _repo: {"src/old.py": "x", "tests/test_app.py": "y"})
    monkeypatch.setattr(github_sync, "github_indexed_paths", lambda _repo: {"src/old.py"})
    monkeypatch.setattr(github_sync, "github_existing_embeddings", lambda _hashes: {})
    monkeypatch.setattr(github_sync, "embed_texts", lambda texts: calls["embedded"].extend(texts) or [[0.0] * 3 for _ in texts])
    monkeypatch.setattr(github_sync, "replace_github_chunks", lambda repo, path, rows: calls["replace"].append((path, rows)))
    monkeypatch.setattr(github_sync, "delete_github_chunks", lambda repo, path: calls["replace"].append((path, None)))
    monkeypatch.setattr(github_sync, "upsert_github_file", lambda *args: calls["file"].append(("upsert", args[1])))
    monkeypatch.setattr(github_sync, "delete_github_file", lambda repo, path: calls["file"].append(("delete", path)))
    monkeypatch.setattr(github_sync, "set_github_commit", lambda *_args: None)
    summary = github_sync.sync_repository(tmp_path, _REPO, _SHA, pulls=None)
    assert summary == {"files_indexed": 1, "files_removed": 1, "pulls_indexed": 0, "pulls_removed": 0}
    written = dict(calls["replace"])
    assert set(written) == {"src/app.py", "src/old.py"}
    assert written["src/old.py"] is None
    assert all("embed_text" not in row for row in written["src/app.py"])
    assert calls["embedded"] and all(text.startswith("File: src/app.py") for text in calls["embedded"])
    # State rows for files with no github_chunks rows and no file are cleaned up.
    assert ("delete", "tests/test_app.py") in calls["file"]
    assert not hasattr(github_sync, "create_indexes")
    assert "document_chunks" not in inspect.getsource(github_sync).split('"""', 2)[2]


# --------------------------------------------------------------------------- storage isolation


@pytest.mark.unit
def test_migration_keeps_code_in_its_own_table_with_disjoint_ids() -> None:
    sql = (_ROOT / f"{_PKG}/storage/sql/009_github_sync.sql").read_text()
    assert "CREATE SEQUENCE IF NOT EXISTS github_chunks_id_seq START WITH 1000000000" in sql
    assert "CREATE TABLE IF NOT EXISTS github_chunks" in sql
    for statement in ("ALTER TABLE document_chunks", "INSERT INTO document_chunks", "ON document_chunks"):
        assert statement not in sql
    assert "kind IN ('code', 'config', 'workflow', 'doc', 'pull_request')" in sql


@pytest.mark.unit
def test_github_storage_functions_never_touch_document_chunks() -> None:
    from benchmark_cli.storage import postgres

    for name in (
        "github_vector_search",
        "github_fts_search",
        "github_header_search",
        "github_pull_rows",
        "github_chunk_count",
        "github_indexed_paths",
        "github_existing_embeddings",
        "replace_github_chunks",
        "delete_github_chunks",
    ):
        source = inspect.getsource(getattr(postgres, name))
        assert "github_chunks" in source
        assert "document_chunks" not in source
    assert not hasattr(postgres, "github_pull_request_chunks")


# --------------------------------------------------------------------------- baseline identity


@pytest.mark.unit
def test_retrieval_sources_are_the_baseline_six() -> None:
    assert RETRIEVAL_SOURCES == ("discord", "kb_page", "trello", "transcript", "policy_doc", "user_contributed")
    assert not hasattr(retrieval, "context_limit")


@pytest.mark.unit
def test_retrieve_evidence_issues_two_global_and_twelve_source_searches_per_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, str | None]] = []

    def rows(kind: str, limit: int, source_type: str | None) -> list[SearchChunk]:
        calls.append((kind, limit, source_type))
        base = {"vector": 0, "fts": 100}[kind] + (RETRIEVAL_SOURCES.index(source_type) + 1) * 1000 if source_type else {"vector": 0, "fts": 100}[kind]
        return [
            SearchChunk(base + index + 1, f"s/{base + index}", 0, "kiosk text", {"source_type": source_type or "trello"}, 0.5, content_sha256="a" * 64)
            for index in range(limit)
        ]

    class _Conn:
        def __enter__(self) -> _Conn:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(retrieval, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(retrieval, "connection", lambda: _Conn())
    monkeypatch.setattr(retrieval, "vector_search", lambda _vec, limit, source_type=None, **_k: rows("vector", limit, source_type))
    monkeypatch.setattr(retrieval, "fts_search", lambda _text, limit, source_type=None, **_k: rows("fts", limit, source_type))
    decision = retrieval.retrieve_evidence("kiosk", queries=["kiosk", "kiosk plan"])
    assert len(calls) == 2 * 14
    for start in (0, 14):
        per_query = calls[start : start + 14]
        assert per_query[:2] == [("vector", 40, None), ("fts", 40, None)]
        assert [item for item in per_query[2:] if item[1] != 4] == []
        assert [item[2] for item in per_query[2:]] == [source for source in RETRIEVAL_SOURCES for _ in (0, 1)]
    assert len(decision.results) <= 16


@pytest.mark.unit
def test_system_prompt_is_the_baseline_prompt() -> None:
    assert hashlib.sha256(agent.SYSTEM_PROMPT.encode()).hexdigest() == _BASELINE_SYSTEM_PROMPT_SHA256
    assert "github" not in agent.SYSTEM_PROMPT.casefold()


@pytest.mark.unit
def test_finish_does_not_append_links_to_non_code_answers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    chunk = _github_chunk()
    response = _answer(_cite(chunk, "def sync_board() -> None:"), text="Board answer.")
    finished = agent._finish(response, outcome="answered", synthesis_attempts=1)
    assert finished.answer == "Board answer."
    assert set(inspect.signature(agent._finish).parameters) == {"response", "outcome", "synthesis_attempts"}


# --------------------------------------------------------------------------- classifier

_CODE_QUESTIONS = (
    "where is the Trello ingest code",
    "how does the Ask bot decide which source to trust",
    "which file handles Add answer",
    "which model does chat use",
    "what embedding model is configured",
    "where is Discord message sync",
    "which workflow runs Trello ingestion",
    "where is saved-answer question matching",
    "where does the Ask bot store chunks",
    "what changed in the last few PRs",
    "what does the kiosk backend use for the database",
    "which SQL migration creates document_chunks",
    "what does `retrieve_evidence` return",
    "where is rrf_fuse defined",
)

_NON_CODE_QUESTIONS = (
    "Sample Person's PR tickets",
    "who is assigned the repo cleanup card",
    "What tickets are assigned to Sample Person",
    "what's on Sample's plate",
    "Which tickets are still open?",
    "What tickets are we currently working on?",
    "what's in Backlog?",
    "what's the status of the portal?",
    "How many employees does BMTech currently have?",
    "Who is on the core team?",
    "what comes after the pilot launch?",
    "what remains before onboarding a customer?",
    "which vendor should we pick for heavy analysis work?",
    "what is the trello board link",
    "link to the knowledge base",
    "what did we decide in last week's meeting?",
    "who is working on the website",
    "What is the sev 1 ticket?",
)


@pytest.mark.unit
@pytest.mark.parametrize("question", _CODE_QUESTIONS)
def test_code_questions_are_classified_as_code(question: str) -> None:
    assert is_code_question(question)


@pytest.mark.unit
@pytest.mark.parametrize("question", _NON_CODE_QUESTIONS)
def test_people_board_and_company_questions_are_not_code(question: str) -> None:
    assert not is_code_question(question)


@pytest.mark.unit
def test_every_board_question_in_the_test_fixtures_is_not_code() -> None:
    fixtures = _ROOT / "projects/internal-tools/benchmark-cli-tool/tests/fixtures/replay/paraphrases.json"
    cases = json.loads(fixtures.read_text())["cases"]
    questions = {case["question"] for case in cases} | {case["thread_question"] for case in cases if case.get("thread_question")}
    assert questions
    assert [question for question in questions if is_code_question(question)] == []


@pytest.mark.unit
def test_private_non_code_cases_are_not_code() -> None:
    """Runs only with ASK_PRIVATE_CASES pointing at the private case directory."""
    directory = os.getenv("ASK_PRIVATE_CASES", "").strip()
    if not directory:
        pytest.skip("ASK_PRIVATE_CASES is not set.")
    root = Path(directory)
    questions = {row["q"] for row in json.loads((root / "suite.json").read_text())}
    questions |= {row["starter"] for row in json.loads((root / "suite.json").read_text()) if row.get("starter")}
    questions |= {row["q"] for row in json.loads((root / "names.json").read_text())["cases"]}
    questions |= {row["q"] for row in json.loads((root / "saved.json").read_text())}
    flagged = sorted(question for question in questions if is_code_question(question))
    assert len(flagged) == 0, f"{len(flagged)} private non-code cases classified as code"


# --------------------------------------------------------------------------- lane


@pytest.mark.unit
def test_validator_drops_a_citation_outside_the_lane() -> None:
    lane = _github_chunk()
    outside = _chunk(77, "trello/b/77.md", "Kiosk MVP card", {"source_type": "trello"}, digest="b" * 64)
    response = _answer(_cite(outside, "Kiosk MVP card"))
    checked, reasons = validate_code_answer(response, (lane,))
    assert checked is None
    assert any("not a lane chunk" in reason for reason in reasons)
    mixed = _answer(_cite(outside, "Kiosk MVP card"), _cite(lane, "def sync_board() -> None:"))
    kept, _ = validate_code_answer(mixed, (lane,))
    assert kept is not None and [item.chunk_id for item in kept.citations] == [lane.id]


@pytest.mark.unit
def test_validator_keeps_an_ambiguous_quote_on_the_cited_chunk_then_falls_back() -> None:
    first = _github_chunk(1_000_000_001, f"{_PKG}/a.py", "def shared() -> None:\n    return None\n", symbol="shared")
    second = _github_chunk(1_000_000_002, f"{_PKG}/b.py", "def shared() -> None:\n    return None\n", symbol="shared")
    on_second = _answer(_cite(second, "def shared() -> None:"))
    checked, _ = validate_code_answer(on_second, (first, second))
    assert checked is not None and checked.citations[0].chunk_id == second.id
    wrong_chunk = _answer(_cite(first, "return None"), text="x")
    only_second = _github_chunk(1_000_000_003, f"{_PKG}/c.py", "VALUE = 3\n", symbol="")
    moved, _ = validate_code_answer(_answer(_cite(first, "VALUE = 3")), (first, only_second))
    assert moved is not None and moved.citations[0].chunk_id == only_second.id
    assert moved.citations[0].source_file == only_second.source_file
    assert validate_code_answer(wrong_chunk, (first,))[0] is not None


@pytest.mark.unit
def test_validator_removes_invented_urls_and_keeps_chunk_urls() -> None:
    chunk = _github_chunk()
    url = chunk.metadata["url"]
    text = f"See https://github.com/{_REPO}/blob/main/x.py#L1 and {url} for sync_board."
    checked, reasons = validate_code_answer(_answer(_cite(chunk, "def sync_board() -> None:"), text=text), (chunk,))
    assert checked is not None
    assert "blob/main/x.py" not in checked.answer
    assert url in checked.answer
    assert any("URL" in reason for reason in reasons)


@pytest.mark.unit
def test_rerank_caps_files_and_docs_and_prefers_code() -> None:
    results = []
    for index in range(4):
        results.append(HybridSearchResult(_github_chunk(1_000_000_100 + index, f"docs/d{index}.md", f"text {index}\n", symbol="", kind="doc"), 0.05))
    for index in range(3):
        results.append(HybridSearchResult(_github_chunk(1_000_000_200 + index, f"{_PKG}/same.py", f"x = {index}\n", symbol=""), 0.03))
    chosen = [result for result, _ in rerank(results)]
    paths = [chunk.chunk.metadata["path"] for chunk in chosen]
    assert sum(path.endswith(".md") for path in paths) == 2
    assert paths.count(f"{_PKG}/same.py") == 2
    assert chosen[0].chunk.metadata["kind"] == "code"
    assert len(rerank(results * 3)) <= 8


def _identity_rows(chunks: list[SearchChunk]) -> list[tuple[str, str, str]]:
    return [(str(c.metadata.get("path") or c.source_file), str(c.metadata.get("symbol") or ""), str(c.metadata.get("header_words") or "")) for c in chunks]


def _lane_storage(monkeypatch: pytest.MonkeyPatch, chunks: list[SearchChunk], *, count: int = 10) -> dict[str, Any]:
    seen: dict[str, Any] = {}

    def search(name: str):
        def run(_query: Any, limit: int, kinds: tuple[str, ...]) -> list[SearchChunk]:
            seen[name] = kinds
            return chunks[:limit]

        return run

    monkeypatch.setattr(github_answers, "github_chunk_count", lambda: count)
    monkeypatch.setattr(github_answers, "embed_texts", lambda texts: [[0.1] for _ in texts])
    monkeypatch.setattr(github_answers, "github_vector_search", search("vector"))
    monkeypatch.setattr(github_answers, "github_fts_search", search("fts"))
    monkeypatch.setattr(github_answers, "github_header_search", search("header"))
    monkeypatch.setattr(github_answers, "github_identity_rows", lambda _kinds: _identity_rows(chunks))
    return seen


@pytest.mark.unit
def test_retrieve_code_excludes_pull_requests_and_needs_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _github_chunk()
    seen = _lane_storage(monkeypatch, [chunk])
    found = github_answers.retrieve_code("where is the Trello ingest code")
    assert set(seen) == {"vector", "fts", "header"}
    assert all("pull_request" not in kinds for kinds in seen.values())
    assert found.status == "accepted" and found.chunks == (chunk,)
    assert found.header_ids == (chunk.id,)


@pytest.mark.unit
def test_recent_pull_question_uses_the_pull_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    pull = _chunk(
        1_000_000_900,
        source_file_for(_REPO, "pulls/11"),
        "Pull request #11: Newer work\nState: merged\nMerged: 2026-11-12T00:00:00Z\nhttps://github.com/benchmarktechllc/bmtechllc/pull/11\n",
        {"source_type": "github", "kind": "pull_request", "merged_at": "2026-11-12T00:00:00Z", "url": "https://github.com/benchmarktechllc/bmtechllc/pull/11"},
        digest="ef" * 32,
    )
    monkeypatch.setattr(github_answers, "github_pull_rows", lambda: [pull])
    monkeypatch.setattr(github_answers, "embed_texts", lambda _t: pytest.fail("pull answers need no embedding"))
    found = github_answers.retrieve_code("what changed in the last few PRs")
    result = answer_from_retrieval("what changed in the last few PRs", found, lambda _m: pytest.fail("no model call"))
    assert result.outcome == "github_pulls"
    assert all("/pulls/" in item.source_file for item in result.response.citations)


@pytest.mark.unit
def test_recent_pull_questions_use_stored_prs_in_order() -> None:
    assert asks_for_recent_pulls("what changed in the last few PRs")
    assert not asks_for_recent_pulls("where is the Trello ingest code")
    older = _chunk(
        7,
        source_file_for(_REPO, "pulls/10"),
        "Pull request #10: Older work\nState: merged\nMerged: 2026-11-01T00:00:00Z\n"
        "https://github.com/benchmarktechllc/bmtechllc/pull/10\n",
        {"source_type": "github", "kind": "pull_request", "merged_at": "2026-11-01T00:00:00Z", "url": "https://github.com/benchmarktechllc/bmtechllc/pull/10"},
        digest="cd" * 32,
    )
    newer = _chunk(
        8,
        source_file_for(_REPO, "pulls/11"),
        "Pull request #11: Newer work\nState: merged\nMerged: 2026-11-12T00:00:00Z\n"
        "https://github.com/benchmarktechllc/bmtechllc/pull/11\n",
        {"source_type": "github", "kind": "pull_request", "merged_at": "2026-11-12T00:00:00Z", "url": "https://github.com/benchmarktechllc/bmtechllc/pull/11"},
        digest="ef" * 32,
    )
    answer = recent_pulls_answer((older, newer))
    assert answer is not None
    assert answer.answer.index("#11") < answer.answer.index("#10")
    assert "https://github.com/benchmarktechllc/bmtechllc/pull/11" in answer.answer
    assert answer.citations[0].source_file.endswith("pulls/11")


@pytest.mark.unit
def test_code_answer_cites_the_blob_url_from_the_chunk() -> None:
    chunk = _github_chunk()
    linked = append_github_links(_answer(_cite(chunk, "def sync_board() -> None:")), {chunk.id: chunk})
    assert chunk.metadata["url"] in linked.answer
    rendered = format_discord_answer(linked)
    assert "trello_sync.py" in rendered
    assert chunk.metadata["url"] in rendered


@pytest.mark.unit
def test_lane_retries_once_then_uses_the_pointer_for_a_where_question() -> None:
    chunk = _github_chunk(names=())
    found = CodeRetrieval("accepted", (chunk,), (chunk.id,), 0.6, 2)
    drafts = [_answer(_cite(chunk, "invented quote")), _answer(_cite(chunk, "another invented quote"))]
    prompts: list[list[dict[str, Any]]] = []

    def draft(messages: list[dict[str, Any]]) -> AnswerResponse:
        prompts.append(messages)
        return drafts.pop(0)

    result = answer_from_retrieval("where is the trello sync code", found, draft)
    assert len(prompts) == 2
    assert result.outcome == "code_pointer"
    assert result.response.answer.startswith(f"Closest match: `{_TRELLO_SYNC}` lines 1–2 (sync_board)")
    assert result.response.citations[0].quote == chunk.chunk_text.splitlines()[0]
    assert chunk.metadata["url"] in result.response.answer


@pytest.mark.unit
def test_pointer_needs_a_where_question_and_a_header_match() -> None:
    chunk = _github_chunk()
    strong = CodeRetrieval("accepted", (chunk,), (chunk.id,), 0.6, 2)
    assert pointer_answer("where is the trello sync code", strong) is not None
    assert pointer_answer("what does trello sync do", strong) is None
    assert pointer_answer("where is the kiosk code", strong) is None
    other_top = CodeRetrieval("accepted", (chunk,), (999,), 0.6, 2)
    assert pointer_answer("where is the trello sync code", other_top) is None


@pytest.mark.unit
def test_lane_abstains_without_falling_through_and_sends_only_github_chunks() -> None:
    chunk = _github_chunk()
    found = CodeRetrieval("accepted", (chunk,), (chunk.id,), 0.6, 2)
    sent: list[str] = []

    def draft(messages: list[dict[str, Any]]) -> AnswerResponse:
        sent.append(json.dumps(messages))
        return _NO_INFO

    result = answer_from_retrieval("what does the kiosk backend use for the database", found, draft)
    assert result.outcome == "code_abstained"
    assert result.response.status == "no_relevant_information"
    assert len(sent) == 1
    assert "trello_board" not in sent[0] and "person_cards" not in sent[0]
    assert github_answers.CODE_SYSTEM_PROMPT in sent[0] or "code repository" in sent[0]
    rejected = answer_from_retrieval("q", CodeRetrieval("no_relevant_information"), lambda _m: pytest.fail("no call"))
    assert rejected.outcome == "code_abstained"


# --------------------------------------------------------------------------- the gate in run_agent


def _run(
    monkeypatch: pytest.MonkeyPatch,
    question: str,
    hits: tuple[SearchChunk, ...] = (),
    *,
    person: str | None = None,
    board: bool = False,
    model: AnswerResponse = _NO_INFO,
) -> tuple[AnswerResponse, dict[str, Any]]:
    from benchmark_cli.query_understanding import QueryPlan
    from benchmark_cli.replay import sample_board

    seen: dict[str, Any] = {}
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **kwargs: seen.update(kwargs))
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()) if board else [])
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(tool_calls=None, content=""))]),
    )
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: model)
    monkeypatch.setattr(agent, "understand_query", lambda *_args, **_kwargs: QueryPlan(searches=[question], person=person))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted" if hits else "no_relevant_information",
            tuple(HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0) for chunk in hits),
            0.8 if hits else None,
            2 if hits else 0,
        ),
    )
    monkeypatch.setattr(agent, "fts_search", lambda *_args, **_kwargs: [])
    return agent.run_agent(question), seen


def _kb_chunk() -> SearchChunk:
    return _chunk(5, "portal/home.md", "The knowledge base lives at the portal.", {"source_type": "kb_page"}, digest="c" * 64)


@pytest.mark.unit
def test_code_question_is_answered_by_the_lane_with_a_blob_link(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _github_chunk()
    _lane_storage(monkeypatch, [chunk])
    monkeypatch.setattr(
        github_answers,
        "call_llm_structured",
        lambda **kwargs: _answer(_cite(chunk, "def sync_board() -> None:")) if kwargs["component"] == "code_answer" else pytest.fail("one lane call"),
    )
    response, seen = _run(monkeypatch, "where is the Trello ingest code", (_kb_chunk(),), board=True)
    assert seen["outcome"] == "code_answered"
    assert response.status == "answered"
    assert "trello_sync.py" in response.answer
    assert chunk.metadata["url"] in response.answer
    assert [item.source_file for item in response.citations] == [chunk.source_file]


@pytest.mark.unit
def test_non_code_question_never_enters_the_lane(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "answer_code_question", lambda _q: pytest.fail("non-code question entered the lane"))
    response, seen = _run(monkeypatch, "What tickets are we currently working on?", (_kb_chunk(),), board=True)
    assert response.status == "answered"
    assert not str(seen["outcome"]).startswith("code_")


@pytest.mark.unit
@pytest.mark.parametrize("count", (0, None))
def test_empty_or_missing_index_keeps_the_baseline_path(monkeypatch: pytest.MonkeyPatch, count: int | None) -> None:
    def chunk_count() -> int:
        if count is None:
            raise RuntimeError('relation "github_chunks" does not exist')
        return count

    monkeypatch.setattr(github_answers, "github_chunk_count", chunk_count)
    monkeypatch.setattr(github_answers, "retrieve_code", lambda _q: pytest.fail("no lane retrieval on an empty index"))
    model = _answer(_cite(_kb_chunk(), "The knowledge base lives at the portal."), text="It is in the portal.")
    response, seen = _run(monkeypatch, "where is the Trello ingest code", (_kb_chunk(),), model=model)
    assert seen["outcome"] == "answered"
    assert response.citations[0].source_file == "portal/home.md"


@pytest.mark.unit
def test_person_questions_skip_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "answer_code_question", lambda _q: pytest.fail("person question entered the lane"))
    question = "github commits by Dante"
    assert is_code_question(question)
    _response, seen = _run(monkeypatch, question, (_kb_chunk(),), person="Dante Paniccia", board=True)
    assert not str(seen["outcome"]).startswith("code_")


@pytest.mark.unit
def test_saved_answer_beats_the_code_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    question = "Where is the deploy script?"
    assert is_code_question(question)
    saved = _chunk(
        90,
        "user_contributed/7/9.md",
        f"Question: {question}\nAccording to Ada Example on 2026-11-12:\nSample answer.",
        {"source_type": "user_contributed", "question": question, "author_name": "Ada Example", "contributed_at": "2026-11-12", "date": "2026-11-12"},
        digest=hashlib.sha256(b"saved-deploy").hexdigest(),
    )
    monkeypatch.setattr(agent, "answer_code_question", lambda _q: pytest.fail("saved answer must win"))
    response, seen = _run(monkeypatch, question, (saved,))
    assert seen["outcome"] == "user_contributed"
    assert response.citations[0].source_file.startswith("user_contributed/")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "outcome_prefix"),
    (
        ("How many employees does BMTech have according to the repo?", "employee_count"),
        ("Who is on the core team in the repo?", "core_team"),
        ("which discord channels are synced by the sync script", "discord_channels"),
    ),
)
def test_shortcuts_keep_priority_over_the_gate(monkeypatch: pytest.MonkeyPatch, question: str, outcome_prefix: str) -> None:
    from benchmark_cli.grounding import SyncedDiscordChannel

    assert is_code_question(question)
    monkeypatch.setattr(agent, "answer_code_question", lambda _q: pytest.fail("shortcut must win"))
    channel = SyncedDiscordChannel("general", 5, "discord/general.md", "c" * 64, "general")
    monkeypatch.setattr(agent, "_synced_channels", lambda: (channel,))
    _response, seen = _run(monkeypatch, question, (_kb_chunk(),))
    assert str(seen["outcome"]).startswith(outcome_prefix)


@pytest.mark.unit
def test_portal_link_and_board_forms_skip_the_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "answer_code_question", lambda _q: pytest.fail("link and list forms must skip the gate"))
    monkeypatch.setattr(agent, "asks_for_portal_link", lambda _q: True)
    _response, seen = _run(monkeypatch, "portal link from the config", (_kb_chunk(),))
    assert seen["outcome"] == "portal_link"


@pytest.mark.unit
def test_lane_quota_error_does_not_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli.llm import ModelQuotaError

    chunk = _github_chunk()
    _lane_storage(monkeypatch, [chunk])

    def quota(**_kwargs: Any) -> AnswerResponse:
        raise ModelQuotaError("quota")

    monkeypatch.setattr(github_answers, "call_llm_structured", quota)
    result = github_answers.answer_code_question("where is the Trello ingest code")
    assert result is not None and result.outcome == "code_model_quota"


# --------------------------------------------------------------------------- board and saved answers stay put


def _board_run(monkeypatch: pytest.MonkeyPatch, question: str, hits: tuple[SearchChunk, ...], *, person: str | None) -> AnswerResponse:
    response, _seen = _run(monkeypatch, question, hits, person=person, board=True)
    return response


@pytest.mark.unit
@pytest.mark.parametrize(
    ("question", "person", "expected"),
    (
        ("Tickets assigned to Fanual", "Fanuel Mengistu", "Kiosk MVP"),
        ("Which tickets are still open?", None, "Ship the portal"),
        ("What tickets are we currently working on?", None, "Ship the portal"),
    ),
)
def test_github_chunks_do_not_override_board_answers(
    monkeypatch: pytest.MonkeyPatch, question: str, person: str | None, expected: str
) -> None:
    response = _board_run(monkeypatch, question, (_github_chunk(),), person=person)
    assert response.status == "answered"
    assert expected in response.answer
    assert all(not citation.source_file.startswith("github/") for citation in response.citations)


@pytest.mark.unit
def test_saved_headcount_wins_over_a_github_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    saved_text = (
        "Question: How many employees does BMTech currently have?\n"
        "According to Ada Example on 2026-11-12:\n"
        "Sample count is 3."
    )
    saved = _chunk(
        90,
        "user_contributed/7/9.md",
        saved_text,
        {
            "source_type": "user_contributed",
            "question": "How many employees does BMTech currently have?",
            "author_name": "Ada Example",
            "contributed_at": "2026-11-12",
            "date": "2026-11-12",
        },
        digest=hashlib.sha256(b"saved-count").hexdigest(),
    )
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("saved answer must not call the model"))
    monkeypatch.setattr(agent, "call_llm_structured", lambda **_kwargs: pytest.fail("saved answer must not call the model"))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (
                HybridSearchResult(_github_chunk(), 0.4, vector_score=0.95, fts_score=1.0),
                HybridSearchResult(saved, 0.3, vector_score=0.2, fts_score=1.0),
            ),
            0.95,
            2,
        ),
    )
    response = agent.run_agent("How many employees does BMTech currently have ?")
    assert "Sample count is 3." in response.answer
    assert response.citations[0].source_file.startswith("user_contributed/")


@pytest.mark.unit
def test_kiosk_database_question_does_not_answer_from_the_trello_card(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _github_chunk()
    _lane_storage(monkeypatch, [chunk])
    monkeypatch.setattr(github_answers, "call_llm_structured", lambda **_kwargs: _NO_INFO)
    response, seen = _run(monkeypatch, "what does the kiosk backend use for the database", (chunk,), board=True)
    assert seen["outcome"] == "code_abstained"
    assert response.status == "no_relevant_information"
    assert "Kiosk MVP" not in response.answer


# --------------------------------------------------------------------------- lane ranking rules
# Synthetic paths and words only ("zebra", "gadget"); no gate question or expected path appears here.

_HOME = github_answers.HOME_PROJECT
_OTHER = "projects/labs/gadget-helper"


def _vocab(*chunks: SearchChunk, filler: int = 40) -> github_answers.LaneVocabulary:
    """The chunks plus unrelated files, so document frequency behaves like a real index."""
    rows = [(c.metadata["path"], c.metadata["symbol"], c.metadata["header_words"]) for c in chunks]
    rows += [(f"{_HOME}/src/pkg/unrelated_{index}.py", f"helper_{index}", f"unrelated {index} helper") for index in range(filler)]
    return github_answers.LaneVocabulary.from_rows(rows)


def _ranked_paths(question: str, fused: list[HybridSearchResult], vocab: github_answers.LaneVocabulary) -> list[str]:
    query = github_answers.LaneQuery.build(question, vocab)
    return [result.chunk.metadata["path"] for result, _ in rerank(fused, query)]


@pytest.mark.unit
def test_stems_and_prefixes_join_word_forms() -> None:
    stem = github_answers._stem
    assert stem("ingestion") == stem("ingest") and stem("storage") == stem("store")
    assert stem("matching") == stem("matches") and stem("chunks") == stem("chunking")
    assert github_answers.terms_match(stem("configured"), stem("config"))
    assert not github_answers.terms_match("run", "runner")  # short stems must match exactly
    assert github_answers._stems("AddAnswerView.add_answer PRs") == ["add", "answ", "view", "add", "answ", "prs"]


@pytest.mark.unit
def test_rare_subject_term_outranks_a_common_one_with_better_fts() -> None:
    """The file named for the rare word wins over files that share only the common word,
    even when those files were higher in the FTS list."""
    subject = _github_chunk(1_000_001_001, f"{_HOME}/src/pkg/zebra_sync.py", "x = 1\n", symbol="")
    commons = [_github_chunk(1_000_001_010 + i, f"{_HOME}/src/pkg/loading/loader_{i}.py", "y = 1\n", symbol="load_rows") for i in range(6)]
    vocab = _vocab(subject, *commons)
    fused = [HybridSearchResult(chunk, 1.0 / (61 + rank)) for rank, chunk in enumerate(commons)]
    fused.append(HybridSearchResult(subject, 1.0 / 70))
    paths = _ranked_paths("where is the zebra loading code", fused, vocab)
    assert paths[0] == subject.metadata["path"]


@pytest.mark.unit
def test_generic_code_words_earn_no_identity_credit() -> None:
    generic = _github_chunk(1_000_001_101, ".github/workflows/code-files.yml", "on: push\n", symbol="", kind="workflow")
    vocab = _vocab(generic)
    query = github_answers.LaneQuery.build("which file has the code for the zebra", vocab)
    assert set(query.terms) == {"zebra"}
    assert query.lexical(generic) == 0.0


@pytest.mark.unit
def test_path_and_symbol_count_fully_and_defined_names_count_half() -> None:
    by_path = _github_chunk(1_000_001_201, f"{_HOME}/src/pkg/zebra.py", "x = 1\n", symbol="")
    by_symbol = _github_chunk(1_000_001_202, f"{_HOME}/src/pkg/other.py", "x = 1\n", symbol="ZebraClient")
    by_name = _github_chunk(1_000_001_203, f"{_HOME}/src/pkg/settings.py", "ZEBRA_KEY = 1\n", symbol="", names=("ZEBRA_KEY",))
    query = github_answers.LaneQuery.build("where is zebra", _vocab(by_path, by_symbol, by_name))
    assert query.lexical(by_path) == query.lexical(by_symbol) == 1.0
    assert query.lexical(by_name) == pytest.approx(github_answers.NAME_MATCH_WEIGHT)


@pytest.mark.unit
def test_header_list_adds_candidates_but_no_rrf_vote() -> None:
    both = _github_chunk(1_000_001_301, f"{_HOME}/src/pkg/a.py", "a = 1\n", symbol="")
    header_only = _github_chunk(1_000_001_302, f"{_HOME}/src/pkg/b.py", "b = 1\n", symbol="")
    fused = github_answers.candidates([], [both], [both, header_only])
    scores = {result.chunk.id: result.rrf_score for result in fused}
    assert scores == {both.id: pytest.approx(1.0 / 61), header_only.id: 0.0}
    assert next(r for r in fused if r.chunk.id == both.id).fts_score == both.score


@pytest.mark.unit
def test_home_project_is_the_default_scope_and_other_projects_are_down_weighted() -> None:
    home = _github_chunk(1_000_001_401, f"{_HOME}/src/pkg/gizmo.py", "x = 1\n", symbol="")
    other = _github_chunk(1_000_001_402, f"{_OTHER}/src/gizmo.py", "x = 1\n", symbol="")
    workflow = _github_chunk(1_000_001_403, ".github/workflows/gizmo.yml", "on: push\n", symbol="", kind="workflow")
    top_level = _github_chunk(1_000_001_404, "portal/src/gizmo.ts", "x\n", symbol="")
    vocab = _vocab(home, other, workflow, top_level)
    query = github_answers.LaneQuery.build("where is the gizmo", vocab)
    assert query.scope_weight(home) == query.scope_weight(workflow) == 1.0
    assert query.scope_weight(other) == query.scope_weight(top_level) == github_answers.OFF_SCOPE_WEIGHT
    # Other project first in the vector list, home second: home still wins.
    fused = [HybridSearchResult(other, 1.0 / 61), HybridSearchResult(home, 1.0 / 62)]
    assert _ranked_paths("where is the gizmo", fused, vocab)[0] == home.metadata["path"]


@pytest.mark.unit
def test_naming_the_bot_scopes_home_and_naming_a_project_scopes_that_project() -> None:
    home = _github_chunk(1_000_001_501, f"{_HOME}/src/pkg/gizmo.py", "x = 1\n", symbol="")
    other = _github_chunk(1_000_001_502, f"{_OTHER}/src/gizmo.py", "x = 1\n", symbol="")
    vocab = _vocab(home, other)
    for question in ("how does the Ask bot store gizmos", "where does the bot keep gizmos", "how does Ask pick a gizmo"):
        query = github_answers.LaneQuery.build(question, vocab)
        assert _HOME in query.scope and _OTHER not in query.scope, question
        assert not {"ask", "bot"} & set(query.terms), question  # scope words are not subject terms
    query = github_answers.LaneQuery.build("where does the gadget helper store gizmos", vocab)
    assert _OTHER in query.scope and _HOME not in query.scope and "" in query.scope
    assert query.scope_weight(other) == 1.0 and query.scope_weight(home) == github_answers.OFF_SCOPE_WEIGHT
    assert "gadget" not in query.terms and "gizmo" in query.terms
    # "ask" as an ordinary verb does not name the bot.
    assert github_answers.LaneQuery.build("where do we ask for gizmos", vocab).scope == frozenset({_HOME, ""})


@pytest.mark.unit
def test_project_of_maps_paths_to_projects() -> None:
    assert github_answers.project_of(f"{_HOME}/src/x.py") == _HOME
    assert github_answers.project_of(".github/workflows/x.yml") == ""
    assert github_answers.project_of("README.md") == ""
    assert github_answers.project_of("portal/src/x.ts") == "portal"


@pytest.mark.unit
def test_retrieve_code_reports_one_score_scale_in_rank_order(monkeypatch: pytest.MonkeyPatch) -> None:
    subject = _github_chunk(1_000_001_601, f"{_HOME}/src/pkg/zebra_sync.py", "x = 1\n", symbol="")
    common = _github_chunk(1_000_001_602, ".github/workflows/zebra-free-loading.yml", "on: push\n", symbol="", kind="workflow")
    _lane_storage(monkeypatch, [common, subject])
    found = github_answers.retrieve_code("where is the zebra loading code")
    assert len(found.ranking) == len(found.chunks)
    scores = [parts["score"] for parts in found.ranking]
    assert scores == sorted(scores, reverse=True)
    for parts in found.ranking:
        expected = parts["kind_weight"] * parts["scope"] * (parts["rrf"] + github_answers.LEXICAL_WEIGHT * parts["lexical"])
        assert parts["score"] == pytest.approx(expected)
