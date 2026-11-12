"""Code questions cite GitHub files and do not override Trello or saved answers."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from benchmark_cli import agent
from benchmark_cli.discord_ask import format_discord_answer
from benchmark_cli.github_answers import (
    append_github_links,
    asks_for_recent_pulls,
    recent_pulls_answer,
)
from benchmark_cli.github_files import (
    blob_url,
    chunk_python,
    chunk_text,
    render_piece,
    should_index,
    source_file_for,
)
from benchmark_cli.github_sync import plan_file_changes
from benchmark_cli.models import AnswerResponse, ChunkMetadata, Citation
from benchmark_cli.retrieval import (
    RETRIEVAL_SOURCES,
    HybridSearchResult,
    RetrievalDecision,
    context_limit,
)
from benchmark_cli.storage.postgres import SearchChunk

_REPO = "benchmarktechllc/bmtechllc"
_SHA = "abc123def456abc123def456abc123def456abcd"
_ROOT = Path(__file__).resolve().parents[5]


def _chunk(chunk_id: int, source: str, text: str, metadata: dict[str, object], *, digest: str) -> SearchChunk:
    return SearchChunk(chunk_id, source, 0, text, metadata, 0.4, content_sha256=digest)


def _github_chunk() -> SearchChunk:
    piece_text = "def sync_board() -> None:\n    return None\n"
    rendered = render_piece(
        _REPO,
        _SHA,
        "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/trello_sync.py",
        start_line=1,
        end_line=2,
        symbol="sync_board",
        body=piece_text,
    )
    url = blob_url(
        _REPO,
        _SHA,
        "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/trello_sync.py",
        1,
        2,
    )
    digest = hashlib.sha256(rendered.encode()).hexdigest()
    return _chunk(
        50,
        source_file_for(_REPO, "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/trello_sync.py"),
        rendered,
        {
            "source_type": "github",
            "kind": "code",
            "repo": _REPO,
            "path": "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/trello_sync.py",
            "commit_sha": _SHA,
            "url": url,
            "symbol": "sync_board",
        },
        digest=digest,
    )


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
def test_python_chunks_keep_function_line_ranges_and_blob_urls() -> None:
    source = "import os\n\n\ndef sync_board() -> None:\n    value = 1\n    return value\n"
    pieces = chunk_python(source)
    symbols = {piece.symbol: piece for piece in pieces}
    assert "sync_board" in symbols
    piece = symbols["sync_board"]
    assert piece.start_line == 4
    rendered = render_piece(_REPO, _SHA, "src/trello_sync.py", piece.start_line, piece.end_line, piece.symbol, piece.text)
    assert "def sync_board" in rendered
    assert blob_url(_REPO, _SHA, "src/trello_sync.py", piece.start_line, piece.end_line) in rendered
    assert rendered.splitlines()[0].startswith(f"{_REPO}@{_SHA} ")


@pytest.mark.unit
def test_real_trello_sync_and_add_answer_chunk_to_those_files() -> None:
    trello = (_ROOT / "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/trello_sync.py").read_text()
    ask = (_ROOT / "projects/internal-tools/benchmark-cli-tool/src/benchmark_cli/discord_ask.py").read_text()
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
def test_plan_reindexes_changed_files_and_deletes_missing_ones() -> None:
    plan = plan_file_changes(
        {"keep.py": "aaa", "old.py": "bbb", "edit.py": "ccc"},
        {"keep.py": "aaa", "edit.py": "ddd", "new.py": "eee"},
    )
    assert plan.changed == ("edit.py", "new.py")
    assert plan.removed == ("old.py",)


@pytest.mark.unit
def test_github_metadata_round_trip_uses_existing_chunk_columns() -> None:
    metadata = ChunkMetadata(
        source_type="github",
        date="2026-11-12",
        ingested_at="2026-11-12T00:00:00+00:00",
        repo=_REPO,
        path="src/app.py",
        commit_sha=_SHA,
        url=blob_url(_REPO, _SHA, "src/app.py", 1, 4),
        start_line=1,
        end_line=4,
        symbol="main",
        kind="code",
        section="src/app.py",
        title="main",
    )
    stored = metadata.as_metadata()
    row = {
        "source_file": source_file_for(_REPO, "src/app.py"),
        "chunk_index": 0,
        "chunk_text": "def main() -> None:\n    return None\n",
        "content_sha256": "ab" * 32,
        "token_count": 8,
        "embedding": [0.0],
        "metadata": stored,
    }
    assert set(row) == {
        "source_file",
        "chunk_index",
        "chunk_text",
        "content_sha256",
        "token_count",
        "embedding",
        "metadata",
    }
    assert stored["source_type"] == "github"
    assert stored["url"].startswith("https://github.com/benchmarktechllc/bmtechllc/blob/")


@pytest.mark.unit
def test_retrieval_gives_github_its_own_arm_without_shrinking_others() -> None:
    assert "github" in RETRIEVAL_SOURCES
    assert context_limit(5) == 4 * len(RETRIEVAL_SOURCES)
    assert context_limit(5) >= 28


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
    quote = "def sync_board() -> None:"
    response = AnswerResponse(
        status="answered",
        answer="Trello ingest is sync_board in trello_sync.py.",
        confidence=0.8,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=chunk.content_sha256 or "",
                quote=quote,
            )
        ],
    )
    linked = append_github_links(response, {chunk.id: chunk})
    assert chunk.metadata["url"] in linked.answer
    rendered = format_discord_answer(linked)
    assert "trello_sync.py" in rendered
    assert chunk.metadata["url"] in rendered


def _board_run(monkeypatch: pytest.MonkeyPatch, question: str, hits: tuple[SearchChunk, ...], *, person: str | None) -> AnswerResponse:
    from benchmark_cli.query_understanding import QueryPlan
    from benchmark_cli.replay import sample_board

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: list(sample_board()))
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[__import__("types").SimpleNamespace(message=__import__("types").SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="no_relevant_information",
            answer="No relevant information was found.",
            confidence=0.0,
            citations=[],
        ),
    )
    monkeypatch.setattr(agent, "understand_query", lambda *_args, **_kwargs: QueryPlan(searches=[question], person=person))
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            tuple(HybridSearchResult(chunk, 0.2, vector_score=chunk.score, fts_score=1.0) for chunk in hits),
            0.9,
            2,
        ),
    )
    return agent.run_agent(question)


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
    response = _board_run(
        monkeypatch,
        "what does the kiosk backend use for the database",
        (_github_chunk(),),
        person=None,
    )
    assert response.status == "no_relevant_information"
    assert "Kiosk MVP" not in response.answer


@pytest.mark.unit
def test_run_agent_appends_the_blob_link(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli.query_understanding import QueryPlan

    chunk = _github_chunk()
    quote = "def sync_board() -> None:"
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "trello_card_rows", lambda: [])
    monkeypatch.setattr(
        agent,
        "call_llm",
        lambda **_kwargs: __import__("types").SimpleNamespace(
            choices=[__import__("types").SimpleNamespace(message=__import__("types").SimpleNamespace(tool_calls=None, content=""))]
        ),
    )
    monkeypatch.setattr(
        agent,
        "call_llm_structured",
        lambda **_kwargs: AnswerResponse(
            status="answered",
            answer="Trello ingest is sync_board in trello_sync.py.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=chunk.content_sha256 or "",
                    quote=quote,
                )
            ],
        ),
    )
    monkeypatch.setattr(
        agent,
        "understand_query",
        lambda *_args, **_kwargs: QueryPlan(searches=["trello ingest"], person=None),
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision(
            "accepted",
            (HybridSearchResult(chunk, 0.4, vector_score=0.8, fts_score=1.0),),
            0.8,
            2,
        ),
    )
    response = agent.run_agent("where is the Trello ingest code")
    assert response.status == "answered"
    assert "trello_sync.py" in response.answer
    assert chunk.metadata["url"] in response.answer
    assert response.citations[0].source_file.startswith("github/")
    source = "class AddAnswerModal:\n    def on_submit(self) -> None:\n        return None\n"
    pieces = chunk_python(source)
    method = next(piece for piece in pieces if piece.symbol.endswith("on_submit"))
    assert method.start_line == 2
    ast.parse(method.text)
