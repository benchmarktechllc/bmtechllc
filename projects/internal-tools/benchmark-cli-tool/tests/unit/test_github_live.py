"""Live GitHub PR/issue status in the answer context. Synthetic data; no network."""

from __future__ import annotations

import urllib.error
from typing import Any

import pytest
from test_context_first import _draft, _install, _kb, _pack

from benchmark_cli import agent, github_answers, github_live
from benchmark_cli.github_answers import CodeRetrieval, answer_from_retrieval
from benchmark_cli.models import AnswerResponse, Citation

REPO = "example/repo"


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    github_live._cache.clear()


def _pull(**over: Any) -> dict[str, Any]:
    base = {
        "title": "Add widget",
        "state": "open",
        "merged": False,
        "merged_at": None,
        "user": {"login": "alex"},
        "created_at": "2026-11-01T00:00:00Z",
        "updated_at": "2026-11-02T00:00:00Z",
        "draft": False,
        "head": {"sha": "abc"},
        "html_url": f"https://github.com/{REPO}/pull/68",
        "requested_reviewers": [],
    }
    base.update(over)
    return base


def _fake_api(routes: dict[str, Any], calls: list[str] | None = None):
    def get(path: str) -> Any:
        if calls is not None:
            calls.append(path)
        for prefix, value in routes.items():
            if path.startswith(prefix):
                if isinstance(value, Exception):
                    raise value
                return value
        raise urllib.error.HTTPError(path, 404, "nf", None, None)  # type: ignore[arg-type]

    return get


def _approved_routes() -> dict[str, Any]:
    return {
        f"/repos/{REPO}/pulls/68/reviews": [
            {"user": {"login": "sam"}, "state": "COMMENTED", "submitted_at": "2026-11-02T01:00:00Z"},
            {"user": {"login": "sam"}, "state": "APPROVED", "submitted_at": "2026-11-02T02:00:00Z"},
        ],
        f"/repos/{REPO}/pulls/68": _pull(),
        f"/repos/{REPO}/commits/abc/check-runs": {
            "check_runs": [
                {"name": "lint", "conclusion": "success"},
                {"name": "tests", "conclusion": "failure"},
            ]
        },
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Has PR 68 been approved yet?", [68]),
        ("what about PR #68", [68]),
        ("pr68 status", [68]),
        ("pull request 68 and issue 12", [68, 12]),
        ("see #68 and #68", [68]),
        ("prs 1 2", [1]),
        ("look at <#1234567890123> and <@123456>", []),
        ("#123456 is too long", []),
        ("what is the BMTech portal?", []),
        ("PR 1, PR 2, PR 3, PR 4", [1, 2, 3]),
    ],
)
def test_referenced_numbers(text: str, expected: list[int]) -> None:
    assert github_live.referenced_numbers(text) == expected


def test_approved_pull_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes()))
    [chunk] = github_live.live_status_chunks([68], REPO)
    assert chunk.id == github_live.LIVE_ID_BASE + 68 and chunk.content_sha256
    assert chunk.metadata["url"] == f"https://github.com/{REPO}/pull/68"
    text = chunk.chunk_text
    assert text.startswith("Pull request #68: Add widget\n")
    assert text.rstrip("\n").splitlines()[-1].startswith("Live GitHub status fetched 20")
    assert "Author: alex" in text and "State: open" in text
    assert "sam: APPROVED at 2026-11-02T02:00:00Z" in text and "COMMENTED" not in text
    assert "Checks: 1 failure, 1 success" in text and "Failing checks: tests" in text


def test_no_reviews_and_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        f"/repos/{REPO}/pulls/68/reviews": [],
        f"/repos/{REPO}/pulls/68": _pull(
            state="closed", merged=True, merged_at="2026-11-03T00:00:00Z", merged_by={"login": "sam"}
        ),
        f"/repos/{REPO}/commits/abc/check-runs": {"check_runs": []},
    }
    monkeypatch.setattr(github_live, "_get", _fake_api(routes))
    [chunk] = github_live.live_status_chunks([68], REPO)
    assert "Reviews: none" in chunk.chunk_text
    assert "State: merged (merged at 2026-11-03T00:00:00Z by sam)" in chunk.chunk_text
    assert "Checks: none" in chunk.chunk_text


def test_issue_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    routes = {
        f"/repos/{REPO}/issues/12": {
            "title": "Bug",
            "state": "closed",
            "closed_at": "2026-11-05T00:00:00Z",
            "user": {"login": "alex"},
            "labels": [{"name": "bug"}],
            "assignees": [{"login": "sam"}],
            "html_url": f"https://github.com/{REPO}/issues/12",
        }
    }
    monkeypatch.setattr(github_live, "_get", _fake_api(routes))
    [chunk] = github_live.live_status_chunks([12], REPO)
    assert "Issue #12: Bug" in chunk.chunk_text and "closed at 2026-11-05" in chunk.chunk_text
    assert "Labels: bug" in chunk.chunk_text and "Assignees: sam" in chunk.chunk_text


@pytest.mark.parametrize("error", [TimeoutError("slow"), urllib.error.URLError("down")])
def test_failures_yield_no_chunks(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    monkeypatch.setattr(github_live, "_get", _fake_api({"/repos": error}))
    assert github_live.live_status_chunks([68], REPO) == []


def test_cache_avoids_second_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes(), calls))
    github_live.live_status_chunks([68], REPO)
    first = len(calls)
    github_live.live_status_chunks([68], REPO)
    assert len(calls) == first


def _stub_live(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[int]:
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes()))
    asked: list[int] = []
    real = github_live.live_status_chunks

    def wrapped(numbers: list[int], repo: str = REPO) -> Any:
        asked.extend(numbers)
        return real(numbers, repo)

    monkeypatch.setattr(module, "live_status_chunks", wrapped)
    return asked


def test_code_lane_injects_live_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_live(monkeypatch, github_answers)
    sent: list[list[dict[str, Any]]] = []

    def draft(messages: list[dict[str, Any]]) -> AnswerResponse:
        sent.append(messages)
        return AnswerResponse(
            status="answered",
            answer="Yes, sam approved it.",
            confidence=0.9,
            citations=[
                Citation(
                    source_file="x",
                    chunk_id=9_000_000_000_068,
                    content_sha256="0" * 64,
                    quote="Reviews: sam: APPROVED at 2026-11-02T02:00:00Z",
                )
            ],
        )

    result = answer_from_retrieval(
        "Has PR 68 been approved yet?", CodeRetrieval(status="no_relevant_information"), draft
    )
    assert result.outcome == "code_answered"
    assert "Live GitHub status" in sent[0][1]["content"] and "9000000000068" in sent[0][1]["content"]
    assert result.response.citations[0].chunk_id == github_live.LIVE_ID_BASE + 68
    assert "pulls/68" in result.response.answer


def test_code_lane_unchanged_without_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes(), calls))
    result = answer_from_retrieval(
        "Where is the sync code?", CodeRetrieval(status="no_relevant_information"), lambda _m: None
    )
    assert result.outcome == "code_abstained" and calls == []


def test_general_path_pack_has_github_live(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("Not sure.", status="no_relevant_information")])
    _stub_live(monkeypatch, agent)
    agent.run_agent("what's the status of PR 68")
    pack = _pack(seen["calls"][0])
    [entry] = pack["github_live"]
    assert entry["chunk_id"] == 9_000_000_000_068 and "Reviews: sam: APPROVED" in entry["text"]


def test_general_path_uses_thread_history(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("Not sure.", status="no_relevant_information")])
    _stub_live(monkeypatch, agent)
    agent.run_agent("and is it approved?", history=[("user", "tell me about PR 68", None)])
    assert "github_live" in _pack(seen["calls"][0])


def test_non_pr_question_makes_no_github_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes(), calls))
    seen = _install(monkeypatch, [_draft("Not sure.", status="no_relevant_information")], hits=[_kb()])
    agent.run_agent("What is the BMTech portal?")
    assert calls == []
    assert "github_live" not in _pack(seen["calls"][0])


def _saved_match() -> Any:
    from benchmark_cli.contributions import ContributionMatch

    response = AnswerResponse(status="answered", answer="Saved answer: not yet approved.", confidence=1.0, citations=[])
    return ContributionMatch(response=response, tier="exact", chunk_id=5, similarity=None)


def test_live_status_outranks_saved_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("Yes, approved.", status="no_relevant_information")])
    monkeypatch.setattr(agent, "contribution_match", lambda *_a, **_k: _saved_match())
    _stub_live(monkeypatch, agent)
    result = agent.run_agent("Has PR 68 been approved yet?")
    assert result.answer != "Saved answer: not yet approved."
    assert seen["calls"] and "github_live" in _pack(seen["calls"][0])
    assert seen.get("outcome") != "user_contributed"


def test_saved_answer_shortcut_without_live_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _install(monkeypatch, [_draft("model", status="no_relevant_information")])
    monkeypatch.setattr(agent, "contribution_match", lambda *_a, **_k: _saved_match())
    calls: list[str] = []
    monkeypatch.setattr(github_live, "_get", _fake_api({}, calls))
    result = agent.run_agent("Has the portal launched yet?")
    assert result.answer == "Saved answer: not yet approved."
    assert seen["calls"] == [] and calls == []


def _live_chunk(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setattr(github_live, "_get", _fake_api(_approved_routes()))
    return github_live.live_status_chunks([68], REPO)[0]


def _code_validate(chunk: Any, quote: str) -> tuple[Any, list[str]]:
    response = AnswerResponse(
        status="answered",
        answer="Alex opened it.",
        confidence=0.9,
        citations=[Citation(source_file="x", chunk_id=chunk.id, content_sha256="0" * 64, quote=quote)],
    )
    return github_answers.validate_code_answer(response, [chunk])


def test_core_facts_are_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    text = _live_chunk(monkeypatch).chunk_text
    assert "Author: alex\nState: open\nReviews: sam: APPROVED" in text


@pytest.mark.parametrize(
    "quote",
    [
        "Author: alex\nState: open\nReviews: sam",
        "Pull request #68\nURL\nAuthor: alex",
        "Pull request #68\\nAuthor: alex\\nState: open",
    ],
)
def test_code_lane_accepts_line_wise_live_quotes(monkeypatch: pytest.MonkeyPatch, quote: str) -> None:
    chunk = _live_chunk(monkeypatch)
    cleaned, _ = _code_validate(chunk, quote)
    assert cleaned is not None
    stored = cleaned.citations[0].quote
    assert stored in chunk.chunk_text and "Author: alex" in stored


def test_live_quote_rewritten_to_exact_chunk_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _live_chunk(monkeypatch)
    cleaned, _ = _code_validate(chunk, "Author: alex\nState: open\nReviews: sam\nChecks: 1")
    assert cleaned is not None
    assert cleaned.citations[0].quote == (
        "Author: alex\nState: open\nReviews: sam: APPROVED at 2026-11-02T02:00:00Z\nChecks: 1 failure, 1 success"
    )


def test_live_title_and_core_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    chunk = _live_chunk(monkeypatch)
    cleaned, _ = _code_validate(chunk, "Pull request #68: Add widget\nAuthor: alex\nState: open")
    assert cleaned is not None and cleaned.citations[0].quote in chunk.chunk_text


def test_non_live_chunk_noncontiguous_quote_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    live = _live_chunk(monkeypatch)
    plain = live.__class__(
        id=5,
        source_file="github/pulls/68",
        chunk_index=0,
        chunk_text=live.chunk_text,
        metadata={"source_type": "github"},
        score=1.0,
        token_count=None,
        content_sha256="a" * 64,
    )
    cleaned, reasons = _code_validate(plain, "Author: alex\nState: open\nReviews: sam\nChecks: 1")
    assert cleaned is None and any("matches no lane chunk" in r for r in reasons)


def test_agent_canonical_citations_accept_live_line_quote(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmark_cli import agent

    chunk = _live_chunk(monkeypatch)
    draft = AnswerResponse(
        status="answered",
        answer="Open.",
        confidence=0.9,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=chunk.content_sha256,
                quote="Author: alex\nState: open\nReviews: sam",
            )
        ],
    )
    result = agent._canonical_citations(draft, {chunk.id: chunk}, {})
    assert result.citations and result.citations[0].quote in chunk.chunk_text
    assert agent._citations_are_exact(result, {chunk.id: chunk})
