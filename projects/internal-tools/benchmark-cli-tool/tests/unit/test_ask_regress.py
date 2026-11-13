"""Offline tests for the Ask regression harness. Synthetic data only."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_cli import ask_regress as harness
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SearchChunk

_REPO_ROOT = Path(__file__).resolve().parents[5]


def _decision(*ids: int, cosine: float = 0.5) -> RetrievalDecision:
    results = tuple(
        HybridSearchResult(
            SearchChunk(item, f"kb/{item}.md", 0, "text", {"source_type": "kb_page"}, 0.4, content_sha256="a" * 64),
            1.0 / (60 + rank),
            vector_score=0.4,
            fts_score=0.1,
        )
        for rank, item in enumerate(ids, start=1)
    )
    return RetrievalDecision("accepted", results, cosine, 2)


def _case(key: str = "suite:0", ids: tuple[int, ...] = (1, 2, 3), **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "key": key,
        "q": "Synthetic question?",
        "starter": None,
        "retrievals": [{"args": {"query": "q", "queries": ["q"], "snapshot_present": True}, "decision": harness.trace_decision(_decision(*ids))}],
        "phrase_lookups": [],
        "model_calls": [{"fn": "call_llm", "component": None, "messages_sha256": "m" * 64, "tools_sha256": "t" * 64}],
        "response": {"status": "no_relevant_information", "answer_sha256": "x", "citations": []},
        "outcome": "validation_failed",
        "error": None,
    }
    record.update(overrides)
    return record


_FP = {"count": 10, "max_id": 10, "md5": "abc"}


@pytest.mark.unit
def test_identical_traces_have_no_diff() -> None:
    assert harness.diff_case(_case(), _case()) == []


@pytest.mark.unit
def test_reordered_chunks_report_the_ids() -> None:
    diffs = harness.diff_case(_case(ids=(1, 2, 3)), _case(ids=(1, 3, 2)))
    assert len(diffs) == 1
    assert "baseline=[1, 2, 3]" in diffs[0] and "candidate=[1, 3, 2]" in diffs[0]


@pytest.mark.unit
def test_displaced_chunk_is_named_missing() -> None:
    diffs = harness.diff_case(_case(ids=(1, 2, 3)), _case(ids=(1, 2, 9)))
    assert "missing=[3]" in diffs[0] and "added=[9]" in diffs[0]


@pytest.mark.unit
def test_score_drift_beyond_tolerance_is_a_diff() -> None:
    base = _case()
    cand = _case()
    cand["retrievals"][0]["decision"]["results"][0]["rrf"] += 1e-6  # type: ignore[index]
    assert any("rrf" in line for line in harness.diff_case(base, cand))
    tiny = _case()
    tiny["retrievals"][0]["decision"]["results"][0]["rrf"] += 1e-12  # type: ignore[index]
    assert harness.diff_case(base, tiny) == []


@pytest.mark.unit
def test_prompt_hash_response_and_outcome_changes_are_diffs() -> None:
    base = _case()
    prompt = _case(model_calls=[{"fn": "call_llm", "component": None, "messages_sha256": "n" * 64, "tools_sha256": "t" * 64}])
    response = _case(response={"status": "answered", "answer_sha256": "y", "citations": [{"chunk_id": 1, "source_file": "kb/1.md"}]})
    outcome = _case(outcome="answered")
    assert any("prompts differ" in line for line in harness.diff_case(base, prompt))
    assert any("response status" in line for line in harness.diff_case(base, response))
    assert any("outcome" in line for line in harness.diff_case(base, outcome))


@pytest.mark.unit
def test_check_refuses_a_different_corpus() -> None:
    baseline = {"fingerprint": _FP, "cases": [_case()]}
    report = harness.evaluate_check(baseline, [_case()], fingerprint={**_FP, "md5": "other"}, is_code=None)
    assert report["green"] is False
    assert report["fingerprint_ok"] is False
    assert "recapture" in report["message"]


@pytest.mark.unit
def test_check_is_green_only_when_every_case_matches_and_none_is_code() -> None:
    baseline = {"fingerprint": _FP, "cases": [_case("suite:0"), _case("names:0")]}
    green = harness.evaluate_check(baseline, [_case("suite:0"), _case("names:0")], fingerprint=_FP, is_code=lambda _q: False)
    assert green["green"] is True
    flagged = harness.evaluate_check(baseline, [_case("suite:0"), _case("names:0")], fingerprint=_FP, is_code=lambda _q: True)
    assert flagged["green"] is False
    missing = harness.evaluate_check(baseline, [_case("suite:0")], fingerprint=_FP, is_code=None)
    assert missing["missing"] == ["names:0"] and missing["green"] is False


@pytest.mark.unit
def test_embedding_fixture_miss_raises_and_is_recorded() -> None:
    store = harness.EmbeddingStore({harness.sha256_text("known"): [0.1, 0.2]}, replay=True)
    embed = store.wrap(lambda _texts: pytest.fail("replay must not call the provider"))
    assert embed(["known"]) == [[0.1, 0.2]]
    with pytest.raises(harness.EmbeddingMiss):
        embed(["new text"])
    assert store.misses == ["new text"]
    # BaseException: the agent's ``except Exception`` blocks cannot hide a miss.
    assert not issubclass(harness.EmbeddingMiss, Exception)


@pytest.mark.unit
def test_capture_store_records_vectors_by_text_hash() -> None:
    store = harness.EmbeddingStore()
    embed = store.wrap(lambda texts: [[float(len(text))] for text in texts])
    embed(["ab", "abc"])
    assert store.vectors == {harness.sha256_text("ab"): [2.0], harness.sha256_text("abc"): [3.0]}


@pytest.mark.unit
def test_capture_memoizes_repeated_texts_to_the_recorded_vector() -> None:
    calls: list[list[str]] = []

    def noisy(texts: list[str]) -> list[list[float]]:
        calls.append(list(texts))
        return [[float(len(text)) + 1e-5 * len(calls)] for text in texts]

    store = harness.EmbeddingStore()
    embed = store.wrap(noisy)
    first = embed(["ab", "abc", "ab"])
    again = embed(["abc", "ab"])
    assert calls == [["ab", "abc"]]  # one request, duplicates folded, no second request
    assert first == [[2.00001], [3.00001], [2.00001]] and again == [[3.00001], [2.00001]]
    assert store.vectors[harness.sha256_text("ab")] == [2.00001]
    embed(["new"])
    assert calls[-1] == ["new"]  # only uncached texts reach the provider


@pytest.mark.unit
def test_capture_then_check_on_the_same_build_is_green_with_a_noisy_embedder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The provider returns a slightly different vector on every call for the same text, as
    OpenAI does. Each case embeds its question more than once and two cases share a question,
    so the capture must replay exactly the vectors it ran on."""
    import math

    from benchmark_cli import contributions, retrieval
    from benchmark_cli.config import EMBEDDING_DIMENSIONS
    from benchmark_cli.providers import openai as provider

    requests = {"n": 0}

    def create(*, model: str, input: list[str]) -> SimpleNamespace:  # noqa: A002 - SDK keyword
        requests["n"] += 1
        data = []
        for text in input:
            base = [math.sin(len(text) * (index + 1)) for index in range(EMBEDDING_DIMENSIONS)]
            base[0] += 1e-5 * requests["n"]  # drift between requests, like the real endpoint
            data.append(SimpleNamespace(embedding=base))
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=1))

    fake_client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    monkeypatch.setattr(provider, "_get_client", lambda: fake_client)
    monkeypatch.setattr(provider, "log_usage", lambda **_k: None)

    docs = {11: [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1), 12: [0.0, 1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 2)}

    def fake_case(case: harness.Case, _plan: object) -> dict[str, object]:
        # Different modules embed the same question, as contributions + retrieval do.
        contributions.embed_texts([case.q])
        vector = retrieval.embed_texts([case.q, f"{case.q} rewritten"])[0]
        scores = {chunk_id: sum(a * b for a, b in zip(vector, doc, strict=True)) for chunk_id, doc in docs.items()}
        ranked = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
        decision = {
            "status": "accepted",
            "top_cosine": max(scores.values()),
            "overlap": 1,
            "results": [{"id": i, "sha": "s", "source": "kb", "source_type": "kb_page", "rrf": 0.0, "vector": scores[i], "fts": 0.0} for i in ranked],
        }
        return {"key": case.key, "q": case.q, "starter": None, "plan": None, "retrievals": [{"args": {}, "decision": decision}], "phrase_lookups": [], "model_calls": [], "response": None, "outcome": None, "error": None}

    shared = "Who runs the weekly standup?"
    inputs = harness.PrivateInputs(
        suite=(harness.Case("suite:0", "suite", shared), harness.Case("suite:1", "suite", "Where are the meeting notes?")),
        names=(),
        saved=(harness.Case("saved:0", "saved", shared),),
        titles={},
        baseline={},
    )
    monkeypatch.setattr(harness, "load_private", lambda _d: inputs)
    monkeypatch.setattr(harness, "preflight", lambda: None)
    monkeypatch.setattr(harness, "corpus_fingerprint", lambda: dict(_FP))
    monkeypatch.setattr(harness, "stored_plan", lambda _case, _mode: None)
    monkeypatch.setattr(harness, "run_offline_case", fake_case)
    monkeypatch.setattr(harness, "classifier", lambda: None)

    base_path, check_path = tmp_path / "base.json", tmp_path / "check.json"
    assert harness.main(["capture", "--out", str(base_path), "--private-dir", str(tmp_path), "--plans", "raw"]) == 0
    assert requests["n"] >= 2  # the provider really drifted between requests
    monkeypatch.setattr(provider, "_get_client", lambda: pytest.fail("check makes no embedding request"))
    assert harness.main(["check", "--baseline", str(base_path), "--out", str(check_path), "--private-dir", str(tmp_path)]) == 0
    import json

    report = json.loads(check_path.read_text())
    assert report["green"] is True and report["embedding_misses"] == 0 and report["failed"] == []


@pytest.mark.unit
def test_suite_early_stop_rules() -> None:
    assert harness.suite_stop_reason(41, 15, 15) is None
    assert "mathematically" in harness.suite_stop_reason(40, 15, 15, bar=56)  # type: ignore[operator]
    assert "clearly" in harness.suite_stop_reason(45, 12, 9, bar=56)  # type: ignore[operator]
    assert harness.suite_stop_reason(45, 12, 10, bar=56) is None


@pytest.mark.unit
def test_tally_stops_on_saved_names_hijack_and_third_code_failure() -> None:
    tally = harness.LiveTally()
    with pytest.raises(harness.EarlyStop, match="saved"):
        tally.after_saved(False, "saved:0")
    with pytest.raises(harness.EarlyStop, match="names"):
        tally.after_name(False, "names:3")
    tally.after_code(False, True, "a")
    tally.after_code(False, True, "b")
    tally.after_code(False, False, "twelfth")
    with pytest.raises(harness.EarlyStop, match="third code failure"):
        tally.after_code(False, True, "c")
    with pytest.raises(harness.EarlyStop, match="hijack"):
        harness.LiveTally().after_suite(answered=True, hijack="github citation", key="suite:4", remaining=60, remaining_baseline_answered=50)
    with pytest.raises(harness.EarlyStop, match="mathematically"):
        harness.LiveTally().after_suite(answered=False, hijack=None, key="suite:20", remaining=10, remaining_baseline_answered=10)


@pytest.mark.unit
def test_budget_rules() -> None:
    assert harness.budget_exceeded(700, 8_000_000, 5400) is None
    assert "chat calls" in harness.budget_exceeded(701, 0, 0)  # type: ignore[operator]
    assert "prompt tokens" in harness.budget_exceeded(0, 8_000_001, 0)  # type: ignore[operator]
    assert "s >" in harness.budget_exceeded(0, 0, 5401)  # type: ignore[operator]


def _result(status: str = "answered", answer: str = "Plain answer.", sources: tuple[str, ...] = ("trello/b/1.md",), outcome: str | None = "answered") -> dict[str, object]:
    return {"status": status, "answer": answer, "citations": [{"source_file": item, "chunk_id": 1} for item in sources], "outcome": outcome}


@pytest.mark.unit
def test_saved_scorers() -> None:
    shortcut = {"expect": "saved_shortcut", "source_prefix": "user_contributed/"}
    good = _result(answer="According to Ada Example on 2026-01-01: three.", sources=("user_contributed/1/2.md",), outcome="user_contributed")
    assert harness.score_saved(shortcut, good)[0]
    assert not harness.score_saved(shortcut, {**good, "outcome": "answered"})[0]
    cites = {"expect": "cites_saved"}
    paraphrase = _result(answer="Three people, according to Ada.", sources=("user_contributed/1/2.md",))
    assert harness.score_saved(cites, paraphrase)[0]
    assert not harness.score_saved(cites, _result())[0]


@pytest.mark.unit
def test_core_team_is_not_a_saved_case() -> None:
    rule = {"expect": "not_saved"}
    assert harness.score_saved(rule, _result(answer="Core team: A, B.", sources=("kb/team.md",)))[0]
    assert not harness.score_saved(rule, _result(answer="According to Ada: A, B.", sources=("kb/team.md",)))[0]
    assert not harness.score_saved(rule, _result(answer="Core team: A.", sources=("user_contributed/1/2.md",)))[0]


@pytest.mark.unit
def test_names_scorer_requires_titles_and_rejects_other_people_and_hijacks() -> None:
    titles = {"Sample One": ["Card A", "Card B"], "Sample Two": ["Card C"]}
    case = {"person": "Sample One"}
    assert harness.score_name(case, _result(answer="Card A and Card B"), titles)[0]
    assert not harness.score_name(case, _result(answer="Card A only"), titles)[0]
    assert not harness.score_name(case, _result(answer="Card A, Card B, Card C"), titles)[0]
    assert not harness.score_name(case, _result(answer="According to X: Card A, Card B"), titles)[0]
    assert not harness.score_name(case, _result(answer="Card A, Card B", sources=("github/r/x.py",)), titles)[0]
    assert not harness.score_name(case, _result(status="no_relevant_information"), titles)[0]


@pytest.mark.unit
def test_code_scorer() -> None:
    case = {"expect_paths": ("benchmark_cli/config.py",)}
    assert harness.score_code(case, _result(sources=("github/o/r/src/benchmark_cli/config.py",)))[0]
    assert not harness.score_code(case, _result(sources=("github/o/r/README.md",)))[0]
    assert not harness.score_code(case, _result(sources=("github/o/r/src/benchmark_cli/config.py", "trello/b/1.md")))[0]
    kiosk = {"expect": "abstain_or_kiosk_path", "expect_paths": ()}
    assert harness.score_code(kiosk, _result(status="no_relevant_information", sources=()))[0]
    assert harness.score_code(kiosk, _result(sources=("github/o/r/kiosk/db.py",)))[0]
    assert not harness.score_code(kiosk, _result(sources=("trello/b/kiosk.md",)))[0]
    pulls = {"expect": "pulls", "expect_paths": ()}
    assert harness.score_code(pulls, _result(sources=("github/o/r/pulls/3",)))[0]
    assert not harness.score_code(pulls, _result(sources=("github/o/r/pulls/3", "github/o/r/README.md")))[0]


@pytest.mark.unit
def test_hijack_detector() -> None:
    plain = {"answer": "Plain.", "citations": [{"source_file": "trello/b/1.md"}]}
    saved = {"answer": "According to Ada: x", "citations": [{"source_file": "user_contributed/1/2.md"}]}
    assert harness.suite_hijack(_result(), plain) is None
    assert harness.suite_hijack(_result(sources=("github/o/r/tests/test_x.py",)), plain) == "github citation"
    assert harness.suite_hijack(_result(answer="According to Ada: x"), plain) == "saved-answer takeover"
    assert harness.suite_hijack(_result(sources=("user_contributed/1/2.md",)), plain) == "saved-answer takeover"
    assert harness.suite_hijack(_result(answer="According to Ada: x", sources=("user_contributed/1/2.md",)), saved) is None


@pytest.mark.unit
def test_refuses_to_write_inside_a_git_tree(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="git working tree"):
        harness.ensure_outside_repo(_REPO_ROOT / "trace.json")
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    with pytest.raises(SystemExit):
        harness.write_json(repo / "out" / "trace.json", {})
    private = tmp_path / "private"
    written = harness.write_json(private / "trace.json", {"ok": True})
    assert written.is_file()


@pytest.mark.unit
def test_guard_rejects_non_mini_chat_and_offline_calls() -> None:
    guard = harness.OpenAIGuard(allow_chat=True, allow_embed=True)
    guard.check_chat("gpt-4o-mini")
    with pytest.raises(harness.HarnessAbort):
        guard.check_chat("gpt-4o")
    assert guard.violation and "gpt-4o" in guard.violation
    with pytest.raises(harness.HarnessAbort):
        guard.check_embed("text-embedding-3-large")
    offline = harness.OpenAIGuard(allow_chat=False, allow_embed=False)
    with pytest.raises(harness.HarnessAbort):
        offline.check_chat("gpt-4o-mini")
    assert not issubclass(harness.HarnessAbort, Exception)


@pytest.mark.unit
def test_guard_wraps_both_sdk_chat_entry_points() -> None:
    calls: list[str] = []
    completions = SimpleNamespace(create=lambda **k: calls.append("create"), parse=lambda **k: calls.append("parse"))
    beta = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(parse=lambda **k: calls.append("beta"))))
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions), beta=beta)
    guard = harness.OpenAIGuard(allow_chat=True, allow_embed=False)
    guard.guard_chat_client(client)
    guard.guard_chat_client(client)  # idempotent
    client.chat.completions.create(model="gpt-4o-mini")
    client.beta.chat.completions.parse(model="gpt-4o-mini")
    with pytest.raises(harness.HarnessAbort):
        client.chat.completions.parse(model="gpt-4.1")
    assert calls == ["create", "beta"]
    assert guard.usage.chat_calls == 2


@pytest.mark.unit
def test_guard_budget_aborts_before_the_call() -> None:
    guard = harness.OpenAIGuard(allow_chat=True, allow_embed=True, budget=True)
    guard.usage.chat_calls = harness.MAX_CHAT_CALLS + 1
    with pytest.raises(harness.HarnessAbort, match="budget"):
        guard.check_chat("gpt-4o-mini")


@pytest.mark.unit
def test_preflight_requires_mini_and_small_embeddings() -> None:
    from benchmark_cli.config import EMBEDDING_MODEL, get_model_config

    assert get_model_config("cheap").model == harness.REQUIRED_CHAT_MODEL
    assert EMBEDDING_MODEL == harness.REQUIRED_EMBEDDING_MODEL
    harness.preflight()


@pytest.mark.unit
def test_live_needs_a_green_check_for_the_same_build() -> None:
    build = {"source_sha256": "b" * 64}
    harness.verify_check_report({"kind": "ask-regress-check", "green": True, "build": build}, build)
    with pytest.raises(SystemExit, match="not green"):
        harness.verify_check_report({"kind": "ask-regress-check", "green": False, "build": build}, build)
    with pytest.raises(SystemExit, match="different build"):
        harness.verify_check_report({"kind": "ask-regress-check", "green": True, "build": {"source_sha256": "c"}}, build)


@pytest.mark.unit
def test_private_inputs_load_from_a_directory(tmp_path: Path) -> None:
    import json

    (tmp_path / "suite.json").write_text(json.dumps([{"group": "a", "q": "Q1?", "starter": None}, {"group": "t", "q": "Q2?", "starter": "S?"}]))
    (tmp_path / "names.json").write_text(json.dumps({"cases": [{"q": "N1?", "person": "Sample One"}], "titles": {"Sample One": ["Card A"]}}))
    (tmp_path / "saved.json").write_text(json.dumps([{"q": "S1?", "expect": "not_saved"}]))
    (tmp_path / "baseline-suite.json").write_text(json.dumps({"results": [{"q": "Q2?", "starter": "S?", "status": "answered"}]}))
    inputs = harness.load_private(tmp_path)
    assert [case.key for case in inputs.non_code_cases()] == ["suite:0", "suite:1", "names:0", "saved:0"]
    assert inputs.baseline == {"suite:1": {"q": "Q2?", "starter": "S?", "status": "answered"}}
    assert inputs.saved[0].extra["source_prefix"] == "user_contributed/"


@pytest.mark.unit
def test_baseline_alignment_keeps_repeated_questions_apart() -> None:
    suite = [harness.Case("suite:0", "suite", "Q?"), harness.Case("suite:1", "suite", "Q?")]
    rows = [{"q": "Q?", "starter": None, "status": "answered"}, {"q": "Q?", "starter": None, "status": "no_relevant_information"}]
    aligned = harness.align_baseline(suite, rows)
    assert aligned["suite:0"]["status"] == "answered"
    assert aligned["suite:1"]["status"] == "no_relevant_information"


@pytest.mark.unit
def test_thread_question_matches_the_discord_rule() -> None:
    from benchmark_cli.discord_ask import thread_question_for

    for question, starter in (("a b", None), ("a  b", "a b"), ("next?", "first?")):
        assert harness.thread_question(question, starter) == thread_question_for(question, starter)


@pytest.mark.unit
def test_code_cases_are_twelve_with_eleven_counted() -> None:
    assert len(harness.CODE_CASES) == 12
    assert sum(case["original"] for case in harness.CODE_CASES) == 11
    assert len({case["id"] for case in harness.CODE_CASES}) == 12


@pytest.mark.unit
def test_code_retrieval_gate_flags_forbidden_chunks_and_rank() -> None:
    def chunk(path: str, kind: str = "code") -> SearchChunk:
        return SearchChunk(1, f"github/o/r/{path}", 0, "x", {"path": path, "kind": kind}, 0.1, content_sha256="a" * 64)

    case = {"id": "x", "expect_paths": ("benchmark_cli/config.py",)}
    ok = SimpleNamespace(status="accepted", chunks=(chunk("README.md", "doc"), chunk("src/benchmark_cli/config.py")))
    assert harness.evaluate_code_retrieval(case, ok)["rank"] == 2
    assert harness.evaluate_code_retrieval(case, ok)["passed"]
    low = SimpleNamespace(status="accepted", chunks=tuple(chunk(f"a{i}.py") for i in range(3)) + (chunk("src/benchmark_cli/config.py"),))
    assert not harness.evaluate_code_retrieval(case, low)["passed"]
    for bad in (chunk("pulls/3", "pull_request"), chunk("tests/unit/test_x.py"), chunk("portal/content/a.md", "doc")):
        dirty = SimpleNamespace(status="accepted", chunks=(chunk("src/benchmark_cli/config.py"), bad))
        assert not harness.evaluate_code_retrieval(case, dirty)["passed"]


@pytest.mark.unit
def test_live_summary_requires_every_group_and_no_stop() -> None:
    report = {
        "saved": [{"passed": True}] * 3,
        "names": [{"passed": True}] * 12,
        "code": [{"passed": True, "original": True}] * 9 + [{"passed": False, "original": True}] * 2 + [{"passed": False, "original": False}],
        "suite_runs": [[{"answered": True, "hijack": None}] * 56 + [{"answered": False, "hijack": None}] * 13],
        "early_stop": None,
    }
    assert harness.summarize_live(report, ["saved", "names", "code", "suite"])["pass"] is True
    assert harness.summarize_live(report, ["saved", "names"])["pass"] is False
    stopped = {**report, "early_stop": "budget"}
    assert harness.summarize_live(stopped, ["saved", "names", "code", "suite"])["pass"] is False


@pytest.mark.unit
def test_prompt_hash_ignores_the_random_evidence_nonce_only() -> None:
    from benchmark_cli.safety import delimit_evidence

    first = [{"role": "system", "content": delimit_evidence("same text")}]
    second = [{"role": "system", "content": delimit_evidence("same text")}]
    assert first != second
    assert harness.prompt_sha256(first) == harness.prompt_sha256(second)
    other = [{"role": "system", "content": delimit_evidence("other text")}]
    assert harness.prompt_sha256(first) != harness.prompt_sha256(other)


@pytest.mark.unit
def test_code_retrieval_record_and_replay_run_offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    from benchmark_cli import github_answers
    from benchmark_cli.models import AnswerResponse, Citation

    def chunk(chunk_id: int, path: str, symbol: str) -> SearchChunk:
        text = f"{path}:L1-L2 {symbol}\nhttps://github.com/o/r/blob/abc/{path}#L1-L2\n\ndef {symbol or 'x'}() -> None:\n    return None\n"
        return SearchChunk(
            chunk_id,
            f"github/o/r/{path}",
            0,
            text,
            {"source_type": "github", "path": path, "kind": "code", "symbol": symbol, "header_words": path.replace("/", " "), "start_line": 1, "end_line": 2, "url": f"https://github.com/o/r/blob/abc/{path}#L1-L2"},
            0.6,
            10,
            "d" * 64,
        )

    lane = [chunk(1_000_000_001, "src/benchmark_cli/trello_sync.py", "sync_board"), chunk(1_000_000_002, "src/benchmark_cli/config.py", "")]
    monkeypatch.setattr(github_answers, "embed_texts", lambda texts: [[0.1] for _ in texts])
    for name in ("github_vector_search", "github_fts_search", "github_header_search"):
        monkeypatch.setattr(github_answers, name, lambda _q, limit, _kinds: lane[:limit])
    monkeypatch.setattr(github_answers, "github_pull_rows", lambda: [])
    monkeypatch.setattr(github_answers, "github_identity_rows", lambda _kinds: [(c.metadata["path"], c.metadata["symbol"], c.metadata["header_words"]) for c in lane])
    drafts = iter([AnswerResponse(status="answered", answer="sync_board", confidence=0.8, citations=[Citation(source_file=lane[0].source_file, chunk_id=lane[0].id, content_sha256="d" * 64, quote="not in the chunk")])])
    monkeypatch.setattr(github_answers, "call_llm_structured", lambda **_k: next(drafts, None) or AnswerResponse(status="no_relevant_information", answer="No.", confidence=0.0, citations=[]))

    assert harness.main(["code-retrieval", "--out", str(tmp_path / "code.json")]) == 1  # expected paths missing for most cases
    report = json.loads((tmp_path / "code.json").read_text())
    assert next(row for row in report["cases"] if row["id"] == "trello-ingest")["rank"] == 1

    assert harness.main(["record", "--out", str(tmp_path / "rec.json")]) == 0
    recording = json.loads((tmp_path / "rec.json").read_text())
    first = recording["cases"][0]
    assert first["id"] == "trello-ingest" and first["drafts"][0]["status"] == "answered"
    monkeypatch.setattr(github_answers, "call_llm_structured", lambda **_k: pytest.fail("replay makes no model call"))
    assert harness.main(["replay", "--recording", str(tmp_path / "rec.json"), "--out", str(tmp_path / "replay.json")]) == 0
    replayed = json.loads((tmp_path / "replay.json").read_text())
    assert all(row["recorded_outcome"] == row["replayed_outcome"] for row in replayed["cases"])
    assert replayed["cases"][0]["drafts"][0]["valid"] is False


@pytest.mark.unit
def test_fingerprint_ignores_github_rows_but_reports_them() -> None:
    assert "IS DISTINCT FROM 'github'" in harness.FINGERPRINT_SQL
    base = {"count": 5, "max_id": 9, "md5": "m", "github_rows_in_document_chunks": 0}
    assert harness.fingerprint_matches(base, {**base, "github_rows_in_document_chunks": 3000})
    assert not harness.fingerprint_matches(base, {**base, "max_id": 10})
