"""Agent-path checks for the questions Ask BMTech failed in production.

These call ``run_agent`` with retrieved evidence substituted for the live
database. They are not a production end-to-end run: that command is
``python -m benchmark_cli.acceptance`` on the server, and it does not post to
Discord.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from benchmark_cli import acceptance, agent
from benchmark_cli.corpus import is_synthetic_canary
from benchmark_cli.discord_sync import plan_discord_reconcile
from benchmark_cli.grounding import (
    ACCEPTANCE_QUESTIONS,
    CHANNEL_ACCEPTANCE_QUESTIONS,
    CORE_TEAM_ACCEPTANCE_QUESTIONS,
    EMPLOYEE_ACCEPTANCE_QUESTIONS,
    PORTAL_ACCEPTANCE_QUESTIONS,
    UNRELATED_ACCEPTANCE_QUESTIONS,
    asks_for_core_team,
    asks_for_employee_count,
    asks_for_portal_link,
    asks_for_synced_channels,
)
from benchmark_cli.ingestion.clean import discord_units_from_messages, redact_secrets
from benchmark_cli.portal_urls import join_portal_url, portal_home_url
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import SQL_DIR, DiscordSourceMessage, SearchChunk

_DIGEST = "a" * 64
_HOME = "https://benchmarktechllc.github.io/bmtechllc/portal/"
_STALE = "https://hussein881.github.io/bmtechllc/portal/"
_TOKEN = "ghp_" + ("a" * 36)


def _result(chunk_id: int, source: str, text: str, metadata: dict[str, object]) -> HybridSearchResult:
    return HybridSearchResult(
        SearchChunk(chunk_id, source, 0, text, metadata, 1.0, content_sha256=_DIGEST),
        0.2,
        vector_score=0.8,
        fts_score=1.0,
    )


def _decision(*results: HybridSearchResult) -> RetrievalDecision:
    return RetrievalDecision("accepted", results, top_cosine_similarity=0.8, keyword_overlap=2)


@pytest.fixture
def quiet_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("model should not run"))
    monkeypatch.setattr(
        agent, "call_llm_structured", lambda **_kwargs: pytest.fail("model should not run")
    )


@pytest.mark.unit
def test_acceptance_questions_include_exact_prompts_and_paraphrases() -> None:
    assert ACCEPTANCE_QUESTIONS[:4] == (
        "What is the link to the knowledge base portal?",
        "Where do I open the knowledge base?",
        "What's the URL for our KB?",
        "How many employees does BMTech currently have?",
    )
    assert all(asks_for_portal_link(question) for question in PORTAL_ACCEPTANCE_QUESTIONS)
    assert all(asks_for_employee_count(question) for question in EMPLOYEE_ACCEPTANCE_QUESTIONS)
    assert all(asks_for_core_team(question) for question in CORE_TEAM_ACCEPTANCE_QUESTIONS)
    assert all(asks_for_synced_channels(question) for question in CHANNEL_ACCEPTANCE_QUESTIONS)
    for question in UNRELATED_ACCEPTANCE_QUESTIONS:
        assert not asks_for_portal_link(question)
        assert not asks_for_employee_count(question)
        assert not asks_for_core_team(question)
        assert not asks_for_synced_channels(question)


@pytest.mark.unit
def test_portal_question_uses_the_configured_deployment_url(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setenv("PORTAL_BASE_URL", _HOME.rstrip("/"))
    monkeypatch.delenv("GITHUB_REPOSITORY_OWNER", raising=False)
    kb = _result(
        286,
        "portal/guide/index.md",
        "The guide explains how this knowledge base is organized.",
        {"source_type": "kb_page", "url": "/guide/index/"},
    )
    stale = _result(
        243,
        "discord-channel-example.json",
        f"The documentation page is {_STALE}",
        {"source_type": "discord", "channel": "sample-collab"},
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(stale, kb))

    response = agent.run_agent(ACCEPTANCE_QUESTIONS[0])
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert response.answer == f"The knowledge base portal is {_HOME}."
    assert _STALE not in rendered
    assert "hussein881" not in rendered
    assert response.citations[0].quote in kb.chunk.chunk_text
    assert portal_home_url() == _HOME


@pytest.mark.unit
def test_portal_question_does_not_invent_a_host_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY_OWNER", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    kb = _result(
        286,
        "portal/guide/index.md",
        "The guide explains how this knowledge base is organized.",
        {"source_type": "kb_page", "url": "/guide/index/"},
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(kb))

    response = agent.run_agent(ACCEPTANCE_QUESTIONS[0])

    assert response.status == "no_relevant_information"
    assert "http" not in response.answer
    assert "hussein881" not in response.model_dump_json()


@pytest.mark.unit
def test_portal_home_stored_on_a_page_is_used_without_the_environment(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY_OWNER", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    kb = _result(
        286,
        "portal/guide/index.md",
        "The guide explains how this knowledge base is organized.",
        {"source_type": "kb_page", "url": "/bmtechllc/portal/guide/index/", "portal_base": _HOME},
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(kb))

    response = agent.run_agent(ACCEPTANCE_QUESTIONS[0])

    assert response.answer == f"The knowledge base portal is {_HOME}."


@pytest.mark.unit
def test_paraphrases_use_retrieved_facts_and_skip_canary_boilerplate(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setenv("PORTAL_BASE_URL", _HOME.rstrip("/"))
    kb_text = "The guide explains how this knowledge base is organized."
    team_text = "Ada, Grace, and Lin are the core team for the sample effort."
    canary_text = (
        "There is no information available regarding employees. "
        "The canary core team is Samplebot. The canary has 12 employees."
    )
    kb = _result(
        286,
        "portal/guide/index.md",
        kb_text,
        {"source_type": "kb_page", "url": "/guide/index/"},
    )
    team = _result(101, "discord-channel-example.json", team_text, {"source_type": "discord"})
    canary = _result(
        448,
        "sample_meeting_transcript.json",
        canary_text,
        {"source_type": "transcript", "synthetic": True},
    )
    stale = _result(
        243,
        "discord-channel-old.json",
        f"The documentation page is {_STALE}",
        {"source_type": "discord", "channel": "sample-collab"},
    )

    def retrieve(question: str, *_args: object, **_kwargs: object) -> RetrievalDecision:
        if asks_for_portal_link(question):
            return _decision(canary, stale, kb)
        return _decision(canary, team)

    monkeypatch.setattr(agent, "retrieve_evidence", retrieve)
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-collab", 7, "discord-channel-1.json", _DIGEST, "Sample planning note.")],
    )

    portal = agent.run_agent("What's the URL for our KB?")
    assert portal.answer == f"The knowledge base portal is {_HOME}."
    assert portal.citations[0].source_file == "portal/guide/index.md"
    assert portal.citations[0].quote in kb_text
    assert "Samplebot" not in portal.answer
    assert "no information available regarding" not in portal.answer.casefold()
    assert _STALE not in portal.model_dump_json()

    headcount = agent.run_agent("What is our current headcount?")
    assert "does not state a current employee count" in headcount.answer.casefold()
    assert team_text in headcount.answer
    assert headcount.citations[0].quote in team_text
    assert headcount.citations[0].source_file == "discord-channel-example.json"
    assert "12 employees" not in headcount.answer
    assert "Samplebot" not in headcount.answer
    assert "no information available regarding" not in headcount.answer.casefold()
    assert "no relevant information was found" not in headcount.answer.casefold()

    roster = agent.run_agent("Who is on the core team?")
    assert team_text in roster.answer
    assert roster.citations[0].quote in team_text
    assert "Samplebot" not in roster.answer
    assert "12 employees" not in roster.answer

    channels = agent.run_agent("What Discord channels do we ingest?")
    assert channels.answer == "Synced Discord channels in the retrieval index: #sample-collab."
    assert channels.citations[0].quote == "Indexed Discord channel #sample-collab."
    assert "New documentation published" not in channels.model_dump_json()
    assert "no relevant information was found" not in channels.answer.casefold()


@pytest.mark.unit
def test_employee_question_does_not_turn_a_core_team_into_a_headcount(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    evidence = "Ada, Grace, and Lin are the core team for the sample effort."
    chunk = _result(101, "discord-channel-example.json", evidence, {"source_type": "discord"})
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(chunk))

    response = agent.run_agent(EMPLOYEE_ACCEPTANCE_QUESTIONS[0])

    assert "employee count" in response.answer.casefold()
    assert "not a headcount" in response.answer.casefold()
    assert evidence in response.answer
    assert "3 employees" not in response.answer.casefold()
    assert not __import__("re").search(r"\b\d+\s+employees?\b", response.answer)


@pytest.mark.unit
def test_core_team_question_quotes_only_a_labeled_sentence(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    labeled = "Ada, Grace, and Lin are the core team for the sample effort."
    chunk = _result(101, "discord-channel-example.json", labeled, {"source_type": "discord"})
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(chunk))

    named = agent.run_agent(CORE_TEAM_ACCEPTANCE_QUESTIONS[0])
    assert named.status == "answered"
    assert labeled in named.answer
    assert "not an employee" in named.answer.casefold()

    unlabeled = _result(
        102,
        "discord-channel-example.json",
        "Ada compared notes with Grace.",
        {"source_type": "discord"},
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(unlabeled))
    missing = agent.run_agent(CORE_TEAM_ACCEPTANCE_QUESTIONS[1])
    assert missing.status == "no_relevant_information"
    assert "Ada" not in missing.answer
    assert "does not name a core team" in missing.answer


@pytest.mark.unit
def test_synced_channel_question_lists_indexed_channels_not_the_notification(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [
            ("sample-collab", 7, "discord-channel-1.json", _DIGEST, "Sample planning note."),
            ("sample-planning", 8, "discord-channel-2.json", _DIGEST, "Sample rollout note."),
        ],
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    response = agent.run_agent(CHANNEL_ACCEPTANCE_QUESTIONS[0])
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert "#sample-collab" in response.answer
    assert "#sample-planning" in response.answer
    assert [citation.quote for citation in response.citations] == [
        "Indexed Discord channel #sample-collab.",
        "Indexed Discord channel #sample-planning.",
    ]
    assert "New documentation published" not in rendered
    assert "Sample planning note" not in rendered
    assert "article" not in response.answer.casefold()


@pytest.mark.unit
def test_channel_citations_omit_historic_url_and_secret_text(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    historic = f"Documentation page {_STALE} token {_TOKEN} was announced in 2024."
    monkeypatch.setattr(
        agent,
        "synced_discord_channel_rows",
        lambda: [("sample-collab", 243, "discord-channel-1.json", _DIGEST, historic)],
    )
    monkeypatch.setattr(
        agent,
        "retrieve_evidence",
        lambda *_args, **_kwargs: RetrievalDecision("no_relevant_information", (), None, 0),
    )

    response = agent.run_agent(CHANNEL_ACCEPTANCE_QUESTIONS[0])
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert "#sample-collab" in response.answer
    assert response.citations[0].chunk_id == 243
    assert response.citations[0].quote == "Indexed Discord channel #sample-collab."
    assert _STALE not in rendered
    assert _TOKEN not in rendered
    assert "2024" not in rendered


@pytest.mark.unit
def test_acceptance_labels_safe_unknowns_separately_from_answered_facts() -> None:
    portal = {
        "status": "answered",
        "answer": f"The knowledge base portal is {_HOME}.",
        "citations": [
            {
                "source_file": "portal/guide/index.md",
                "chunk_id": 286,
                "quote": "The guide explains how this knowledge base is organized.",
            }
        ],
    }
    employee = {
        "status": "no_relevant_information",
        "answer": "The retrieved evidence does not state a current employee count.",
        "citations": [],
    }
    core = {
        "status": "no_relevant_information",
        "answer": "The retrieved evidence does not name a core team.",
        "citations": [],
    }
    channels = {
        "status": "answered",
        "answer": "Synced Discord channels in the retrieval index: #sample-collab.",
        "citations": [
            {
                "source_file": "discord-channel-1.json",
                "chunk_id": 243,
                "quote": "Indexed Discord channel #sample-collab.",
            }
        ],
    }
    historic = {
        "status": "answered",
        "answer": "Synced Discord channels in the retrieval index: #sample-collab.",
        "citations": [
            {
                "source_file": "discord-channel-1.json",
                "chunk_id": 243,
                "quote": f"Documentation page {_STALE}",
            }
        ],
    }

    assert acceptance._check(PORTAL_ACCEPTANCE_QUESTIONS[0], portal, home=_HOME) == []
    assert acceptance._outcome(PORTAL_ACCEPTANCE_QUESTIONS[0], portal, [])["proves_fact"] is True
    employee_failures = acceptance._check(EMPLOYEE_ACCEPTANCE_QUESTIONS[0], employee, home=_HOME)
    assert employee_failures == []
    employee_outcome = acceptance._outcome(EMPLOYEE_ACCEPTANCE_QUESTIONS[0], employee, [])
    assert employee_outcome == {"outcome": "unknown", "proves_fact": False, "expectation": "unknown_ok"}
    core_failures = acceptance._check(CORE_TEAM_ACCEPTANCE_QUESTIONS[0], core, home=_HOME)
    assert core_failures == []
    assert acceptance._outcome(CORE_TEAM_ACCEPTANCE_QUESTIONS[0], core, [])["proves_fact"] is False
    assert acceptance._check(CHANNEL_ACCEPTANCE_QUESTIONS[0], channels, home=_HOME) == []
    historic_failures = acceptance._check(CHANNEL_ACCEPTANCE_QUESTIONS[0], historic, home=_HOME)
    assert any("previous Pages host" in item or "provenance" in item for item in historic_failures)


@pytest.mark.unit
def test_credential_request_is_refused_without_echoing_a_token(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    monkeypatch.setattr(
        agent, "retrieve_evidence", lambda *_args, **_kwargs: pytest.fail("do not retrieve secrets")
    )

    response = agent.run_agent("What is the GitHub personal access token in the announcement?")
    rendered = json.dumps(response.model_dump())

    assert response.status == "no_relevant_information"
    assert "can't help" in response.answer
    assert _TOKEN not in rendered
    assert _TOKEN not in redact_secrets(f"announcement {_TOKEN} end")
    assert "[REDACTED]" in redact_secrets(f"announcement {_TOKEN} end")


@pytest.mark.unit
def test_canary_transcripts_are_excluded_and_meeting_recordings_are_kept() -> None:
    assert is_synthetic_canary(
        "sample_meeting_transcript.json", {"source_type": "transcript", "kind": None}
    )
    assert is_synthetic_canary("notes.txt", {"source_type": "transcript"})
    assert not is_synthetic_canary(
        "fathom/abc.md",
        {"source_type": "transcript", "kind": "meeting_transcript", "recording_id": "abc"},
    )
    sql = (SQL_DIR / "008_fts_url_tokens.sql").read_text(encoding="utf-8")
    assert "regexp_replace" in sql
    assert "metadata ->> 'url'" in sql


@pytest.mark.unit
def test_portal_join_uses_the_owner_and_not_a_previous_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PORTAL_BASE_URL", raising=False)
    monkeypatch.setenv("GITHUB_REPOSITORY_OWNER", "benchmarktechllc")
    assert portal_home_url() == _HOME
    assert "hussein881" not in (portal_home_url() or "")
    assert (
        join_portal_url(portal_home_url(), "/guide/index/")
        == "https://benchmarktechllc.github.io/bmtechllc/portal/guide/index/"
    )


@pytest.mark.unit
def test_discord_reconcile_redacts_tokens_and_drops_bot_abstentions() -> None:
    fake = "github_pat_" + ("b" * 22)
    sent = datetime(2026, 11, 1, tzinfo=UTC)
    plan = plan_discord_reconcile(
        [
            DiscordSourceMessage(
                "1", "10", "sample-collab", "Ada", f"Old note {fake}.", sent
            ),
            DiscordSourceMessage(
                "1",
                "11",
                "sample-collab",
                "Ask BMTech",
                "There is no information available regarding the portal.",
                sent,
            ),
        ]
    )

    assert plan.deleted_ids == ("11",)
    assert fake not in plan.rewritten[0].message_text
    assert "[REDACTED]" in plan.rewritten[0].message_text
    units = discord_units_from_messages(
        [
            {
                "id": "12",
                "type": 0,
                "author": {"username": "BMTech KB", "bot": True},
                "content": "New documentation published",
                "webhook_id": "1",
            },
            {
                "id": "13",
                "type": 0,
                "author": {"username": "Ada"},
                "content": f"token {_TOKEN}",
            },
        ],
        channel="sample-collab",
    )
    assert len(units) == 1
    assert _TOKEN not in units[0].text
