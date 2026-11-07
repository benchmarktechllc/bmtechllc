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

from benchmark_cli import agent
from benchmark_cli.corpus import is_synthetic_canary
from benchmark_cli.discord_sync import plan_discord_reconcile
from benchmark_cli.grounding import ACCEPTANCE_QUESTIONS
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
def test_acceptance_questions_are_the_production_wording() -> None:
    assert ACCEPTANCE_QUESTIONS == (
        "What is the link to the knowledge base portal?",
        "How many employees does BMTech currently have?",
        "Who are the core team members at BMTech?",
        "Which Discord channels are synced?",
    )


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
def test_employee_question_does_not_turn_a_core_team_into_a_headcount(
    monkeypatch: pytest.MonkeyPatch, quiet_agent: None
) -> None:
    evidence = "Ada, Grace, and Lin are the core team for the sample effort."
    chunk = _result(101, "discord-channel-example.json", evidence, {"source_type": "discord"})
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_args, **_kwargs: _decision(chunk))

    response = agent.run_agent(ACCEPTANCE_QUESTIONS[1])

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

    named = agent.run_agent(ACCEPTANCE_QUESTIONS[2])
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
    missing = agent.run_agent(ACCEPTANCE_QUESTIONS[2])
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

    response = agent.run_agent(ACCEPTANCE_QUESTIONS[3])
    rendered = response.model_dump_json()

    assert response.status == "answered"
    assert "#sample-collab" in response.answer
    assert "#sample-planning" in response.answer
    assert "New documentation published" not in rendered
    assert "article" not in response.answer.casefold()


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
