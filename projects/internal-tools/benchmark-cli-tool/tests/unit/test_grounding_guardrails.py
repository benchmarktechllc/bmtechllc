"""Adversarial checks for classifiers, headcount, roster, and Discord retention."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from benchmark_cli import agent
from benchmark_cli.acceptance import _check, _outcome
from benchmark_cli.discord_sync import plan_discord_reconcile
from benchmark_cli.grounding import (
    asks_for_core_team,
    asks_for_employee_count,
    asks_for_portal_link,
    asks_for_synced_channels,
    stated_employee_count_answer,
)
from benchmark_cli.ingestion.clean import discord_message_is_authoritative, redact_secrets
from benchmark_cli.models import AnswerResponse, Citation
from benchmark_cli.retrieval import HybridSearchResult, RetrievalDecision
from benchmark_cli.storage.postgres import DiscordSourceMessage, SearchChunk

_DIGEST = "b" * 64
_HOME = "https://benchmarktechllc.github.io/bmtechllc/portal/"


def _result(text: str) -> HybridSearchResult:
    return HybridSearchResult(
        SearchChunk(3, "discord-channel-1.json", 0, text, {"source_type": "discord"}, 1.0, content_sha256=_DIGEST),
        0.1,
        vector_score=0.8,
        fts_score=1.0,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "question",
    (
        "Where is the deployment guide in the knowledge base?",
        "What did Discord channel #ops decide about deploy?",
        "What is the current employee handbook?",
        "What did the core team decide about pricing?",
        "What's the URL for the billing page in our KB?",
        "How many staff are on the on-call rotation?",
        "Who did the core team hire?",
        "How do I index a Discord channel?",
        "Send me the link to the portal status page",
        "Please open the portal and tell me who is on call",
        "How many employees does Acme have?",
        "How many employees joined last quarter?",
        "Who should we hire next for the core team?",
        "How do I index a new Discord channel?",
    ),
)
def test_ordinary_prompts_are_not_intercepted(question: str) -> None:
    assert asks_for_portal_link(question) is False
    assert asks_for_synced_channels(question) is False
    assert asks_for_employee_count(question) is False
    assert asks_for_core_team(question) is False


@pytest.mark.unit
def test_catalog_questions_still_match() -> None:
    assert asks_for_portal_link("What is the link to the knowledge base portal?")
    assert asks_for_portal_link("Where do I open the knowledge base?")
    assert asks_for_portal_link("What's the URL for our KB?")
    assert asks_for_employee_count("How many employees does BMTech currently have?")
    assert asks_for_employee_count("What is our current headcount?")
    assert asks_for_core_team("Who are the core team members at BMTech?")
    assert asks_for_core_team("Who is on the core team?")
    assert asks_for_synced_channels("Which Discord channels are synced?")
    assert asks_for_synced_channels("What Discord channels do we ingest?")


@pytest.mark.unit
def test_negated_conditional_and_other_party_counts_are_not_headcount() -> None:
    samples = (
        "BMTech hasn't got 12 employees.",
        "BMTech does not have 12 employees.",
        "When the company has 10 employees, file the form.",
        "Acme has 10 employees while BMTech integrates the product.",
        "The client has 50 employees.",
        "BMTech has one employee handbook.",
        "BMTech has 12 employee benefits.",
        "BMTech would have 12 employees.",
        "BMTech might have 12 employees.",
        "BMTech has 12 employees if the plan is approved.",
        "BMTech had 12 employees in 2019.",
        "The company has 10 employees.",
    )
    assert stated_employee_count_answer(tuple(_result(text) for text in samples)) is None
    stated = stated_employee_count_answer((_result("BMTech currently has 12 employees."),))
    assert stated is not None
    assert stated.citations[0].quote == "BMTech currently has 12 employees."


@pytest.mark.unit
def test_heading_and_denial_are_not_a_core_team_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "call_llm", lambda **_kwargs: pytest.fail("roster extractor must run first"))
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: RetrievalDecision(
        "accepted",
        (
            _result("Core team"),
            _result("We do not have a core team."),
            _result("The core team is meeting tomorrow."),
            _result("Our core team is ready."),
            _result("The core team is Responsible for deployments."),
            _result("The core team members are in Boston."),
        ),
        0.8,
        2,
    ))

    response = agent.run_agent("Who is on the core team?")

    assert response.status == "no_relevant_information"
    assert response.citations == []
    assert "does not name a core team" in response.answer.casefold()
    assert "Ada" not in response.answer
    assert "Boston" not in response.answer


@pytest.mark.unit
def test_answered_status_alone_does_not_prove_employee_or_roster_facts() -> None:
    uncertain = {
        "status": "answered",
        "answer": "The retrieved evidence does not state a current employee count. Ada is a collaborator.",
        "citations": [{"source_file": "discord-channel-1.json", "chunk_id": 1, "quote": "Ada collaborates."}],
    }
    denial = {
        "status": "answered",
        "answer": "The retrieved evidence does not name a core team.",
        "citations": [{"source_file": "notes.md", "chunk_id": 2, "quote": "Core team"}],
    }
    employee = _outcome("How many employees does BMTech currently have?", uncertain, [])
    roster = _outcome("Who are the core team members at BMTech?", denial, [])
    assert employee == {"outcome": "unknown", "proves_fact": False, "expectation": "unknown_ok"}
    assert roster["proves_fact"] is False
    assert roster["outcome"] == "unknown"
    assert _check("How many employees does BMTech currently have?", uncertain, home=_HOME) == []


@pytest.mark.unit
def test_reconcile_keeps_human_messages_and_drops_only_exact_bot_templates() -> None:
    sent = datetime(2026, 11, 1, tzinfo=UTC)
    kept = (
        "There is no information available regarding Q3 plan until Friday.",
        "New documentation published for API.",
        "Ship portal fix today.",
    )
    names = ("Marci", "Deploy Bot", "CI", "my-bot", "user_github", "x-ci")
    texts = (
        *kept,
        "New documentation published",
        "Ship portal fix today.",
    )
    for name in names:
        for text in texts:
            assert discord_message_is_authoritative(text, name) is True
    plan = plan_discord_reconcile(
        [
            DiscordSourceMessage("1", "1", "ops", "Marci", kept[0], sent),
            DiscordSourceMessage("1", "2", "ops", "Marci", kept[1], sent),
            DiscordSourceMessage("1", "3", "ops", "Deploy Bot", kept[2], sent),
            DiscordSourceMessage("1", "4", "ops", "CI", "New documentation published", sent),
            DiscordSourceMessage("1", "5", "ops", "x-ci", "Ship portal fix today.", sent),
        ]
    )
    assert plan.deleted_ids == ()
    assert plan.rewritten == ()


@pytest.mark.unit
def test_ordinary_prompt_reaches_the_model_with_redacted_tool_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = "The deployment guide is the runbook page."
    secret = "ghp_" + ("c" * 36)
    chunk = SearchChunk(
        9,
        "portal/runbooks/deploy.md",
        0,
        f"Token {secret}. {evidence}",
        {"source_type": "kb_page", "url": "/runbooks/deploy/"},
        1.0,
        content_sha256=_DIGEST,
    )
    decision = RetrievalDecision(
        "accepted",
        (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
        0.9,
        1,
    )
    seen: list[str] = []

    def fake_llm(**kwargs: object) -> SimpleNamespace:
        seen.append(str(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))])

    def fake_structured(**_kwargs: object) -> AnswerResponse:
        return AnswerResponse(
            status="answered",
            answer="The deployment guide is the runbook page.",
            confidence=0.8,
            citations=[
                Citation(
                    source_file=chunk.source_file,
                    chunk_id=chunk.id,
                    content_sha256=_DIGEST,
                    quote=evidence,
                )
            ],
        )

    monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
    monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: decision)
    monkeypatch.setattr(agent, "call_llm", fake_llm)
    monkeypatch.setattr(agent, "call_llm_structured", fake_structured)

    response = agent.run_agent("Where is the deployment guide in the knowledge base?")

    assert seen
    assert secret not in seen[0]
    assert "[REDACTED]" in seen[0]
    assert "not established as the requested document" in response.answer.casefold()
    assert evidence in response.answer
    assert secret not in response.answer
    assert _HOME not in response.answer


def _cited(chunk: SearchChunk, quote: str) -> AnswerResponse:
    return AnswerResponse(
        status="answered",
        answer="placeholder",
        confidence=0.9,
        citations=[
            Citation(
                source_file=chunk.source_file,
                chunk_id=chunk.id,
                content_sha256=_DIGEST,
                quote=quote,
            )
        ],
    )


@pytest.mark.unit
def test_plans_and_proposals_are_not_reported_as_completed_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = (
        "Kiosk next work is MVP definition and setting the timeline. "
        "Prototype todos remain open. "
        "The revised project proposal covers the architecture."
    )
    done = "BMTech approved the launch and the decision is final."
    plan_chunk = SearchChunk(205, "meeting.txt", 0, planning, {"source_type": "transcript", "kind": "meeting_transcript", "meeting": "sample", "title": "Planning", "url": "https://example.test/plan", "recording_id": "rec"}, 1.0, content_sha256=_DIGEST)
    done_chunk = SearchChunk(206, "meeting.txt", 1, done, {"source_type": "transcript", "kind": "meeting_transcript", "meeting": "sample", "title": "Decision", "url": "https://example.test/done", "recording_id": "rec"}, 1.0, content_sha256="c" * 64)

    def answer_for(question: str, chunk: SearchChunk, claim: str) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
                0.9,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: _cited(chunk, chunk.chunk_text).model_copy(update={"answer": claim}),
        )
        return agent.run_agent(question)

    qualified = answer_for(
        "What kiosk decisions are complete?",
        plan_chunk,
        "The MVP and timeline were defined, and the proposal architecture was decided.",
    )
    assert "does not explicitly state" in qualified.answer.casefold()
    assert "were defined" not in qualified.answer.casefold()
    assert "was decided" not in qualified.answer.casefold()
    assert "setting the timeline" in qualified.answer

    kept = answer_for(
        "What launch decision was approved?",
        done_chunk,
        "The launch was approved.",
    )
    assert kept.answer == "The launch was approved."


@pytest.mark.unit
def test_closest_page_is_not_renamed_as_the_requested_document(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    article = SearchChunk(
        110,
        "portal/guide/lifecycle.md",
        0,
        "This article explains a forward-deployed lifecycle. It is not a runbook.",
        {
            "source_type": "kb_page",
            "title": "Forward-deployed lifecycle",
            "url": "https://example.test/guide/lifecycle/",
        },
        1.0,
        content_sha256=_DIGEST,
    )
    runbook = SearchChunk(
        615,
        "portal/runbooks/index.md",
        0,
        "These runbooks and playbooks are the operating set.",
        {
            "source_type": "kb_page",
            "title": "Playbooks and runbooks",
            "url": "https://example.test/runbooks/",
        },
        1.0,
        content_sha256="d" * 64,
    )

    def answer_for(question: str, chunk: SearchChunk, claim: str) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                (HybridSearchResult(chunk, 0.2, vector_score=0.9, fts_score=1.0),),
                0.9,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]),
        )
        monkeypatch.setattr(
            agent,
            "call_llm_structured",
            lambda **_k: _cited(chunk, chunk.chunk_text).model_copy(update={"answer": claim}),
        )
        return agent.run_agent(question)

    guide = answer_for(
        "Where is the deployment guide?",
        article,
        "The deployment guide is the forward-deployed lifecycle article.",
    )
    assert "not established as the requested document" in guide.answer.casefold()
    assert "closest cited page" in guide.answer.casefold()
    assert "forward-deployed lifecycle" in guide.answer.casefold()
    assert "the deployment guide is the forward-deployed" not in guide.answer.casefold()

    books = answer_for(
        "Where are the playbooks and runbooks?",
        runbook,
        "The playbooks and runbooks are the operating set.",
    )
    assert books.answer == "The playbooks and runbooks are the operating set."


@pytest.mark.unit
def test_discord_shaped_tokens_are_redacted_before_the_model() -> None:
    discord_token = "M" + ("a" * 23) + "." + ("b" * 6) + "." + ("c" * 27)
    mfa = "mfa." + ("d" * 40)
    redacted = redact_secrets(f"bot {discord_token} and {mfa} and version v1.2.3")
    assert discord_token not in redacted
    assert mfa not in redacted
    assert "v1.2.3" in redacted
    assert redacted.count("[REDACTED]") == 2


@pytest.mark.unit
def test_orphan_citation_marker_is_removed_and_unrelated_hire_abstains(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planning = "Kiosk next work is MVP definition and setting the timeline. The revised project proposal covers architecture."
    prototype = "A demonstrated prototype exists only as a note in this planning paragraph."
    plan_chunk = SearchChunk(
        110,
        "notes/planning.md",
        0,
        planning,
        {"source_type": "kb_page", "title": "Planning note", "url": "https://example.test/planning/"},
        1.0,
        content_sha256=_DIGEST,
    )
    other = SearchChunk(
        200,
        "notes/other.md",
        0,
        prototype,
        {"source_type": "kb_page", "title": "Prototype note", "url": "https://example.test/prototype/"},
        1.0,
        content_sha256="e" * 64,
    )

    def run(question: str, chunks: tuple[SearchChunk, ...], claim: str, quotes: list[str]) -> AnswerResponse:
        monkeypatch.setattr(agent, "log_agent_diagnostic", lambda **_kwargs: None)
        monkeypatch.setattr(agent, "classify_query", lambda _question: "cheap")
        monkeypatch.setattr(
            agent,
            "retrieve_evidence",
            lambda *_a, **_k: RetrievalDecision(
                "accepted",
                tuple(HybridSearchResult(chunk, 0.2, vector_score=0.8, fts_score=1.0) for chunk in chunks),
                0.8,
                1,
            ),
        )
        monkeypatch.setattr(
            agent,
            "call_llm",
            lambda **_k: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
            ),
        )

        def structured(**_kwargs: object) -> AnswerResponse:
            return AnswerResponse(
                status="answered",
                answer=claim,
                confidence=0.9,
                citations=[
                    Citation(
                        source_file=chunk.source_file,
                        chunk_id=chunk.id,
                        content_sha256=chunk.content_sha256 or _DIGEST,
                        quote=quote,
                    )
                    for chunk, quote in zip(chunks, quotes, strict=True)
                ],
            )

        monkeypatch.setattr(agent, "call_llm_structured", structured)
        return agent.run_agent(question)

    kiosk = run(
        "What kiosk decisions are complete?",
        (plan_chunk, other),
        "Architecture decisions were made for the kiosk [1]. The demonstrated prototype shipped [3].",
        [planning, prototype],
    )
    assert "does not explicitly state" in kiosk.answer.casefold()
    assert "[3]" not in kiosk.answer
    assert "demonstrated prototype shipped" not in kiosk.answer.casefold()
    assert [citation.chunk_id for citation in kiosk.citations] == [110, 200]

    decided = SearchChunk(
        300,
        "notes/decision.md",
        0,
        "The group decided to keep the prototype.",
        {"source_type": "kb_page", "title": "Decision note"},
        1.0,
        content_sha256="f" * 64,
    )
    kept = run(
        "What did the group decide about the prototype?",
        (decided,),
        "The group decided to keep the prototype.",
        ["The group decided to keep the prototype."],
    )
    assert kept.answer == "The group decided to keep the prototype."

    hire = run(
        "Who should we hire next for the core team?",
        (plan_chunk,),
        "Hire Nick to work on ingestion.",
        [planning],
    )
    assert hire.status == "no_relevant_information"
    assert hire.citations == []
    assert "Nick" not in hire.answer
    assert "does not address" in hire.answer.casefold()

    schedule = SearchChunk(
        521,
        "notes/schedule.md",
        0,
        "A scheduler can be triggered manually. Suggest a bot if you want automation.",
        {"source_type": "kb_page", "title": "Scheduler"},
        1.0,
        content_sha256="a" * 64,
    )
    howto = run(
        "How do I index a new Discord channel?",
        (schedule,),
        "Trigger the ingestion workflow and add a bot.",
        ["A scheduler can be triggered manually. Suggest a bot if you want automation."],
    )
    assert howto.status == "no_relevant_information"
    assert "ingestion workflow" not in howto.answer.casefold()
    assert "unknown" in howto.answer.casefold()
    assert howto.citations == []

    lifecycle = SearchChunk(
        110,
        "portal/guide/lifecycle.md",
        0,
        "Later we might write a deployment guide. This article explains a lifecycle.",
        {
            "source_type": "kb_page",
            "title": "Forward-deployed lifecycle",
            "url": "https://example.test/guide/lifecycle/",
        },
        1.0,
        content_sha256=_DIGEST,
    )
    guide = run(
        "Where is the deployment guide?",
        (lifecycle,),
        "The deployment guide is the forward-deployed lifecycle article.",
        ["Later we might write a deployment guide. This article explains a lifecycle."],
    )
    assert "not established as the requested document" in guide.answer.casefold()
    assert "the deployment guide is the forward-deployed" not in guide.answer.casefold()


@pytest.mark.unit
def test_tool_results_redact_secrets_before_they_reach_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "ghp_" + ("d" * 36)
    chunk = SearchChunk(
        4,
        "discord-channel-1.json",
        0,
        f"Announcement {secret} stays in the database row.",
        {"source_type": "discord"},
        1.0,
        content_sha256=_DIGEST,
    )
    decision = RetrievalDecision(
        "accepted",
        (HybridSearchResult(chunk, 0.1, vector_score=0.8, fts_score=1.0),),
        0.8,
        1,
    )
    monkeypatch.setattr(agent, "retrieve_evidence", lambda *_a, **_k: decision)
    evidence: dict[int, SearchChunk] = {}

    payload = agent._execute_tool("search_docs", {"query": "announcement"}, evidence)

    assert secret not in payload
    assert "[REDACTED]" in payload
    assert secret not in evidence[4].chunk_text
