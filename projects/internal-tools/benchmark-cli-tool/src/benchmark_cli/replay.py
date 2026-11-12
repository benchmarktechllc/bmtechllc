"""Replay recorded retrieval and model JSON without the network.

``record`` is the only command that calls the model or the database. The
paraphrase file and ``replay`` stay offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

from .agent import run_agent
from .models import AnswerResponse, Citation
from .query_understanding import QueryPlan, understand_query
from .retrieval import HybridSearchResult, RetrievalDecision, retrieve_evidence
from .storage.postgres import SearchChunk
from .trello_answers import TrelloCardRow

logger = logging.getLogger(__name__)


def sample_board() -> tuple[TrelloCardRow, ...]:
    """Six synthetic cards. Member lines use raw Trello names, not aliases."""
    specs = (
        ("Doing", "Ship the portal", "ship", "Dantino12"),
        ("Done", "Old portal import", "old", "Dantino12"),
        ("Doing", "Kiosk MVP", "kiosk", "Fanuel Mengistu"),
        ("Done", "Retired kiosk notes", "retired", "Fanuel Mengistu"),
        ("Backlog", "Review the lobby copy", "lobby", "Anwar Hussein"),
        ("Doing", "Unassigned sample", "open", ""),
    )
    rows: list[TrelloCardRow] = []
    for index, (list_name, title, card_id, member) in enumerate(specs, start=1):
        url = f"https://trello.com/c/{card_id}"
        lines = [f"Card: {title}", f"List: {list_name}", f"URL: {url}"]
        if member:
            lines.append(f"Members: {member}")
        text = "\n".join(lines)
        rows.append(
            TrelloCardRow(
                board="Sample Board",
                list_name=list_name,
                card_id=card_id,
                title=title,
                url=url,
                chunk_id=index,
                source_file=f"trello/tIvVZqio/{card_id}.md",
                content_sha256=hashlib.sha256(card_id.encode()).hexdigest(),
                chunk_text=text,
            )
        )
    return tuple(rows)


def _citations_for(answer: str, rows: Sequence[TrelloCardRow]) -> list[Citation]:
    cited: list[Citation] = []
    folded = answer.casefold()
    for card in rows:
        if card.title.casefold() not in folded and card.url not in answer:
            continue
        quote = next(
            (line.strip() for line in card.chunk_text.splitlines() if line.strip().startswith("Card: ")),
            f"Card: {card.title}",
        )
        cited.append(
            Citation(
                source_file=card.source_file,
                chunk_id=card.chunk_id,
                content_sha256=card.content_sha256,
                quote=quote,
            )
        )
    return cited


def answer_from_recording(payload: dict[str, Any], rows: Sequence[TrelloCardRow]) -> AnswerResponse:
    """Build the recorded model JSON into a response the validator can check."""
    if payload.get("citations"):
        try:
            return AnswerResponse.model_validate(payload)
        except ValueError:
            logger.debug("Recorded citations did not validate; rebuilding them from the board.")
    status = str(payload.get("status") or "no_relevant_information")
    answer = str(payload.get("answer") or "")
    if status != "answered":
        return AnswerResponse(
            status="no_relevant_information",
            answer=answer or "No relevant information was found in the company document library.",
            confidence=0.0,
            citations=[],
        )
    citations = _citations_for(answer, rows)
    if not citations:
        raise ValueError("The recorded answer does not name a card on the board.")
    return AnswerResponse(
        status="answered",
        answer=answer,
        confidence=float(payload.get("confidence") or 0.8),
        citations=citations,
    )


def _decision_from(payload: dict[str, Any] | None) -> RetrievalDecision:
    body = payload or {}
    results = []
    for item in body.get("results") or []:
        if not isinstance(item, dict):
            continue
        chunk = SearchChunk(
            int(item["id"]),
            str(item.get("source_file") or "recorded.txt"),
            int(item.get("chunk_index") or 0),
            str(item.get("chunk_text") or ""),
            dict(item.get("metadata") or {}),
            float(item.get("score") or 0.0),
            content_sha256=item.get("content_sha256"),
        )
        results.append(
            HybridSearchResult(
                chunk,
                float(item.get("rrf_score") or 0.0),
                item.get("vector_score"),
                item.get("fts_score"),
            )
        )
    status = str(body.get("status") or "no_relevant_information")
    return RetrievalDecision(
        status=status,
        results=tuple(results) if status == "accepted" else (),
        top_cosine_similarity=body.get("top_cosine_similarity"),
        keyword_overlap=int(body.get("keyword_overlap") or 0),
    )


def rows_from_payload(payload: dict[str, Any]) -> tuple[TrelloCardRow, ...]:
    """Cards shipped with a recording, or the synthetic board."""
    if payload.get("sample_board"):
        return sample_board()
    rows = []
    for item in payload.get("cards") or []:
        if isinstance(item, dict):
            rows.append(TrelloCardRow(**item))
    return tuple(rows)


def replay_case(
    case: dict[str, Any],
    rows: Sequence[TrelloCardRow] | None = None,
) -> tuple[AnswerResponse, dict[str, Any]]:
    """Run one case against recorded plan, retrieval, and model JSON. No network."""
    catalog = tuple(rows) if rows is not None else sample_board()
    plan = QueryPlan.model_validate(case["plan"])
    decision = _decision_from(case.get("retrieval"))
    if "model_calls" in case:
        model_calls = list(case["model_calls"])
    else:
        model_calls = [case.get("model")]
    trace: dict[str, Any] = {"prompts": []}

    def _understand(question: str, thread_question: str | None = None) -> QueryPlan:
        del question, thread_question
        return plan

    def _retrieve(query: str, **kwargs: Any) -> RetrievalDecision:
        trace["query"] = query
        trace["queries"] = list(kwargs.get("queries") or [query])
        trace["snapshot_present"] = bool(kwargs.get("snapshot_present"))
        return decision

    def _completion(**_kwargs: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=None, content=""))]
        )

    def _structured(**kwargs: Any) -> AnswerResponse | None:
        trace["prompts"].append(kwargs.get("messages"))
        if not model_calls:
            return None
        item = model_calls.pop(0)
        if item is None:
            return None
        if not isinstance(item, dict):
            raise TypeError("Recorded model JSON must be an object or null.")
        return answer_from_recording(item, catalog)

    from . import agent

    with (
        patch.object(agent, "understand_query", _understand),
        patch.object(agent, "retrieve_evidence", _retrieve),
        patch.object(agent, "call_llm", _completion),
        patch.object(agent, "call_llm_structured", _structured),
        patch.object(agent, "classify_query", lambda _question: "cheap"),
        patch.object(agent, "trello_card_rows", lambda: list(catalog)),
        patch.object(agent, "log_agent_diagnostic", lambda **_kwargs: None),
    ):
        response = run_agent(
            str(case["question"]),
            thread_question=case.get("thread_question"),
        )
    return response, trace


def check_expect(response: AnswerResponse, case: dict[str, Any]) -> list[str]:
    """Compare one replayed answer with the expectations stored beside the fixture."""
    expect = case.get("expect") or {}
    failures: list[str] = []
    if "status" in expect and response.status != expect["status"]:
        failures.append(f"status {response.status} != {expect['status']}")
    for title in expect.get("titles") or []:
        if title not in response.answer:
            failures.append(f"missing {title}")
    for title in expect.get("absent") or []:
        if title in response.answer:
            failures.append(f"unexpected {title}")
    positions = [response.answer.find(title) for title in expect.get("order") or []]
    if any(position < 0 for position in positions):
        failures.append("order title missing")
    elif positions != sorted(positions):
        failures.append("cards are out of order")
    return failures


def load_replay(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"{path} must be an object with a cases list.")
    return payload


def replay_file(path: Path) -> list[dict[str, Any]]:
    """Replay every case in a fixture file. Returns one result dict per case."""
    payload = load_replay(path)
    rows = rows_from_payload(payload)
    results = []
    for case in payload["cases"]:
        response, trace = replay_case(case, rows)
        failures = check_expect(response, case)
        results.append(
            {
                "id": case.get("id"),
                "question": case.get("question"),
                "status": response.status,
                "failures": failures,
                "queries": trace.get("queries"),
                "snapshot_present": trace.get("snapshot_present"),
            }
        )
    return results


def _read_questions(path: Path) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        question, separator, thread = line.partition("\t")
        pairs.append((question.strip(), thread.strip() if separator and thread.strip() else None))
    return pairs


def _dump_decision(decision: RetrievalDecision) -> dict[str, Any]:
    results = []
    for result in decision.results:
        chunk = result.chunk
        results.append(
            {
                "id": chunk.id,
                "source_file": chunk.source_file,
                "chunk_index": chunk.chunk_index,
                "chunk_text": chunk.chunk_text,
                "metadata": chunk.metadata,
                "score": chunk.score,
                "content_sha256": chunk.content_sha256,
                "rrf_score": result.rrf_score,
                "vector_score": result.vector_score,
                "fts_score": result.fts_score,
            }
        )
    return {
        "status": decision.status,
        "top_cosine_similarity": decision.top_cosine_similarity,
        "keyword_overlap": decision.keyword_overlap,
        "results": results,
    }


def record_questions(path: Path) -> dict[str, Any]:
    """Call the live agent and store plan, retrieval, model JSON, and the board."""
    from . import agent
    from .llm import call_llm_structured as real_structured
    from .trello_answers import cards_from_rows

    cases: list[dict[str, Any]] = []
    catalog: list[TrelloCardRow] = []

    for question, thread in _read_questions(path):
        captured: dict[str, Any] = {"model_calls": []}

        def _understand(current: str, thread_question: str | None = None) -> QueryPlan:
            plan = understand_query(current, thread_question)
            captured["plan"] = plan.model_dump()
            return plan

        def _retrieve(query: str, **kwargs: Any) -> RetrievalDecision:
            decision = retrieve_evidence(query, **kwargs)
            captured["retrieval"] = _dump_decision(decision)
            return decision

        def _structured(**kwargs: Any) -> AnswerResponse:
            parsed = real_structured(**kwargs)
            if getattr(parsed, "status", None) is not None and not isinstance(parsed, QueryPlan):
                captured["model_calls"].append(parsed.model_dump())
            return parsed

        def _cards() -> list[tuple[object, ...]]:
            from .storage.postgres import trello_card_rows

            raw = trello_card_rows()
            catalog[:] = list(cards_from_rows(raw))
            return raw

        with (
            patch.object(agent, "understand_query", _understand),
            patch.object(agent, "retrieve_evidence", _retrieve),
            patch.object(agent, "call_llm_structured", _structured),
            patch.object(agent, "trello_card_rows", _cards),
        ):
            response = run_agent(question, thread_question=thread)
        cases.append(
            {
                "id": f"live-{len(cases) + 1}",
                "question": question,
                "thread_question": thread,
                "plan": captured.get("plan") or {"searches": [question], "person": None},
                "retrieval": captured.get("retrieval")
                or {"status": "no_relevant_information", "top_cosine_similarity": None, "keyword_overlap": 0, "results": []},
                "model_calls": captured["model_calls"],
                "final": response.model_dump(),
            }
        )
    return {
        "sample_board": False,
        "cards": [asdict(card) for card in catalog],
        "cases": cases,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("replay", "record"))
    parser.add_argument("--fixtures", type=Path, help="Recorded JSON to replay offline.")
    parser.add_argument("--questions", type=Path, help="Question file. One question per line. Optional tab then thread starter.")
    parser.add_argument("--output", type=Path, help="Where record writes the fixture JSON.")
    args = parser.parse_args()
    if args.command == "replay":
        if args.fixtures is None:
            raise SystemExit("--fixtures is required")
        results = replay_file(args.fixtures)
        failed = [item for item in results if item["failures"]]
        for item in results:
            state = "ok" if not item["failures"] else "FAIL"
            print(f"{state}\t{item['id']}\t{item['question']}")
            for failure in item["failures"]:
                print(f"  {failure}")
        print(f"{len(results) - len(failed)}/{len(results)} passed")
        if failed:
            raise SystemExit(1)
        return
    if args.questions is None or args.output is None:
        raise SystemExit("--questions and --output are required")
    payload = record_questions(args.questions)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Recorded {len(payload['cases'])} questions to {args.output}")


if __name__ == "__main__":
    main()
