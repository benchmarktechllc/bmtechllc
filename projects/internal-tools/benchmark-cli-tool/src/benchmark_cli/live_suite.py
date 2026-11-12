"""Index GitHub, then run the live Ask suite and write JSON.

The 69-case file is not in this repository. Pass it with ``--questions``.
Code questions, the 12 name questions, and the 3 saved-answer questions always run.
Chat stays on the configured cheap model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from .agent import run_agent
from .config import CHEAP_MODEL
from .github_sync import DEFAULT_REPO, fetch_pull_requests, head_sha, sync_repository

logger = logging.getLogger(__name__)

CODE_CASES: tuple[dict[str, str], ...] = (
    {"id": "trello-ingest", "question": "where is the Trello ingest code", "path": "trello_sync.py", "symbol": "sync_board"},
    {"id": "source-trust", "question": "how does the Ask bot decide which source to trust", "path": "retrieval.py", "symbol": "RETRIEVAL_SOURCES"},
    {"id": "add-answer", "question": "which file handles Add answer", "path": "discord_ask.py", "symbol": "AddAnswerModal"},
    {"id": "chat-model", "question": "which model does chat use", "path": "config.py", "symbol": "gpt-4o-mini"},
    {"id": "embedding-model", "question": "what embedding model is configured", "path": "config.py", "symbol": "text-embedding-3-small"},
    {"id": "discord-sync", "question": "where is Discord message sync", "path": "discord_sync.py", "symbol": ""},
    {"id": "trello-cron", "question": "which workflow runs Trello ingestion", "path": "trello-ingestion.yml", "symbol": "47 4"},
    {"id": "saved-match", "question": "where is saved-answer question matching", "path": "contributions.py", "symbol": "QUESTION_MATCH_SIMILARITY"},
    {"id": "chunk-store", "question": "where does the Ask bot store chunks", "path": "postgres.py", "symbol": "document_chunks"},
    {"id": "recent-prs", "question": "what changed in the last few PRs", "path": "pulls/", "symbol": ""},
    {"id": "kiosk-database", "question": "what does the kiosk backend use for the database", "path": "", "symbol": ""},
)

NAME_QUESTIONS: tuple[str, ...] = (
    "What tickets are assigned to Dante",
    "Dante's tickets",
    "tickets for Dantino",
    "what's on Dantino12's plate",
    "Tickets assigned to Fanual",
    "fanual's tickets",
    "what tickets does Fano have",
    "Tickets assigned to Anwar",
    "Anwar's tickets please",
    "What is Fanuel currently working on",
    "what is Fano doing right now",
    "Fanuel Mengistu's current work",
)

SAVED_QUESTIONS: tuple[str, ...] = (
    "How many employees does BMTech currently have?",
    "how many people work at bmtech",
    "Who is on the core team?",
)


def load_questions(path: Path | None) -> list[str]:
    """A JSON list of strings, or objects with a question field."""
    if path is None or not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("questions file must be a list or an object with questions.")
    questions: list[str] = []
    for item in rows:
        if isinstance(item, str):
            questions.append(item)
        elif isinstance(item, dict) and item.get("question"):
            questions.append(str(item["question"]))
    return questions


def _view(question: str) -> dict[str, Any]:
    response = run_agent(question)
    citations = [
        {
            "source_file": citation.source_file,
            "chunk_id": citation.chunk_id,
            "quote": citation.quote[:500],
        }
        for citation in response.citations
    ]
    return {
        "question": question,
        "status": response.status,
        "answer": response.answer[:2000],
        "citations": citations,
        "model": CHEAP_MODEL,
    }


def _code_passed(case: dict[str, str], result: dict[str, Any]) -> bool:
    answer = str(result["answer"])
    citations = result["citations"]
    blob = "\n".join(
        [answer, *(item["source_file"] for item in citations), *(item["quote"] for item in citations)]
    )
    if case["id"] == "kiosk-database":
        if "Kiosk MVP" in answer:
            return False
        if any(str(item["source_file"]).startswith("trello/") for item in citations):
            return False
        if result["status"] != "answered":
            return True
        return any("kiosk" in str(item["source_file"]).casefold() for item in citations)
    if case["id"] == "recent-prs":
        return result["status"] == "answered" and "pull/" in blob
    if case["path"] and case["path"] not in blob:
        return False
    if case["symbol"] and case["symbol"] not in blob:
        return False
    return result["status"] == "answered"


def run_suite(questions_path: Path | None) -> dict[str, Any]:
    """Ask every live group and score the checks that do not need a human."""
    code_results = []
    for case in CODE_CASES:
        result = _view(case["question"])
        result["id"] = case["id"]
        result["passed"] = _code_passed(case, result)
        code_results.append(result)
    name_results = [_view(question) for question in NAME_QUESTIONS]
    saved_results = [_view(question) for question in SAVED_QUESTIONS]
    for result in saved_results:
        result["from_contribution"] = any(
            str(item["source_file"]).startswith("user_contributed/") for item in result["citations"]
        )
    suite_questions = load_questions(questions_path)
    suite_results = [_view(question) for question in suite_questions]
    code_pass = sum(bool(item["passed"]) for item in code_results)
    names_pass = sum(item["status"] == "answered" for item in name_results)
    saved_pass = sum(bool(item["from_contribution"]) for item in saved_results)
    suite_pass = sum(item["status"] == "answered" for item in suite_results)
    return {
        "model": CHEAP_MODEL,
        "embedding_model": "text-embedding-3-small",
        "code": {"passed": code_pass, "total": len(code_results), "cases": code_results},
        "names": {"passed": names_pass, "total": len(name_results), "cases": name_results},
        "saved": {"passed": saved_pass, "total": len(saved_results), "cases": saved_results},
        "suite": {
            "passed": suite_pass,
            "total": len(suite_results),
            "source": str(questions_path) if questions_path and questions_path.is_file() else None,
            "cases": suite_results,
        },
        "bar": {
            "suite_at_least_55": None if not suite_results else suite_pass >= 55 and len(suite_results) >= 69,
            "suite": f"{suite_pass}/{len(suite_results)}" if suite_results else "not_provided",
            "names_12_of_12": names_pass == len(name_results),
            "names": f"{names_pass}/{len(name_results)}",
            "saved_3_of_3": saved_pass == len(saved_results),
            "saved": f"{saved_pass}/{len(saved_results)}",
            "code": f"{code_pass}/{len(code_results)}",
            "model": CHEAP_MODEL,
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--sha", default="")
    parser.add_argument("--questions", type=Path, default=None, help="69-case JSON file.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-pulls", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report: dict[str, Any] = {"index": None, "error": None}
    if not args.skip_index:
        sha = args.sha.strip() or head_sha(root)
        pulls: list[dict[str, Any]] = []
        if not args.skip_pulls:
            try:
                from .github_sync import _token

                pulls = fetch_pull_requests(args.repo, _token())
            except Exception:
                logger.warning("Pull request list was not fetched. File index will still run.")
        report["index"] = sync_repository(root, args.repo, sha, pulls=pulls)
        report["index"]["sha"] = sha
    report.update(run_suite(args.questions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
