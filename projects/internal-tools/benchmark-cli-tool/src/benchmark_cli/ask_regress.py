"""Ask regression harness: offline identity checks, code retrieval gate, and live bar.

Subcommands
-----------
capture         Run every private non-code case with the answer model "down" and record
                the retrieval trace, prompt hashes, the model-down response, the outcome,
                every embedding vector, and the ``document_chunks`` fingerprint.
check           Replay the same cases on another build with zero OpenAI calls (embeddings
                come from the capture) and fail on any difference.
code-retrieval  Rank the lane's chunks for the code cases. Embeddings only, no chat.
record          Store the code lane's chunks and raw model drafts for the code cases.
replay          Run the lane validator over a recording. Zero OpenAI calls.
live            Saved, names, code, then suite against the real model, with early stop.

The private case files (suite, names, saved, baseline results) are passed by
directory (``--private-dir`` or ``ASK_PRIVATE_DIR``). Every output is private, and
the harness refuses to write inside a git working tree.

This file uses absolute imports only, so it also runs by path inside an older
wheel that does not ship it (``python /work/ask_regress.py capture ...``).
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    # Run by path from inside the package directory, the sibling ``logging.py``
    # would shadow the standard library. Drop the script directory first.
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [item for item in sys.path if os.path.abspath(item or ".") != _HERE]

import argparse
import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest import mock

logger = logging.getLogger("benchmark_cli.ask_regress")

REQUIRED_CHAT_MODEL = "gpt-4o-mini"
REQUIRED_EMBEDDING_MODEL = "text-embedding-3-small"

SUITE_BAR = 56
NAMES_BAR = 12
CODE_ORIGINAL_BAR = 9
CODE_MAX_ORIGINAL_FAILURES = 2
MAX_CHAT_CALLS = 700
MAX_PROMPT_TOKENS = 8_000_000
MAX_SECONDS = 90 * 60
SCORE_TOLERANCE = 1e-9
CODE_RETRIEVAL_TOP = 3

SAVED_PREFIX = "user_contributed/"
GITHUB_PREFIX = "github/"
MODEL_DOWN = "ask-regress: answer model is down for this offline run"

# Code questions about the public repository. Expected paths are any-of suffixes.
# ``original`` marks the eleven cases the live bar counts; the twelfth is reported only.
CODE_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "trello-ingest",
        "q": "where is the Trello ingest code",
        "expect_paths": ("benchmark_cli/trello_sync.py",),
        "original": True,
    },
    {
        "id": "source-trust",
        "q": "how does the Ask bot decide which source to trust",
        "expect_paths": ("benchmark_cli/retrieval.py", "benchmark_cli/agent.py"),
        "original": True,
    },
    {
        "id": "add-answer",
        "q": "which file handles Add answer",
        "expect_paths": ("benchmark_cli/discord_ask.py",),
        "original": True,
    },
    {
        "id": "chat-model",
        "q": "which model does chat use",
        "expect_paths": ("benchmark_cli/config.py",),
        "original": True,
    },
    {
        "id": "embedding-model",
        "q": "what embedding model is configured",
        "expect_paths": ("benchmark_cli/config.py",),
        "original": True,
    },
    {
        "id": "discord-sync",
        "q": "where is Discord message sync",
        "expect_paths": ("benchmark_cli/discord_sync.py",),
        "original": True,
    },
    {
        "id": "trello-cron",
        "q": "which workflow runs Trello ingestion",
        "expect_paths": (".github/workflows/trello-ingestion.yml",),
        "original": True,
    },
    {
        "id": "saved-match",
        "q": "where is saved-answer question matching",
        "expect_paths": ("benchmark_cli/contributions.py",),
        "original": True,
    },
    {
        "id": "chunk-store",
        "q": "where does the Ask bot store chunks",
        "expect_paths": ("benchmark_cli/storage/postgres.py", "benchmark_cli/storage/sql/001_init.sql"),
        "original": True,
    },
    {
        "id": "recent-prs",
        "q": "what changed in the last few PRs",
        "expect": "pulls",
        "expect_paths": (),
        "original": True,
    },
    {
        "id": "kiosk-database",
        "q": "what does the kiosk backend use for the database",
        "expect": "abstain_or_kiosk_path",
        "expect_paths": (),
        "original": True,
    },
    {
        "id": "migration-document-chunks",
        "q": "which SQL migration creates document_chunks",
        "expect_paths": ("benchmark_cli/storage/sql/001_init.sql",),
        "original": False,
    },
)


class HarnessAbort(BaseException):
    """Stops the run. A BaseException so ``except Exception`` in the agent cannot swallow it."""


class EmbeddingMiss(BaseException):
    """``check`` needed a vector the baseline capture did not record."""


# --------------------------------------------------------------------------- helpers


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ``safety.delimit_evidence`` wraps each chunk in a fresh random nonce per prompt.
_EVIDENCE_NONCE = re.compile(r"(</?untrusted-evidence:)[A-Za-z0-9_-]+>")


def normalize_prompt(text: str) -> str:
    """Prompt text with the per-call evidence nonce fixed, so equal prompts hash equal."""
    return _EVIDENCE_NONCE.sub(r"\1NONCE>", text)


def sha256_json(value: Any) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str))


def prompt_sha256(value: Any) -> str:
    """Hash of model-call messages or tools with evidence nonces normalized."""
    return sha256_text(normalize_prompt(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)))


def ensure_outside_repo(path: Path) -> Path:
    """Refuse any output inside a git working tree (the public repo above all)."""
    resolved = Path(path).expanduser().resolve()
    for ancestor in (resolved, *resolved.parents):
        if (ancestor / ".git").exists():
            raise SystemExit(
                f"Refusing to write {resolved}: it is inside the git working tree {ancestor}. "
                "Harness output is private; write it to the private directory."
            )
    return resolved


def write_json(path: Path, payload: Any) -> Path:
    target = ensure_outside_repo(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1, ensure_ascii=False, default=str) + "\n", "utf-8")
    return target


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def thread_question(question: str, starter: str | None) -> str | None:
    """Same rule as ``discord_ask.thread_question_for`` without importing discord."""
    current = " ".join(question.split())
    prior = " ".join((starter or "").split())
    if not prior or prior == current:
        return None
    return prior


def build_fingerprint() -> dict[str, Any]:
    """Version plus a digest of the installed package source (this file excluded)."""
    import benchmark_cli

    root = Path(benchmark_cli.__file__).resolve().parent
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts:
            continue
        if item.suffix not in {".py", ".sql", ".json"} or item.name == "ask_regress.py":
            continue
        digest.update(item.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    try:
        from importlib.metadata import version

        package_version = version("benchmark-cli-tool")
    except Exception:
        package_version = "unknown"
    return {"version": package_version, "source_sha256": digest.hexdigest()}


def _code_lane() -> Any | None:
    """The candidate's GitHub lane module, or None on builds without one."""
    try:
        from benchmark_cli import github_answers
    except Exception:
        return None
    return github_answers


def classifier() -> Callable[[str], bool] | None:
    lane = _code_lane()
    return getattr(lane, "is_code_question", None) if lane is not None else None


# --------------------------------------------------------------------------- private inputs


@dataclass(frozen=True)
class Case:
    key: str
    group: str
    q: str
    starter: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PrivateInputs:
    suite: tuple[Case, ...]
    names: tuple[Case, ...]
    saved: tuple[Case, ...]
    titles: dict[str, list[str]]
    baseline: dict[str, dict[str, Any]]  # suite case key -> baseline result

    def non_code_cases(self) -> tuple[Case, ...]:
        return (*self.suite, *self.names, *self.saved)


def private_dir(value: str | None) -> Path:
    chosen = value or os.getenv("ASK_PRIVATE_DIR", "")
    if not chosen:
        raise SystemExit("Pass --private-dir or set ASK_PRIVATE_DIR.")
    return Path(chosen).expanduser().resolve()


def load_private(directory: Path) -> PrivateInputs:
    """Read suite.json, names.json, saved.json, and baseline-suite.json."""
    suite_rows = read_json(directory / "suite.json")
    suite = tuple(
        Case(f"suite:{index}", "suite", str(row["q"]), row.get("starter") or None)
        for index, row in enumerate(suite_rows)
    )
    names_payload = read_json(directory / "names.json")
    names = tuple(
        Case(f"names:{index}", "names", str(row["q"]), None, {"person": row["person"]})
        for index, row in enumerate(names_payload["cases"])
    )
    titles = {str(person): list(values) for person, values in names_payload["titles"].items()}
    saved = tuple(
        Case(
            f"saved:{index}",
            "saved",
            str(row["q"]),
            None,
            {"expect": row["expect"], "source_prefix": row.get("source_prefix") or SAVED_PREFIX},
        )
        for index, row in enumerate(read_json(directory / "saved.json"))
    )
    baseline_path = directory / "baseline-suite.json"
    rows: list[dict[str, Any]] = []
    if baseline_path.is_file():
        payload = read_json(baseline_path)
        rows = list(payload.get("results", []) if isinstance(payload, dict) else payload)
    return PrivateInputs(suite, names, saved, titles, align_baseline(suite, rows))


def align_baseline(suite: Sequence[Case], rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Match baseline rows to suite cases by position, else by question and starter.

    The suite can repeat a question, so position wins when it agrees.
    """
    def same(case: Case, row: dict[str, Any]) -> bool:
        return str(row.get("q")) == case.q and (row.get("starter") or None) == case.starter

    aligned: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(suite):
        if index < len(rows) and same(case, rows[index]):
            aligned[case.key] = rows[index]
            continue
        match = next((row for row in rows if same(case, row)), None)
        if match is not None:
            aligned[case.key] = match
    return aligned


# --------------------------------------------------------------------------- OpenAI guard


@dataclass
class Usage:
    chat_calls: int = 0
    embed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    embed_tokens: int = 0
    cost_usd: float = 0.0
    models: dict[str, int] = field(default_factory=dict)
    by_component: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chat_calls": self.chat_calls,
            "embed_calls": self.embed_calls,
            "chat_prompt_tokens": self.prompt_tokens,
            "chat_completion_tokens": self.completion_tokens,
            "embed_tokens": self.embed_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "models": dict(self.models),
            "by_component": self.by_component,
        }


class OpenAIGuard:
    """Enforces gpt-4o-mini for chat, the embedding model, and the run budget."""

    def __init__(self, *, allow_chat: bool, allow_embed: bool, budget: bool = False) -> None:
        self.allow_chat = allow_chat
        self.allow_embed = allow_embed
        self.budget = budget
        self.usage = Usage()
        self.started = time.monotonic()
        self.violation: str | None = None

    # The checks are plain methods so unit tests can drive them without a client.
    def check_chat(self, model: Any) -> None:
        if not self.allow_chat:
            self._abort("A chat request was attempted in an offline mode.")
        if str(model) != REQUIRED_CHAT_MODEL:
            self._abort(f"Chat model {model!r} is not {REQUIRED_CHAT_MODEL}.")
        reason = self.budget_reason()
        if reason:
            self._abort(reason)
        self.usage.chat_calls += 1
        self.usage.models[str(model)] = self.usage.models.get(str(model), 0) + 1

    def check_embed(self, model: Any) -> None:
        if not self.allow_embed:
            self._abort("An embedding request was attempted in an offline mode.")
        if str(model) != REQUIRED_EMBEDDING_MODEL:
            self._abort(f"Embedding model {model!r} is not {REQUIRED_EMBEDDING_MODEL}.")
        self.usage.embed_calls += 1
        self.usage.models[str(model)] = self.usage.models.get(str(model), 0) + 1

    def budget_reason(self) -> str | None:
        if not self.budget:
            return None
        return budget_exceeded(
            self.usage.chat_calls, self.usage.prompt_tokens, time.monotonic() - self.started
        )

    def _abort(self, message: str) -> None:
        self.violation = message
        raise HarnessAbort(message)

    def wrap(self, method: Callable[..., Any], check: Callable[[Any], None]) -> Callable[..., Any]:
        def guarded(*args: Any, **kwargs: Any) -> Any:
            check(kwargs.get("model"))
            return method(*args, **kwargs)

        guarded.__ask_regress_guarded__ = True  # type: ignore[attr-defined]
        return guarded

    def guard_chat_client(self, client: Any) -> Any:
        """Wrap every chat entry point the SDK exposes on this client instance."""
        holders = []
        chat = getattr(client, "chat", None)
        if chat is not None:
            holders.append(getattr(chat, "completions", None))
        beta_chat = getattr(getattr(client, "beta", None), "chat", None)
        if beta_chat is not None:
            holders.append(getattr(beta_chat, "completions", None))
        for holder in holders:
            if holder is None:
                continue
            for name in ("create", "parse"):
                method = getattr(holder, name, None)
                if method is None or getattr(method, "__ask_regress_guarded__", False):
                    continue
                setattr(holder, name, self.wrap(method, self.check_chat))
        return client

    def guard_embed_client(self, client: Any) -> Any:
        holder = getattr(client, "embeddings", None)
        method = getattr(holder, "create", None)
        if method is not None and not getattr(method, "__ask_regress_guarded__", False):
            holder.create = self.wrap(method, self.check_embed)
        return client

    def record_usage(self, tier: str, component: str, prompt: int, completion: int) -> None:
        if tier == "embedding" or component == "embedding":
            self.usage.embed_tokens += prompt
        else:
            self.usage.prompt_tokens += prompt
            self.usage.completion_tokens += completion
        try:
            from benchmark_cli.config import get_model_config
            from benchmark_cli.logging import calculate_cost_usd

            cost = calculate_cost_usd(get_model_config(tier), prompt, completion)
        except Exception:
            cost = 0.0
        self.usage.cost_usd += cost
        bucket = self.usage.by_component.setdefault(
            component, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt
        bucket["completion_tokens"] += completion
        bucket["cost_usd"] = round(bucket["cost_usd"] + cost, 8)


def preflight() -> None:
    """Configured models must be the required ones before any call is made."""
    from benchmark_cli.config import EMBEDDING_MODEL, get_model_config

    chat = get_model_config("cheap").model
    if chat != REQUIRED_CHAT_MODEL:
        raise SystemExit(f"Configured cheap chat model is {chat!r}; the harness requires {REQUIRED_CHAT_MODEL}.")
    if EMBEDDING_MODEL != REQUIRED_EMBEDDING_MODEL:
        raise SystemExit(
            f"Configured embedding model is {EMBEDDING_MODEL!r}; the harness requires {REQUIRED_EMBEDDING_MODEL}."
        )


@contextmanager
def guarded_openai(guard: OpenAIGuard) -> Iterator[OpenAIGuard]:
    """Route every OpenAI client through the guard and tally the usage log."""
    from benchmark_cli import llm
    from benchmark_cli.providers import openai as provider

    real_chat_client = llm._get_client
    real_embed_client = provider._get_client

    def chat_client() -> Any:
        if not guard.allow_chat:
            guard._abort("A chat request was attempted in an offline mode.")
        return guard.guard_chat_client(real_chat_client())

    def embed_client() -> Any:
        if not guard.allow_embed:
            guard._abort("An embedding request was attempted in an offline mode.")
        return guard.guard_embed_client(real_embed_client())

    def usage_wrapper(original: Callable[..., Any]) -> Callable[..., Any]:
        def log_usage(**kwargs: Any) -> Any:
            guard.record_usage(
                str(kwargs.get("tier") or ""),
                str(kwargs.get("component") or ""),
                int(kwargs.get("prompt_tokens") or 0),
                int(kwargs.get("completion_tokens") or 0),
            )
            return original(**kwargs)

        return log_usage

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(llm, "_get_client", chat_client))
        stack.enter_context(mock.patch.object(provider, "_get_client", embed_client))
        stack.enter_context(mock.patch.object(llm, "log_usage", usage_wrapper(llm.log_usage)))
        stack.enter_context(mock.patch.object(provider, "log_usage", usage_wrapper(provider.log_usage)))
        yield guard


# --------------------------------------------------------------------------- embeddings


class EmbeddingStore:
    """Vectors keyed by sha256 of the embedded text.

    Capture memoizes by text: the first vector the provider returns for a text is
    recorded and returned for every later request of the same text. The OpenAI
    embeddings endpoint is not bit-for-bit deterministic, and one capture embeds the
    same string several times (contribution matching, the retrieval query list, the
    same question in more than one case). Without the memo the capture ran on
    vectors that differ from the single one it recorded, so ``check`` on the very
    same build replayed different scores and reordered near-ties.
    """

    def __init__(self, vectors: dict[str, list[float]] | None = None, *, replay: bool = False) -> None:
        self.vectors: dict[str, list[float]] = dict(vectors or {})
        self.replay = replay
        self.misses: list[str] = []

    def wrap(self, original: Callable[[Sequence[str]], list[list[float]]]) -> Callable[..., Any]:
        def embed_texts(texts: Sequence[str]) -> list[list[float]]:
            items = list(texts)
            if not self.replay:
                keys = [sha256_text(text) for text in items]
                pending: dict[str, str] = {}
                for key, text in zip(keys, items, strict=True):
                    if key not in self.vectors:
                        pending.setdefault(key, text)
                if pending:
                    vectors = original(list(pending.values()))
                    if len(vectors) != len(pending):
                        raise RuntimeError("Embedding response length did not match the request.")
                    for key, vector in zip(pending, vectors, strict=True):
                        self.vectors[key] = list(vector)
                return [list(self.vectors[key]) for key in keys]
            found: list[list[float]] = []
            for text in items:
                vector = self.vectors.get(sha256_text(text))
                if vector is None:
                    self.misses.append(text)
                    raise EmbeddingMiss(f"No recorded embedding for a {len(text)}-character text.")
                found.append(list(vector))
            return found

        return embed_texts


@contextmanager
def patched_embeddings(store: EmbeddingStore) -> Iterator[EmbeddingStore]:
    """Replace ``embed_texts`` in every loaded benchmark_cli module that imported it."""
    from benchmark_cli.providers import openai as provider

    original = provider.embed_texts
    wrapper = store.wrap(original)
    with ExitStack() as stack:
        for name, module in list(sys.modules.items()):
            if not (name == "benchmark_cli" or name.startswith("benchmark_cli.")) or module is None:
                continue
            if getattr(module, "embed_texts", None) is original:
                stack.enter_context(mock.patch.object(module, "embed_texts", wrapper))
        yield store


# --------------------------------------------------------------------------- traces


def trace_decision(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    results = []
    for result in decision.results:
        chunk = result.chunk
        results.append(
            {
                "id": chunk.id,
                "sha": chunk.content_sha256,
                "source": chunk.source_file,
                "source_type": str((chunk.metadata or {}).get("source_type") or ""),
                "rrf": result.rrf_score,
                "vector": result.vector_score,
                "fts": result.fts_score,
            }
        )
    return {
        "status": decision.status,
        "top_cosine": decision.top_cosine_similarity,
        "overlap": decision.keyword_overlap,
        "results": results,
    }


def response_view(response: Any) -> dict[str, Any]:
    return {
        "status": response.status,
        "answer_sha256": sha256_text(response.answer),
        "answer_preview": response.answer[:160],
        "citations": [
            {"chunk_id": item.chunk_id, "source_file": item.source_file} for item in response.citations
        ],
    }


def _close(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=SCORE_TOLERANCE)


def diff_trace(base: dict[str, Any] | None, cand: dict[str, Any] | None, label: str) -> list[str]:
    """Differences between two retrieval traces. Empty when identical."""
    if base is None or cand is None:
        return [] if base is cand else [f"{label}: one build had no retrieval"]
    diffs: list[str] = []
    if base["status"] != cand["status"]:
        diffs.append(f"{label}: status {base['status']} -> {cand['status']}")
    if not _close(base["top_cosine"], cand["top_cosine"]):
        diffs.append(f"{label}: top cosine {base['top_cosine']} -> {cand['top_cosine']}")
    if base["overlap"] != cand["overlap"]:
        diffs.append(f"{label}: keyword overlap {base['overlap']} -> {cand['overlap']}")
    base_ids = [row["id"] for row in base["results"]]
    cand_ids = [row["id"] for row in cand["results"]]
    if base_ids != cand_ids:
        missing = [item for item in base_ids if item not in cand_ids]
        added = [item for item in cand_ids if item not in base_ids]
        diffs.append(
            f"{label}: chunk ids differ; baseline={base_ids} candidate={cand_ids} "
            f"missing={missing} added={added}"
        )
        return diffs
    for left, right in zip(base["results"], cand["results"], strict=True):
        if left["sha"] != right["sha"] or left["source_type"] != right["source_type"]:
            diffs.append(f"{label}: chunk {left['id']} content or source type differs")
        for score in ("rrf", "vector", "fts"):
            if not _close(left[score], right[score]):
                diffs.append(f"{label}: chunk {left['id']} {score} {left[score]} -> {right[score]}")
    return diffs


def diff_case(base: dict[str, Any], cand: dict[str, Any]) -> list[str]:
    """Every field ``check`` requires to be identical for a non-code case."""
    diffs: list[str] = []
    if base["q"] != cand["q"] or base.get("starter") != cand.get("starter"):
        return ["case question differs; the private files changed between capture and check"]
    if base.get("error") != cand.get("error"):
        diffs.append(f"error {base.get('error')!r} -> {cand.get('error')!r}")
    base_r, cand_r = base.get("retrievals") or [], cand.get("retrievals") or []
    if len(base_r) != len(cand_r):
        diffs.append(f"retrieval count {len(base_r)} -> {len(cand_r)}")
    for index, (left, right) in enumerate(zip(base_r, cand_r, strict=False)):
        if left.get("args") != right.get("args"):
            diffs.append(f"retrieval[{index}]: search strings differ")
        diffs.extend(diff_trace(left.get("decision"), right.get("decision"), f"retrieval[{index}]"))
    if base.get("phrase_lookups") != cand.get("phrase_lookups"):
        diffs.append("supplemental phrase lookups differ")
    if base.get("model_calls") != cand.get("model_calls"):
        diffs.append(
            "model-call prompts differ: baseline="
            f"{[c.get('messages_sha256', '')[:12] for c in base.get('model_calls') or []]} candidate="
            f"{[c.get('messages_sha256', '')[:12] for c in cand.get('model_calls') or []]}"
        )
    left_resp, right_resp = base.get("response") or {}, cand.get("response") or {}
    for key in ("status", "answer_sha256", "citations"):
        if left_resp.get(key) != right_resp.get(key):
            diffs.append(f"model-down response {key} differs")
    if base.get("outcome") != cand.get("outcome"):
        diffs.append(f"outcome {base.get('outcome')} -> {cand.get('outcome')}")
    return diffs


FINGERPRINT_SQL = (
    "SELECT count(*), max(id), md5(string_agg(content_sha256, ',' ORDER BY id)) "
    "FROM document_chunks WHERE metadata ->> 'source_type' IS DISTINCT FROM 'github'"
)
GITHUB_ROWS_SQL = "SELECT count(*) FROM document_chunks WHERE metadata ->> 'source_type' = 'github'"


def corpus_fingerprint() -> dict[str, Any]:
    """Company rows of document_chunks. GitHub rows an older build wrote there are counted apart,
    so a database carrying that index can still be compared and its displacement named."""
    from benchmark_cli.storage.postgres import connection

    with connection() as conn, conn.cursor() as cursor:
        cursor.execute(FINGERPRINT_SQL)
        count, max_id, digest = cursor.fetchone()
        cursor.execute(GITHUB_ROWS_SQL)
        github_rows = int(cursor.fetchone()[0])
    return {
        "count": int(count),
        "max_id": None if max_id is None else int(max_id),
        "md5": digest,
        "github_rows_in_document_chunks": github_rows,
    }


def fingerprint_matches(base: dict[str, Any], cand: dict[str, Any]) -> bool:
    return (base.get("count"), base.get("max_id"), base.get("md5")) == (
        cand.get("count"),
        cand.get("max_id"),
        cand.get("md5"),
    )


@contextmanager
def offline_agent(record: dict[str, Any], plan: Any) -> Iterator[None]:
    """Pin the query plan, stub the answer model, and record what the agent did."""
    from benchmark_cli import agent

    real_retrieve = agent.retrieve_evidence
    real_fts = agent.fts_search

    def retrieve(*args: Any, **kwargs: Any) -> Any:
        decision = real_retrieve(*args, **kwargs)
        record["retrievals"].append(
            {
                "args": {
                    "query": args[0] if args else kwargs.get("query"),
                    "queries": list(kwargs.get("queries") or []),
                    "snapshot_present": bool(kwargs.get("snapshot_present")),
                },
                "decision": trace_decision(decision),
            }
        )
        return decision

    def fts(phrase: str, *args: Any, **kwargs: Any) -> Any:
        rows = real_fts(phrase, *args, **kwargs)
        record["phrase_lookups"].append({"phrase": phrase, "ids": [row.id for row in rows]})
        return rows

    def model_stub(name: str) -> Callable[..., Any]:
        def stub(**kwargs: Any) -> Any:
            record["model_calls"].append(
                {
                    "fn": name,
                    "component": kwargs.get("component"),
                    "messages_sha256": prompt_sha256(kwargs.get("messages")),
                    "tools_sha256": prompt_sha256(kwargs.get("tools")) if kwargs.get("tools") else None,
                }
            )
            raise RuntimeError(MODEL_DOWN)

        return stub

    def diagnostic(**kwargs: Any) -> None:
        record["outcome"] = kwargs.get("outcome")
        record["attempts"] = kwargs.get("synthesis_attempts")

    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(agent, "understand_query", lambda *_a, **_k: plan))
        stack.enter_context(mock.patch.object(agent, "retrieve_evidence", retrieve))
        stack.enter_context(mock.patch.object(agent, "fts_search", fts))
        stack.enter_context(mock.patch.object(agent, "call_llm", model_stub("call_llm")))
        stack.enter_context(
            mock.patch.object(agent, "call_llm_structured", model_stub("call_llm_structured"))
        )
        stack.enter_context(mock.patch.object(agent, "log_agent_diagnostic", diagnostic))
        lane = _code_lane()
        if lane is not None and hasattr(lane, "call_llm_structured"):
            stack.enter_context(
                mock.patch.object(lane, "call_llm_structured", model_stub("code_lane"))
            )
        yield


def run_offline_case(case: Case, plan: Any) -> dict[str, Any]:
    """One case with the answer model down. Returns the trace record."""
    from benchmark_cli import agent

    record: dict[str, Any] = {
        "key": case.key,
        "group": case.group,
        "q": case.q,
        "starter": case.starter,
        "plan": plan.model_dump() if hasattr(plan, "model_dump") else dict(plan),
        "retrievals": [],
        "phrase_lookups": [],
        "model_calls": [],
        "response": None,
        "outcome": None,
        "attempts": None,
        "error": None,
    }
    tq = thread_question(case.q, case.starter)
    with offline_agent(record, plan):
        try:
            response = agent.run_agent(case.q, thread_question=tq) if tq else agent.run_agent(case.q)
        except EmbeddingMiss as exc:
            record["error"] = f"embedding_miss: {exc}"
        except Exception as exc:  # recorded and compared, not fatal
            record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        else:
            record["response"] = response_view(response)
    return record


def stored_plan(case: Case, mode: str) -> Any:
    from benchmark_cli.query_understanding import fallback_plan, understand_query

    tq = thread_question(case.q, case.starter)
    if mode == "understand":
        return understand_query(case.q, tq)
    return fallback_plan(case.q, tq)


def plan_from_dict(payload: dict[str, Any]) -> Any:
    from benchmark_cli.query_understanding import QueryPlan

    return QueryPlan(**payload)


# --------------------------------------------------------------------------- capture / check


def command_capture(args: argparse.Namespace) -> int:
    out = ensure_outside_repo(args.out)
    inputs = load_private(private_dir(args.private_dir))
    preflight()
    guard = OpenAIGuard(allow_chat=args.plans == "understand", allow_embed=True)
    store = EmbeddingStore()
    from benchmark_cli import agent  # noqa: F401  (load modules before patching embed_texts)

    cases = []
    with guarded_openai(guard), patched_embeddings(store):
        fingerprint = corpus_fingerprint()
        for case in inputs.non_code_cases():
            plan = stored_plan(case, args.plans)
            cases.append(run_offline_case(case, plan))
            logger.info("captured %s outcome=%s", case.key, cases[-1]["outcome"])
        after = corpus_fingerprint()
    if not fingerprint_matches(fingerprint, after):
        raise SystemExit("document_chunks changed during capture; rerun on a frozen database.")
    payload = {
        "kind": "ask-regress-trace",
        "version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "build": build_fingerprint(),
        "plans": args.plans,
        "fingerprint": fingerprint,
        "usage": guard.usage.as_dict(),
        "cases": cases,
        "embeddings": store.vectors,
    }
    write_json(out, payload)
    print(f"capture: {len(cases)} cases, {len(store.vectors)} vectors -> {out}")
    return 0


def evaluate_check(
    baseline: dict[str, Any],
    candidate_cases: Sequence[dict[str, Any]],
    *,
    fingerprint: dict[str, Any],
    is_code: Callable[[str], bool] | None,
) -> dict[str, Any]:
    """Pure comparison used by ``check``. Returns the report body."""
    if not fingerprint_matches(baseline["fingerprint"], fingerprint):
        return {
            "green": False,
            "fingerprint_ok": False,
            "message": "document_chunks fingerprint differs from the baseline capture; recapture the baseline.",
            "cases": [],
        }
    by_key = {case["key"]: case for case in baseline["cases"]}
    rows = []
    for cand in candidate_cases:
        base = by_key.get(cand["key"])
        diffs = ["case missing from baseline capture"] if base is None else diff_case(base, cand)
        classified = None
        if is_code is not None:
            classified = bool(is_code(cand["q"]))
            if classified:
                diffs.append("is_code_question is True for a non-code case")
        rows.append({"key": cand["key"], "q": cand["q"], "ok": not diffs, "classified_as_code": classified, "diffs": diffs})
    missing = sorted(set(by_key) - {case["key"] for case in candidate_cases})
    green = all(row["ok"] for row in rows) and not missing
    return {
        "green": green,
        "fingerprint_ok": True,
        "classifier_checked": is_code is not None,
        "failed": [row["key"] for row in rows if not row["ok"]],
        "missing": missing,
        "cases": rows,
    }


def command_check(args: argparse.Namespace) -> int:
    out = ensure_outside_repo(args.out)
    baseline = read_json(args.baseline)
    inputs = load_private(private_dir(args.private_dir))
    guard = OpenAIGuard(allow_chat=False, allow_embed=False)
    store = EmbeddingStore(baseline.get("embeddings") or {}, replay=True)
    from benchmark_cli import agent  # noqa: F401

    base_plans = {case["key"]: case.get("plan") for case in baseline["cases"]}
    cases = []
    with guarded_openai(guard), patched_embeddings(store):
        fingerprint = corpus_fingerprint()
        if fingerprint_matches(baseline["fingerprint"], fingerprint):
            for case in inputs.non_code_cases():
                payload = base_plans.get(case.key)
                plan = plan_from_dict(payload) if payload else stored_plan(case, "raw")
                cases.append(run_offline_case(case, plan))
    body = evaluate_check(baseline, cases, fingerprint=fingerprint, is_code=classifier())
    report = {
        "kind": "ask-regress-check",
        "created_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "build": build_fingerprint(),
        "baseline_build": baseline.get("build"),
        "baseline_label": baseline.get("label"),
        "fingerprint": fingerprint,
        "embedding_misses": len(store.misses),
        **body,
    }
    write_json(out, report)
    for row in report["cases"]:
        if not row["ok"]:
            print(f"FAIL {row['key']}: {row['q']}")
            for line in row["diffs"]:
                print(f"    {line}")
    if not report["fingerprint_ok"]:
        print(report["message"])
    print(f"check: {'GREEN' if report['green'] else 'RED'} ({len(report['failed'])} failed) -> {out}")
    return 0 if report["green"] else 1


# --------------------------------------------------------------------------- code retrieval


def path_of(chunk: Any) -> str:
    metadata = chunk.metadata or {}
    return str(metadata.get("path") or chunk.source_file)


def path_matches(path: str, expect_paths: Sequence[str]) -> bool:
    return any(path.endswith(suffix) for suffix in expect_paths)


_TEST_PATH = re.compile(
    r"(^|/)(tests?|__tests__|fixtures)/|(^|/)test_[^/]*\.py$|_test\.py$|\.(test|spec)\.tsx?$|(^|/)conftest\.py$"
)


def forbidden_lane_chunk(chunk: Any) -> str | None:
    """A pull request, a test, or a knowledge-base duplicate must never be lane evidence."""
    metadata = chunk.metadata or {}
    path = path_of(chunk)
    if str(metadata.get("kind") or "") == "pull_request" or "/pulls/" in chunk.source_file:
        return "pull_request"
    if _TEST_PATH.search(path):
        return "test"
    if "portal/content/" in path:
        return "kb_duplicate"
    return None


def evaluate_code_retrieval(case: dict[str, Any], retrieval: Any) -> dict[str, Any]:
    chunks = list(getattr(retrieval, "chunks", ()) or ())
    ranking = list(getattr(retrieval, "ranking", ()) or ())
    rank = next(
        (index for index, chunk in enumerate(chunks, start=1) if path_matches(path_of(chunk), case["expect_paths"])),
        None,
    )
    forbidden = [
        {"path": path_of(chunk), "why": why} for chunk in chunks if (why := forbidden_lane_chunk(chunk))
    ]
    gated = bool(case["expect_paths"])
    return {
        "id": case["id"],
        "status": getattr(retrieval, "status", None),
        "rank": rank,
        "gated": gated,
        "passed": (not gated or (rank is not None and rank <= CODE_RETRIEVAL_TOP)) and not forbidden,
        "forbidden": forbidden,
        # "score" is the lane's own ranking score; the raw list scores (cosine, ts_rank) are its parts.
        "top": [
            {
                "path": path_of(chunk),
                "kind": (chunk.metadata or {}).get("kind"),
                "symbol": (chunk.metadata or {}).get("symbol"),
                **(ranking[index] if index < len(ranking) else {"score": None, "raw_score": chunk.score}),
            }
            for index, chunk in enumerate(chunks)
        ],
    }


def command_code_retrieval(args: argparse.Namespace) -> int:
    out = ensure_outside_repo(args.out)
    lane = _code_lane()
    if lane is None or not hasattr(lane, "retrieve_code"):
        raise SystemExit("This build has no code lane (github_answers.retrieve_code).")
    preflight()
    guard = OpenAIGuard(allow_chat=False, allow_embed=True)
    rows = []
    with guarded_openai(guard):
        for case in CODE_CASES:
            rows.append(evaluate_code_retrieval(case, lane.retrieve_code(case["q"])))
    gated = [row for row in rows if row["gated"]]
    green = all(row["passed"] for row in rows)
    report = {
        "kind": "ask-regress-code-retrieval",
        "build": build_fingerprint(),
        "green": green,
        "top3": f"{sum(row['passed'] for row in gated)}/{len(gated)}",
        "usage": guard.usage.as_dict(),
        "cases": rows,
    }
    write_json(out, report)
    for row in rows:
        mark = "ok  " if row["passed"] else "FAIL"
        print(f"{mark} {row['id']}: rank={row['rank']} status={row['status']} forbidden={row['forbidden']}")
    print(f"code-retrieval: {'GREEN' if green else 'RED'} top{CODE_RETRIEVAL_TOP}={report['top3']} -> {out}")
    return 0 if green else 1


# --------------------------------------------------------------------------- record / replay


def chunk_to_dict(chunk: Any) -> dict[str, Any]:
    return {
        "id": chunk.id,
        "source_file": chunk.source_file,
        "chunk_index": chunk.chunk_index,
        "chunk_text": chunk.chunk_text,
        "metadata": dict(chunk.metadata or {}),
        "score": chunk.score,
        "token_count": chunk.token_count,
        "content_sha256": chunk.content_sha256,
    }


def chunk_from_dict(payload: dict[str, Any]) -> Any:
    from benchmark_cli.storage.postgres import SearchChunk

    return SearchChunk(**payload)


def retrieval_to_dict(retrieval: Any) -> dict[str, Any]:
    pulls = getattr(retrieval, "pulls", None)
    return {
        "status": retrieval.status,
        "chunks": [chunk_to_dict(chunk) for chunk in retrieval.chunks],
        "header_ids": list(retrieval.header_ids),
        "top_cosine": retrieval.top_cosine,
        "overlap": retrieval.overlap,
        "pulls": None if pulls is None else pulls.model_dump(),
        "ranking": [dict(parts) for parts in getattr(retrieval, "ranking", ()) or ()],
    }


def retrieval_from_dict(payload: dict[str, Any]) -> Any:
    from benchmark_cli.github_answers import CodeRetrieval
    from benchmark_cli.models import AnswerResponse

    return CodeRetrieval(
        status=payload["status"],
        chunks=tuple(chunk_from_dict(item) for item in payload["chunks"]),
        header_ids=tuple(payload["header_ids"]),
        top_cosine=payload["top_cosine"],
        overlap=payload["overlap"],
        pulls=None if payload.get("pulls") is None else AnswerResponse(**payload["pulls"]),
        ranking=tuple(payload.get("ranking") or ()),
    )


def command_record(args: argparse.Namespace) -> int:
    out = ensure_outside_repo(args.out)
    lane = _code_lane()
    if lane is None or not hasattr(lane, "answer_from_retrieval"):
        raise SystemExit("This build has no code lane.")
    preflight()
    guard = OpenAIGuard(allow_chat=True, allow_embed=True)
    rows = []
    with guarded_openai(guard):
        for case in CODE_CASES:
            retrieval = lane.retrieve_code(case["q"])
            drafts: list[Any] = []

            def draft(messages: list[dict[str, Any]], _drafts: list[Any] = drafts) -> Any:
                response = lane.draft_code_answer(messages)
                _drafts.append(None if response is None else response.model_dump())
                return response

            result = lane.answer_from_retrieval(case["q"], retrieval, draft)
            rows.append(
                {
                    "id": case["id"],
                    "q": case["q"],
                    "retrieval": retrieval_to_dict(retrieval),
                    "drafts": drafts,
                    "outcome": result.outcome,
                    "reasons": list(result.reasons),
                    "final": result.response.model_dump(),
                }
            )
            print(f"{case['id']}: outcome={result.outcome} drafts={len(drafts)} reasons={result.reasons}")
    write_json(out, {"kind": "ask-regress-recording", "build": build_fingerprint(), "usage": guard.usage.as_dict(), "cases": rows})
    return 0


def command_replay(args: argparse.Namespace) -> int:
    lane = _code_lane()
    if lane is None or not hasattr(lane, "answer_from_retrieval"):
        raise SystemExit("This build has no code lane.")
    from benchmark_cli.models import AnswerResponse

    recording = read_json(args.recording)
    guard = OpenAIGuard(allow_chat=False, allow_embed=False)
    rows = []
    with guarded_openai(guard):
        for row in recording["cases"]:
            retrieval = retrieval_from_dict(row["retrieval"])
            queue = [None if item is None else AnswerResponse(**item) for item in row["drafts"]]

            def draft(_messages: list[dict[str, Any]], _queue: list[Any] = queue) -> Any:
                return _queue.pop(0) if _queue else None

            per_draft = []
            for item in row["drafts"]:
                if item is None:
                    per_draft.append({"draft": None})
                    continue
                checked, reasons = lane.validate_code_answer(AnswerResponse(**item), retrieval.chunks)
                per_draft.append({"valid": checked is not None, "reasons": reasons})
            result = lane.answer_from_retrieval(row["q"], retrieval, draft)
            rows.append(
                {
                    "id": row["id"],
                    "recorded_outcome": row["outcome"],
                    "replayed_outcome": result.outcome,
                    "drafts": per_draft,
                    "reasons": list(result.reasons),
                }
            )
            print(f"{row['id']}: recorded={row['outcome']} replayed={result.outcome} drafts={per_draft}")
    if args.out:
        write_json(args.out, {"kind": "ask-regress-replay", "build": build_fingerprint(), "cases": rows})
    return 0


# --------------------------------------------------------------------------- scorers


def _citations(result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(result.get("citations") or [])


def _sources(result: dict[str, Any]) -> list[str]:
    return [str(item.get("source_file") or "") for item in _citations(result)]


def score_saved(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str]:
    """``saved_shortcut``, ``cites_saved`` or ``not_saved``."""
    expect = case["expect"]
    prefix = case.get("source_prefix") or SAVED_PREFIX
    answer = str(result.get("answer") or "")
    cites_saved = any(source.startswith(prefix) for source in _sources(result))
    answered = result.get("status") == "answered"
    if expect == "saved_shortcut":
        if result.get("outcome") not in (None, "user_contributed"):
            return False, f"outcome {result.get('outcome')} is not user_contributed"
        ok = answered and cites_saved and answer.startswith("According to")
        return ok, "" if ok else "not answered from the saved row"
    if expect == "cites_saved":
        ok = answered and cites_saved
        return ok, "" if ok else "does not cite the saved row"
    if expect == "not_saved":
        if answer.startswith("According to"):
            return False, "answer starts with According to"
        if cites_saved:
            return False, "cites a saved row"
        return True, ""
    raise ValueError(f"Unknown saved expectation {expect!r}.")


def score_name(case: dict[str, Any], result: dict[str, Any], titles: dict[str, list[str]]) -> tuple[bool, str]:
    person = case["person"]
    answer = str(result.get("answer") or "")
    if result.get("status") != "answered":
        return False, f"status {result.get('status')}"
    missing = [title for title in titles.get(person, []) if title not in answer]
    if missing:
        return False, f"missing titles {missing}"
    extra = [
        title
        for other, values in titles.items()
        if other != person
        for title in values
        if title not in titles.get(person, []) and title in answer
    ]
    if extra:
        return False, f"titles of another person {extra}"
    if answer.startswith("According to"):
        return False, "answer starts with According to"
    bad = [source for source in _sources(result) if source.startswith((SAVED_PREFIX, GITHUB_PREFIX))]
    if bad:
        return False, f"cites {bad}"
    return True, ""


def score_code(case: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str]:
    sources = _sources(result)
    answered = result.get("status") == "answered"
    outside = [source for source in sources if not source.startswith(GITHUB_PREFIX)]
    if outside:
        return False, f"non-GitHub citation {outside}"
    expect = case.get("expect")
    if expect == "abstain_or_kiosk_path":
        if not answered:
            return True, ""
        ok = bool(sources) and all("kiosk" in source.casefold() for source in sources)
        return ok, "" if ok else "answered without kiosk-path citations"
    if expect == "pulls":
        ok = answered and bool(sources) and all("/pulls/" in source for source in sources)
        return ok, "" if ok else "not answered from pull requests"
    if not answered:
        return False, f"status {result.get('status')}"
    ok = any(path_matches(source, case["expect_paths"]) for source in sources)
    return ok, "" if ok else f"expected one of {list(case['expect_paths'])}, cited {sources}"


def suite_hijack(result: dict[str, Any], baseline: dict[str, Any] | None) -> str | None:
    """GitHub evidence in a non-code answer, or a saved-answer takeover the baseline did not have."""
    sources = _sources(result)
    if any(source.startswith(GITHUB_PREFIX) for source in sources):
        return "github citation"
    answer = str(result.get("answer") or "")
    saved_now = answer.startswith("According to") or any(source.startswith(SAVED_PREFIX) for source in sources)
    if not saved_now:
        return None
    base = baseline or {}
    base_sources = [str(item.get("source_file") or "") for item in base.get("citations") or []]
    saved_before = str(base.get("answer") or "").startswith("According to") or any(
        source.startswith(SAVED_PREFIX) for source in base_sources
    )
    return None if saved_before else "saved-answer takeover"


def suite_stop_reason(answered: int, remaining: int, remaining_baseline_answered: int, bar: int = SUITE_BAR) -> str | None:
    if answered + remaining < bar:
        return f"suite mathematically out: {answered} answered + {remaining} remaining < {bar}"
    if answered + remaining_baseline_answered + 1 < bar:
        return (
            f"suite clearly out: {answered} answered + {remaining_baseline_answered} remaining "
            f"baseline-answered + 1 < {bar}"
        )
    return None


def budget_exceeded(chat_calls: int, prompt_tokens: int, elapsed_seconds: float) -> str | None:
    if chat_calls > MAX_CHAT_CALLS:
        return f"budget: {chat_calls} chat calls > {MAX_CHAT_CALLS}"
    if prompt_tokens > MAX_PROMPT_TOKENS:
        return f"budget: {prompt_tokens} prompt tokens > {MAX_PROMPT_TOKENS}"
    if elapsed_seconds > MAX_SECONDS:
        return f"budget: {elapsed_seconds:.0f}s > {MAX_SECONDS}s"
    return None


class EarlyStop(Exception):
    """A live bar is out of reach. The partial report is written."""


@dataclass
class LiveTally:
    """Running counts that drive the early-stop rules."""

    code_original_failures: int = 0
    suite_answered: int = 0
    suite_hijacks: int = 0

    def after_saved(self, ok: bool, key: str) -> None:
        if not ok:
            raise EarlyStop(f"saved failure: {key}")

    def after_name(self, ok: bool, key: str) -> None:
        if not ok:
            raise EarlyStop(f"names failure: {key}")

    def after_code(self, ok: bool, original: bool, key: str) -> None:
        if not ok and original:
            self.code_original_failures += 1
            if self.code_original_failures > CODE_MAX_ORIGINAL_FAILURES:
                raise EarlyStop(f"third code failure among the original 11: {key}")

    def after_suite(
        self, *, answered: bool, hijack: str | None, key: str, remaining: int, remaining_baseline_answered: int
    ) -> None:
        if answered:
            self.suite_answered += 1
        if hijack:
            self.suite_hijacks += 1
            raise EarlyStop(f"suite hijack ({hijack}): {key}")
        reason = suite_stop_reason(self.suite_answered, remaining, remaining_baseline_answered)
        if reason:
            raise EarlyStop(reason)


# --------------------------------------------------------------------------- live


def run_live_case(question: str, starter: str | None, guard: OpenAIGuard) -> dict[str, Any]:
    from benchmark_cli import agent

    seen: dict[str, Any] = {}

    def diagnostic(**kwargs: Any) -> None:
        seen["outcome"] = kwargs.get("outcome")
        seen["attempts"] = kwargs.get("synthesis_attempts")

    before = guard.usage.chat_calls
    started = time.monotonic()
    tq = thread_question(question, starter)
    result: dict[str, Any] = {"q": question, "starter": starter}
    with mock.patch.object(agent, "log_agent_diagnostic", diagnostic):
        try:
            response = agent.run_agent(question, thread_question=tq) if tq else agent.run_agent(question)
        except Exception as exc:
            result.update({"status": "error", "answer": "", "citations": [], "error": f"{type(exc).__name__}: {exc}"[:300]})
        else:
            result.update(
                {
                    "status": response.status,
                    "answer": response.answer,
                    "citations": [
                        {"source_file": item.source_file, "chunk_id": item.chunk_id, "quote": item.quote[:300]}
                        for item in response.citations
                    ],
                }
            )
    result.update(
        {
            "outcome": seen.get("outcome"),
            "attempts": seen.get("attempts"),
            "chat_calls": guard.usage.chat_calls - before,
            "secs": round(time.monotonic() - started, 1),
        }
    )
    reason = guard.budget_reason()
    if reason:
        raise EarlyStop(reason)
    return result


def verify_check_report(report: dict[str, Any], build: dict[str, Any]) -> None:
    if report.get("kind") != "ask-regress-check":
        raise SystemExit("--check-report is not a check report.")
    if not report.get("green"):
        raise SystemExit("The check report is not green; live will not start.")
    if (report.get("build") or {}).get("source_sha256") != build["source_sha256"]:
        raise SystemExit("The check report was produced by a different build; rerun check on this build.")


def _run_suite(inputs: PrivateInputs, guard: OpenAIGuard, tally: LiveTally, results: list[dict[str, Any]]) -> None:
    cases = list(inputs.suite)
    for index, case in enumerate(cases):
        result = run_live_case(case.q, case.starter, guard)
        baseline = inputs.baseline.get(case.key)
        hijack = suite_hijack(result, baseline)
        result.update({"key": case.key, "answered": result["status"] == "answered", "hijack": hijack})
        results.append(result)
        rest = cases[index + 1 :]
        remaining_baseline = sum(
            1 for item in rest if (inputs.baseline.get(item.key) or {}).get("status") == "answered"
        )
        tally.after_suite(
            answered=result["answered"],
            hijack=hijack,
            key=case.key,
            remaining=len(rest),
            remaining_baseline_answered=remaining_baseline,
        )


def command_live(args: argparse.Namespace) -> int:
    out = ensure_outside_repo(args.out)
    build = build_fingerprint()
    verify_check_report(read_json(args.check_report), build)
    inputs = load_private(private_dir(args.private_dir))
    if not inputs.baseline:
        raise SystemExit("baseline-suite.json is required for hijack and early-stop scoring.")
    preflight()
    groups = [item.strip() for item in args.groups.split(",") if item.strip()]
    guard = OpenAIGuard(allow_chat=True, allow_embed=True, budget=True)
    tally = LiveTally()
    report: dict[str, Any] = {
        "kind": "ask-regress-live",
        "created_at": datetime.now(UTC).isoformat(),
        "build": build,
        "check_report": str(args.check_report),
        "groups_requested": groups,
        "saved": [],
        "names": [],
        "code": [],
        "suite_runs": [],
        "early_stop": None,
    }
    lane_present = classifier() is not None
    try:
        with guarded_openai(guard):
            if "saved" in groups:
                for case in inputs.saved:
                    result = run_live_case(case.q, None, guard)
                    ok, why = score_saved(case.extra, result)
                    result.update({"key": case.key, "expect": case.extra["expect"], "passed": ok, "why": why})
                    report["saved"].append(result)
                    tally.after_saved(ok, case.key)
            if "names" in groups:
                for case in inputs.names:
                    result = run_live_case(case.q, None, guard)
                    ok, why = score_name(case.extra, result, inputs.titles)
                    result.update({"key": case.key, "passed": ok, "why": why})
                    report["names"].append(result)
                    tally.after_name(ok, case.key)
            if "code" in groups:
                if not lane_present:
                    raise EarlyStop("code group requested on a build without the code lane")
                for case in CODE_CASES:
                    result = run_live_case(case["q"], None, guard)
                    ok, why = score_code(case, result)
                    result.update({"key": case["id"], "original": case["original"], "passed": ok, "why": why})
                    report["code"].append(result)
                    tally.after_code(ok, case["original"], case["id"])
            if "suite" in groups:
                attempts = 1 if args.no_rerun else 2
                for attempt in range(attempts):
                    run: list[dict[str, Any]] = []
                    report["suite_runs"].append(run)
                    tally.suite_answered = 0
                    try:
                        _run_suite(inputs, guard, tally, run)
                    except EarlyStop as stop:
                        count_only = tally.suite_hijacks == 0 and str(stop).startswith("suite ")
                        if count_only and attempt + 1 < attempts:
                            logger.info("suite count failed (%s); one rerun", stop)
                            continue
                        raise
                    answered = sum(item["answered"] for item in run)
                    if answered >= SUITE_BAR or attempt + 1 >= attempts:
                        break
                    logger.info("suite answered %s < %s with zero hijacks; one rerun", answered, SUITE_BAR)
    except EarlyStop as stop:
        report["early_stop"] = str(stop)
    except HarnessAbort as abort:
        report["early_stop"] = f"aborted: {abort}"
    report.update(summarize_live(report, groups))
    report["usage"] = guard.usage.as_dict()
    report["elapsed_seconds"] = round(time.monotonic() - guard.started, 1)
    write_json(out, report)
    print(json.dumps({key: report[key] for key in ("bar", "pass", "early_stop", "usage")}, indent=1))
    return 0 if report["pass"] else 1


def summarize_live(report: dict[str, Any], groups: Sequence[str]) -> dict[str, Any]:
    saved = report["saved"]
    names = report["names"]
    code = report["code"]
    suite = report["suite_runs"][-1] if report["suite_runs"] else []
    code_original = [item for item in code if item["original"]]
    suite_answered = sum(item["answered"] for item in suite)
    hijacks = sum(bool(item["hijack"]) for run in report["suite_runs"] for item in run)
    bar = {
        "saved": f"{sum(item['passed'] for item in saved)}/{len(saved)}",
        "names": f"{sum(item['passed'] for item in names)}/{len(names)}",
        "code_original": f"{sum(item['passed'] for item in code_original)}/{len(code_original)}",
        "code_twelfth": [item["passed"] for item in code if not item["original"]],
        "suite": f"{suite_answered}/{len(suite)}",
        "suite_hijacks": hijacks,
        "suite_runs": len(report["suite_runs"]),
        "models": REQUIRED_CHAT_MODEL,
    }
    complete = set(groups) >= {"saved", "names", "code", "suite"}
    passed = (
        complete
        and report["early_stop"] is None
        and all(item["passed"] for item in saved)
        and len(names) >= NAMES_BAR
        and all(item["passed"] for item in names)
        and sum(item["passed"] for item in code_original) >= CODE_ORIGINAL_BAR
        and suite_answered >= SUITE_BAR
        and hijacks == 0
    )
    return {"bar": bar, "pass": passed}


# --------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark-ask-regress", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="Record the offline baseline trace.")
    capture.add_argument("--out", type=Path, required=True)
    capture.add_argument("--private-dir")
    capture.add_argument("--plans", choices=("understand", "raw"), default="understand")
    capture.add_argument("--label", default="")
    capture.set_defaults(func=command_capture)

    check = sub.add_parser("check", help="Compare this build to a capture. Zero OpenAI calls.")
    check.add_argument("--baseline", type=Path, required=True)
    check.add_argument("--out", type=Path, required=True)
    check.add_argument("--private-dir")
    check.add_argument("--label", default="")
    check.set_defaults(func=command_check)

    code = sub.add_parser("code-retrieval", help="Offline code-lane ranking gate.")
    code.add_argument("--out", type=Path, required=True)
    code.set_defaults(func=command_code_retrieval)

    record = sub.add_parser("record", help="Record code-lane chunks and model drafts.")
    record.add_argument("--out", type=Path, required=True)
    record.set_defaults(func=command_record)

    replay = sub.add_parser("replay", help="Validate recorded drafts offline.")
    replay.add_argument("--recording", type=Path, required=True)
    replay.add_argument("--out", type=Path)
    replay.set_defaults(func=command_replay)

    live = sub.add_parser("live", help="Live bar with early stop.")
    live.add_argument("--check-report", type=Path, required=True)
    live.add_argument("--out", type=Path, required=True)
    live.add_argument("--private-dir")
    live.add_argument("--groups", default="saved,names,code,suite")
    live.add_argument("--no-rerun", action="store_true", help="Do not rerun the suite once on a count-only failure.")
    live.set_defaults(func=command_live)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
