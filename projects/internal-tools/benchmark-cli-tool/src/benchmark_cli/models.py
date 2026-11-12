"""Validated ingestion metadata and retrieval-evaluation dataset models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkMetadata(BaseModel):
    """Provenance retained for a persisted document chunk."""

    source_type: Literal["discord", "transcript", "policy_doc", "kb_page", "trello", "github"]
    date: str
    ingested_at: str
    section: str | None = None
    speakers: list[str] = Field(default_factory=list)
    channel: str | None = None
    timezone: str | None = None
    date_start: str | None = None
    date_end: str | None = None
    meeting: str | None = None
    message_count: int | None = Field(default=None, ge=1)
    split_unit: bool = False
    embed_prefix: str = ""
    title: str | None = None
    url: str | None = None
    owner: str | None = None
    tags: list[str] = Field(default_factory=list)
    visibility: str | None = None
    kb_section: str | None = None
    kb_status: str | None = None
    portal_base: str | None = None
    board: str | None = None
    column: str | None = None
    card_id: str | None = None
    labels: list[str] = Field(default_factory=list)
    due: str | None = None
    attendees: list[str] = Field(default_factory=list)
    recording_id: str | None = None
    kind: str | None = None
    repo: str | None = None
    path: str | None = None
    commit_sha: str | None = None
    start_line: int | None = Field(default=None, ge=1)
    end_line: int | None = Field(default=None, ge=1)
    symbol: str | None = None
    merged_at: str | None = None

    @model_validator(mode="after")
    def validate_source_fields(self) -> ChunkMetadata:
        if self.source_type == "discord" and (not self.channel or not self.date_end):
            raise ValueError("discord chunk metadata requires channel and date_end.")
        if self.source_type == "transcript" and not self.meeting:
            raise ValueError("transcript chunk metadata requires meeting.")
        if self.kind == "meeting_transcript" and not (
            self.meeting and self.title and self.url and self.recording_id
        ):
            raise ValueError(
                "meeting transcript metadata requires meeting, title, url, and recording_id."
            )
        if self.source_type == "policy_doc" and not self.section:
            raise ValueError("policy_doc chunk metadata requires section.")
        if self.source_type == "kb_page" and (not self.section or not self.title):
            raise ValueError("kb_page chunk metadata requires section and title.")
        if self.source_type == "trello" and not (
            self.board and self.column and self.card_id and self.title and self.url
        ):
            raise ValueError(
                "trello chunk metadata requires board, column, card_id, title, and url."
            )
        if self.source_type == "github" and not (self.repo and self.path and self.commit_sha and self.url):
            raise ValueError("github chunk metadata requires repo, path, commit_sha, and url.")
        return self

    def as_metadata(self) -> dict[str, Any]:
        """JSON object stored on ``document_chunks.metadata``."""
        payload = self.model_dump(mode="json")
        if self.source_type == "trello":
            # ``column`` is the Trello list. Keep ``list`` as the name the board uses.
            payload["list"] = self.column
        if self.source_type != "github":
            for key in ("repo", "path", "commit_sha", "start_line", "end_line", "symbol", "merged_at"):
                payload.pop(key, None)
        return payload


QueryCategory = Literal["lookup", "multi_chunk", "filter", "unanswerable"]


class GoldenChunkReference(BaseModel):
    """A durable source-content reference resolved to a database ID at evaluation time."""

    source_file: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GoldenQuery(BaseModel):
    """One retrieval-only evaluation query with durable expected chunk references."""

    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_chunks: list[GoldenChunkReference] = Field(default_factory=list)
    query_category: QueryCategory
    required_answer_facts: list[str] = Field(default_factory=list)
    speaker: str | None = None
    source: str | None = None
    date: str | None = None
    expected_chunk_ids: list[int] = Field(default_factory=list, exclude=True)

    @model_validator(mode="after")
    def validate_expected_chunks(self) -> GoldenQuery:
        if self.query_category == "unanswerable" and self.expected_chunks:
            raise ValueError("unanswerable queries must not specify expected_chunks.")
        if self.query_category in {"lookup", "filter"} and len(self.expected_chunks) != 1:
            raise ValueError("lookup queries require exactly one expected chunk reference.")
        if self.query_category == "multi_chunk" and len(self.expected_chunks) < 2:
            raise ValueError("multi_chunk queries require at least two expected chunk references.")
        references = {(chunk.source_file, chunk.content_sha256) for chunk in self.expected_chunks}
        if len(references) != len(self.expected_chunks):
            raise ValueError("expected_chunks must not repeat a source-content reference.")
        return self


class Citation(BaseModel):
    """Verbatim evidence that can be validated against one persisted chunk."""

    source_file: str = Field(min_length=1)
    chunk_id: int = Field(gt=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    quote: str = Field(min_length=1)

    @field_validator("quote")
    @classmethod
    def quote_must_contain_visible_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("citation quote must contain non-whitespace text.")
        return value


class AnswerResponse(BaseModel):
    """A grounded answer or an explicit insufficient-evidence outcome."""

    status: Literal["answered", "no_relevant_information"]
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_abstention(self) -> AnswerResponse:
        if self.status == "no_relevant_information":
            if self.confidence != 0.0 or self.citations:
                raise ValueError("Abstentions must have zero confidence and no citations.")
        elif not self.citations:
            raise ValueError("Answered responses require at least one citation.")
        return self
