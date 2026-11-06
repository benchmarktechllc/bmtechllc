"""Configuration for ingestion and hybrid retrieval."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

from .paths import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env.local", override=False)

OPENAI_API_KEY: Final[str | None] = os.getenv("OPENAI_API_KEY")
DATABASE_URL: Final[str | None] = os.getenv("DATABASE_URL")
DISCORD_INGEST_BOT_TOKEN: Final[str | None] = os.getenv("DISCORD_INGEST_BOT_TOKEN")
DISCORD_INGEST_CHANNEL_IDS: Final[tuple[str, ...]] = tuple(
    channel_id.strip()
    for channel_id in os.getenv("DISCORD_INGEST_CHANNEL_IDS", "").split(",")
    if channel_id.strip()
)
DISCORD_API_BASE_URL: Final[str] = os.getenv("DISCORD_API_BASE_URL", "https://discord.com/api/v10")
DB_STATEMENT_TIMEOUT_MS: Final[int] = int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "5000"))
HNSW_EF_SEARCH: Final[int] = int(os.getenv("HNSW_EF_SEARCH", "40"))

EMBEDDING_MODEL: Final[str] = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS: Final[int] = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
EMBEDDING_ENCODING: Final[str] = os.getenv("EMBEDDING_ENCODING", "cl100k_base")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """A chat model and its published per-million-token prices."""

    model: str
    input_cost_per_million: float
    output_cost_per_million: float


CHEAP_MODEL: Final[str] = os.getenv("CHEAP_MODEL", "gpt-4o-mini")
FLAGSHIP_MODEL: Final[str] = os.getenv("FLAGSHIP_MODEL", "gpt-4o")
MODEL_TIERS: Final[dict[str, ModelConfig]] = {
    "cheap": ModelConfig(
        CHEAP_MODEL,
        float(os.getenv("CHEAP_INPUT_COST_PER_MILLION", "0.15")),
        float(os.getenv("CHEAP_OUTPUT_COST_PER_MILLION", "0.60")),
    ),
    "flagship": ModelConfig(
        FLAGSHIP_MODEL,
        float(os.getenv("FLAGSHIP_INPUT_COST_PER_MILLION", "2.50")),
        float(os.getenv("FLAGSHIP_OUTPUT_COST_PER_MILLION", "10.00")),
    ),
    "embedding": ModelConfig(
        EMBEDDING_MODEL,
        float(os.getenv("EMBEDDING_INPUT_COST_PER_MILLION", "0.02")),
        0.0,
    ),
}


def get_model_config(tier: str) -> ModelConfig:
    """Return a configured model tier or explain the supported values."""
    try:
        return MODEL_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"Unknown model tier {tier!r}. Expected one of: {', '.join(MODEL_TIERS)}.") from exc

CHUNK_TARGET_TOKENS: Final[int] = 400
CHUNK_MAX_TOKENS: Final[int] = 500
CHUNK_MIN_TOKENS: Final[int] = 80
CHUNK_OVERLAP_TOKENS: Final[int] = 50
PARENT_CONTEXT_MAX_TOKENS: Final[int] = 1200
MIN_COSINE_SIMILARITY: Final[float] = float(os.getenv("MIN_COSINE_SIMILARITY", "0.30"))
MIN_KEYWORD_OVERLAP: Final[int] = int(os.getenv("MIN_KEYWORD_OVERLAP", "1"))

DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
GOLDEN_DATASET_PATH: Final[Path] = DATA_DIR / "evaluation" / "golden_queries.json"
EVALUATION_LOG_PATH: Final[Path] = PROJECT_ROOT / "artifacts" / "retrieval_evaluation.csv"
