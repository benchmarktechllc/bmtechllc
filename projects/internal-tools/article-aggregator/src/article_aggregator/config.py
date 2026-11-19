"""Load and validate config/feeds.yaml and config/keywords.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, field_validator

FEEDS_FILE = "feeds.yaml"
KEYWORDS_FILE = "keywords.yaml"


class ConfigError(Exception):
    """Config is missing or invalid. The message names the file and the problem."""


class FeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    url: str
    enabled: bool = True
    # Most articles this feed may take in one digest; None means no per-feed cap.
    max_per_digest: int | None = Field(default=None, ge=1)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("name must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def _http_url(cls, value: str) -> str:
        value = value.strip()
        parts = urlsplit(value)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise ValueError("url must be an absolute http(s) URL")
        return value


@dataclass(frozen=True)
class AppConfig:
    feeds: tuple[FeedConfig, ...]
    keywords: tuple[str, ...]

    @property
    def enabled_feeds(self) -> tuple[FeedConfig, ...]:
        return tuple(feed for feed in self.feeds if feed.enabled)

    @property
    def feed_caps(self) -> dict[str, int]:
        """Per-digest article caps keyed by feed name, for feeds that set one."""
        return {f.name: f.max_per_digest for f in self.feeds if f.max_per_digest is not None}


_FEEDS = TypeAdapter(list[FeedConfig])
_KEYWORDS = TypeAdapter(list[str])


def load_config(config_dir: Path) -> AppConfig:
    feeds = _load_feeds(config_dir / FEEDS_FILE)
    keywords = _load_keywords(config_dir / KEYWORDS_FILE)
    return AppConfig(feeds=feeds, keywords=keywords)


def _read_yaml(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"{path}: file not found") from None
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from None


def _load_feeds(path: Path) -> tuple[FeedConfig, ...]:
    try:
        feeds = _FEEDS.validate_python(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    if not feeds:
        raise ConfigError(f"{path}: at least one feed is required")
    seen: set[str] = set()
    names: set[str] = set()
    for feed in feeds:
        if feed.url in seen:
            raise ConfigError(f"{path}: duplicate feed url {feed.url}")
        # Names key the per-feed caps and show in embed footers, so they must be unique too.
        if feed.name in names:
            raise ConfigError(f"{path}: duplicate feed name {feed.name!r}")
        seen.add(feed.url)
        names.add(feed.name)
    if not any(feed.enabled for feed in feeds):
        raise ConfigError(f"{path}: every feed is disabled")
    return tuple(feeds)


def _load_keywords(path: Path) -> tuple[str, ...]:
    try:
        raw = _KEYWORDS.validate_python(_read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"{path}: {exc}") from None
    keywords: list[str] = []
    seen: set[str] = set()
    for item in raw:
        keyword = " ".join(item.split())
        if not keyword:
            raise ConfigError(f"{path}: blank keyword")
        if keyword.casefold() in seen:
            raise ConfigError(f"{path}: duplicate keyword {keyword!r}")
        seen.add(keyword.casefold())
        keywords.append(keyword)
    if not keywords:
        raise ConfigError(f"{path}: at least one keyword is required")
    return tuple(keywords)
