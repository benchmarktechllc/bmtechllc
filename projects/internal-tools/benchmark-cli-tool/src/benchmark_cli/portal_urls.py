"""Canonical knowledge-base URLs for the current GitHub Pages deployment.

The site path ``/bmtechllc/portal`` matches ``portal/astro.config.mjs``. The host
follows ``PORTAL_BASE_URL`` or the current repository owner. This module does
not default to a previous owner's Pages host.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

SITE_PREFIX = "/bmtechllc/portal"
_OWNER = re.compile(r"^[A-Za-z0-9-]+$")


def canonical_portal_base_url() -> str | None:
    """Return the configured portal origin plus site prefix, without a trailing slash."""
    explicit = os.getenv("PORTAL_BASE_URL", "").strip().rstrip("/")
    if explicit:
        if "://" not in explicit:
            explicit = "https://" + explicit
        return explicit.rstrip("/")
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "").strip()
    if not owner:
        repository = os.getenv("GITHUB_REPOSITORY", "").strip()
        owner = repository.split("/", 1)[0] if "/" in repository else ""
    if not _OWNER.fullmatch(owner):
        return None
    return f"https://{owner}.github.io{SITE_PREFIX}"


def join_portal_url(base: str | None, path: str) -> str:
    """Join a relative site path to the deployment base without duplicating the prefix."""
    text = path.strip()
    if text.startswith(("http://", "https://")):
        return text if text.endswith("/") else text + "/"
    if not text.startswith("/"):
        text = "/" + text
    if not base:
        return text
    parsed = urlsplit(base if "://" in base else f"https://{base}")
    origin = f"{parsed.scheme}://{parsed.netloc}"
    base_path = parsed.path.rstrip("/")
    if base_path and (text == base_path or text.startswith(base_path + "/")):
        absolute = origin + text
    else:
        absolute = base.rstrip("/") + text
    return absolute if absolute.endswith("/") else absolute + "/"


def portal_home_url(base: str | None = None) -> str | None:
    """Return the clickable portal home, or None when no deployment base is known."""
    selected = base if base is not None else canonical_portal_base_url()
    if not selected:
        return None
    return join_portal_url(selected, SITE_PREFIX + "/")


def portal_home_from_metadata(metadata: Mapping[str, object], configured: str | None) -> str | None:
    """Prefer the live deployment, then an absolute base already stored on a knowledge-base chunk."""
    if configured:
        return portal_home_url(configured)
    recorded = str(metadata.get("portal_base") or "").strip()
    if recorded.startswith(("http://", "https://")):
        return portal_home_url(recorded)
    url = str(metadata.get("url") or "").strip()
    marker = f"{SITE_PREFIX}/"
    if url.startswith(("http://", "https://")) and marker in url:
        return url[: url.index(marker) + len(marker)]
    return None
