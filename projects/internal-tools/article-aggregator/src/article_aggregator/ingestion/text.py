"""Text cleanup shared by matching and embed formatting."""

from __future__ import annotations

import unicodedata
from html.parser import HTMLParser

_SKIPPED_TAGS = frozenset({"script", "style", "noscript", "template"})
# Block-level tags separate words ("<p>a</p><p>b</p>" is "a b"); inline tags do not.
_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt", "figcaption",
    "figure", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "ol", "p",
    "pre", "section", "table", "td", "th", "tr", "ul",
})


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS and self._skip_depth:
            self._skip_depth -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def normalize_text(text: str) -> str:
    """NFKC-normalize and collapse all whitespace runs to single spaces."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def strip_html(html: str) -> str:
    """Return the visible text of an HTML fragment, entities decoded, whitespace collapsed."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return normalize_text("".join(parser.parts))


def truncate(text: str, limit: int) -> str:
    """Cut text to at most `limit` characters at a word boundary, ending with an ellipsis."""
    if len(text) <= limit:
        return text
    cut = text[: limit - 1]
    space = cut.rfind(" ")
    if space > 0:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"
