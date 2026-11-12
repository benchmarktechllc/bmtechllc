"""Choose and chunk files from the public BMTech repository."""

from __future__ import annotations

import ast
import textwrap
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from .ingestion.chunk import token_count

_CODE_SUFFIXES = {
    ".py",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".sql",
    ".css",
    ".html",
    ".sh",
    ".md",
    ".yml",
    ".yaml",
    ".toml",
}
_SKIP_DIRS = {
    "node_modules",
    "dist",
    "build",
    ".next",
    "coverage",
    "__pycache__",
    ".venv",
    "venv",
    "vendor",
    "vendored",
    ".git",
}
_LOCKFILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
    "cargo.lock",
}
_MAX_BYTES = 200_000


@dataclass(frozen=True, slots=True)
class CodePiece:
    """One citation-sized slice of a file."""

    text: str
    start_line: int
    end_line: int
    symbol: str


def should_index(path: str, size: int) -> bool:
    """True for source, docs, and config. Secrets, lockfiles, and build output stay out."""
    if size > _MAX_BYTES or size < 1:
        return False
    pure = PurePosixPath(path)
    if any(part in _SKIP_DIRS for part in pure.parts[:-1]):
        return False
    name = pure.name
    folded = name.casefold()
    if folded.startswith(".env") or folded.endswith((".pem", ".key")):
        return False
    if "secret" in folded or "credential" in folded:
        return False
    if folded in _LOCKFILES:
        return False
    if folded == "dockerfile" or folded.startswith("dockerfile."):
        return True
    if ".github/workflows/" in path.replace("\\", "/") and pure.suffix.casefold() in {".yml", ".yaml"}:
        return True
    return pure.suffix.casefold() in _CODE_SUFFIXES


def source_file_for(repo: str, path: str) -> str:
    """Stable chunk identity. A new commit does not change this path."""
    return f"github/{repo}/{path.lstrip('/')}"


def blob_url(repo: str, commit_sha: str, path: str, start_line: int, end_line: int) -> str:
    """GitHub blob link for the indexed commit and line range."""
    return (
        f"https://github.com/{repo}/blob/{commit_sha}/{path.lstrip('/')}"
        f"#L{start_line}-L{end_line}"
    )


def render_piece(
    repo: str,
    commit_sha: str,
    path: str,
    start_line: int,
    end_line: int,
    symbol: str,
    body: str,
) -> str:
    """Header plus body. The blob URL is part of the stored text so a quote can carry it."""
    url = blob_url(repo, commit_sha, path, start_line, end_line)
    label = f" {symbol}" if symbol else ""
    return f"{repo}@{commit_sha} {path}:L{start_line}-L{end_line}{label}\n{url}\n\n{body.rstrip()}\n"


def chunk_text(text: str, path: str) -> list[CodePiece]:
    """Chunk one file. Python splits on functions; everything else uses line windows."""
    if PurePosixPath(path).suffix.casefold() == ".py":
        return chunk_python(text)
    if PurePosixPath(path).suffix.casefold() == ".md":
        return _chunk_markdown(text)
    return _windows(text.splitlines(), "")


def chunk_python(source: str) -> list[CodePiece]:
    """Split a Python file on top-level functions and class methods."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _windows(source.splitlines(), "")
    lines = source.splitlines()
    covered: set[int] = set()
    pieces: list[CodePiece] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            pieces.extend(_class_pieces(node, lines, covered))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pieces.extend(_symbol_pieces(node, lines, node.name, covered))
    pieces.extend(_gaps(lines, covered, ""))
    return pieces


def _class_pieces(node: ast.ClassDef, lines: list[str], covered: set[int]) -> list[CodePiece]:
    pieces: list[CodePiece] = []
    method_lines: set[int] = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pieces.extend(_symbol_pieces(child, lines, f"{node.name}.{child.name}", method_lines))
    start, end = _span(node)
    class_lines = set(range(start, end + 1)) - method_lines
    if class_lines:
        pieces.extend(_from_line_set(lines, class_lines, node.name))
    covered.update(range(start, end + 1))
    return pieces


def _symbol_pieces(
    node: ast.AST, lines: list[str], symbol: str, covered: set[int]
) -> list[CodePiece]:
    start, end = _span(node)
    signature_end = getattr(node, "lineno", start)
    pieces = _split_with_signature(lines, start, end, signature_end, symbol)
    covered.update(range(start, end + 1))
    return pieces


def _span(node: ast.AST) -> tuple[int, int]:
    decorators = getattr(node, "decorator_list", None) or []
    start = decorators[0].lineno if decorators else int(node.lineno)
    end = int(getattr(node, "end_lineno", None) or node.lineno)
    return start, end


def _split_with_signature(
    lines: list[str], start: int, end: int, signature_end: int, symbol: str
) -> list[CodePiece]:
    full = _join(lines, start, end)
    if token_count(full) <= CHUNK_MAX_TOKENS:
        return [CodePiece(_storable(full), start, end, symbol)]
    signature = _join(lines, start, signature_end)
    pieces: list[CodePiece] = []
    cursor = signature_end + 1
    while cursor <= end:
        window_end = cursor
        while window_end < end:
            nxt = window_end + 1
            body = _join(lines, cursor, nxt)
            candidate = f"{signature}\n{body}"
            if token_count(candidate) > CHUNK_TARGET_TOKENS and window_end >= cursor:
                break
            window_end = nxt
        body = _join(lines, cursor, window_end)
        text = signature if not body else f"{signature}\n{body}"
        pieces.append(CodePiece(_storable(text), cursor, window_end, symbol))
        if window_end >= end:
            break
        cursor = window_end + 1
    return pieces


def _gaps(lines: list[str], covered: set[int], symbol: str) -> list[CodePiece]:
    remaining = set(range(1, len(lines) + 1)) - covered
    return _from_line_set(lines, remaining, symbol)


def _from_line_set(lines: list[str], selected: set[int], symbol: str) -> list[CodePiece]:
    if not selected:
        return []
    ordered = sorted(selected)
    groups: list[list[int]] = [[ordered[0]]]
    for number in ordered[1:]:
        if number == groups[-1][-1] + 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    pieces: list[CodePiece] = []
    for group in groups:
        text = _join(lines, group[0], group[-1]).strip("\n")
        if not text.strip():
            continue
        pieces.extend(_windows(text.splitlines(), symbol, origin=group[0]))
    return pieces


def _chunk_markdown(text: str) -> list[CodePiece]:
    lines = text.splitlines()
    pieces: list[CodePiece] = []
    start = 1
    heading = ""
    for number, line in enumerate(lines, start=1):
        if number > 1 and line.startswith("#"):
            pieces.extend(_windows(lines[start - 1 : number - 1], heading, origin=start))
            start = number
            heading = line.lstrip("#").strip()
    pieces.extend(_windows(lines[start - 1 :], heading, origin=start))
    return [piece for piece in pieces if piece.text.strip()]


def _windows(lines: list[str], symbol: str, *, origin: int = 1) -> list[CodePiece]:
    cleaned = list(lines)
    if not any(line.strip() for line in cleaned):
        return []
    pieces: list[CodePiece] = []
    start = 0
    total = len(cleaned)
    while start < total:
        end = start
        while end < total:
            if end > start and token_count("\n".join(cleaned[start : end + 1])) > CHUNK_TARGET_TOKENS:
                break
            end += 1
        if end == start:
            end = start + 1
        pieces.append(
            CodePiece(
                "\n".join(cleaned[start:end]),
                origin + start,
                origin + end - 1,
                symbol,
            )
        )
        if end >= total:
            break
        overlap_at = end
        while (
            overlap_at > start
            and token_count("\n".join(cleaned[overlap_at - 1 : end])) < CHUNK_OVERLAP_TOKENS
        ):
            overlap_at -= 1
        nxt = overlap_at if start < overlap_at < end else end
        start = nxt
    return pieces


def _join(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end])


def _storable(text: str) -> str:
    """Store a method so the slice parses, without changing a top-level function."""
    dedented = textwrap.dedent(text)
    try:
        ast.parse(dedented)
    except SyntaxError:
        return text
    return dedented
