"""Choose and chunk files from the public BMTech repository.

Each chunk has two texts. ``chunk_text`` is stored, shown to the model, and
quoted: a ``path:Lx-Ly symbol`` line, the blob URL, and the code. ``embed_text``
is only embedded: path, kind, symbol, defined names, and a one-line summary
above the code, with no commit SHA or URL, so a question about a constant or
a workflow finds the file that defines it.
"""

from __future__ import annotations

import ast
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from .config import CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS, CHUNK_TARGET_TOKENS
from .ingestion.chunk import token_count

GAP_TARGET_TOKENS = 200
MAX_DEFINED_NAMES = 12

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
    # Tests and fixtures imitate company facts and repeat suite questions verbatim.
    "tests",
    "test",
    "__tests__",
    "fixtures",
    "archive",
    "project_memory",
}
# Already indexed as knowledge-base pages.
_SKIP_PREFIXES = ("portal/content/",)
# The regression harness lists the private suite's code questions.
_SKIP_NAMES = {"ask_regress.py", "conftest.py"}
_TEST_NAME = re.compile(r"^test_.*\.py$|_test\.py$|\.(test|spec)\.(ts|tsx)$", re.IGNORECASE)
_LOCKFILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "poetry.lock",
    "uv.lock",
    "cargo.lock",
}
_MAX_BYTES = 200_000

KIND_LABELS = {
    "code": "source code",
    "config": "configuration",
    "workflow": "GitHub Actions workflow",
    "doc": "documentation",
    "pull_request": "pull request",
}


@dataclass(frozen=True, slots=True)
class CodePiece:
    """One citation-sized slice of a file."""

    text: str
    start_line: int
    end_line: int
    symbol: str
    names: tuple[str, ...] = ()
    summary: str = ""


def should_index(path: str, size: int) -> bool:
    """True for source, docs, and config. Secrets, tests, lockfiles, and build output stay out."""
    if size > _MAX_BYTES or size < 1:
        return False
    normalized = path.replace("\\", "/").lstrip("/")
    pure = PurePosixPath(normalized)
    if any(part in _SKIP_DIRS for part in pure.parts[:-1]):
        return False
    if normalized.startswith(_SKIP_PREFIXES) or any(f"/{prefix}" in f"/{normalized}" for prefix in _SKIP_PREFIXES):
        return False
    name = pure.name
    folded = name.casefold()
    if folded in _SKIP_NAMES or _TEST_NAME.search(name):
        return False
    if folded.startswith(".env") or folded.endswith((".pem", ".key")):
        return False
    if "secret" in folded or "credential" in folded:
        return False
    if folded in _LOCKFILES:
        return False
    if folded == "dockerfile" or folded.startswith("dockerfile."):
        return True
    if ".github/workflows/" in normalized and pure.suffix.casefold() in {".yml", ".yaml"}:
        return True
    return pure.suffix.casefold() in _CODE_SUFFIXES


def kind_for(path: str) -> str:
    """code, config, workflow, or doc."""
    normalized = path.replace("\\", "/")
    pure = PurePosixPath(normalized)
    suffix = pure.suffix.casefold()
    if ".github/workflows/" in normalized and suffix in {".yml", ".yaml"}:
        return "workflow"
    if suffix == ".md":
        return "doc"
    if suffix in {".yml", ".yaml", ".toml", ".sql"} or pure.name.casefold().startswith("dockerfile"):
        return "config"
    return "code"


def source_file_for(repo: str, path: str) -> str:
    """Stable chunk identity. A new commit does not change this path."""
    return f"github/{repo}/{path.lstrip('/')}"


def blob_url(repo: str, commit_sha: str, path: str, start_line: int, end_line: int) -> str:
    """GitHub blob link for the indexed commit and line range."""
    return (
        f"https://github.com/{repo}/blob/{commit_sha}/{path.lstrip('/')}"
        f"#L{start_line}-L{end_line}"
    )


def render_chunk_text(
    repo: str,
    commit_sha: str,
    path: str,
    start_line: int,
    end_line: int,
    symbol: str,
    body: str,
) -> str:
    """Stored and quoted text: location line, blob URL, then the code."""
    url = blob_url(repo, commit_sha, path, start_line, end_line)
    location = f"{path}:L{start_line}-L{end_line} {symbol}".rstrip()
    return f"{location}\n{url}\n\n{body.rstrip()}\n"


def render_embed_text(path: str, kind: str, piece: CodePiece) -> str:
    """Embedded-only text. No SHA and no URL."""
    names = ", ".join(piece.names) or "-"
    return (
        f"File: {path}\n"
        f"Kind: {KIND_LABELS.get(kind, kind)}\n"
        f"Symbol: {piece.symbol or '-'}\n"
        f"Defines: {names}\n"
        f"Summary: {piece.summary or '-'}\n\n"
        f"{piece.text.rstrip()}\n"
    )


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_SPLIT = re.compile(r"[/._\-\s]+")


def header_words(path: str, symbol: str, names: Sequence[str]) -> str:
    """Path, symbol, and defined names as lower-case words, de-duplicated in order."""
    words: list[str] = []
    for part in (path, symbol, *names):
        for piece in _SPLIT.split(part or ""):
            for word in _CAMEL.sub(" ", piece).split():
                cleaned = re.sub(r"[^0-9a-z]+", "", word.casefold())
                if cleaned and cleaned not in words:
                    words.append(cleaned)
    return " ".join(words)


def chunk_text(text: str, path: str) -> list[CodePiece]:
    """Chunk one file. Python splits on functions; markdown on headings; the rest on lines."""
    suffix = PurePosixPath(path).suffix.casefold()
    if suffix == ".py":
        return chunk_python(text)
    if suffix == ".md":
        return _chunk_markdown(text)
    pieces = _windows(text.splitlines(), "", overlap=False)
    if kind_for(path) == "workflow":
        name, crons, commands = workflow_parts(text)
        names = tuple(item for item in (name, *crons, *commands) if item)[:MAX_DEFINED_NAMES]
        summary = workflow_summary(crons, commands)
        return [CodePiece(p.text, p.start_line, p.end_line, p.symbol, names, summary) for p in pieces]
    return pieces


def workflow_parts(text: str) -> tuple[str, list[str], list[str]]:
    """The workflow ``name:``, every ``cron:`` string, and every ``run:`` command."""
    name_match = re.search(r"^name:\s*[\"']?([^\"'\n#]+)", text, flags=re.MULTILINE)
    name = name_match.group(1).strip() if name_match else ""
    crons = [item.strip() for item in re.findall(r"cron:\s*[\"']?([^\"'\n#]+)", text)]
    commands: list[str] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        found = re.match(r"^\s*(?:-\s*)?run:\s*(.*)$", line)
        if not found:
            continue
        value = found.group(1).strip()
        if not value:
            continue  # ``defaults: run:`` is a mapping, not a command
        if value in {"|", ">", "|-", ">-"}:
            value = next((item.strip() for item in lines[index + 1 :] if item.strip()), "")
        if value:
            commands.append(value)
    return name, crons, commands


def workflow_summary(crons: Sequence[str], commands: Sequence[str]) -> str:
    schedule = ", ".join(crons) or "none"
    runs = "; ".join(commands) or "none"
    return f"schedule {schedule}; runs {runs}"


def defined_names(tree: ast.Module, start: int, end: int) -> tuple[str, ...]:
    """Targets of top-level assignments inside a line range, capped."""
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if not start <= node.lineno <= end:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            for item in ast.walk(target):
                if isinstance(item, ast.Name) and item.id not in names:
                    names.append(item.id)
    return tuple(names[:MAX_DEFINED_NAMES])


def _first_doc_line(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)  # type: ignore[arg-type]
    except TypeError:
        return ""
    return doc.strip().splitlines()[0].strip() if doc and doc.strip() else ""


def chunk_python(source: str) -> list[CodePiece]:
    """Split a Python file on top-level functions and class methods."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _windows(source.splitlines(), "", overlap=False)
    lines = source.splitlines()
    covered: set[int] = set()
    pieces: list[CodePiece] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            pieces.extend(_class_pieces(node, lines, covered))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pieces.extend(_symbol_pieces(node, lines, node.name, covered))
    pieces.extend(_gaps(lines, covered, tree))
    return sorted(pieces, key=lambda piece: (piece.start_line, piece.end_line))


def _class_pieces(node: ast.ClassDef, lines: list[str], covered: set[int]) -> list[CodePiece]:
    pieces: list[CodePiece] = []
    method_lines: set[int] = set()
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pieces.extend(_symbol_pieces(child, lines, f"{node.name}.{child.name}", method_lines))
    start, end = _span(node)
    class_lines = set(range(start, end + 1)) - method_lines
    if class_lines:
        summary = _first_doc_line(node)
        pieces.extend(
            _with(piece, summary=summary) for piece in _from_line_set(lines, class_lines, node.name)
        )
    covered.update(range(start, end + 1))
    return pieces


def _symbol_pieces(
    node: ast.AST, lines: list[str], symbol: str, covered: set[int]
) -> list[CodePiece]:
    start, end = _span(node)
    signature_end = getattr(node, "lineno", start)
    summary = _first_doc_line(node)
    pieces = [
        _with(piece, summary=summary)
        for piece in _split_with_signature(lines, start, end, signature_end, symbol)
    ]
    covered.update(range(start, end + 1))
    return pieces


def _with(piece: CodePiece, *, names: tuple[str, ...] | None = None, summary: str | None = None) -> CodePiece:
    return CodePiece(
        piece.text,
        piece.start_line,
        piece.end_line,
        piece.symbol,
        piece.names if names is None else names,
        piece.summary if summary is None else summary,
    )


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


def _gaps(lines: list[str], covered: set[int], tree: ast.Module) -> list[CodePiece]:
    """Module-level code between functions: blank-line blocks packed to a small target."""
    remaining = sorted(set(range(1, len(lines) + 1)) - covered)
    module_summary = _first_doc_line(tree)
    pieces: list[CodePiece] = []
    for group in _runs(remaining):
        for block_start, block_end in _packed_blocks(lines, group[0], group[-1]):
            body = _join(lines, block_start, block_end)
            if token_count(body) <= GAP_TARGET_TOKENS:
                windows = [CodePiece(body, block_start, block_end, "")]
            else:
                windows = _windows(
                    lines[block_start - 1 : block_end],
                    "",
                    origin=block_start,
                    overlap=False,
                    target=GAP_TARGET_TOKENS,
                )
            for window in windows:
                pieces.append(
                    _with(
                        window,
                        names=defined_names(tree, window.start_line, window.end_line),
                        summary=module_summary,
                    )
                )
    return pieces


def _runs(numbers: Sequence[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for number in numbers:
        if groups and number == groups[-1][-1] + 1:
            groups[-1].append(number)
        else:
            groups.append([number])
    return groups


def _packed_blocks(lines: list[str], start: int, end: int) -> list[tuple[int, int]]:
    """Split ``start..end`` at blank lines and pack the blocks up to the gap target."""
    blocks: list[tuple[int, int]] = []
    current: int | None = None
    for number in range(start, end + 1):
        blank = not lines[number - 1].strip()
        if blank and current is not None:
            blocks.append((current, number - 1))
            current = None
        elif not blank and current is None:
            current = number
    if current is not None:
        blocks.append((current, end))
    packed: list[tuple[int, int]] = []
    for block in blocks:
        if packed:
            merged = (packed[-1][0], block[1])
            if token_count(_join(lines, *merged)) <= GAP_TARGET_TOKENS:
                packed[-1] = merged
                continue
        packed.append(block)
    return packed


def _from_line_set(lines: list[str], selected: set[int], symbol: str) -> list[CodePiece]:
    if not selected:
        return []
    pieces: list[CodePiece] = []
    for group in _runs(sorted(selected)):
        text = _join(lines, group[0], group[-1]).strip("\n")
        if not text.strip():
            continue
        pieces.extend(_windows(text.splitlines(), symbol, origin=group[0], overlap=False))
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
    return [_with(piece, summary=piece.symbol) for piece in pieces if piece.text.strip()]


def _windows(
    lines: list[str],
    symbol: str,
    *,
    origin: int = 1,
    overlap: bool = True,
    target: int = CHUNK_TARGET_TOKENS,
) -> list[CodePiece]:
    """Line windows up to ``target`` tokens. Code uses ``overlap=False`` so no line repeats."""
    cleaned = list(lines)
    if not any(line.strip() for line in cleaned):
        return []
    pieces: list[CodePiece] = []
    start = 0
    total = len(cleaned)
    while start < total:
        end = start
        while end < total:
            if end > start and token_count("\n".join(cleaned[start : end + 1])) > target:
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
        if not overlap:
            start = end
            continue
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
