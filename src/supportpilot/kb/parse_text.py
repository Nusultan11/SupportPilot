"""TXT and JSON -> blocks (stdlib).

TXT: a line underlined with `===` is the title, `---` a level-1 heading, `~~~` a level-2
heading; `Q:` starts a FAQ question and `A:` its answer (indented lines continue them);
lines starting with `|` form a pipe table (the `|---|` separator is dropped); other
consecutive lines form a paragraph until a blank line. Locator: 1-based line range.

JSON: a FAQ export (`items` of `{id, question, answer}`, plus title / effective_from /
intro) maps to title, meta, headings with their section id and answer paragraphs. Any other
JSON is flattened to one `path: value` paragraph per scalar leaf. Locator: JSON pointer.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from supportpilot.common.schemas import Locator
from supportpilot.kb.blocks import (
    Block,
    BlockKind,
    ExtractionError,
    ParsedDoc,
    clean,
    parse_effective,
)

PARSER_REVISION = 1
_UNDERLINE = re.compile(r"^(=+|-+|~+)$")
_SEPARATOR = re.compile(r"^\|(\s*:?-{3,}:?\s*\|)+$")


def _lines_loc(a: int, b: int) -> Locator:
    return Locator(line_range=(a, b))


def _is_underline(lines: list[str], j: int) -> bool:
    s = lines[j].strip() if j < len(lines) else ""
    return len(s) >= 3 and _UNDERLINE.match(s) is not None


def _continues(lines: list[str], j: int, mode: str) -> bool:
    """Does line j continue the current Q (q), A (a) or plain paragraph (p)?"""
    if j >= len(lines) or not lines[j].strip():
        return False
    s = lines[j]
    if mode in ("q", "a"):
        return s.startswith((" ", "\t"))
    return not s.startswith(("|", "Q:", "A:")) and not _is_underline(lines, j + 1)


def parse_txt(data: bytes) -> ParsedDoc:
    try:
        lines = data.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as e:
        raise ExtractionError(f"TXT is not UTF-8: {e}") from e
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        if _is_underline(lines, i + 1) and not line.startswith(("|", "Q:", "A:")):
            mark = lines[i + 1].strip()[0]
            if mark == "=" and not any(b.kind == "title" for b in blocks):
                blocks.append(Block("title", _lines_loc(i + 1, i + 2), text=clean(line)))
            else:
                level = 2 if mark == "~" else 1
                blocks.append(Block("heading", _lines_loc(i + 1, i + 2), text=clean(line),
                                    level=level))  # fmt: skip
            i += 2
            continue
        if line.startswith("|"):
            start = i
            rows: list[list[str]] = []
            while i < n and lines[i].startswith("|"):
                row = lines[i].strip()
                if not _SEPARATOR.match(row):
                    rows.append([clean(c) for c in row.strip("|").split("|")])
                i += 1
            blocks.append(Block("table", _lines_loc(start + 1, i), header=rows[0], rows=rows[1:]))
            continue
        start = i
        mode = "q" if line.startswith("Q:") else "a" if line.startswith("A:") else "p"
        parts = [line[2:] if mode != "p" else line]
        i += 1
        while _continues(lines, i, mode):
            parts.append(lines[i].strip())
            i += 1
        joined = clean(" ".join(parts))
        kind: BlockKind = "heading" if mode == "q" else "paragraph"
        if mode == "p" and parse_effective(joined) is not None:
            kind = "meta"
        blocks.append(Block(kind, _lines_loc(start + 1, i), text=joined, level=1))
    if not blocks:
        raise ExtractionError("the TXT file has no text")
    return ParsedDoc(blocks=blocks, parser=f"txt/stdlib+kb{PARSER_REVISION}")


def _pointer(*parts: str | int) -> str:
    return "/" + "/".join(str(p).replace("~", "~0").replace("/", "~1") for p in parts)


def parse_json(data: bytes) -> ParsedDoc:
    try:
        obj = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ExtractionError(f"invalid JSON: {e}") from e
    blocks: list[Block] = []
    items = obj.get("items") if isinstance(obj, dict) else None
    if isinstance(items, list) and all(
        isinstance(x, dict) and {"question", "answer"} <= set(x) for x in items
    ):
        if obj.get("title"):
            blocks.append(Block("title", Locator(json_pointer="/title"), text=clean(obj["title"])))
        if obj.get("effective_from"):
            try:
                d = date.fromisoformat(str(obj["effective_from"]))
            except ValueError as e:
                raise ExtractionError(f"effective_from is not an ISO date: {e}") from e
            blocks.append(
                Block("meta", Locator(json_pointer="/effective_from"), text=d.isoformat())
            )
        for k, p in enumerate(obj.get("intro") or []):
            blocks.append(Block("paragraph", Locator(json_pointer=_pointer("intro", k)),
                                text=clean(p)))  # fmt: skip
        for k, item in enumerate(items):
            sid = item.get("id")
            blocks.append(Block("heading", Locator(json_pointer=_pointer("items", k)),
                                text=clean(item["question"]), level=1,
                                section_id=str(sid) if sid else None))  # fmt: skip
            answer = item["answer"]
            parts = answer if isinstance(answer, list) else str(answer).split("\n")
            for p in parts:
                if clean(p):
                    blocks.append(Block("paragraph",
                                        Locator(json_pointer=_pointer("items", k, "answer")),
                                        text=clean(p)))  # fmt: skip
    else:

        def walk(v: Any, path: tuple[str | int, ...]) -> None:
            if isinstance(v, dict):
                for key, x in v.items():
                    walk(x, (*path, key))
            elif isinstance(v, list):
                for k, x in enumerate(v):
                    walk(x, (*path, k))
            elif v is not None and str(v).strip():
                label = ".".join(str(p) for p in path)
                pointer = _pointer(*path) if path else "/"
                text = clean(f"{label}: {v}" if label else str(v))
                blocks.append(Block("paragraph", Locator(json_pointer=pointer), text=text))

        walk(obj, ())
    if not blocks:
        raise ExtractionError("the JSON file has no text")
    return ParsedDoc(blocks=blocks, parser=f"json/stdlib+kb{PARSER_REVISION}")
