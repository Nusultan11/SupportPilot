"""KBDocument -> KBChunks, by document type (DATA_CONTRACTS §8, ADR-027).

- policy / terms: one chunk per section (section-aware). The first line is the heading path
  ("4. Personal data > 4.1. What data we store"), then the section text, then its tables as
  pipe tables with the header row. A level-1 section's own text and each level-2 section are
  separate chunks.
- faq: one chunk per question + answer; never split unless it exceeds the token budget.
- catalog and other record documents: one chunk per record: "title · sheet", then one
  `header: value` line per non-empty cell (the header and the row key stay in every chunk).
- intro text (before the first heading): one chunk under the document title.
A unit over `max_tokens` is split at paragraph, then sentence, boundaries; table rows are
split in groups with the header repeated. Every part repeats the heading line.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_chunk_id
from supportpilot.common.schemas import KBChunk, KBDocument, KBSection, KBTable
from supportpilot.data.textfmt import group_digits

CHUNKER_REVISION = 1
TokenCounter = Callable[[Sequence[str]], list[int]]
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_INT = re.compile(r"^\d+$")


def pipe_table(t: KBTable, rows: Sequence[Sequence[str]] | None = None) -> list[str]:
    body = t.rows if rows is None else rows
    lines = ["| " + " | ".join(t.header) + " |", "|" + "|".join("---" for _ in t.header) + "|"]
    return lines + ["| " + " | ".join(r) + " |" for r in body]


def record_lines(t: KBTable, lang: str) -> list[str]:
    """`header: value` per non-empty cell; an all-digit cell is a number (digit grouping)."""
    return [
        f"{h}: {group_digits(int(v), lang) if _INT.match(v) else v}"
        for h, v in zip(t.header, t.rows[0], strict=True)
        if v
    ]


def _split_text(lines: list[str], head: str, fits: Callable[[str], bool]) -> Iterator[str]:
    """Greedy packing of lines (paragraphs, sentences) under the heading line."""
    buf: list[str] = []
    for line in lines:
        if fits("\n".join([head, *buf, line])):
            buf.append(line)
            continue
        if buf:
            yield "\n".join([head, *buf])
            buf = []
        if fits("\n".join([head, line])):
            buf = [line]
            continue
        parts = _SENTENCE.split(line)
        if len(parts) == 1:
            yield "\n".join([head, line])  # one sentence over budget: kept whole, gate reports it
            continue
        yield from _split_text(parts, head, fits)
    if buf:
        yield "\n".join([head, *buf])


Unit = tuple[KBSection, list[str], list[str]]


def _units(doc: KBDocument) -> Iterator[Unit]:
    """(section, section_path, text lines) per chunking unit, in document order. A section
    with only a heading and children yields no unit of its own (its heading is in theirs)."""

    def walk(s: KBSection, parents: list[str]) -> Iterator[Unit]:
        path = [*parents, s.heading] if s.heading else [*parents]
        lines = [p for p in s.text.split("\n") if p]
        if s.locator.record_id is not None or lines or s.tables or not s.children:
            yield s, path, lines
        for c in s.children:
            yield from walk(c, path)

    for s in doc.sections:
        yield from walk(s, [])


def chunk_document(
    doc: KBDocument,
    *,
    max_tokens: int,
    count_tokens: TokenCounter,
) -> list[KBChunk]:
    def fits(text: str) -> bool:
        return count_tokens([text])[0] <= max_tokens

    drafts: list[tuple[KBSection, list[str], str, int]] = []
    for s, path, lines in _units(doc):
        tables = s.tables
        if s.locator.record_id is not None and tables:
            sheet = s.locator.sheet
            head = f"{doc.title} · {sheet}" if sheet else doc.title
            text = "\n".join([head, *record_lines(tables[0], doc.lang)])
            drafts.append((s, [*([sheet] if sheet else []), s.heading or ""], text, 0))
            continue
        head = " > ".join(path) if path else doc.title
        whole_section = "\n".join([head, *lines, *(x for t in tables for x in pipe_table(t))])
        if fits(whole_section):
            drafts.append((s, path or [doc.title], whole_section, 0))
            continue
        texts = list(_split_text(lines, head, fits)) if lines else []
        for t in tables:
            whole = "\n".join([head, *pipe_table(t)])
            if fits(whole):
                texts.append(whole)
                continue
            buf: list[list[str]] = []
            for row in t.rows:
                if buf and not fits("\n".join([head, *pipe_table(t, [*buf, row])])):
                    texts.append("\n".join([head, *pipe_table(t, buf)]))
                    buf = []
                buf.append(row)
            if buf:
                texts.append("\n".join([head, *pipe_table(t, buf)]))
        if not texts:
            texts = [head]
        for part, text in enumerate(texts):
            drafts.append((s, path or [doc.title], text, part))

    counts = count_tokens([d[2] for d in drafts])
    chunks = []
    for pos, ((s, path, text, part), n) in enumerate(zip(drafts, counts, strict=True)):
        group = s.section_id or ("intro" if s.level == 0 else f"pos{pos}")
        chunks.append(
            KBChunk(
                chunk_id=make_chunk_id(doc.doc_id, doc.version, pos),
                doc_id=doc.doc_id,
                version=doc.version,
                lang=doc.lang,
                position=pos,
                part=part,
                section_id=s.section_id,
                section_path=path,
                title=doc.title,
                text=text,
                locator=s.locator,
                doc_type=doc.doc_type,
                format=doc.format,
                source_path=doc.source_path,
                effective_from=doc.effective_from,
                effective_to=doc.effective_to,
                token_count=max(1, n),
                translation_group=f"{doc.doc_id}#{group}",
                checksum=sha256_text(text),
            )
        )
    return chunks
