"""Parser blocks + register entry -> KBDocument (section tree with tables and locators).

- Title: the parser's title block; only a CSV, which cannot hold one, takes the register's.
- Effective date: the parser's meta line ("Effective from ..." or JSON `effective_from`);
  only a CSV takes the register's. A document without one is an extraction error.
- Text before the first heading is a level-0 section without id (the introduction).
- Level-2 headings nest under the preceding level-1 heading.
- Section ids: the id the format carries (JSON FAQ), else the register section whose heading
  equals the found heading (printed numbers ignored, case-insensitive). Unmatched headings
  keep `section_id=None`; the `sections_linked` gate fails on them.
- Records become level-1 sections whose one-row table is the record; id = slug of the key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from supportpilot.common.ids import SLUG_RE
from supportpilot.common.schemas import DocFormat, KBDocument, KBSection, KBTable, Locator
from supportpilot.kb.blocks import (
    Block,
    ExtractionError,
    ParsedDoc,
    merge_locators,
    parse_effective,
)
from supportpilot.kb.register import RegisterEntry
from supportpilot.kb.render import plain_heading


def heading_key(text: str) -> str:
    return plain_heading(text).casefold()


@dataclass
class _Node:
    heading: str | None
    level: int
    section_id: str | None
    locs: list[Locator]
    paragraphs: list[str] = field(default_factory=list)
    tables: list[KBTable] = field(default_factory=list)
    children: list[_Node] = field(default_factory=list)

    def build(self) -> KBSection:
        return KBSection(
            section_id=self.section_id,
            heading=self.heading,
            level=self.level,
            text="\n".join(self.paragraphs),
            tables=self.tables,
            locator=merge_locators(self.locs),
            children=[c.build() for c in self.children],
        )

    def empty(self) -> bool:
        return not (self.paragraphs or self.tables or self.children or self.heading)


def _effective(blocks: list[Block], entry: RegisterEntry) -> date:
    for b in blocks:
        if b.kind == "meta":
            d = parse_effective(b.text)
            if d is None:
                try:
                    d = date.fromisoformat(b.text)
                except ValueError:
                    continue
            return d
    if entry.format is DocFormat.CSV:
        return entry.effective_from
    raise ExtractionError(f"{entry.path}: no effective-date line found")


def build_document(parsed: ParsedDoc, entry: RegisterEntry) -> KBDocument:
    blocks = parsed.blocks
    titles = [b for b in blocks if b.kind == "title"]
    if titles:
        title = titles[0].text
    elif entry.format is DocFormat.CSV:
        title = entry.title
    else:
        raise ExtractionError(f"{entry.path}: no title found")
    effective = _effective(blocks, entry)

    by_heading: dict[str, str] = {heading_key(s.heading): s.id for s in entry.sections}
    used: set[str] = set()
    intro = _Node(heading=None, level=0, section_id=None, locs=[])
    top: list[_Node] = []
    current = intro
    for b in blocks:
        if b.kind in ("title", "meta"):
            continue
        if b.kind == "heading":
            sid = b.section_id or by_heading.get(heading_key(b.text))
            if sid is not None and sid in used:
                raise ExtractionError(f"{entry.path}: section {sid!r} found twice")
            if sid is not None:
                used.add(sid)
            node = _Node(heading=b.text, level=b.level, section_id=sid, locs=[b.locator])
            if b.level >= 2 and top:
                top[-1].children.append(node)
            else:
                node.level = 1
                top.append(node)
            current = node
        elif b.kind == "paragraph":
            current.paragraphs.append(b.text)
            current.locs.append(b.locator)
        elif b.kind == "table":
            current.tables.append(KBTable(header=b.header, rows=b.rows, locator=b.locator))
            current.locs.append(b.locator)
        elif b.kind == "record":
            assert b.key is not None
            sid = b.key.lower() if SLUG_RE.fullmatch(b.key.lower()) else None
            top.append(
                _Node(
                    heading=b.key,
                    level=1,
                    section_id=sid,
                    locs=[b.locator],
                    tables=[KBTable(header=b.header, rows=b.rows, locator=b.locator)],
                )
            )
            current = top[-1]
    sections = ([intro] if intro.paragraphs or intro.tables else []) + top
    if not sections:
        raise ExtractionError(f"{entry.path}: no content")
    return KBDocument(
        doc_id=entry.doc_id,
        version=entry.sha256,
        lang=entry.lang,
        title=title,
        doc_type=entry.doc_type,
        format=entry.format,
        source_path=entry.path,
        effective_from=effective,
        parser=parsed.parser,
        sections=[s.build() for s in sections],
    )
