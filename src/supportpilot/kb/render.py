"""Doc spec + facts -> one rendered document per language (format-independent).

The rendered document is what the writers put into a file and what the gates expect to get
back: its `placements` say which fact appears where with which exact surface string, and
`content_strings` is every visible string except the title and the effective-date line
(those are document metadata, checked against KBDocument.title / effective_from).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from supportpilot.common.schemas import LANGUAGES, DocFormat, DocType, LanguageCode
from supportpilot.kb.spec import (
    DocSpec,
    Filler,
    Hit,
    KBSources,
    Lookup,
    Record,
    effective_line,
    pick,
    record_values,
    zone_values,
)

HEADING_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+")

Cell = str | int


@dataclass(frozen=True)
class Placement:
    key: str
    value: Any
    surface: str
    section_id: str | None  # None = intro text before the first section
    where: str  # intro | paragraph | table | record


@dataclass(frozen=True)
class RTable:
    header: list[str]
    rows: list[list[Cell]]


@dataclass(frozen=True)
class RSection:
    id: str
    heading: str  # as printed, with the number for numbered documents
    level: int
    paragraphs: list[str]
    table: RTable | None = None


@dataclass(frozen=True)
class RRecord:
    key: str  # record key as printed (SKU, city code)
    section_id: str
    cells: list[Cell]


@dataclass(frozen=True)
class RRecords:
    sheet: str | None
    header: list[str]
    rows: list[RRecord]
    int_columns: frozenset[int] = frozenset()


@dataclass
class RenderedDoc:
    doc_id: str
    lang: LanguageCode
    doc_type: DocType
    format: DocFormat
    store_name: str
    title: str
    effective_from: date
    intro: list[str]
    sections: list[RSection] = field(default_factory=list)
    records: RRecords | None = None
    placements: list[Placement] = field(default_factory=list)
    effective_to: date | None = None

    @property
    def effective_text(self) -> str:
        return effective_line(self.effective_from, self.lang)

    @property
    def path(self) -> str:
        return f"{self.lang}/{self.doc_id}.{self.format.value}"

    def content_strings(self) -> list[str]:
        out = list(self.intro)
        for s in self.sections:
            out.append(s.heading)
            out.extend(s.paragraphs)
            if s.table is not None:
                out.extend(s.table.header)
                out.extend(cell_text(c, self.lang) for row in s.table.rows for c in row)
        if self.records is not None:
            if self.records.sheet and self.format is DocFormat.XLSX:  # only a workbook has sheets
                out.append(self.records.sheet)
            out.extend(self.records.header)
            for r in self.records.rows:
                out.extend(cell_text(c, self.lang) for c in r.cells)
        return [s for s in out if s]


def cell_text(cell: Cell, lang: LanguageCode) -> str:
    """How a cell reads in text: integers get the language's digit grouping."""
    from supportpilot.data.textfmt import group_digits

    return group_digits(cell, lang) if isinstance(cell, int) else cell


def plain_heading(heading: str) -> str:
    return HEADING_NUMBER_RE.sub("", heading).strip()


def render_doc(
    spec: DocSpec, sources: KBSources, lang: LanguageCode, lookup: Lookup
) -> RenderedDoc:
    filler = Filler(sources.facts, lookup)
    store = filler.fill("{store_name:store}", lang)[0]
    doc = RenderedDoc(
        doc_id=spec.doc_id,
        lang=lang,
        doc_type=spec.doc_type,
        format=spec.format,
        store_name=store,
        title=spec.title.get(lang),
        effective_from=spec.effective_from or sources.effective_from,
        effective_to=spec.effective_to,
        intro=[],
    )

    def put(hits: list[Hit], section_id: str | None, where: str) -> None:
        doc.placements.extend(Placement(h.key, h.value, h.surface, section_id, where) for h in hits)

    for p in spec.intro:
        text, hits = filler.fill(p.get(lang), lang)
        doc.intro.append(text)
        put(hits, None, "intro")

    n1 = n2 = 0
    for s in spec.sections:
        heading = s.heading.get(lang)
        if spec.numbered:
            if s.level == 1:
                n1, n2 = n1 + 1, 0
                heading = f"{n1}. {heading}"
            else:
                n2 += 1
                heading = f"{n1}.{n2}. {heading}"
        paragraphs = []
        for p in s.paragraphs:
            text, hits = filler.fill(p.get(lang), lang)
            paragraphs.append(text)
            put(hits, s.id, "paragraph")
        table = None
        if s.table is not None:
            rows: list[list[Cell]] = []
            for row in s.table.rows:
                cells: list[Cell] = []
                for cell in row:
                    text, hits = filler.fill(pick(cell, lang), lang)
                    cells.append(text)
                    put(hits, s.id, "table")
                rows.append(cells)
            table = RTable(header=[c.get(lang) for c in s.table.columns], rows=rows)
        doc.sections.append(RSection(s.id, heading, s.level, paragraphs, table))

    if spec.records is not None:
        doc.records = _render_records(spec, sources, lang, filler, put)
    return doc


def _render_records(
    spec: DocSpec, sources: KBSources, lang: LanguageCode, filler: Filler, put: Any
) -> RRecords:
    assert spec.records is not None
    rs, cat = spec.records, sources.catalog
    recs: list[tuple[Record, bool]] = []
    if rs.source == "products":
        for p in cat.products:
            rec = Record(prefix=f"product.{p.sku}", values=record_values(p, cat))
            recs.append((rec, p.cat in cat.warranty_categories))
    else:
        for z in cat.delivery_zones:
            values = zone_values(z, cat)
            alias = {"delivery_days": values.pop("delivery_days")}
            recs.append((Record(prefix=f"zone.{z.id}", values=values, aliases=alias), True))
    rows = []
    for rec, warranty in recs:
        key = str(rec.values[rs.key])
        section_id = key.lower()
        cells: list[Cell] = []
        for col in rs.columns:
            if col.when == "warranty" and not warranty:
                cells.append("")
                continue
            text, hits = filler.fill(col.value, lang, rec)
            put(hits, section_id, "record")
            if col.type == "int":
                if len(hits) != 1 or not isinstance(hits[0].value, int):
                    raise ValueError(f"{spec.doc_id}: int column {col.value!r} is not one integer")
                cells.append(hits[0].value)
            else:
                cells.append(text)
        rows.append(RRecord(key=key, section_id=section_id, cells=cells))
    ints = frozenset(i for i, c in enumerate(rs.columns) if c.type == "int")
    return RRecords(
        sheet=rs.sheet.get(lang) if rs.sheet else None,
        header=[c.header.get(lang) for c in rs.columns],
        rows=rows,
        int_columns=ints,
    )


def render_all(sources: KBSources, lookup: Lookup) -> list[RenderedDoc]:
    return [render_doc(spec, sources, lang, lookup) for spec in sources.docs for lang in LANGUAGES]
