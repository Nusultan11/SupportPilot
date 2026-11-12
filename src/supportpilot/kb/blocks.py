"""Format-independent parser output: a flat list of blocks in reading order.

Parsers find structure in the file itself (font size and weight in PDF, paragraph styles in
DOCX, header rows in XLSX/CSV, setext headings and Q:/A: items in TXT, keys in JSON); they
never see the document spec. `structure.build_document` turns blocks into a KBDocument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

from supportpilot.common.schemas import LANGUAGES, Locator
from supportpilot.data import textfmt

BlockKind = Literal["title", "meta", "heading", "paragraph", "table", "record"]


class ExtractionError(ValueError):
    """A document cannot be extracted completely (never silently skipped)."""


@dataclass
class Block:
    kind: BlockKind
    locator: Locator
    text: str = ""
    level: int = 0
    header: list[str] = field(default_factory=list)  # table / record
    rows: list[list[str]] = field(default_factory=list)  # table rows (records: one row)
    key: str | None = None  # record key
    section_id: str | None = None  # when the format carries a section id (JSON FAQ)
    sheet: str | None = None


@dataclass
class ParsedDoc:
    blocks: list[Block]
    parser: str


_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """One line of text: NBSP -> space, whitespace runs collapsed, stripped."""
    return _WS.sub(" ", text.replace("\xa0", " ")).strip()


def merge_locators(locs: list[Locator]) -> Locator:
    """Smallest locator covering all of `locs` (same document)."""
    first = locs[0]

    def span(name: str) -> tuple[int, int] | None:
        vals = [getattr(x, name) for x in locs if getattr(x, name) is not None]
        return (min(v[0] for v in vals), max(v[1] for v in vals)) if vals else None

    pages = [p for x in locs for p in (x.page, x.page_end) if p is not None]
    return Locator(
        page=min(pages) if pages else None,
        page_end=max(pages) if pages and max(pages) != min(pages) else None,
        sheet=first.sheet,
        row_range=span("row_range"),
        block_range=span("block_range"),
        line_range=span("line_range"),
        json_pointer=first.json_pointer,
        record_id=first.record_id if len(locs) == 1 else None,
    )


# ---------------------------------------------------------------- effective-date lines


def _month_maps() -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {lang: {} for lang in LANGUAGES}
    for m in range(1, 13):
        d = date(2000, m, 1)
        out["en"][textfmt.MONTHS["en"][m - 1].casefold()] = m
        out["ru"][textfmt.MONTHS["ru"][m - 1]] = m
        out["kk"][textfmt.fmt_date(d, "kk", "abl").split(" ", 1)[1]] = m
    return out


_MONTHS = _month_maps()
_EFFECTIVE = [
    ("en", re.compile(r"^Effective from (\d{1,2}) (\w+) (\d{4})$"), (1, 2, 3)),
    ("ru", re.compile(r"^Действует с (\d{1,2}) (\w+) (\d{4}) г\.$"), (1, 2, 3)),
    ("kk", re.compile(r"^(\d{4}) жылғы (\d{1,2}) (\w+) бастап қолданылады$"), (2, 3, 1)),
]


def parse_effective(text: str) -> date | None:
    """The date of an "Effective from ..." line in en/ru/kk, else None."""
    t = clean(text)
    for lang, rx, (gd, gm, gy) in _EFFECTIVE:
        m = rx.match(t)
        if m:
            month = _MONTHS[lang].get(m.group(gm).casefold() if lang == "en" else m.group(gm))
            if month is None:
                return None
            return date(int(m.group(gy)), month, int(m.group(gd)))
    return None
