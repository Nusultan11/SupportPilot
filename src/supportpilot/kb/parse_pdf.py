"""PDF -> blocks with pdfplumber (MIT; chosen by measurement, ADR-027).

Born-digital PDFs only. A page without a text layer (a scan) raises ExtractionError: OCR is
not configured, and an empty page must never pass as "no content".

- Tables: `find_tables` (ruled tables). A table continued on the next page with the same
  header row is merged, and the repeated header is dropped.
- Lines outside tables come from `extract_text_lines`. The body font size is the most
  common size. Page furniture (running header, page numbers) is a line in the top/bottom
  margin zone that is smaller than the body text or repeats on every page.
- Headings are bold lines larger than the body; the largest is the title, the next sizes
  are levels 1, 2, ... Body lines are joined into paragraphs; a vertical gap larger than the
  line gap starts a new paragraph. A paragraph continues across a page break when it does
  not end with sentence punctuation.
"""

from __future__ import annotations

import io
import re
from collections import Counter
from dataclasses import dataclass
from functools import partial
from itertools import pairwise
from typing import Any

from supportpilot.common.schemas import Locator
from supportpilot.kb.blocks import Block, ExtractionError, ParsedDoc, clean, parse_effective

PARSER_REVISION = 1
MARGIN_ZONE = 0.08  # share of the page height at the top and at the bottom
_PAGE_NO = re.compile(r"\d+")
_SENTENCE_END = re.compile(r"[.!?:;»\")]$")


@dataclass
class _Line:
    page: int
    top: float
    bottom: float
    text: str
    size: float
    bold: bool
    height: float


@dataclass
class _Table:
    page: int
    top: float
    rows: list[list[str]]


def parser_id() -> str:
    import pdfplumber

    return f"pdfplumber/{pdfplumber.__version__}+kb{PARSER_REVISION}"


def _inside(obj: dict[str, Any], bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (obj["x0"] + obj["x1"]) / 2
    cy = (obj["top"] + obj["bottom"]) / 2
    return bool(x0 <= cx <= x1 and top <= cy <= bottom)


def _outside_tables(obj: dict[str, Any], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Keep every non-char object and the chars outside the table boxes."""
    return obj.get("object_type") != "char" or not any(_inside(obj, b) for b in boxes)


def _read(data: bytes) -> tuple[list[_Line], list[_Table], int]:
    import pdfplumber

    lines: list[_Line] = []
    tables: list[_Table] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        n_pages = len(pdf.pages)
        for pno, page in enumerate(pdf.pages, start=1):
            if not page.chars:
                if page.images:
                    raise ExtractionError(
                        f"page {pno} has no text layer (scanned image); OCR is not configured"
                    )
                continue
            found = page.find_tables()
            boxes = [t.bbox for t in found]
            for t in found:
                rows = [[clean(c or "") for c in row] for row in t.extract()]
                tables.append(_Table(page=pno, top=t.bbox[1], rows=rows))
            body = page.filter(partial(_outside_tables, boxes=boxes))
            for ln in body.extract_text_lines(return_chars=True, strip=True):
                chars = ln["chars"]
                if not chars or not ln["text"].strip():
                    continue
                size = Counter(round(c["size"], 1) for c in chars).most_common(1)[0][0]
                bold = sum("Bold" in c["fontname"] for c in chars) > len(chars) / 2
                lines.append(
                    _Line(pno, ln["top"], ln["bottom"], clean(ln["text"]), size, bold, page.height)
                )
    return lines, tables, n_pages


def _furniture(lines: list[_Line], body_size: float, n_pages: int) -> set[int]:
    """Indexes of running headers / footers / page numbers."""

    def zone(ln: _Line) -> bool:
        return ln.top < MARGIN_ZONE * ln.height or ln.bottom > (1 - MARGIN_ZONE) * ln.height

    keyed = Counter(_PAGE_NO.sub("#", ln.text) for ln in lines if zone(ln))
    out = set()
    for i, ln in enumerate(lines):
        if not zone(ln):
            continue
        repeated = n_pages > 1 and keyed[_PAGE_NO.sub("#", ln.text)] >= n_pages
        if ln.size < body_size - 0.5 or repeated:
            out.add(i)
    return out


def parse_pdf(data: bytes) -> ParsedDoc:
    lines, tables, n_pages = _read(data)
    if not lines and not tables:
        raise ExtractionError("the PDF has no extractable text")
    weights: Counter[float] = Counter()
    for ln in lines:
        weights[ln.size] += len(ln.text)
    body_size = weights.most_common(1)[0][0]
    drop = _furniture(lines, body_size, n_pages)
    content = [ln for i, ln in enumerate(lines) if i not in drop]
    head_sizes = sorted({ln.size for ln in content if ln.bold and ln.size > body_size + 0.5},
                        reverse=True)  # fmt: skip
    title_size = head_sizes[0] if head_sizes else None
    level_of = {s: k for k, s in enumerate(head_sizes)}  # 0 = title, 1, 2, ...

    items: list[tuple[int, float, _Line | _Table]] = [(ln.page, ln.top, ln) for ln in content]
    items += [(t.page, t.top, t) for t in tables]
    items.sort(key=lambda x: (x[0], x[1]))

    gaps = [
        b.top - a.bottom
        for a, b in pairwise(content)
        if a.page == b.page and a.size == b.size == body_size and not a.bold and not b.bold
    ]
    line_gap = min(gaps) if gaps else 0.0

    blocks: list[Block] = []
    para: list[_Line] = []
    heading_bottom = 0.0

    def flush() -> None:
        if para:
            loc = Locator(page=para[0].page, page_end=para[-1].page)
            if loc.page_end == loc.page:
                loc = Locator(page=para[0].page)
            blocks.append(Block("paragraph", loc, text=" ".join(x.text for x in para)))
            para.clear()

    for _, _, item in items:
        if isinstance(item, _Table):
            flush()
            prev = blocks[-1] if blocks else None
            if (
                prev is not None
                and prev.kind == "table"
                and prev.header == item.rows[0]
                and (prev.locator.page_end or prev.locator.page or 0) == item.page - 1
            ):
                prev.rows.extend(item.rows[1:])
                prev.locator = Locator(page=prev.locator.page, page_end=item.page)
                continue
            if len(item.rows) < 1:
                continue
            blocks.append(
                Block("table", Locator(page=item.page), header=item.rows[0], rows=item.rows[1:])
            )
            continue
        ln = item
        if ln.bold and ln.size in level_of:
            flush()
            level = level_of[ln.size]
            prev = blocks[-1] if blocks else None
            if (
                prev is not None
                and prev.kind in ("title", "heading")
                and prev.level == level
                and prev.locator.page == ln.page
                and ln.top - heading_bottom < ln.size
            ):
                prev.text = f"{prev.text} {ln.text}"  # a wrapped heading
            elif ln.size == title_size and not any(b.kind == "title" for b in blocks):
                blocks.append(Block("title", Locator(page=ln.page), text=ln.text))
            else:
                blocks.append(Block("heading", Locator(page=ln.page), text=ln.text,
                                    level=max(1, level)))  # fmt: skip
            heading_bottom = ln.bottom
            continue
        if parse_effective(ln.text) is not None and not para:
            blocks.append(Block("meta", Locator(page=ln.page), text=ln.text))
            continue
        if para:
            last = para[-1]
            new_page = ln.page != last.page
            if (new_page and _SENTENCE_END.search(last.text)) or (
                not new_page and ln.top - last.bottom > line_gap + 2.0
            ):
                flush()
        para.append(ln)
    flush()
    return ParsedDoc(blocks=blocks, parser=parser_id())
