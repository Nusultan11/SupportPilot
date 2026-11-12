"""Read the text at a locator straight from the source file (independent of the parsers'
block logic), so the `locators` gate can confirm that a chunk's locator really points at the
content: PDF page text, DOCX body blocks, XLSX sheet rows, CSV lines, JSON pointer, TXT lines.
"""

from __future__ import annotations

import io
import json
from typing import Any

from supportpilot.common.schemas import DocFormat, LanguageCode, Locator
from supportpilot.data.textfmt import group_digits
from supportpilot.kb.blocks import clean


class SourceReader:
    """Caches the opened sources of one document."""

    def __init__(self, data: bytes, fmt: DocFormat, lang: LanguageCode) -> None:
        self.data, self.fmt, self.lang = data, fmt, lang
        self._cache: dict[str, Any] = {}

    def text_at(self, loc: Locator) -> str:
        reader = {
            DocFormat.PDF: self._pdf,
            DocFormat.DOCX: self._docx,
            DocFormat.XLSX: self._xlsx,
            DocFormat.CSV: self._lines,
            DocFormat.TXT: self._lines,
            DocFormat.JSON: self._json,
        }[self.fmt]
        return clean(reader(loc))

    def _pdf(self, loc: Locator) -> str:
        if "pages" not in self._cache:
            import pdfplumber

            with pdfplumber.open(io.BytesIO(self.data)) as pdf:
                self._cache["pages"] = [p.extract_text() or "" for p in pdf.pages]
        pages: list[str] = self._cache["pages"]
        if loc.page is None:
            return ""
        return " ".join(pages[loc.page - 1 : (loc.page_end or loc.page)])

    def _docx(self, loc: Locator) -> str:
        if "blocks" not in self._cache:
            import docx
            from docx.table import Table
            from docx.text.paragraph import Paragraph

            d = docx.Document(io.BytesIO(self.data))
            out = []
            for child in d.element.body.iterchildren():
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "p":
                    out.append(Paragraph(child, d).text)
                elif tag == "tbl":
                    t = Table(child, d)
                    out.append(" ".join(c.text for r in t.rows for c in r.cells))
            self._cache["blocks"] = out
        blocks: list[str] = self._cache["blocks"]
        if loc.block_range is None:
            return ""
        a, b = loc.block_range
        return " ".join(blocks[a - 1 : b])

    def _xlsx(self, loc: Locator) -> str:
        if "sheets" not in self._cache:
            import openpyxl

            wb = openpyxl.load_workbook(io.BytesIO(self.data), read_only=True, data_only=True)
            self._cache["sheets"] = {
                ws.title: [list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets
            }
            wb.close()
        if loc.sheet is None or loc.row_range is None:
            return ""
        rows = self._cache["sheets"].get(loc.sheet, [])
        a, b = loc.row_range
        cells = [c for r in rows[a - 1 : b] for c in r if c is not None]
        return " ".join(
            group_digits(int(c), self.lang) if isinstance(c, int | float) else str(c) for c in cells
        )

    def _lines(self, loc: Locator) -> str:
        if "lines" not in self._cache:
            self._cache["lines"] = self.data.decode("utf-8-sig").splitlines()
        lines: list[str] = self._cache["lines"]
        span = loc.line_range or loc.row_range
        if span is None:
            return ""
        return " ".join(lines[span[0] - 1 : span[1]])

    def _json(self, loc: Locator) -> str:
        if "obj" not in self._cache:
            self._cache["obj"] = json.loads(self.data.decode("utf-8-sig"))
        if loc.json_pointer is None:
            return ""
        node: Any = self._cache["obj"]
        for raw in loc.json_pointer.split("/")[1:]:
            part = raw.replace("~1", "/").replace("~0", "~")
            node = node[int(part)] if isinstance(node, list) else node[part]
        return " ".join(_leaves(node))


def _leaves(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for v in node.values() for s in _leaves(v)]
    if isinstance(node, list):
        return [s for v in node for s in _leaves(v)]
    return [str(node)]
