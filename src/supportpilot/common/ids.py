"""Deterministic identifiers (DATA_CONTRACTS §1). Same input → same ID on every machine."""

from __future__ import annotations

import re
import unicodedata
from datetime import date

from supportpilot.common.hashing import SHA256_RE, sha256_text

SLUG_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")
SOURCE_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+$")
EXAMPLE_ID_RE = re.compile(r"^[0-9a-f]{16}$")
RECORD_ID_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*:[A-Za-z0-9._-]+$")
CHUNK_ID_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*:[0-9a-f]{8}:\d{4,}$")
DATASET_VERSION_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*-\d{8}-[0-9a-f]{8}$")

_INLINE_WS = re.compile(r"[^\S\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Canonical text form used for hashing and dedup.

    NFC, LF newlines, runs of inline whitespace collapsed to one space, lines trimmed,
    at most one blank line in a row, outer whitespace stripped. Case is preserved.
    """
    t = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    t = _INLINE_WS.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", t).strip()


def validate_slug(value: str, what: str = "slug") -> str:
    if not SLUG_RE.fullmatch(value):
        raise ValueError(
            f"invalid {what} {value!r}: expected lowercase [a-z0-9] words joined by _ or -"
        )
    return value


def make_record_id(source_id: str, source_row_key: str | int) -> str:
    """`{source_id}:{source_row_key}`. The source key keeps its original case."""
    validate_slug(source_id, "source_id")
    key = str(source_row_key)
    if not SOURCE_KEY_RE.fullmatch(key):
        raise ValueError(f"invalid source_row_key {key!r}: allowed characters are [A-Za-z0-9._-]")
    return f"{source_id}:{key}"


def make_example_id(source_id: str, record_id: str, lang: str, origin: str, text: str) -> str:
    """First 16 hex chars of sha256(source_id|record_id|lang|origin|normalize_text(text))."""
    parts = (source_id, record_id, lang, origin, normalize_text(text))
    if any("|" in p for p in parts[:4]):
        raise ValueError("ID components must not contain '|'")
    return sha256_text("|".join(parts))[:16]


def make_chunk_id(doc_id: str, version: str, position: int) -> str:
    """`{doc_id}:{version[:8]}:{position:04d}` where version is the document's sha256."""
    validate_slug(doc_id, "doc_id")
    if not SHA256_RE.fullmatch(version):
        raise ValueError("version must be a 64-char lowercase sha256 hex digest")
    if position < 0:
        raise ValueError("position must be >= 0")
    return f"{doc_id}:{version[:8]}:{position:04d}"


def make_dataset_version(name: str, created: date, content_sha256: str) -> str:
    """`{name}-{yyyymmdd}-{hash8}`."""
    validate_slug(name, "dataset name")
    if not SHA256_RE.fullmatch(content_sha256):
        raise ValueError("content_sha256 must be a 64-char lowercase sha256 hex digest")
    return f"{name}-{created:%Y%m%d}-{content_sha256[:8]}"
