"""Deterministic identifiers (DATA_CONTRACTS §1). Same input → same ID on every machine."""

from __future__ import annotations

import re

SLUG_RE = re.compile(r"^[a-z0-9]+(?:[_-][a-z0-9]+)*$")


def validate_slug(value: str, what: str = "slug") -> str:
    if not SLUG_RE.fullmatch(value):
        raise ValueError(
            f"invalid {what} {value!r}: expected lowercase [a-z0-9] words joined by _ or -"
        )
    return value
