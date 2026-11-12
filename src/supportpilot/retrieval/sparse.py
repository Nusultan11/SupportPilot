"""Stable Unicode sparse encoding; Qdrant supplies global collection IDF.

Fixed length pivot avoids re-encoding unchanged documents after a corpus update.
The corpus-average BM25S baseline is evaluated separately. No language-specific
stopword list removes Kazakh words. Prefix features are an explicit ablation.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter

from supportpilot.retrieval.models import Recipe

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def tokenize(text: str, mode: str = "word") -> list[str]:
    words = WORD.findall(unicodedata.normalize("NFKC", text).casefold())
    if mode == "prefix":
        return words + ["p:" + w[:4] for w in words if len(w) >= 6 and w.isalpha()]
    return words


def term_id(token: str) -> int:
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")


def sparse_vector(
    text: str, recipe: Recipe, *, query: bool = False
) -> tuple[list[int], list[float]]:
    counts = Counter(tokenize(text, recipe.sparse))
    pairs = []
    seen: set[int] = set()
    norm = recipe.bm25_k1 * (
        1 - recipe.bm25_b + recipe.bm25_b * sum(counts.values()) / recipe.length_pivot
    )
    for token, count in counts.items():
        idx = term_id(token)
        if idx in seen:
            raise ValueError("sparse hash collision")
        seen.add(idx)
        value = 1.0 if query else count * (recipe.bm25_k1 + 1) / (count + norm)
        pairs.append((idx, value))
    pairs.sort()
    return [p[0] for p in pairs], [p[1] for p in pairs]
