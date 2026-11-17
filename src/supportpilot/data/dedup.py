"""Near-duplicate detection (MinHash + LSH, verified by exact Jaccard) and union-find groups.

Deterministic: shingle hashes use blake2b, permutation parameters come from a seeded
generator, and every candidate pair is confirmed with exact character-shingle Jaccard.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

_MASK32 = np.uint64(0xFFFFFFFF)
_SHIFT = np.uint64(32)


class UnionFind:
    """Union-find over string keys; the root of a set is its lexicographically smallest key."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, key: str) -> None:
        self._parent.setdefault(key, key)

    def find(self, key: str) -> str:
        self.add(key)
        root = key
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[key] != root:
            self._parent[key], key = root, self._parent[key]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self._parent[hi] = lo


def shingles(key: str, k: int) -> frozenset[str]:
    padded = f" {key} "
    if len(padded) <= k:
        return frozenset({padded})
    return frozenset(padded[i : i + k] for i in range(len(padded) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _hash64(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


class MinHasher:
    def __init__(self, num_perm: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, 2**63, size=num_perm, dtype=np.uint64) | np.uint64(1)
        self.b = rng.integers(0, 2**63, size=num_perm, dtype=np.uint64)

    def signature(self, sh: Iterable[str]) -> npt.NDArray[np.uint64]:
        h = np.fromiter((_hash64(s) for s in sh), dtype=np.uint64)
        # multiply-shift hashing: uint64 arithmetic wraps, the top 32 bits are the hash
        with np.errstate(over="ignore"):
            vals = ((h[:, None] * self.a[None, :] + self.b[None, :]) >> _SHIFT) & _MASK32
        return np.asarray(vals.min(axis=0), dtype=np.uint64)


def near_duplicate_pairs(
    keys: Sequence[str],
    *,
    shingle: int,
    threshold: float,
    num_perm: int,
    bands: int,
    seed: int,
) -> list[tuple[int, int, float]]:
    """Index pairs (i < j) whose shingle Jaccard is >= threshold, sorted."""
    sh = [shingles(k, shingle) for k in keys]
    hasher = MinHasher(num_perm, seed)
    rows = num_perm // bands
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for i, s in enumerate(sh):
        sig = hasher.signature(s)
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(i)
    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                candidates.add((members[x], members[y]))
    out: list[tuple[int, int, float]] = []
    for i, j in sorted(candidates):
        score = jaccard(sh[i], sh[j])
        if score >= threshold:
            out.append((i, j, score))
    return out
