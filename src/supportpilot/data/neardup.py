"""Near-duplicate candidates: character shingles -> MinHash -> LSH, verified by exact Jaccard.

Deterministic on every machine: shingle ids follow input order, hash coefficients come from
a seeded PCG64 generator, and no Python `hash()` is used. LSH only proposes candidates; every
reported pair carries its *exact* Jaccard similarity (sparse-matrix intersection), so the
threshold applies to real similarity, not to the MinHash estimate (ADR-022).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

_PRIME = np.uint64((1 << 31) - 1)
_PAIR_CHUNK = 500_000

IntArray = npt.NDArray[np.int64]


def shingles(text: str, k: int) -> list[str]:
    """Distinct character k-grams in order of first occurrence (never set order: string
    hashing is randomised per process, and the id order feeds the MinHash)."""
    padded = f" {text} "
    if len(padded) <= k:
        return [padded]
    return list(dict.fromkeys(padded[i : i + k] for i in range(len(padded) - k + 1)))


@dataclass
class ShingleMatrix:
    """Binary doc x shingle matrix (CSR) plus per-doc shingle counts."""

    matrix: sparse.csr_matrix
    sizes: npt.NDArray[np.float64]
    docs: list[npt.NDArray[np.uint64]]


def build_shingles(texts: Sequence[str], k: int) -> ShingleMatrix:
    vocab: dict[str, int] = {}
    docs: list[npt.NDArray[np.uint64]] = []
    indptr = [0]
    indices: list[int] = []
    for text in texts:
        ids = sorted(vocab.setdefault(s, len(vocab)) for s in shingles(text, k))
        docs.append(np.asarray(ids, dtype=np.uint64))
        indices.extend(ids)
        indptr.append(len(indices))
    if len(vocab) >= int(_PRIME):  # pragma: no cover - 2^31 distinct shingles
        raise ValueError("too many distinct shingles for 31-bit hashing")
    data = np.ones(len(indices), dtype=np.float32)
    matrix = sparse.csr_matrix(
        (data, np.asarray(indices, dtype=np.int64), np.asarray(indptr, dtype=np.int64)),
        shape=(len(texts), max(len(vocab), 1)),
    )
    sizes = np.diff(matrix.indptr).astype(np.float64)
    return ShingleMatrix(matrix, sizes, docs)


def minhash(
    docs: Sequence[npt.NDArray[np.uint64]], num_perm: int, seed: int
) -> npt.NDArray[np.uint32]:
    """(n_docs, num_perm) signatures with universal hashing h(x) = (a*x + b) mod (2^31 - 1)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    a = rng.integers(1, int(_PRIME), size=num_perm, dtype=np.uint64)[:, None]
    b = rng.integers(0, int(_PRIME), size=num_perm, dtype=np.uint64)[:, None]
    sig = np.empty((len(docs), num_perm), dtype=np.uint32)
    for i, ids in enumerate(docs):
        sig[i] = ((a * ids[None, :] + b) % _PRIME).min(axis=1)
    return sig


def _bucket_pairs(keys: npt.NDArray[np.void]) -> Iterator[IntArray]:
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    ends = np.r_[starts[1:], len(keys)]
    for s, e in zip(starts, ends, strict=True):
        if e - s < 2:
            continue
        members = np.sort(order[s:e])
        i, j = np.triu_indices(len(members), k=1)
        yield np.stack([members[i], members[j]], axis=1).astype(np.int64)


def lsh_candidates(sig: npt.NDArray[np.uint32], bands: int) -> IntArray:
    """Unique (i, j) pairs, i < j, that share at least one LSH band."""
    n, perms = sig.shape
    rows = perms // bands
    if rows < 1:
        raise ValueError("lsh_bands must not exceed minhash_perms")
    found: list[IntArray] = []
    for band in range(bands):
        block = np.ascontiguousarray(sig[:, band * rows : (band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        found.extend(_bucket_pairs(keys))
    if not found:
        return np.empty((0, 2), dtype=np.int64)
    flat = np.concatenate([p[:, 0] * n + p[:, 1] for p in found])
    flat = np.unique(flat)
    return np.stack([flat // n, flat % n], axis=1)


def exact_jaccard(
    sm: ShingleMatrix, pairs: IntArray, *, nnz_budget: float = 2e7
) -> npt.NDArray[np.float64]:
    """Exact Jaccard per pair. Work is chunked so one chunk touches <= `nnz_budget` entries."""
    out = np.empty(len(pairs), dtype=np.float64)
    if len(pairs) == 0:
        return out
    cost = sm.sizes[pairs[:, 0]] + sm.sizes[pairs[:, 1]]
    start = 0
    while start < len(pairs):
        end = int(np.searchsorted(np.cumsum(cost[start : start + _PAIR_CHUNK]), nnz_budget))
        end = start + max(1, end)
        p = pairs[start:end]
        inter = np.asarray(sm.matrix[p[:, 0]].multiply(sm.matrix[p[:, 1]]).sum(axis=1)).ravel()
        union = sm.sizes[p[:, 0]] + sm.sizes[p[:, 1]] - inter
        out[start:end] = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        start = end
    return out


@dataclass
class NearDupResult:
    pairs: IntArray  # verified pairs with jaccard >= min_jaccard
    jaccard: npt.NDArray[np.float64]
    candidates: int


def near_duplicates(
    texts: Sequence[str],
    *,
    shingle_size: int,
    num_perm: int,
    bands: int,
    seed: int,
    min_jaccard: float,
) -> NearDupResult:
    sm = build_shingles(texts, shingle_size)
    sig = minhash(sm.docs, num_perm, seed)
    cand = lsh_candidates(sig, bands)
    jac = exact_jaccard(sm, cand)
    keep = jac >= min_jaccard
    return NearDupResult(pairs=cand[keep], jaccard=jac[keep], candidates=len(cand))
