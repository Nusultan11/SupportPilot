"""Split assignment and leakage checks (Stage 4, ADR-026; DATA_CONTRACTS §1, §10).

Split unit: `scenario_id` (the contract). Bitext has only 55 scenarios (one per source label
and action), so a scenario-disjoint split of Bitext would remove whole intents from test.
Decision (ADR-026): every Bitext-derived example (EN originals and their RU/KK translations,
which share the family and scenario) goes to train; validation and test are built from the
authored scenarios, assigned by scenario_id, stratified by intent and action (ADR-029).
Generalisation to new phrasings is measured on the independent gold benchmark.

Leakage checks between train and every held-out set (validation, test, gold):
- shared scenario_id / family_id;
- identical normalised user text;
- lexical near-copy: char-5-gram Jaccard >= `leak_jaccard` (MinHash/LSH + exact Jaccard);
- word n-gram overlap (n = `leak_ngram`);
- semantic near-copy: multilingual-e5 cosine >= `leak_cosine`.
Training texts that overlap assigned held-out scenarios are removed and counted, so
evaluation coverage survives filtering; the gates re-check that overlap is zero.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import Split
from supportpilot.data.neardup import near_duplicates
from supportpilot.data.textnorm import sim_key

_WORD_RE = re.compile(r"\w+", re.UNICODE)


def assign_authored(
    scenarios: Sequence[tuple[str, str, str, bool]], ratios: Mapping[str, float], seed: int
) -> dict[str, Split]:
    """(scenario_id, intent, action, trilingual) -> split (ADR-026, amended by ADR-029).

    Stratified per intent so that validation and test cover every intent: within an intent the
    scenarios that have en, ru and kk templates are ordered round-robin over their actions
    (sha256 order inside each action, no RNG state); the first ones go to test, the next to
    validation. An intent with >= 3 trilingual scenarios gets at least one of each. Scenarios
    without all three languages (code-switching only, single language) stay in train."""
    by_intent: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    out: dict[str, Split] = {}
    for sid, intent, action, trilingual in sorted(set(scenarios)):
        if trilingual:
            by_intent[intent][action].append(sid)
        else:
            out[sid] = Split.TRAIN
    for _intent, actions in sorted(by_intent.items()):
        queues = []
        for _action, ids in sorted(actions.items()):
            queues.append(sorted(ids, key=lambda s: sha256_text(f"{seed}|split|{s}")))
        queues.sort(key=lambda q: sha256_text(f"{seed}|action|{q[0]}"))
        order: list[str] = []
        while any(queues):
            for q in queues:
                if q:
                    order.append(q.pop(0))
        n = len(order)
        n_val = round(n * ratios.get("validation", 0.0))
        n_test = round(n * ratios.get("test", 0.0))
        if n >= 3:
            n_val, n_test = max(1, n_val), max(1, n_test)
        for i, sid in enumerate(order):
            if i < n_test:
                out[sid] = Split.TEST
            elif i < n_test + n_val:
                out[sid] = Split.VALIDATION
            else:
                out[sid] = Split.TRAIN
    return out


def norm_text(text: str) -> str:
    return sim_key(text)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    w = _WORD_RE.findall(text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def lexical_leaks(
    train: Sequence[str], held: Sequence[str], *, threshold: float, seed: int
) -> dict[int, float]:
    """held index -> max Jaccard with any train text (only pairs >= threshold)."""
    texts = [*map(norm_text, train), *map(norm_text, held)]
    if not train or not held:
        return {}
    res = near_duplicates(
        texts, shingle_size=5, num_perm=128, bands=32, seed=seed, min_jaccard=threshold
    )
    n = len(train)
    out: dict[int, float] = {}
    for (a, b), j in zip(res.pairs.tolist(), res.jaccard.tolist(), strict=True):
        if (a < n) == (b < n):
            continue
        h = (b if a < n else a) - n
        out[h] = max(out.get(h, 0.0), float(j))
    return out


def exact_leaks(train: Sequence[str], held: Sequence[str]) -> set[int]:
    seen = {norm_text(t) for t in train}
    return {i for i, t in enumerate(held) if norm_text(t) in seen}


def ngram_leaks(train: Sequence[str], held: Sequence[str], n: int) -> set[int]:
    grams: set[tuple[str, ...]] = set()
    for t in train:
        grams |= ngrams(t, n)
    return {i for i, t in enumerate(held) if ngrams(t, n) & grams}


def semantic_leaks(
    train_vec: np.ndarray[Any, Any], held_vec: np.ndarray[Any, Any], threshold: float
) -> dict[int, float]:
    from supportpilot.data.embed import max_cosine

    if len(train_vec) == 0 or len(held_vec) == 0:
        return {}
    best, _ = max_cosine(held_vec, train_vec)
    return {i: float(v) for i, v in enumerate(best.tolist()) if v >= threshold}
