"""Near-duplicate detection, union-find and split assignment."""

from __future__ import annotations

from collections import Counter

from supportpilot.data.corpus import assign_split, balanced_splits
from supportpilot.data.dedup import UnionFind, jaccard, near_duplicate_pairs, shingles
from supportpilot.data.recipe import Recipe
from supportpilot.data.records import Split

RECIPE = Recipe.model_validate(
    {"seed": 7, "split": {"train": 0.8, "validation": 0.1, "test": 0.1}, "sources": {}}
)


def test_near_duplicates_found_and_unrelated_ignored() -> None:
    keys = [
        "i want to return the jacket i bought",
        "i want to return the jacket i bought yesterday",
        "what is the weather in almaty",
        "wake me up at seven",
    ]
    pairs = near_duplicate_pairs(keys, shingle=4, threshold=0.7, num_perm=64, bands=16, seed=1)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]
    assert pairs == near_duplicate_pairs(
        keys, shingle=4, threshold=0.7, num_perm=64, bands=16, seed=1
    )
    assert jaccard(shingles(keys[0], 4), shingles(keys[1], 4)) >= 0.7


def test_union_find_root_is_smallest_key() -> None:
    uf = UnionFind()
    uf.union("c", "b")
    uf.union("b", "a")
    uf.add("z")
    assert {uf.find(k) for k in "abc"} == {"a"}
    assert uf.find("z") == "z"


def test_assign_split_is_deterministic_and_heldout_never_trains() -> None:
    groups = [f"g{i}" for i in range(4000)]
    splits = [assign_split(RECIPE, g, heldout=False) for g in groups]
    assert splits == [assign_split(RECIPE, g, heldout=False) for g in groups]
    share = Counter(splits)
    assert 0.77 < share[Split.TRAIN] / 4000 < 0.83
    assert all(assign_split(RECIPE, g, heldout=True) is not Split.TRAIN for g in groups)


def test_balanced_splits_follow_ratios_for_few_large_groups() -> None:
    sizes = {f"doc{i}": 20 + (i * 37) % 60 for i in range(200)}
    out = balanced_splits(RECIPE, sizes)
    rows: Counter[Split] = Counter()
    for g, s in out.items():
        rows[s] += sizes[g]
    total = sum(sizes.values())
    assert abs(rows[Split.VALIDATION] / total - 0.1) < 0.02
    assert abs(rows[Split.TEST] / total - 0.1) < 0.02
    assert out == balanced_splits(RECIPE, dict(reversed(sizes.items())))
