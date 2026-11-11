"""Language identification and near-duplicate primitives."""

from __future__ import annotations

import numpy as np
import pytest

from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.neardup import (
    build_shingles,
    exact_jaccard,
    lsh_candidates,
    minhash,
    near_duplicates,
    shingles,
)

LID = LanguageIdentifier()


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("Where is my order? I placed it last week and it still has not arrived.", "en"),
        ("Где мой заказ? Я оформил его неделю назад, но он так и не пришёл.", "ru"),
        ("Менің тапсырысым қайда? Мен оны бір апта бұрын рәсімдедім.", "kk"),
        ("wanna cancle my ordr plz", "en"),  # typo-heavy English stays English
    ],
)
def test_supported_languages(text: str, lang: str) -> None:
    r = LID.detect(text)
    assert r.lang == lang
    assert not r.code_switch


def test_undetermined_and_other_languages() -> None:
    assert LID.detect("?").lang == "und"
    assert LID.detect("👍👍").lang == "und"
    es = LID.detect("Hola, necesito ayuda con un pago que aparece dos veces en mi cuenta.")
    assert es.lang == "other:es"
    assert es.conf >= 0.9


def test_names_in_latin_do_not_make_code_switch() -> None:
    r = LID.detect("Итальяндық «Cinque Terre» термині нені білдіреді?")
    assert r.lang == "kk"
    assert not r.code_switch
    assert LID.detect("Что такое ASUS Eee PC?").lang == "ru"


def test_code_switch_en_ru() -> None:
    r = LID.detect("привет, я хочу вернуть товар, please help me with the refund")
    assert r.code_switch
    assert set(r.langs) == {"en", "ru"}


def test_detection_is_deterministic() -> None:
    texts = ["Where is my parcel", "Где посылка", "Сәлем, тапсырыс қайда?"]
    assert LID.detect_many(texts) == LID.detect_many(texts)


# ------------------------------------------------------------------ near-duplicates


def test_shingles_are_ordered_and_distinct() -> None:
    assert shingles("abcabc", 3) == [" ab", "abc", "bca", "cab", "bc "]
    assert shingles("ab", 5) == [" ab "]


def test_exact_jaccard_matches_set_jaccard() -> None:
    texts = ["cancel my order now", "cancel my order", "track my parcel please", ""]
    sm = build_shingles(texts, 4)
    pairs = np.array([[0, 1], [0, 2], [1, 2], [0, 3]], dtype=np.int64)
    jac = exact_jaccard(sm, pairs)
    for (a, b), j in zip(pairs.tolist(), jac.tolist(), strict=True):
        sa, sb = set(shingles(texts[a], 4)), set(shingles(texts[b], 4))
        assert j == pytest.approx(len(sa & sb) / len(sa | sb))
    # a tiny memory budget forces one pair per chunk; the result must not change
    assert exact_jaccard(sm, pairs, nnz_budget=1).tolist() == jac.tolist()


def test_minhash_is_deterministic_and_lsh_finds_near_duplicates() -> None:
    texts = [
        "i would like to cancel my order please help me",
        "i would like to cancel my order please help me now",
        "what are your store opening hours on sunday",
        "i would like to cancel my order please help",
    ]
    sm = build_shingles(texts, 5)
    s1 = minhash(sm.docs, 64, seed=7)
    s2 = minhash(build_shingles(texts, 5).docs, 64, seed=7)
    assert (s1 == s2).all()
    cand = {tuple(p) for p in lsh_candidates(s1, 16).tolist()}
    assert (0, 1) in cand
    assert (0, 3) in cand
    assert all(2 not in p for p in cand)


def test_near_duplicates_threshold_and_empty_input() -> None:
    texts = ["track my order please", "track my order please!", "completely different text here"]
    nd = near_duplicates(texts, shingle_size=5, num_perm=64, bands=16, seed=1, min_jaccard=0.8)
    assert nd.pairs.tolist() == [[0, 1]]
    empty = near_duplicates(
        ["a", "b"], shingle_size=5, num_perm=16, bands=4, seed=1, min_jaccard=0.5
    )
    assert empty.pairs.shape == (0, 2)


def test_lsh_rejects_more_bands_than_permutations() -> None:
    with pytest.raises(ValueError, match="lsh_bands"):
        lsh_candidates(np.zeros((2, 4), dtype=np.uint32), 8)
