"""Text repair, artefact flags, PII masking and dedup keys (supportpilot.data.textnorm)."""

from __future__ import annotations

import pytest

from supportpilot.data.textnorm import (
    PiiMasker,
    artefact_flags,
    clean_text,
    exact_key,
    garbled,
    is_offensive,
    levenshtein,
    luhn_ok,
    sim_key,
)


def test_clean_text_repairs_artefacts_and_reports_them() -> None:
    raw = "Hello​ world &amp; co\r\nline2\x07  end  "
    out = clean_text(raw)
    assert out.text == "Hello world & co\nline2 end"
    assert {"fix:crlf", "fix:zero_width", "fix:unicode_spaces", "fix:html_entities"} <= set(
        out.fixes
    )
    assert "fix:control_chars" in out.fixes


def test_clean_text_keeps_content_typos_case_and_emoji() -> None:
    raw = "wanna cancle my ORDER plz 😅"
    assert clean_text(raw).text == raw
    assert clean_text(raw).fixes == ()


def test_clean_text_applies_nfc() -> None:
    decomposed = "\u0438\u0306"  # cyrillic i + combining breve
    assert clean_text(decomposed).text == "й"
    assert "fix:nfc" in clean_text(decomposed).fixes


def test_exact_key_ignores_case_punctuation_and_quotes() -> None:
    assert exact_key("I can’t find my order!!") == exact_key("i can't find my order")
    assert exact_key("track order <order_id>?") == "track order <order_id>"


@pytest.mark.parametrize(
    "variant",
    [
        "I want to cancel my fucking order",
        "I want to cancel my fuckijg order",
        "I want to cancel my damn order",
    ],
)
def test_sim_key_drops_inserted_profanity(variant: str) -> None:
    assert sim_key(variant) == sim_key("I want to cancel my order")


@pytest.mark.parametrize("word", ["funky", "fluke", "flicking", "focus", "fork", "Ducking"])
def test_profanity_detector_keeps_ordinary_words(word: str) -> None:
    assert not is_offensive(f"a {word} item")
    assert word.casefold() in sim_key(f"a {word} item")


def test_levenshtein_with_limit() -> None:
    assert levenshtein("fucking", "fuckijg") == 1
    assert levenshtein("abc", "abcdefgh", limit=2) == 3  # stops early above the limit


def test_artefact_flags() -> None:
    assert "offensive" in artefact_flags("this fucking app")
    assert "short_text" in artefact_flags("hi")
    assert "all_caps" in artefact_flags("WHY IS MY ORDER LATE AGAIN")
    assert "url" in artefact_flags("see https://example.com/x")
    assert "brace_artifact" in artefact_flags("SiteNumber {7")
    assert "brace_artifact" not in artefact_flags("order {{Order Number}}")
    assert "replacement_char" in artefact_flags("bad � byte")
    assert "repeated_chars" in artefact_flags("hiiiiii")


def test_garbled_detects_multi_script_token_soup_only() -> None:
    soup = "payment 我找设施 контроль פתרון मदद help with the order please now"
    assert garbled(soup)
    assert not garbled("Үзіндіге сәйкес, Jardín de la Unión-де не табылмады?")  # 2 scripts
    assert not garbled("短")  # too few letters to judge


def test_pii_masker_replaces_with_reserved_synthetic_values() -> None:
    m = PiiMasker()
    out = m.mask(
        "mail me at jane.doe2004@gmail.com or call +7 701 234 56 78, card 4111 1111 1111 1111"
    )
    assert "jane.doe2004" not in out
    assert "@example.com" in out
    assert "701 234" not in out
    assert "+1 202 555 01" in out
    assert "4111 1111" not in out
    assert "4000 0000 0000 0002" in out
    assert sorted(set(m.found)) == ["pii:card", "pii:email", "pii:phone"]
    assert PiiMasker().mask("mail me at jane.doe2004@gmail.com") == m.mask(
        "mail me at jane.doe2004@gmail.com"
    )  # deterministic


def test_pii_masker_leaves_order_numbers_and_non_luhn_digits() -> None:
    m = PiiMasker()
    text = "cancel purchase113542617735902 and order 1234567890123 #12588"
    assert m.mask(text) == text
    assert m.found == []


def test_luhn() -> None:
    assert luhn_ok("4111111111111111")
    assert not luhn_ok("4111111111111112")
