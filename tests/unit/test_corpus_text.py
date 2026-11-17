"""Text normalisation, language signals and PII masking (data/text.py)."""

from __future__ import annotations

import pytest

from supportpilot.data.text import (
    content_words,
    dedup_key,
    detect_language,
    has_mojibake,
    mask_intro_name,
    mask_pii,
    mask_values,
    normalize_text,
    pii_values,
    residual_pii,
)


def test_normalize_text_is_nfc_and_collapses_whitespace() -> None:
    decomposed = "и" + chr(0x306)  # и + combining breve -> й after NFC
    crlf = chr(13) + chr(10)
    raw = "  a" + chr(0xA0) * 2 + "b" + crlf + crlf + " " + decomposed + chr(0xFEFF) + " "
    assert normalize_text(raw) == "a b" + chr(10) + chr(0x439)


def test_dedup_key_ignores_case_punctuation_and_yo() -> None:
    assert dedup_key("Где мой ЗАКАЗ?!") == dedup_key("где мой заказ")
    assert dedup_key("ещё") == dedup_key("еще")


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("where is my order", "en"),
        ("где мой заказ", "ru"),
        ("Тапсырысым қайда?", "kk"),
        ("бүгін ауа райы", "kk"),
        ("ол не істеп жатыр", "kk"),  # no special letters; particles decide
        ("мне не нравится бар", "ru"),  # не/бар are Russian words too
        ("12345", "unknown"),
    ],
)
def test_detect_language(text: str, lang: str) -> None:
    assert detect_language(text).guess == lang


def test_code_switch_needs_both_languages() -> None:
    assert detect_language("Сәлеметсіз бе, мне нужно вернуть заказ, қалай болады?").code_switch
    assert not detect_language("мне не нравятся эти цвета").code_switch
    assert not detect_language("Тапсырысым қайда екенін айтыңызшы").code_switch


def test_mojibake() -> None:
    assert has_mojibake("Ð¿Ñ€Ð¸Ð²ÐµÑ‚")
    assert has_mojibake("broken � text")
    assert not has_mojibake("привет, hello, сәлем")


def test_mask_pii_kinds() -> None:
    text = (
        "mail me at joe.doe+1@mail.kz or +7 701 123 45 67, card 4111 1111 1111 1111, "
        "IIN 990101300123, see https://x.kz/a"
    )
    masked, counts = mask_pii(text)
    assert "<email>" in masked
    assert "<url>" in masked
    assert counts["card"] == 1
    assert counts["phone"] == 1
    assert counts["number"] == 1
    assert residual_pii(masked) == []
    assert mask_pii("2 jeans for 5 dollars at 10 am")[1] == {}


def test_public_values_are_kept_but_others_masked() -> None:
    doc = "Call us at 1-518-473-6586 or write to dmv@ny.gov."
    public = pii_values(doc)
    masked, counts = mask_pii("Call 1-518-473-6586 or 1-212-555-0199", public)
    assert "1-518-473-6586" in masked
    assert counts == {"phone": 1}
    assert residual_pii(masked, public) == []


def test_mask_intro_name_and_known_values() -> None:
    text, n = mask_intro_name("Hi, my name is Crystal Minh and this is Trying")
    assert n == 1
    assert "Crystal" not in text
    assert "this is Trying" in text
    text, n = mask_intro_name("Сәлем, менің атым Айгүл")
    assert n == 1
    assert text.endswith("<name>")
    masked, counts = mask_values(
        "crystal minh, order 3348917502", {"name": ["Crystal Minh"], "order_id": ["3348917502"]}
    )
    assert masked == "<name>, order <order_id>"
    assert counts == {"name": 1, "order_id": 1}


def test_content_words_skip_tags() -> None:
    assert content_words("<name> <order_id>") == []
    assert content_words("I am <name>, help") == ["i", "am", "help"]
