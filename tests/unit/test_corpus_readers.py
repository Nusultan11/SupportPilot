"""Raw readers: ABCD opening request, xSID CoNLL parsing, doc2dial document families."""

from __future__ import annotations

from pathlib import Path

from supportpilot.data.readers import abcd_opening, doc_family, is_greeting_only, read_xsid


def test_abcd_opening_skips_greetings_and_stops_at_agent() -> None:
    turns = [
        ["agent", "Hi! How can I help?"],
        ["customer", "Hello there"],
        ["customer", "I need to return a jacket."],
        ["customer", "It is too small."],
        ["agent", "Sure, your name?"],
        ["customer", "Crystal Minh"],
    ]
    assert abcd_opening(turns) == "I need to return a jacket. It is too small."
    assert abcd_opening([["agent", "hi"], ["customer", "hello"]]) is None
    assert is_greeting_only("Hi, how are you today?")
    assert not is_greeting_only("Hi, my order is late")


def test_read_xsid_conll(tmp_path: Path) -> None:
    d = tmp_path / "data" / "xSID-0.7"
    d.mkdir(parents=True)
    block = "# id = {i}\n# text-en = x\n# text = {t}\n# intent = weather/find\n1\tx\tO\tO\n\n"
    (d / "kk.valid.conll").write_text(block.format(i=1, t="Бүгін жаңбыр жауады ма?"), "utf-8")
    (d / "kk.test.conll").write_text(block.format(i=1, t="Ертең қар бола ма"), "utf-8")
    items = list(read_xsid(tmp_path))
    assert [(i.source_item_id, i.text) for i in items] == [
        ("valid-1", "Бүгін жаңбыр жауады ма?"),
        ("test-1", "Ертең қар бола ма"),
    ]
    assert items[0].parallel_id != items[1].parallel_id


def test_doc_family_merges_page_variants() -> None:
    assert doc_family("Learn what documents#6_0") == doc_family("Learn what documents#6_0_1_2")
    assert doc_family("A#1_0") != doc_family("B#1_0")
