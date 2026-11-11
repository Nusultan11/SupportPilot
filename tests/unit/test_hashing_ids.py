from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import pytest

from supportpilot.common.hashing import HashingReader, sha256_bytes, sha256_file, sha256_text
from supportpilot.common.ids import (
    CHUNK_ID_RE,
    EXAMPLE_ID_RE,
    make_chunk_id,
    make_dataset_version,
    make_example_id,
    make_record_id,
    normalize_text,
)

ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
DOC_SHA = "3fa2c1d0" + "0" * 56


def test_sha256_known_vector() -> None:
    assert sha256_bytes(b"abc") == ABC_SHA
    assert sha256_text("abc") == ABC_SHA


def test_sha256_file_streams_large_file(tmp_path: Path) -> None:
    data = bytes(range(256)) * (5 * 4096)  # 5 MiB, larger than one read chunk
    p = tmp_path / "big.bin"
    p.write_bytes(data)
    assert sha256_file(p) == sha256_bytes(data)


def test_hashing_reader_hashes_while_reading() -> None:
    r = HashingReader(io.BytesIO(b"abc"))
    assert r.read(2) + r.read() == b"abc"
    assert r.bytes_read == 3
    assert r.hexdigest() == ABC_SHA


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  Сәлем   әлем  ", "Сәлем әлем"),
        ("a\r\nb\rc", "a\nb\nc"),
        ("line1 \n\n\n\n  line2", "line1\n\nline2"),
        ("\u0438\u0306", "\u0439"),  # NFD -> NFC (i + combining breve)
        ("Tab\tand\u00a0nbsp", "Tab and nbsp"),
        ("Қайтару Саясаты", "Қайтару Саясаты"),  # case and KK letters preserved
    ],
)
def test_normalize_text(raw: str, expected: str) -> None:
    assert normalize_text(raw) == expected


def test_example_id_is_deterministic_and_normalisation_aware() -> None:
    rid = make_record_id("bitext_retail_ecom", 17)
    a = make_example_id("bitext_retail_ecom", rid, "kk", "original", "Тапсырыс  қайда?")
    b = make_example_id("bitext_retail_ecom", rid, "kk", "original", "Тапсырыс қайда? ")
    c = make_example_id("bitext_retail_ecom", rid, "ru", "original", "Тапсырыс қайда?")
    assert a == b
    assert a != c
    assert EXAMPLE_ID_RE.fullmatch(a)


def test_example_id_rejects_separator_in_components() -> None:
    with pytest.raises(ValueError, match=r"\|"):
        make_example_id("src", "src:1", "en|x", "original", "hi")


def test_record_id() -> None:
    assert make_record_id("bitext_retail_ecom", "Row-12.a") == "bitext_retail_ecom:Row-12.a"


@pytest.mark.parametrize(
    ("source", "key"), [("Bitext", "1"), ("bad id", "1"), ("ok", "a b"), ("ok", "x:y"), ("ok", "")]
)
def test_record_id_invalid(source: str, key: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        make_record_id(source, key)


def test_chunk_id() -> None:
    cid = make_chunk_id("returns_policy", DOC_SHA, 2)
    assert cid == "returns_policy:3fa2c1d0:0002"
    assert CHUNK_ID_RE.fullmatch(cid)
    assert make_chunk_id("faq", DOC_SHA, 12345).endswith(":12345")


@pytest.mark.parametrize(
    ("doc", "version", "pos"),
    [("Returns", DOC_SHA, 0), ("faq", "abc", 0), ("faq", DOC_SHA.upper(), 0), ("faq", DOC_SHA, -1)],
)
def test_chunk_id_invalid(doc: str, version: str, pos: int) -> None:
    with pytest.raises(ValueError, match=r"invalid|version|position"):
        make_chunk_id(doc, version, pos)


def test_dataset_version() -> None:
    assert make_dataset_version("sft", date(2026, 11, 11), DOC_SHA) == "sft-20261111-3fa2c1d0"
    with pytest.raises(ValueError, match="invalid dataset name"):
        make_dataset_version("SFT", date(2026, 11, 11), DOC_SHA)
