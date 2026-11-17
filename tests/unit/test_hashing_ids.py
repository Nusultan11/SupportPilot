from __future__ import annotations

import io
from pathlib import Path

import pytest

from supportpilot.common.hashing import HashingReader, sha256_bytes, sha256_file, sha256_text
from supportpilot.common.ids import validate_slug

ABC_SHA = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


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


@pytest.mark.parametrize("bad", ["Bitext", "bad id", "a__", ""])
def test_validate_slug_rejects(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid"):
        validate_slug(bad)


def test_validate_slug_accepts() -> None:
    assert validate_slug("massive_en_ru") == "massive_en_ru"
