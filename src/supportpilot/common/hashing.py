"""SHA-256 helpers. Files are hashed in a streaming way so large artefacts never load into RAM."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import BinaryIO

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHUNK_SIZE = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_stream(stream: BinaryIO) -> str:
    h = hashlib.sha256()
    while chunk := stream.read(CHUNK_SIZE):
        h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return sha256_stream(fh)


class HashingReader:
    """File-like wrapper that hashes bytes as they are read (single pass hash + copy)."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._hash = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        self._hash.update(chunk)
        self.bytes_read += len(chunk)
        return chunk

    def hexdigest(self) -> str:
        return self._hash.hexdigest()
