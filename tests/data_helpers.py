"""Shared builders for data-pipeline tests (registry entries, fake remotes, raw dirs)."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from supportpilot.common.manifest import scan_files, write_model_json
from supportpilot.common.schemas import RawManifest, SourceEntry
from supportpilot.data.acquire import RemoteFile

REV = "0123456789abcdef0123456789abcdef01234567"


def entry(source_id: str = "demo_src", /, **overrides: Any) -> SourceEntry:
    data: dict[str, Any] = {
        "source_id": source_id,
        "url": f"https://huggingface.co/datasets/org/{source_id}",
        "revision": REV,
        "licence": "cc-by-4.0",
        "licence_evidence": "test",
        "allowed_use": "train",
        "status": "selected",
        "languages": ["en"],
        "reviewed_at": date(2026, 11, 11),
        "purpose": "test",
        "decision": "test",
        "fetch": {"repo_id": f"org/{source_id}", "include": ["*.csv", "README.md"]},
    }
    data.update(overrides)
    return SourceEntry.model_validate(data)


def git_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()


class Interrupted(BaseException):
    """Simulates the process being killed (not caught by the retry loop)."""


class FakeRemote:
    """In-memory remote. `lfs` files carry sha256, others a git blob sha1 (like HF)."""

    def __init__(self, files: dict[str, bytes], lfs: frozenset[str] = frozenset()) -> None:
        self.files = files
        self.lfs = lfs
        self.list_calls = 0
        self.downloads: list[str] = []
        self.fail_next: dict[str, int] = {}  # path -> number of transient failures
        self.corrupt_next: dict[str, int] = {}  # path -> number of corrupted deliveries
        self.interrupt_on: str | None = None  # write half the file, then die

    def list_files(self) -> list[RemoteFile]:
        self.list_calls += 1
        out = []
        for path, data in self.files.items():
            if path in self.lfs:
                out.append(RemoteFile(path, len(data), sha256=hashlib.sha256(data).hexdigest()))
            else:
                out.append(RemoteFile(path, len(data), git_sha1=git_sha1(data)))
        return out

    def download(self, file: RemoteFile, dest_root: Path) -> None:
        self.downloads.append(file.path)
        target = dest_root / file.path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = self.files[file.path]
        if self.fail_next.get(file.path, 0) > 0:
            self.fail_next[file.path] -= 1
            raise ConnectionError("simulated network error")
        if self.interrupt_on == file.path:
            target.write_bytes(data[: len(data) // 2])
            self.interrupt_on = None
            raise Interrupted
        if self.corrupt_next.get(file.path, 0) > 0:
            self.corrupt_next[file.path] -= 1
            target.write_bytes(b"X" + data[1:])
            return
        target.write_bytes(data)


def make_raw(data_dir: Path, e: SourceEntry, files: dict[str, bytes | str]) -> Path:
    """Create an acquired raw revision (files + RawManifest) without any network."""
    raw = data_dir / "raw" / e.source_id / e.revision
    for rel, content in files.items():
        p = raw / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            with p.open("w", encoding="utf-8", newline="") as fh:
                fh.write(content)
        else:
            p.write_bytes(content)
    manifest = RawManifest(
        source_id=e.source_id,
        revision=e.revision,
        url=e.url,
        retrieved_at=datetime.now(UTC),
        licence=e.licence,
        files=scan_files(raw, exclude=frozenset({"manifest.json"})),
    )
    write_model_json(raw / "manifest.json", manifest)
    return raw
