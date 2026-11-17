from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from supportpilot.common.manifest import load_source_registry
from supportpilot.common.schemas import (
    AllowedUse,
    FileEntry,
    RawManifest,
    SourceRegistry,
)
from tests.data_helpers import entry

SHA = "a" * 64


def test_source_entry_rules() -> None:
    assert entry().allowed_use is AllowedUse.TRAIN
    assert entry(licence="CC-BY-4.0").licence == "CC-BY-4.0"  # policy lookup is case-insensitive
    entry(licence="unknown", allowed_use="blocked", status="rejected", fetch=None)
    with pytest.raises(ValidationError):
        entry(languages=[])
    with pytest.raises(ValidationError):
        entry(source_id="Bitext Retail")
    with pytest.raises(ValidationError):
        entry(reviewed_at="not a date")


def test_registry_duplicates_and_lookup(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        SourceRegistry(sources=[entry(), entry()])
    p = tmp_path / "sources.yaml"
    lines = [
        "schema_version: sources/1",
        "sources:",
        "  - source_id: kazparc",
        "    url: https://huggingface.co/datasets/issai/kazparc",
        "    revision: 41df65bd299ae0b5f2222d86f3db2f4fdd44e8e6",
        "    licence: cc-by-4.0",
        "    licence_evidence: paper",
        "    allowed_use: eval_only",
        "    status: unavailable",
        "    access: gated",
        "    languages: [kk, ru, en]",
        "    reviewed_at: 2026-11-11",
        "    purpose: MT evaluation",
        "    decision: gated",
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    reg = load_source_registry(p)
    assert reg.get("kazparc").languages == ["kk", "ru", "en"]
    with pytest.raises(KeyError):
        reg.get("missing")


@pytest.mark.parametrize("path", ["../etc/passwd", "/abs/file", "a\\b.txt", "", "."])
def test_file_entry_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        FileEntry(path=path, bytes=1, sha256=SHA)


def test_file_entry_rejects_bad_hash_and_size() -> None:
    with pytest.raises(ValidationError):
        FileEntry(path="a.csv", bytes=1, sha256="A" * 64)
    with pytest.raises(ValidationError):
        FileEntry(path="a.csv", bytes=-1, sha256=SHA)


def test_raw_manifest_requires_aware_time_and_files() -> None:
    f = FileEntry(path="train.csv", bytes=10, sha256=SHA)
    common = {"source_id": "abcd", "revision": "r1", "url": "https://x"}
    RawManifest(**common, retrieved_at=datetime.now(UTC), files=[f])
    with pytest.raises(ValidationError):
        RawManifest(**common, retrieved_at=datetime(2026, 1, 1), files=[f])
    with pytest.raises(ValidationError):
        RawManifest(**common, retrieved_at=datetime.now(UTC), files=[])


def test_taxonomy_config_matches_intents() -> None:
    from supportpilot.common.config import find_project_root
    from supportpilot.common.schemas import INTENTS
    from supportpilot.data.taxonomy import load_taxonomy

    tax = load_taxonomy(find_project_root(Path(__file__).parent) / "configs" / "taxonomy.yaml")
    assert set(tax.intents) == INTENTS
