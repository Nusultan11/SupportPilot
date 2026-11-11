from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from supportpilot.common.ids import make_example_id, make_record_id
from supportpilot.common.manifest import load_source_registry
from supportpilot.common.schemas import (
    AllowedUse,
    CanonicalExample,
    DatasetManifest,
    FileEntry,
    RawManifest,
    SourceRegistry,
)
from tests.data_helpers import entry

SHA = "a" * 64


def _example(**overrides: Any) -> dict[str, Any]:
    rid = make_record_id("bitext_retail_ecom", 42)
    base: dict[str, Any] = {
        "example_id": make_example_id("bitext_retail_ecom", rid, "kk", "original", "Қайда?"),
        "record_id": rid,
        "source_id": "bitext_retail_ecom",
        "scenario_id": "track_order:001",
        "family_id": "track_order:001:a",
        "lang": "kk",
        "code_switch": False,
        "text": "Менің тапсырысым қайда?",
        "intent": "track_order",
        "action": "call_tool",
        "escalate": False,
        "origin": "original",
        "quality": {"lid_conf": 0.97},
    }
    base.update(overrides)
    return base


def test_valid_canonical_example() -> None:
    ex = CanonicalExample.model_validate(_example())
    assert ex.schema_version == "canonical/1"
    assert ex.quality.flags == []


def test_valid_escalation_and_translation() -> None:
    src_id = _example()["example_id"]
    CanonicalExample.model_validate(
        _example(
            action="escalate",
            escalate=True,
            escalation_reason="explicit request for operator",
            intent="human_agent",
            origin="machine_translated",
            translation_of=src_id,
            translation_system="tilmash@rev",
        )
    )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"lang": "kz"}, "lang"),
        ({"intent": "where_is_my_parcel"}, "unknown intent"),
        ({"escalate": True}, "escalation_reason"),
        ({"escalate": True, "escalation_reason": "  "}, "escalation_reason"),
        ({"action": "escalate"}, "requires escalate"),
        ({"origin": "machine_translated"}, "translation_of"),
        ({"record_id": "other_source:42"}, "record_id must start"),
        ({"record_id": "Bad Id"}, "record_id"),
        ({"example_id": "XYZ"}, "example_id"),
        ({"text": ""}, "text"),
        ({"text": "Order {{Order Number}}"}, "placeholder"),
        ({"quality": {"lid_conf": 1.5}}, "lid_conf"),
        ({"split": "dev"}, "split"),
        ({"unexpected": 1}, "unexpected"),
        ({"action": "answer"}, "action"),
    ],
)
def test_invalid_canonical_example(override: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        CanonicalExample.model_validate(_example(**override))


def test_canonical_example_is_frozen() -> None:
    ex = CanonicalExample.model_validate(_example())
    with pytest.raises(ValidationError):
        ex.lang = "ru"  # type: ignore[misc]


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
    common = {"source_id": "bitext_retail_ecom", "revision": "r1", "url": "https://x"}
    RawManifest(**common, retrieved_at=datetime.now(UTC), files=[f])
    with pytest.raises(ValidationError):
        RawManifest(**common, retrieved_at=datetime(2026, 1, 1), files=[f])
    with pytest.raises(ValidationError):
        RawManifest(**common, retrieved_at=datetime.now(UTC), files=[])


def test_dataset_manifest_version_pattern() -> None:
    f = FileEntry(path="train.parquet", bytes=10, sha256=SHA)
    kwargs: dict[str, Any] = {
        "data_schema_version": "sft/1",
        "code_commit": "abc",
        "inputs": ["bitext_retail_ecom@r1"],
        "row_counts": {"train/kk": 10},
        "files": [f],
        "created_at": datetime.now(UTC),
    }
    DatasetManifest(dataset_version="sft-20261111-3fa2c1d0", **kwargs)
    with pytest.raises(ValidationError):
        DatasetManifest(dataset_version="sft-final", **kwargs)
