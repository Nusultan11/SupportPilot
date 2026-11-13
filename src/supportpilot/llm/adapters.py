"""Portable adapter manifests; validate bytes and base revision before inference."""

from __future__ import annotations

from pathlib import Path

from supportpilot.common.io import read_json, write_json
from supportpilot.common.manifest import scan_files, verify_files
from supportpilot.common.schemas import FileEntry
from supportpilot.llm.models import ModelSpec

MANIFEST = "adapter_manifest.json"


def write_adapter_manifest(path: Path, model: ModelSpec, training_identity: str) -> None:
    write_json(
        path / MANIFEST,
        {
            "schema_version": "llm_adapter/1",
            "base_model": model.model_dump(mode="json"),
            "training_identity": training_identity,
            "files": [
                entry.model_dump(mode="json") for entry in scan_files(path, frozenset({MANIFEST}))
            ],
        },
    )


def validate_adapter(path: Path, model: ModelSpec) -> None:
    if not (path / MANIFEST).is_file():
        raise ValueError("adapter has no verified manifest")
    manifest = read_json(path / MANIFEST)
    if manifest["schema_version"] != "llm_adapter/1" or manifest["base_model"] != model.model_dump(
        mode="json"
    ):
        raise ValueError("adapter base model/revision mismatch")
    entries = [FileEntry.model_validate(item) for item in manifest["files"]]
    if verify_files(path, entries, frozenset({MANIFEST})):
        raise ValueError("adapter integrity check failed")
