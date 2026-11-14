"""Portable adapter manifests; validate bytes and base revision before inference."""

from __future__ import annotations

import shutil
from pathlib import Path

from supportpilot.common.hashing import sha256_file, sha256_text
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


def export_inference_adapter(
    source: Path, target: Path, model: ModelSpec, *, model_card: str | None = None
) -> Path:
    """Preserve verified inference bytes; keep training pickle only in the source archive."""
    if target.resolve().is_relative_to(source.resolve()):
        raise ValueError("inference export cannot be inside the source archive")
    validate_adapter(source, model)
    origin = sha256_file(source / MANIFEST)
    card_hash = sha256_text(model_card) if model_card is not None else None
    if (target / MANIFEST).exists():
        validate_adapter(target, model)
        previous = read_json(target / MANIFEST)
        if (
            previous.get("source_adapter_manifest_sha256") != origin
            or previous.get("model_card_sha256") != card_hash
        ):
            raise ValueError("inference export is immutable; use a distinct destination")
        return target
    if target.exists() and any(target.iterdir()):
        raise ValueError("inference export destination is not empty")
    selected: list[tuple[Path, Path]] = []
    excluded: dict[str, str] = {}
    for file in source.rglob("*"):
        if file.is_symlink():
            raise ValueError("inference export does not follow symlinks")
        if not file.is_file() or file.name == MANIFEST:
            continue
        relative = file.relative_to(source)
        if relative.as_posix() == "training_args.bin":
            excluded[relative.as_posix()] = sha256_file(file)
            continue
        if file.suffix not in {".json", ".safetensors", ".txt", ".model", ".jinja", ".md"}:
            raise ValueError("unexpected file in inference adapter source")
        selected.append((file, relative))
    target.mkdir(parents=True, exist_ok=True)
    for file, relative in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file, destination)
    if model_card is not None:
        (target / "README.md").write_text(model_card, encoding="utf-8", newline="\n")
    original = read_json(source / MANIFEST)
    write_adapter_manifest(target, model, original["training_identity"])
    manifest = read_json(target / MANIFEST)
    manifest.update(
        source_adapter_manifest_sha256=origin,
        model_card_sha256=card_hash,
        excluded_source_files=excluded,
    )
    write_json(target / MANIFEST, manifest)
    validate_adapter(target, model)
    return target
