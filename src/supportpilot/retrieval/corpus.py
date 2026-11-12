"""Verified KB snapshots and derived search views; qrels always keep canonical IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.manifest import read_model_json, verify_files
from supportpilot.common.schemas import DatasetManifest, KBChunk
from supportpilot.retrieval.models import Recipe, SearchUnit


def load_kb(
    settings: Settings, version: str | None = None
) -> tuple[str, list[KBChunk], dict[str, Any]]:
    root = settings.resolve(settings.paths.data_dir) / "kb" / settings.kb.name
    if version is None:
        version = json.loads((root / "current.json").read_text())["kb_version"]
    if Path(version).name != version or version in (".", ".."):
        raise ValueError("invalid KB version")
    path = root / "versions" / version
    manifest = read_model_json(path / "manifest.json", DatasetManifest)
    if manifest.dataset_version != version:
        raise ValueError("KB version/manifest mismatch")
    if verify_files(path, manifest.files, exclude=frozenset({"manifest.json"})):
        raise ValueError("KB manifest verification failed")
    chunks = [
        KBChunk.model_validate_json(v)
        for v in pl.read_parquet(path / "chunks.parquet")["chunk_json"]
    ]
    return version, chunks, json.loads((path / "changes.json").read_text())


def search_units(chunks: list[KBChunk], recipe: Recipe, tokenizer: Any) -> list[SearchUnit]:
    ordered = sorted(chunks, key=lambda c: (c.lang, c.doc_id, c.position))
    units = []
    for i, chunk in enumerate(ordered):
        texts: list[tuple[str, list[str]]] = []
        if recipe.view == "window":
            encoded = tokenizer.encode(chunk.text, add_special_tokens=False)
            offsets = encoded.offsets
            stride = recipe.window_tokens - recipe.overlap_tokens
            for start in range(0, len(offsets), stride):
                end = min(start + recipe.window_tokens, len(offsets))
                texts.append(
                    (chunk.text[offsets[start][0] : offsets[end - 1][1]], [chunk.chunk_id])
                )
                if end == len(offsets):
                    break
        elif recipe.view == "adjacent" and i + 1 < len(ordered):
            nxt = ordered[i + 1]
            if (nxt.lang, nxt.doc_id, nxt.effective_from, nxt.effective_to) == (
                chunk.lang,
                chunk.doc_id,
                chunk.effective_from,
                chunk.effective_to,
            ):
                text = chunk.text + "\n" + nxt.text
                if len(tokenizer.encode(text, add_special_tokens=False).ids) <= 400:
                    texts.append((text, [chunk.chunk_id, nxt.chunk_id]))
        if not texts:
            texts.append((chunk.text, [chunk.chunk_id]))
        for part, (text, ids) in enumerate(texts):
            # Title/path are search context, never synthetic evidence or new source facts.
            body = f"{chunk.title}\n{' > '.join(chunk.section_path)}\n{text}"
            units.append(
                SearchUnit(
                    unit_id=sha256_text(f"{recipe.identity}|{ids}|{part}|{body}"),
                    chunk_ids=ids,
                    text=body,
                    lang=chunk.lang,
                    doc_id=chunk.doc_id,
                    doc_type=chunk.doc_type.value,
                    effective_from=chunk.effective_from,
                    effective_to=chunk.effective_to,
                )
            )
    return units
