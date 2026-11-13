"""Portable index snapshots and restartable, fail-closed Qdrant synchronisation.

A recipe owns one materialised collection. Immutable local snapshots and vector cache
allow rollback without model inference. During mutations readers refuse service; the
ready state is published only after an exact server inventory check. No stale KB mixing.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import numpy.typing as npt
from qdrant_client import models as qm

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.manifest import read_model_json, scan_files, verify_files, write_model_json
from supportpilot.common.provenance import code_commit
from supportpilot.common.schemas import DatasetManifest, KBChunk
from supportpilot.data.eda import load_tokenizer
from supportpilot.retrieval.corpus import load_kb, search_units
from supportpilot.retrieval.encode import Encoder, Transformer, VectorCache
from supportpilot.retrieval.models import Recipe, SearchUnit, load_models
from supportpilot.retrieval.sparse import sparse_vector, term_id, tokenize


def default_recipe(settings: Settings) -> Recipe:
    cfg = settings.retrieval
    spec = load_models(settings.resolve(cfg.models))[cfg.encoder]
    return Recipe(
        encoder=spec,
        view=cfg.view,
        window_tokens=cfg.window_tokens,
        overlap_tokens=cfg.overlap_tokens,
        sparse=cfg.sparse,
        tokenizer_sha256=settings.eda.tokenizers[0].sha256,
    )


def index_root(settings: Settings, recipe: Recipe) -> Path:
    root = settings.resolve(settings.paths.data_dir) / "indexes" / "retrieval"
    if settings.retrieval.namespace:
        root /= settings.retrieval.namespace
    return root / recipe.identity


def point_id(unit_id: str) -> str:
    return str(UUID(unit_id[:32]))


@contextmanager
def writer_lock(path: Path) -> Iterator[None]:
    """OS-owned advisory lock survives abrupt process loss without stale PID files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = path.open("a+b")
    try:
        fd.seek(0, 2)
        if fd.tell() == 0:
            fd.write(b"0")
            fd.flush()
        fd.seek(0)
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("retrieval index writer is busy") from exc
        yield
    finally:
        fd.close()


@dataclass
class IndexArtifact:
    path: Path
    manifest: DatasetManifest
    recipe: Recipe
    chunks: list[KBChunk]
    units: list[SearchUnit]
    vectors: npt.NDArray[np.float32]

    @property
    def kb_version(self) -> str:
        return self.manifest.inputs[0].removeprefix("kb@")


def read_index(path: Path) -> IndexArtifact:
    manifest = read_model_json(path / "manifest.json", DatasetManifest)
    if path.name != manifest.dataset_version:
        raise ValueError("index directory/manifest version mismatch")
    if verify_files(path, manifest.files, exclude=frozenset({"manifest.json"})):
        raise ValueError("index artifact checksum mismatch")
    recipe = Recipe.model_validate_json((path / "recipe.json").read_text())
    if recipe.model_dump(mode="json") != manifest.params:
        raise ValueError("index manifest/recipe mismatch")
    units = [
        SearchUnit.model_validate_json(v)
        for v in (path / "units.jsonl").read_text("utf-8").splitlines()
    ]
    chunks = [
        KBChunk.model_validate_json(v)
        for v in (path / "chunks.jsonl").read_text("utf-8").splitlines()
    ]
    vectors = np.load(path / "vectors.npy", allow_pickle=False)
    if vectors.shape != (len(units), recipe.encoder.dimensions) or not np.isfinite(vectors).all():
        raise ValueError("invalid index vector matrix")
    if len({point_id(u.unit_id) for u in units}) != len(units):
        raise ValueError("duplicate index point IDs")
    by_chunk = {c.chunk_id: c for c in chunks}
    if len(by_chunk) != len(chunks):
        raise ValueError("duplicate canonical index chunks")
    for unit in units:
        for chunk_id in unit.chunk_ids:
            chunk = by_chunk.get(chunk_id)
            if chunk is None or (
                chunk.lang,
                chunk.doc_id,
                chunk.effective_from,
                chunk.effective_to,
            ) != (unit.lang, unit.doc_id, unit.effective_from, unit.effective_to):
                raise ValueError("search unit has missing or inconsistent canonical evidence")
    return IndexArtifact(path, manifest, recipe, chunks, units, vectors)


def _build_artifact(
    settings: Settings,
    recipe: Recipe,
    *,
    version: str | None = None,
    encoder: Encoder | None = None,
    tokenizer: Any = None,
) -> IndexArtifact:
    kb_version, chunks, _ = load_kb(settings, version)
    root = index_root(settings, recipe)
    identity = sha256_text(recipe.identity + kb_version)[:8]
    iv = f"rindex-{kb_version.split('-')[-2]}-{identity}"
    out = root / "versions" / iv
    if out.exists():
        return read_index(out)
    if tokenizer is None:
        tokenizer = load_tokenizer(settings.eda.tokenizers[0])
    units = search_units(chunks, recipe, tokenizer)
    # Check hash collisions across the complete sparse vocabulary, not just one unit.
    vocabulary: dict[int, str] = {}
    for unit in units:
        for token in tokenize(unit.text, recipe.sparse):
            idx = term_id(token)
            if idx in vocabulary and vocabulary[idx] != token:
                raise ValueError("corpus sparse vocabulary hash collision")
            vocabulary[idx] = token
    owned = encoder is None
    model = Transformer(recipe.encoder, settings.retrieval.device) if owned else encoder
    assert model is not None
    cache = VectorCache(
        settings.resolve(settings.paths.data_dir) / "indexes/embedding_cache", recipe.encoder, model
    )
    try:
        vectors = cache.encode([u.text for u in units])
    finally:
        if owned and isinstance(model, Transformer):
            model.close()
    partial = root / f".partial-{iv}"
    partial.mkdir(parents=True, exist_ok=True)
    (partial / "units.jsonl").write_text(
        "".join(u.model_dump_json() + "\n" for u in units), "utf-8"
    )
    (partial / "chunks.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in chunks), "utf-8"
    )
    write_json(partial / "recipe.json", recipe.model_dump(mode="json"))
    np.save(partial / "vectors.npy", vectors, allow_pickle=False)
    manifest = DatasetManifest(
        dataset_version=iv,
        data_schema_version="retrieval_index/1",
        code_commit=code_commit(settings.root) or "unknown",
        inputs=[f"kb@{kb_version}", f"model@{recipe.encoder.repo_id}@{recipe.encoder.revision}"],
        params=recipe.model_dump(mode="json"),
        row_counts={"chunks": len(chunks), "units": len(units)},
        files=scan_files(partial, exclude=frozenset({"manifest.json"})),
        created_at=datetime.now(UTC),
    )
    write_model_json(partial / "manifest.json", manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial.rename(out)
    return read_index(out)


def build_artifact(
    settings: Settings,
    recipe: Recipe,
    *,
    version: str | None = None,
    encoder: Encoder | None = None,
    tokenizer: Any = None,
) -> IndexArtifact:
    with writer_lock(index_root(settings, recipe) / "artifact.lock"):
        return _build_artifact(
            settings, recipe, version=version, encoder=encoder, tokenizer=tokenizer
        )


def qdrant_call(method: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    for attempt in range(3):
        try:
            return method(*args, **kwargs)
        except (
            ResponseHandlingException,
            UnexpectedResponse,
            TimeoutError,
            ConnectionError,
        ) as exc:
            if (
                isinstance(exc, UnexpectedResponse)
                and exc.status_code is not None
                and exc.status_code < 500
                and exc.status_code != 429
            ):
                raise ValueError(f"Qdrant rejected request: HTTP {exc.status_code}") from exc
            if attempt == 2:
                raise ValueError("Qdrant unavailable after three bounded attempts") from exc
            time.sleep(0.1 * 2**attempt)
    raise AssertionError("unreachable")


def collection_name(settings: Settings, recipe: Recipe) -> str:
    suffix = f"_{settings.retrieval.namespace}" if settings.retrieval.namespace else ""
    return f"sp_{sha256_text(str(settings.root))[:8]}_{recipe.identity}{suffix}"


def inventory(client: Any, name: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = qdrant_call(
            client.scroll, name, limit=256, offset=offset, with_payload=True, with_vectors=False
        )
        for point in points:
            payload = point.payload or {}
            result[str(point.id)] = payload
        if offset is None:
            return result


def point_payload(unit: SearchUnit, artifact: IndexArtifact) -> dict[str, Any]:
    payload = unit.model_dump(mode="json")
    payload.update(
        kb_version=artifact.kb_version,
        index_version=artifact.manifest.dataset_version,
        valid_from=unit.effective_from.toordinal(),
        valid_to=unit.effective_to.toordinal() if unit.effective_to else 3652059,
    )
    return payload


def sync_index(
    settings: Settings,
    artifact: IndexArtifact,
    client: Any,
    *,
    after_batch: Callable[[], None] | None = None,
) -> dict[str, Any]:
    root = index_root(settings, artifact.recipe)
    state_path = root / "state.json"
    name = collection_name(settings, artifact.recipe)
    started = time.perf_counter()
    with writer_lock(root / "writer.lock"):
        if not qdrant_call(client.collection_exists, name):
            qdrant_call(
                client.create_collection,
                name,
                vectors_config={
                    "dense": qm.VectorParams(
                        size=artifact.recipe.encoder.dimensions, distance=qm.Distance.COSINE
                    )
                },
                sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
            )
        existing = inventory(client, name)
        _, _, changes = load_kb(settings, artifact.kb_version)
        prior = json.loads(state_path.read_text()) if state_path.exists() else {}
        transition = "inventory_diff"
        if prior.get("ready") and prior.get("kb_version") == changes.get("parent"):
            previous_chunks = {cid for p in existing.values() for cid in p["chunk_ids"]}
            target_chunks = {c.chunk_id for c in artifact.chunks}
            if (
                set(changes.get("chunks_delete", [])) != previous_chunks - target_chunks
                or set(changes.get("chunks_upsert", [])) != target_chunks - previous_chunks
            ):
                raise ValueError("changes.json disagrees with index inventory")
            transition = "changes.json"
        desired = {point_id(u.unit_id): u for u in artifact.units}
        expected = {pid: point_payload(unit, artifact) for pid, unit in desired.items()}
        if existing == expected:
            write_json(
                state_path,
                {
                    "ready": True,
                    "index_version": artifact.manifest.dataset_version,
                    "kb_version": artifact.kb_version,
                    "collection": name,
                },
            )
            return {
                "status": "skipped",
                "collection": name,
                "upserted": 0,
                "deleted": 0,
                "seconds": time.perf_counter() - started,
            }
        write_json(
            state_path,
            {"ready": False, "target": artifact.manifest.dataset_version, "collection": name},
        )
        removed = sorted(set(existing) - set(desired))
        if removed:
            qdrant_call(client.delete, name, qm.PointIdsList(points=removed), wait=True)
        added = [i for i, u in enumerate(artifact.units) if point_id(u.unit_id) not in existing]
        for start in range(0, len(added), 32):
            points = []
            for i in added[start : start + 32]:
                unit = artifact.units[i]
                ids, values = sparse_vector(unit.text, artifact.recipe)
                payload = expected[point_id(unit.unit_id)]
                points.append(
                    qm.PointStruct(
                        id=point_id(unit.unit_id),
                        vector={
                            "dense": artifact.vectors[i].tolist(),
                            "sparse": qm.SparseVector(indices=ids, values=values),
                        },
                        payload=payload,
                    )
                )
            qdrant_call(client.upsert, name, points=points, wait=True)
            if after_batch:
                after_batch()
        kept = sorted(set(existing) & set(desired))
        metadata_keys = {"kb_version", "index_version"}
        metadata_changed = [
            pid
            for pid in kept
            if {k: v for k, v in existing[pid].items() if k not in metadata_keys}
            != {k: v for k, v in expected[pid].items() if k not in metadata_keys}
        ]
        for pid in metadata_changed:
            # Validity/type edits need no vector inference, but must replace stale filters.
            qdrant_call(
                client.overwrite_payload, name, payload=expected[pid], points=[pid], wait=True
            )
        if kept:
            qdrant_call(
                client.set_payload,
                name,
                payload={
                    "kb_version": artifact.kb_version,
                    "index_version": artifact.manifest.dataset_version,
                },
                points=kept,
                wait=True,
            )
        verified = inventory(client, name)
        if verified != expected:
            raise ValueError("Qdrant inventory verification failed; index remains unavailable")
        write_json(
            state_path,
            {
                "ready": True,
                "index_version": artifact.manifest.dataset_version,
                "kb_version": artifact.kb_version,
                "collection": name,
            },
        )
        _, _, changes = load_kb(settings, artifact.kb_version)
        return {
            "status": "synchronised",
            "collection": name,
            "upserted": len(added),
            "deleted": len(removed),
            "unchanged_vectors": len(kept),
            "metadata_updates": len(metadata_changed),
            "changes_parent": changes.get("parent"),
            "transition": transition,
            "seconds": time.perf_counter() - started,
        }


def require_ready(settings: Settings, artifact: IndexArtifact) -> str:
    root = index_root(settings, artifact.recipe)
    with writer_lock(root / "writer.lock"):
        state = json.loads((root / "state.json").read_text())
        if (
            not state.get("ready")
            or state.get("index_version") != artifact.manifest.dataset_version
        ):
            raise ValueError("retrieval index is incomplete or stale; run retrieval index")
        return str(state["collection"])
