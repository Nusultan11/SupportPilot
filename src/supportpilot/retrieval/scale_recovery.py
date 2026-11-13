"""Restore a portable semantic index into a separate data root and Qdrant namespace."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.eda import load_tokenizer
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.corpus import load_kb
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.index import (
    IndexArtifact,
    point_id,
    point_payload,
    read_index,
    require_ready,
    sync_index,
)
from supportpilot.retrieval.models import SearchFilters, SearchOptions, load_models
from supportpilot.retrieval.scale_data import verify_freeze
from supportpilot.retrieval.scale_profile import uncached_request
from supportpilot.retrieval.search import Retriever
from supportpilot.retrieval.sparse import sparse_vector


def verify_semantic_vectors(client: Any, name: str, artifact: IndexArtifact) -> dict[str, Any]:
    expected = {point_id(unit.unit_id): i for i, unit in enumerate(artifact.units)}
    seen, cursor, maximum = set(), None, 0.0
    while True:
        points, cursor = client.scroll(
            name, limit=128, offset=cursor, with_payload=True, with_vectors=True
        )
        for point in points:
            pid = str(point.id)
            if pid not in expected or pid in seen:
                raise ValueError("unexpected or repeated restored index point")
            seen.add(pid)
            index = expected[pid]
            unit = artifact.units[index]
            if point.payload != point_payload(unit, artifact) or not isinstance(point.vector, dict):
                raise ValueError("restored point payload/vector schema differs")
            dense = np.asarray(point.vector["dense"], dtype=np.float32)
            if not np.allclose(dense, artifact.vectors[index], atol=1e-6, rtol=1e-5):
                raise ValueError("restored dense vector differs from portable artifact")
            maximum = max(maximum, float(np.max(np.abs(dense - artifact.vectors[index]))))
            ids, values = sparse_vector(unit.text, artifact.recipe)
            stored = point.vector["sparse"]
            if stored.indices != ids or not np.allclose(stored.values, values, atol=1e-6):
                raise ValueError("restored sparse vector differs from source features")
        if cursor is None:
            break
    if seen != expected.keys():
        raise ValueError("restored point inventory incomplete")
    return {
        "points": len(seen),
        "dense_and_sparse_verified": True,
        "maximum_dense_absolute_delta": maximum,
        "dense_tolerance": {"atol": 1e-6, "rtol": 1e-5},
    }


def recover_scale(settings: Settings, frozen: Path, scratch: Path) -> dict[str, Any]:
    """The caller first restores KB and namespace backups into scratch/data/... ."""
    import mlflow

    permitted = (settings.root / "artifacts/restore_drills").resolve()
    scratch = scratch.resolve()
    if scratch == permitted or not scratch.is_relative_to(permitted):
        raise ValueError("recovery drill requires a dedicated artifacts/restore_drills child")
    verify_freeze(frozen)
    protocol = json.loads((frozen / "protocol.json").read_text())
    root = settings.resolve(Path(protocol["output"])) / frozen.name
    selection = json.loads((root / "selection.json").read_text())
    holdout = json.loads((root / "holdout/results.json").read_text())
    if json.loads((root / "holdout/attempt.json").read_text())["status"] != "FINISHED":
        raise ValueError("finish holdout before recovery diagnostics")
    chosen = selection["configurations"][selection["selected"]]
    source = read_index(Path(holdout["backends"][chosen["encoder"]]["index_path"]))
    namespace = "restore_" + frozen.name
    data = scratch / "data"
    path = (
        data
        / "indexes/retrieval"
        / namespace
        / source.recipe.identity
        / "versions"
        / source.manifest.dataset_version
    )
    artifact = read_index(path)
    if artifact.manifest != source.manifest:
        raise ValueError("restored artifact manifest differs from original")
    options = SearchOptions.model_validate(chosen["options"])
    cfg = settings.retrieval.model_copy(
        update={
            **options.model_dump(exclude={"k", "rrf_k", "rerank_k"}),
            "namespace": namespace,
            "encoder": chosen["encoder"],
            "rerank_k": options.rerank_k or 20,
            "context_tokens": chosen["budget"],
        }
    )
    restored = settings.model_copy(
        update={
            "paths": settings.paths.model_copy(update={"data_dir": data}),
            "retrieval": cfg,
        }
    )
    load_kb(restored, artifact.kb_version)  # verify the entire restored immutable KB first
    write_json(data / "kb" / settings.kb.name / "current.json", {"kb_version": artifact.kb_version})
    queries = [
        RetrievalQuery.model_validate_json(line)
        for line in (frozen / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    meta = json.loads((frozen / "metadata.json").read_text())
    sample: list[RetrievalQuery] = []
    for scope in ("ru", "kk", "en", "cross", "mixed", "temporal"):
        query = next(
            q
            for q in queries
            if (
                q.cross_lingual
                if scope == "cross"
                else q.code_switch
                if scope == "mixed"
                else "temporal" in meta[q.query_id]["tags"]
                if scope == "temporal"
                else q.lang == scope and not q.cross_lingual and not q.code_switch
            )
        )
        if query.query_id not in {q.query_id for q in sample}:
            sample.append(query)
    out = root / "recovery" / str(time.time_ns())
    out.mkdir(parents=True)
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"{frozen.name}_portable_recovery",
        hypothesis="A backed-up semantic index serves identical evidence in an isolated runtime",
        dataset_version=frozen.name,
        seed=protocol["seed"],
        base_model=chosen["encoder"],
        control_run_id=holdout["run_id"],
        params={
            "kb_version": artifact.kb_version,
            "index_version": artifact.manifest.dataset_version,
            "namespace": namespace,
            "code_sha256": sha256_file(Path(__file__)),
            "source_manifest_sha256": sha256_file(source.path / "manifest.json"),
        },
    )
    client = make_qdrant_client(settings.qdrant, settings)
    encoder = Transformer(artifact.recipe.encoder, settings.retrieval.device)
    reranker = Transformer(
        load_models(settings.resolve(settings.retrieval.models))["bge_reranker"],
        settings.retrieval.device,
    )
    receipt: dict[str, Any] = {"status": "RUNNING"}
    try:
        with start_run(spec, settings) as run:
            receipt["run_id"] = run.info.run_id
            try:
                start = time.perf_counter()
                sync = sync_index(restored, artifact, client)
                receipt["sync"] = sync
                receipt["restore_without_model_inference_seconds"] = time.perf_counter() - start
                receipt["inventory"] = verify_semantic_vectors(client, sync["collection"], artifact)
                origin = settings.model_copy(
                    update={
                        "retrieval": settings.retrieval.model_copy(
                            update={"namespace": "scale_" + frozen.name}
                        )
                    }
                )
                collection = require_ready(origin, source)
                tokenizer = load_tokenizer(settings.eda.tokenizers[0])
                retriever = Retriever(
                    restored,
                    artifact,
                    client,
                    encoder=encoder,
                    reranker=reranker if options.rerank_k else None,
                    tokenizer=tokenizer,
                )
                checked = []
                for query in sample:
                    metadata = meta[query.query_id]
                    filters = SearchFilters(
                        corpus_lang=query.corpus_lang,
                        as_of=metadata["as_of"],
                        all_versions=metadata["all_versions"],
                    )
                    expected = uncached_request(
                        source,
                        collection,
                        client,
                        encoder,
                        reranker,
                        tokenizer,
                        query,
                        options,
                        filters,
                        chosen["budget"],
                    )
                    actual = retriever.context(
                        query.query, filters=filters, budget=chosen["budget"]
                    )
                    if expected["packed"] != [row.chunk_id for row in actual.evidence]:
                        raise ValueError("restored public serving API evidence differs")
                    checked.append(query.query_id)
                receipt.update(
                    status="FINISHED",
                    query_ids_verified=checked,
                    namespace=namespace,
                    index_version=artifact.manifest.dataset_version,
                    kb_version=artifact.kb_version,
                    scratch=str(scratch),
                )
                mlflow.set_tags({"decision": "verified_portable_restore", "cost_usd": "0"})
            except BaseException as exc:
                receipt.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
                raise
            finally:
                write_json(out / "receipt.json", receipt)
                mlflow.log_artifact(str(out / "receipt.json"))
    finally:
        encoder.close()
        reranker.close()
        client.close()
    return {"path": str(out), **receipt}
