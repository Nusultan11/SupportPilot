"""Live serving/context measurements and real version-transition/snapshot drills."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import numpy as np

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.data.eda import load_tokenizer
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.abstain import evidence_gate
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.evaluate import observations, summarise
from supportpilot.retrieval.experiments import benchmark, log_result, measure, prepared, report_root
from supportpilot.retrieval.index import build_artifact, inventory, sync_index
from supportpilot.retrieval.models import SearchFilters, SearchOptions, load_models
from supportpilot.retrieval.search import (
    Retriever,
    assemble_context,
    candidate_lists,
    scored_chunks,
)


def probe_batches(settings: Settings) -> dict[str, Any]:
    import torch

    root = report_root(settings)
    selected = json.loads((root / "selected_base.json").read_text())
    artifact, candidates = prepared(settings, selected["params"]["recipe"])
    options = SearchOptions.model_validate(selected["params"]["options"]).model_copy(
        update={"candidate_k": 20}
    )
    _, queries = benchmark(settings)
    sample = np.random.default_rng(settings.retrieval.seed).choice(len(queries), 32, replace=False)
    pairs, ids = [], []
    for index in sample:
        query = queries[index]
        rows = scored_chunks(
            artifact,
            candidates[query.query_id]["dense"],
            candidates[query.query_id]["sparse"],
            options,
        )
        row = rows[index % 10]
        pairs.append((query.query, f"{row.chunk.title}\n{row.chunk.text}"))
        ids.append([query.query_id, row.chunk_id])
    spec = load_models(settings.resolve(settings.retrieval.models))["qwen_reranker"]
    model = Transformer(spec, settings.retrieval.device)
    results: dict[str, Any] = {}
    try:
        model.score(pairs[:4])
        baseline = None
        for batch in (4, 8, 16):
            model.spec = spec.model_copy(update={"batch_size": batch})
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                with measure() as resources:
                    values = model.score(pairs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                if baseline is None:
                    baseline = np.array(values)
                results[str(batch)] = {
                    **vars(resources),
                    "pairs_per_second": len(pairs) / resources.seconds,
                    "max_abs_delta_from_batch4": float(np.max(np.abs(baseline - values))),
                }
            except torch.cuda.OutOfMemoryError:
                results[str(batch)] = {"status": "CUDA_OOM", "metrics": "UNKNOWN"}
                torch.cuda.empty_cache()
    finally:
        model.close()
    result = {"sample": ids, "results": results, "model": spec.model_dump()}
    write_json(root / "batch_probe.json", result)
    log_result(
        settings,
        "qwen_reranker_batch_probe",
        {"model": spec.repo_id + "@" + spec.revision, "batch_sizes": "4,8,16", "pairs": len(pairs)},
        {},
        root / "batch_probe.json",
    )
    return result


def profile(settings: Settings) -> dict[str, Any]:
    root = report_root(settings)
    selected = json.loads((root / "selected.json").read_text())
    params = selected["params"]
    artifact, candidates = prepared(settings, params["recipe"])
    options = SearchOptions.model_validate(params["options"])
    options = options.model_copy(
        update={
            "rerank_k": params.get("rerank_k"),
            "min_rerank_score": params.get("threshold"),
        }
    )
    _, queries = benchmark(settings)
    tokenizer = load_tokenizer(settings.eda.tokenizers[0])
    reranker_name = params.get("reranker")
    scored = (
        json.loads((root / "rerank" / reranker_name / "scores.json").read_text())
        if reranker_name
        else None
    )
    packed: dict[str, dict[str, list[str]]] = {
        f"{budget}_{diversify}": {} for budget in (512, 1024, 2048) for diversify in (False, True)
    }
    counts: dict[str, list[int]] = {name: [] for name in packed}
    packing_ms: list[float] = []
    for q in queries:
        rows = scored_chunks(
            artifact, candidates[q.query_id]["dense"], candidates[q.query_id]["sparse"], options
        )
        if scored is not None:
            lookup = dict(scored[q.query_id][: options.rerank_k or options.candidate_k])
            rows = [
                row.model_copy(
                    update={"score": lookup[row.chunk_id], "rerank_score": lookup[row.chunk_id]}
                )
                for row in rows
                if row.chunk_id in lookup
                and (
                    options.min_rerank_score is None
                    or lookup[row.chunk_id] >= options.min_rerank_score
                )
            ]
            rows.sort(key=lambda row: (-row.score, row.chunk_id))
        rows = evidence_gate(rows, options.evidence_threshold)
        for budget in (512, 1024, 2048):
            for diversify in (False, True):
                started = time.perf_counter()
                context = assemble_context(
                    rows[:10],
                    tokenizer,
                    budget,
                    kb_version=artifact.kb_version,
                    index_version=artifact.manifest.dataset_version,
                    diversify=diversify,
                )
                packing_ms.append((time.perf_counter() - started) * 1000)
                key = f"{budget}_{diversify}"
                packed[key][q.query_id] = [row.chunk_id for row in context.evidence]
                counts[key].append(context.token_count)
    context_results: dict[str, Any] = {}
    for key, rankings in packed.items():
        metrics = summarise(
            observations(queries, rankings, seed=settings.retrieval.seed),
            seed=settings.retrieval.seed,
        )
        context_results[key] = {
            "metrics": metrics,
            "tokens_p50": float(np.median(counts[key])),
            "tokens_max": max(counts[key]),
        }
    write_json(root / "context_comparison.json", context_results)
    baseline = json.loads((root / "comparisons" / selected["name"] / "metrics.json").read_text())[
        "development"
    ]
    acceptable = [
        budget
        for budget in (512, 1024, 2048)
        if all(
            context_results[f"{budget}_False"]["metrics"]["development"][lang]["recall5"]["mean"]
            >= max(0.85, baseline[lang]["recall5"]["mean"] - 0.01)
            for lang in ("ru", "kk", "en")
        )
    ]
    chosen_budget = min(acceptable) if acceptable else 2048
    client = make_qdrant_client(settings.qdrant, settings)
    encoder = Transformer(artifact.recipe.encoder, settings.retrieval.device)
    reranker = (
        Transformer(
            load_models(settings.resolve(settings.retrieval.models))[reranker_name],
            settings.retrieval.device,
        )
        if reranker_name
        else None
    )
    # Apply the selected options through normal public API, without embedding/pair caches.
    cfg = settings.retrieval.model_copy(
        update={
            "mode": options.mode,
            "fusion": options.fusion,
            "dense_weight": options.dense_weight,
            "candidate_k": options.candidate_k,
            "rerank_k": options.rerank_k or options.candidate_k,
            "min_rerank_score": options.min_rerank_score,
            "evidence_threshold": options.evidence_threshold,
            "context_tokens": chosen_budget,
        }
    )
    serving_settings = settings.model_copy(update={"retrieval": cfg})
    try:
        sync_index(settings, artifact, client)
        retriever = Retriever(
            serving_settings,
            artifact,
            client,
            encoder=encoder,
            reranker=reranker,
            tokenizer=tokenizer,
        )
        representative = queries
        online_rankings: dict[str, list[str]] = {}
        latency, components = [], []
        with measure() as resources:
            cold = time.perf_counter()
            retriever.context(
                representative[0].query,
                filters=SearchFilters(
                    corpus_lang=representative[0].corpus_lang,
                    as_of=settings.retrieval.evaluation_date,
                ),
            )
            cold_ms = (time.perf_counter() - cold) * 1000
            started = time.perf_counter()
            for q in representative:
                before = time.perf_counter()
                context = retriever.context(
                    q.query,
                    filters=SearchFilters(
                        corpus_lang=q.corpus_lang, as_of=settings.retrieval.evaluation_date
                    ),
                )
                online_rankings[q.query_id] = [row.chunk_id for row in context.evidence]
                latency.append((time.perf_counter() - before) * 1000)
                components.append(retriever.last_timings.copy())
            elapsed = time.perf_counter() - started
            encode_started = time.perf_counter()
            rebuilt_vectors = encoder.encode([unit.text for unit in artifact.units])
            encode_seconds = time.perf_counter() - encode_started
            vector_delta = float(np.max(np.abs(rebuilt_vectors - artifact.vectors)))
            if not np.allclose(rebuilt_vectors, artifact.vectors, atol=1e-4):
                raise ValueError("uncached encoder reproduction differs from stored index vectors")
        result = {
            "selected": selected["name"],
            "context_budget": chosen_budget,
            "context_selection": (
                "Smallest tested budget preserving each development "
                "language recall within 0.01 and at least 0.85; diversify=False"
            ),
            "queries": len(representative),
            "query_ids": [q.query_id for q in representative],
            "warm_p50_ms": float(np.quantile(latency, 0.5)),
            "warm_p95_ms": float(np.quantile(latency, 0.95)),
            "throughput_qps": len(latency) / elapsed,
            "cold_request_ms": cold_ms,
            "model_load_ms": (encoder.load_seconds + (reranker.load_seconds if reranker else 0))
            * 1000,
            "resources": vars(resources),
            "component_p50_ms": {
                key: float(np.median([c[key] for c in components])) for key in components[0]
            },
            "context_pack_p50_ms": float(np.median(packing_ms)),
            "index_artifact_bytes": sum(f.bytes for f in artifact.manifest.files),
            "uncached_document_encode_seconds": encode_seconds,
            "uncached_documents_per_second": len(artifact.units) / encode_seconds,
            "reencoded_vector_max_abs_delta": vector_delta,
            "latencies_ms": latency,
            "batch_size": 1,
            "cache": "model weights warm; no query-vector or rerank-score cache",
        }
        online_rows = observations(queries, online_rankings, seed=settings.retrieval.seed)
        online_metrics = summarise(online_rows, seed=settings.retrieval.seed)
        write_json(root / "online_context_observations.json", online_rows)
        write_json(root / "online_context_metrics.json", online_metrics)
        result["cached_context_ranking_differences"] = sum(
            online_rankings[q.query_id] != packed[f"{chosen_budget}_False"][q.query_id]
            for q in queries
        )
        write_json(root / "performance.json", result)
        log_result(
            settings,
            "selected_online_profile",
            {
                "model": params["model"],
                "kb_version": artifact.kb_version,
                "queries": len(representative),
                "seconds": resources.seconds,
            },
            online_metrics,
            root,
            operational_metrics={
                "latency_p50_ms": float(np.quantile(latency, 0.5)),
                "latency_p95_ms": float(np.quantile(latency, 0.95)),
                "throughput_qps": len(latency) / elapsed,
                "peak_rss_bytes": float(resources.rss),
                "peak_vram_bytes": float(resources.peak_vram_allocated),
                "document_encode_qps": len(artifact.units) / encode_seconds,
            },
        )
        return result
    finally:
        encoder.close()
        if reranker:
            reranker.close()
        client.close()


def vector_fingerprint(client: Any, name: str) -> tuple[str, int]:
    digest = hashlib.sha256()
    offset, count = None, 0
    while True:
        records, offset = client.scroll(
            name, limit=64, offset=offset, with_vectors=True, with_payload=False
        )
        for record in records:
            value = {"id": str(record.id), "vectors": record.model_dump(mode="json")["vector"]}
            digest.update((json.dumps(value, sort_keys=True) + "\n").encode())
            count += 1
        if offset is None:
            return digest.hexdigest(), count


def verify_restored_index(
    settings: Settings, artifact: Any, client: Any, source_name: str, restored_name: str
) -> dict[str, Any]:
    if inventory(client, source_name) != inventory(client, restored_name):
        raise ValueError("restored Qdrant snapshot inventory mismatch")
    source_vectors, source_count = vector_fingerprint(client, source_name)
    restored_vectors, restored_count = vector_fingerprint(client, restored_name)
    if (source_vectors, source_count) != (restored_vectors, restored_count):
        raise ValueError("restored named-vector content differs")
    for index in (0, len(artifact.units) // 2, len(artifact.units) - 1):
        unit = artifact.units[index]
        filters = SearchFilters(corpus_lang=unit.lang, as_of=settings.retrieval.evaluation_date)
        source = candidate_lists(
            client, source_name, artifact, unit.text, artifact.vectors[index], filters, 10
        )
        restored = candidate_lists(
            client, restored_name, artifact, unit.text, artifact.vectors[index], filters, 10
        )
        for left, right in zip(source, restored, strict=True):
            if [uid for uid, _ in left] != [uid for uid, _ in right] or not np.allclose(
                [score for _, score in left], [score for _, score in right], atol=1e-6
            ):
                raise ValueError("restored dense/sparse query results differ")
    return {"named_vector_sha256": source_vectors, "vector_points": source_count}


def index_drill(settings: Settings) -> dict[str, Any]:
    import httpx

    root = report_root(settings)
    selected = json.loads((root / "selected.json").read_text())
    artifact, _ = prepared(settings, selected["params"]["recipe"])
    data = settings.resolve(settings.paths.data_dir) / "kb" / settings.kb.name
    changes = json.loads((data / "versions" / artifact.kb_version / "changes.json").read_text())
    parent_version = changes.get("parent")
    if not parent_version:
        raise ValueError("the real incremental drill needs a parent KB version")
    client = make_qdrant_client(settings.qdrant, settings)
    try:
        receipt = root / "index_drill.json"
        if receipt.exists():
            previous = json.loads(receipt.read_text())
            if previous.get("index_version") == artifact.manifest.dataset_version:
                local = settings.root / previous["snapshot"]
                if sha256_file(local) != previous["snapshot_sha256"]:
                    raise ValueError("saved Qdrant snapshot checksum mismatch")
                current = sync_index(settings, artifact, client)
                restore_name = previous["restore_collection"]
                if not client.collection_exists(restore_name):
                    from qdrant_client.models import SnapshotPriority

                    with local.open("rb") as snapshot_file:
                        client.http.snapshots_api.recover_from_uploaded_snapshot(
                            collection_name=restore_name,
                            wait=True,
                            priority=SnapshotPriority.SNAPSHOT,
                            checksum=previous["snapshot_sha256"],
                            snapshot=snapshot_file,
                        )
                verify_restored_index(
                    settings, artifact, client, current["collection"], restore_name
                )
                return {**previous, "rerun_verified": True}
        with measure() as parent_resources:
            parent = build_artifact(settings, artifact.recipe, version=parent_version)
        rollback = sync_index(settings, parent, client)
        forward = sync_index(settings, artifact, client)
        no_work = sync_index(settings, artifact, client)
        old_ids = {c.chunk_id for c in parent.chunks}
        current_ids = {c.chunk_id for c in artifact.chunks}
        if (
            set(changes["chunks_delete"]) != old_ids - current_ids
            or set(changes["chunks_upsert"]) != current_ids - old_ids
        ):
            raise ValueError("changes.json does not match canonical KB diff")
        target = (
            settings.resolve(settings.paths.data_dir)
            / "indexes/retrieval/snapshots"
            / f"{artifact.manifest.dataset_version}.snapshot"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            snapshot = client.create_snapshot(forward["collection"])
            if snapshot is None:
                raise ValueError("Qdrant did not return a snapshot")
            url = (
                f"{settings.qdrant.url}/collections/{forward['collection']}/"
                f"snapshots/{snapshot.name}"
            )
            headers = (
                {"api-key": settings.qdrant.api_key.get_secret_value()}
                if settings.qdrant.api_key
                else {}
            )
            with httpx.stream("GET", url, headers=headers, timeout=60) as response:
                response.raise_for_status()
                with target.open("wb") as out:
                    for block in response.iter_bytes():
                        out.write(block)
            if target.stat().st_size != snapshot.size:
                raise ValueError("downloaded Qdrant snapshot size mismatch")
            checksum = getattr(snapshot, "checksum", None)
            if checksum and checksum != sha256_file(target):
                raise ValueError("downloaded Qdrant snapshot checksum mismatch")
        restore_name = forward["collection"] + "_restore_" + sha256_file(target)[:8]
        if not client.collection_exists(restore_name):
            from qdrant_client.models import SnapshotPriority

            with target.open("rb") as snapshot_file:
                client.http.snapshots_api.recover_from_uploaded_snapshot(
                    collection_name=restore_name,
                    wait=True,
                    priority=SnapshotPriority.SNAPSHOT,
                    checksum=sha256_file(target),
                    snapshot=snapshot_file,
                )
        vector_receipt = verify_restored_index(
            settings, artifact, client, forward["collection"], restore_name
        )
        actual = inventory(client, restore_name)
        result = {
            "parent": parent_version,
            "index_version": artifact.manifest.dataset_version,
            "parent_artifact_build": vars(parent_resources),
            "kb_version": artifact.kb_version,
            "rollback": rollback,
            "incremental_forward": forward,
            "unchanged_rerun": no_work,
            "changes_json_verified": True,
            "snapshot": str(target.relative_to(settings.root)),
            "snapshot_sha256": sha256_file(target),
            "snapshot_bytes": target.stat().st_size,
            "restore_collection": restore_name,
            "restored_points": len(actual),
            "restore_verified": True,
            "restore_method": "uploaded local snapshot file; portable across server loss",
            "restored_dense_sparse_queries": 6,
            **vector_receipt,
        }
        write_json(root / "index_drill.json", result)
        return result
    finally:
        client.close()
