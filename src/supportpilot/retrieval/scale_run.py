"""Development-only comparison followed by a sealed selection and one holdout run."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.eda import load_tokenizer
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.encode import Transformer, VectorCache
from supportpilot.retrieval.evaluate import slices
from supportpilot.retrieval.experiments import measure, score_pairs
from supportpilot.retrieval.index import (
    build_artifact,
    default_recipe,
    require_ready,
    sync_index,
    writer_lock,
)
from supportpilot.retrieval.models import SearchFilters, SearchOptions, load_models
from supportpilot.retrieval.scale_data import family_phase, verify_freeze
from supportpilot.retrieval.scale_metrics import measure_query, paired_scale, summarise_scale
from supportpilot.retrieval.search import assemble_context, candidate_lists, scored_chunks


def configurations() -> dict[str, dict[str, Any]]:
    """Fixed before inference; the control exactly matches ADR-030."""
    base = {
        "mode": "hybrid",
        "fusion": "weighted",
        "dense_weight": 0.75,
        "candidate_k": 20,
        "rerank_k": None,
        "k": 10,
    }
    result: dict[str, dict[str, Any]] = {
        "bge_m3_dense20": {"options": {**base, "mode": "dense"}, "budget": 1024},
        "bge_m3_sparse20": {"options": {**base, "mode": "sparse"}, "budget": 1024},
        "bge_m3_rrf20": {"options": {**base, "fusion": "rrf"}, "budget": 1024},
        "bge_m3_weighted075": {"options": base, "budget": 1024},
        "bge_m3_weighted075_k40": {"options": {**base, "candidate_k": 40}, "budget": 1024},
    }
    for k in (10, 20):
        result[f"bge_m3_weighted075_rerank{k}"] = {
            "options": {**base, "rerank_k": k},
            "budget": 1024,
        }
    for budget in (512, 2048):
        result[f"bge_m3_context{budget}"] = {
            "options": {**base, "rerank_k": 20},
            "budget": budget,
        }
    for threshold in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5):
        result[f"bge_m3_gate{threshold}"] = {
            "options": {**base, "rerank_k": 20, "evidence_threshold": threshold},
            "budget": 1024,
        }
    for value in result.values():
        value["encoder"] = "bge_m3"
    result["e5_base_dense20"] = {
        "encoder": "e5_base",
        "options": {**base, "mode": "dense"},
        "budget": 1024,
    }
    result["e5_base_weighted075_rerank20"] = {
        "encoder": "e5_base",
        "options": {**base, "rerank_k": 20},
        "budget": 1024,
    }
    return result


def save_candidates(path: Path, fingerprint: str, candidates: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "fingerprint": fingerprint,
            "candidates": candidates,
            "sha256": sha256_text(json.dumps(candidates, sort_keys=True)),
        },
    )


def resume_candidates(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    checkpoint = json.loads(path.read_text("utf-8"))
    candidates = dict(checkpoint["candidates"])
    if checkpoint["fingerprint"] != fingerprint or checkpoint["sha256"] != sha256_text(
        json.dumps(candidates, sort_keys=True)
    ):
        raise ValueError("candidate checkpoint changed or belongs to another execution")
    return candidates


def prepare_backend(
    settings: Settings,
    name: str,
    queries: list[RetrievalQuery],
    metadata: dict[str, Any],
    protocol: dict[str, Any],
    kb_version: str,
    client: Any,
    tokenizer: Any,
    out: Path,
) -> dict[str, Any]:
    """Encode and search one pinned backend; keep its namespace/artifact separate."""
    scoped = settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"encoder": name})}
    )
    recipe = default_recipe(scoped)
    encoder = Transformer(recipe.encoder, settings.retrieval.device)
    path = out / "prepared" / name
    path.mkdir(parents=True, exist_ok=True)
    try:
        with measure() as resources:
            artifact = build_artifact(scoped, recipe, version=kb_version, encoder=encoder)
            sync = sync_index(scoped, artifact, client)
            cache = VectorCache(
                settings.resolve(settings.paths.data_dir) / "indexes/embedding_cache_scale",
                recipe.encoder,
                encoder,
            )
            vectors = cache.encode([query.query for query in queries], query=True)
            collection = require_ready(scoped, artifact)
            fingerprint = sha256_text(
                json.dumps(
                    {
                        "index": sha256_file(artifact.path / "manifest.json"),
                        "code": sha256_file(Path(__file__)),
                        "limit": 40,
                        "queries": [
                            (
                                q.query_id,
                                q.query,
                                q.corpus_lang,
                                metadata[q.query_id].get("as_of", protocol["evaluation_date"]),
                                metadata[q.query_id].get("all_versions", False),
                            )
                            for q in queries
                        ],
                    },
                    sort_keys=True,
                )
            )
            checkpoint_path = path / "checkpoint.json"
            candidates = resume_candidates(checkpoint_path, fingerprint)
            if not set(candidates) <= {q.query_id for q in queries}:
                raise ValueError("checkpoint contains unknown queries")
            resumed = len(candidates)
            timings = []
            for index, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
                if query.query_id in candidates:
                    continue
                started = time.perf_counter()
                candidates[query.query_id] = candidate_lists(
                    client,
                    collection,
                    artifact,
                    query.query,
                    vector,
                    SearchFilters(
                        corpus_lang=query.corpus_lang,
                        as_of=metadata[query.query_id].get("as_of", protocol["evaluation_date"]),
                        all_versions=metadata[query.query_id].get("all_versions", False),
                    ),
                    40,
                )
                timings.append(time.perf_counter() - started)
                if index % 500 == 0:
                    save_candidates(checkpoint_path, fingerprint, candidates)
                    print(f"SCALE SEARCH {name} {index}/{len(queries)}", flush=True)
            save_candidates(checkpoint_path, fingerprint, candidates)
            write_json(path / "candidates.json", candidates)
        receipt = {
            "index_path": str(artifact.path),
            "index_version": artifact.manifest.dataset_version,
            "recipe": recipe.model_dump(mode="json"),
            "resources": vars(resources),
            "sync": sync,
            "candidate_sha256": sha256_file(path / "candidates.json"),
            "query_vectors_computed": cache.computed,
            "query_cache_hits": cache.hits,
            "search_p50_ms": float(np.quantile(timings, 0.5) * 1000) if timings else None,
            "search_p95_ms": float(np.quantile(timings, 0.95) * 1000) if timings else None,
            "candidate_queries_resumed": resumed,
            "candidate_queries_computed": len(timings),
            "timing_scope": "precomputed query vectors; not online end-to-end latency",
        }
        write_json(path / "receipt.json", receipt)
        return {"artifact": artifact, "candidates": candidates, "receipt": receipt}
    finally:
        encoder.close()


def select_development(
    evaluations: dict[str, dict[str, Any]],
    rows: dict[str, list[dict[str, Any]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Never accept holdout rows. Inconclusive paired differences retain the control."""
    if any(row["phase"] != "development" for batch in rows.values() for row in batch):
        raise ValueError("selection cannot consume holdout observations")
    control = protocol["selection"]["control"]
    seed = protocol["seed"]
    guards = protocol["selection"]

    def recall(name: str, scope: str) -> float | None:
        value = evaluations[name].get(scope, {}).get("metrics", {}).get("recall5", {}).get("mean")
        return float(value) if value is not None else None

    decisions = {}
    selected = control
    for name in evaluations:
        if name == control:
            continue
        failures = []
        for scope in ("ru", "kk", "en", "cross_lingual", "code_switch"):
            baseline, candidate = recall(control, scope), recall(name, scope)
            loss = (
                guards["maximum_cross_or_mixed_recall_loss"]
                if scope in ("cross_lingual", "code_switch")
                else guards["maximum_recall_loss"]
            )
            if baseline is None or candidate is None:
                failures.append(f"unobserved required slice {scope}")
            elif candidate < baseline - loss:
                failures.append(f"recall guard {scope}")
        paired = paired_scale(rows[control], rows[name], seed=seed)
        ndcg = paired["all"]["metrics"]["ndcg10"]
        minimum = min(recall(name, lang) or 0 for lang in ("ru", "kk", "en"))
        incumbent = min(recall(selected, lang) or 0 for lang in ("ru", "kk", "en"))
        significant = ndcg["low"] is not None and ndcg["low"] > 0
        if name.startswith("bge_m3_gate"):
            base_fpr = evaluations[control]["all"]["metrics"]["unanswerable_fpr"]["mean"]
            fpr = evaluations[name]["all"]["metrics"]["unanswerable_fpr"]["mean"]
            if base_fpr is None or fpr is None or fpr >= base_fpr:
                failures.append("gate does not improve observed FPR")
        if not failures and significant and minimum >= incumbent:
            selected = name
        decisions[name] = {
            "guard_failures": failures,
            "paired": paired,
            "significant_ndcg_vs_control": significant,
        }
    return {
        "selected": selected,
        "control": control,
        "comparisons": decisions,
        "phase": "development",
        "rule": guards,
        "configurations": configurations(),
    }


def run_scale(settings: Settings, frozen: Path, *, phase: str) -> dict[str, Any]:
    """Keep the baseline collection/state intact; evaluate only the requested phase."""
    import mlflow

    sealed = verify_freeze(frozen)
    protocol = json.loads((frozen / "protocol.json").read_text())
    audit = json.loads((frozen / "audit.json").read_text())
    if not audit["passed"] or not audit["semantic_checked"]:
        raise ValueError("only a passed, frozen scale benchmark permits evaluation")
    for relative, checksum in audit["source_hashes"].items():
        source = settings.resolve(Path(relative))
        if not source.is_file() or sha256_file(source) != checksum:
            raise ValueError(f"execution input changed after freeze: {relative}")
    if audit.get("tokenizer") != settings.eda.tokenizers[0].model_dump(mode="json"):
        raise ValueError("context tokenizer changed after freeze")
    meta = json.loads((frozen / "metadata.json").read_text())
    queries = [
        RetrievalQuery.model_validate_json(line)
        for line in (frozen / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    expected_qrels = [
        f"{q.query_id}\t0\t{r.chunk_id}\t{r.grade}" for q in queries for r in q.relevant
    ]
    if (frozen / "qrels.tsv").read_text("utf-8").splitlines() != expected_qrels:
        raise ValueError("qrels disagree with frozen queries")
    if len({q.query_id for q in queries}) != len(queries) or set(meta) != {
        q.query_id for q in queries
    }:
        raise ValueError("frozen query and metadata identities disagree")
    for query in queries:
        row = meta[query.query_id]
        if row["phase"] != family_phase(row["family_id"], protocol):
            raise ValueError("split differs from frozen family assignment")
        if query.kb_version != audit["kb_version"]:
            raise ValueError("mixed KB versions in frozen benchmark")
    queries = [q for q in queries if meta[q.query_id]["phase"] == phase]
    if not queries:
        raise ValueError("requested phase is empty")
    root = settings.resolve(Path(protocol["output"])) / frozen.name
    root.mkdir(parents=True, exist_ok=True)
    config = configurations()
    selection_path = root / "selection.json"
    selection = None
    if phase == "holdout":
        selection = json.loads(selection_path.read_text())
        if selection["freeze_sha256"] != sha256_file(frozen / "freeze.json"):
            raise ValueError("selection uses another benchmark")
        if selection["configurations"] != config:
            raise ValueError("configuration matrix changed after selection")
        if sha256_file(root / "development/results.json") != selection["development_sha256"]:
            raise ValueError("development results changed after selection")
        config = {key: config[key] for key in {selection["selected"], selection["control"]}}
    elif phase != "development":
        raise ValueError("phase must be development or holdout")
    if phase == "development" and selection_path.exists():
        raise ValueError("development selection is already sealed; do not retune")
    out = root / phase
    out.mkdir(exist_ok=True)
    if (out / "results.json").exists():
        raise ValueError("completed phase is immutable")
    # Namespace separates both server collection and local state; recipe identity stays compatible.
    scoped = settings.model_copy(
        update={
            "retrieval": settings.retrieval.model_copy(
                update={"namespace": "scale_" + frozen.name, "encoder": "bge_m3"}
            )
        }
    )
    models = load_models(settings.resolve(settings.retrieval.models))
    recipe = default_recipe(scoped)
    tokenizer = load_tokenizer(settings.eda.tokenizers[0])
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"{frozen.name}_{phase}",
        hypothesis="Measure multilingual ranking and full evidence, independent of similarity",
        dataset_version=sealed["input_identity"],
        seed=protocol["seed"],
        base_model=recipe.encoder.repo_id + "@" + recipe.encoder.revision,
        params={
            "phase": phase,
            "kb_version": audit["kb_version"],
            "freeze_sha256": sha256_file(frozen / "freeze.json"),
            "uv_lock_sha256": sha256_file(settings.root / "uv.lock"),
            "matrix_sha256": sha256_text(json.dumps(config, sort_keys=True)),
            "reranker_revision": models["bge_reranker"].revision,
        },
    )
    client = make_qdrant_client(settings.qdrant, settings)
    with writer_lock(root / "experiment.lock"), start_run(spec, settings) as run:
        write_json(out / "attempt.json", {"run_id": run.info.run_id, "status": "RUNNING"})
        try:
            prepared: dict[str, dict[str, Any]] = {}
            base = SearchOptions(
                mode="hybrid",
                fusion="weighted",
                dense_weight=0.75,
                candidate_k=20,
                rerank_k=20,
                k=10,
            )
            pair_settings = settings.model_copy(
                update={
                    "paths": settings.paths.model_copy(
                        update={
                            "data_dir": settings.resolve(settings.paths.data_dir) / "scale_cache"
                        }
                    )
                }
            )
            for encoder_name in sorted({str(value["encoder"]) for value in config.values()}):
                print(f"SCALE PREPARE {encoder_name} {phase}", flush=True)
                backend = prepare_backend(
                    scoped,
                    encoder_name,
                    queries,
                    meta,
                    protocol,
                    audit["kb_version"],
                    client,
                    tokenizer,
                    out,
                )
                source_artifact = backend["artifact"]
                base_rows = {
                    q.query_id: scored_chunks(
                        source_artifact,
                        backend["candidates"][q.query_id][0],
                        backend["candidates"][q.query_id][1],
                        base,
                    )[:20]
                    for q in queries
                }
                pairs = [
                    (q.query, f"{row.chunk.title}\n{row.chunk.text}")
                    for q in queries
                    for row in base_rows[q.query_id]
                ]
                values, pair_resources = score_pairs(
                    pair_settings, "bge_reranker", pairs, out / "rerank" / encoder_name
                )
                score_map, offset = {}, 0
                for query in queries:
                    source_rows = base_rows[query.query_id]
                    score_map[query.query_id] = {
                        row.chunk_id: value
                        for row, value in zip(
                            source_rows, values[offset : offset + len(source_rows)], strict=True
                        )
                    }
                    offset += len(source_rows)
                backend["scores"] = score_map
                backend["receipt"]["rerank_resources"] = pair_resources
                write_json(out / "prepared" / encoder_name / "receipt.json", backend["receipt"])
                prepared[encoder_name] = backend
            results, observations = {}, {}
            for key, variant in config.items():
                backend = prepared[str(variant["encoder"])]
                artifact, candidates, score_map = (
                    backend["artifact"],
                    backend["candidates"],
                    backend["scores"],
                )
                print(f"SCALE METRICS {key} {phase}", flush=True)
                options = SearchOptions.model_validate(variant["options"])
                observation_rows = []
                for query in queries:
                    relevant = {r.chunk_id: r.grade for r in query.relevant}
                    ranked = scored_chunks(
                        artifact,
                        candidates[query.query_id][0],
                        candidates[query.query_id][1],
                        options,
                    )
                    if options.rerank_k:
                        ranked = [
                            row.model_copy(
                                update={"score": score_map[query.query_id][row.chunk_id]}
                            )
                            for row in ranked[: options.rerank_k]
                        ]
                        ranked.sort(key=lambda row: (-row.score, row.chunk_id))
                        if (
                            options.evidence_threshold is not None
                            and max((row.score for row in ranked), default=0)
                            < options.evidence_threshold
                        ):
                            ranked = []
                    ranked = ranked[:10]
                    packed = assemble_context(
                        ranked,
                        tokenizer,
                        variant["budget"],
                        kb_version=artifact.kb_version,
                        index_version=artifact.manifest.dataset_version,
                    )
                    ranking = [row.chunk_id for row in ranked]
                    context = [row.chunk_id for row in packed.evidence]
                    metadata = meta[query.query_id]
                    observation_rows.append(
                        {
                            "query_id": query.query_id,
                            "scenario_id": query.scenario_id,
                            "kb_version": artifact.kb_version,
                            "benchmark_version": sealed["input_identity"],
                            "family_id": metadata["family_id"],
                            "phase": phase,
                            "answerable": query.answerable,
                            "ranking": ranking,
                            "packed": context,
                            "slices": sorted(
                                set(
                                    slices(query)
                                    + metadata["tags"]
                                    + (
                                        ["negative_" + metadata["negative_reason"]]
                                        if metadata["negative_reason"]
                                        else []
                                    )
                                )
                            ),
                            **measure_query(
                                ranking,
                                context,
                                relevant,
                                metadata["required_groups"],
                                answerable=query.answerable,
                            ),
                        }
                    )
                observations[key] = observation_rows
                results[key] = summarise_scale(observation_rows, seed=protocol["seed"])
            write_json(out / "observations.json", observations)
            receipt = {
                "evaluations": results,
                "configs": config,
                "backends": {name: value["receipt"] for name, value in prepared.items()},
                "kb_version": audit["kb_version"],
                "run_id": run.info.run_id,
                "timing_scope": "cached vectors/pairs; NOT online end-to-end latency",
                "review_status": "draft",
                "semantic_abstention": "NOT_ESTABLISHED",
            }
            write_json(out / "results.json", receipt)
            if phase == "development":
                selection = select_development(results, observations, protocol)
                selection.update(
                    freeze_sha256=sha256_file(frozen / "freeze.json"),
                    development_sha256=sha256_file(out / "results.json"),
                )
                write_json(selection_path, selection)
            mlflow.set_tags(
                {
                    "decision": "follow-up",
                    "conclusion": "draft qrels; no production claim",
                    "cost_usd": "0",
                    "gpu_hours": "UNKNOWN",
                }
            )
            for key in config:
                for lang in ("ru", "kk", "en", "cross_lingual", "code_switch"):
                    value = (
                        results[key].get(lang, {}).get("metrics", {}).get("recall5", {}).get("mean")
                    )
                    if value is not None:
                        mlflow.log_metric(f"{key}_{lang}_recall5", value)
            mlflow.log_artifact(str(out / "results.json"))
            mlflow.log_artifact(str(out / "observations.json"))
            write_json(out / "attempt.json", {"run_id": run.info.run_id, "status": "FINISHED"})
            return {
                "path": str(out),
                "run_id": run.info.run_id,
                "phase": phase,
                "selected": selection["selected"] if selection else None,
            }
        except BaseException as exc:
            write_json(
                out / "attempt.json",
                {
                    "run_id": run.info.run_id,
                    "status": "FAILED",
                    "error": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            mlflow.log_artifact(str(out / "attempt.json"))
            raise
        finally:
            client.close()
