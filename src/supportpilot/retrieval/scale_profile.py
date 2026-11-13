"""Uncached serving measurements and a matched, previously exposed growth diagnostic."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import models as qm

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.eda import load_tokenizer
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.abstain import evidence_gate
from supportpilot.retrieval.encode import Transformer
from supportpilot.retrieval.evaluate import (
    load_benchmark,
    observations,
    paired_comparison,
    slices,
    summarise,
)
from supportpilot.retrieval.experiments import measure
from supportpilot.retrieval.index import (
    IndexArtifact,
    default_recipe,
    index_root,
    qdrant_call,
    read_index,
    require_ready,
)
from supportpilot.retrieval.models import SearchFilters, SearchOptions, load_models
from supportpilot.retrieval.scale_data import verify_freeze
from supportpilot.retrieval.search import assemble_context, candidate_lists, rerank, scored_chunks


def sample_server(stopped: threading.Event, result: dict[str, Any]) -> None:
    """Sample local Qdrant cgroup memory, including file cache, separately from Python."""
    docker = shutil.which("docker")
    if docker is None:
        result["sample_failures"] += 1
        return
    while not stopped.is_set():
        try:
            probe = subprocess.run(  # noqa: S603 - fixed argv with resolved executable, no shell
                [
                    docker,
                    "exec",
                    "supportpilot-qdrant",
                    "cat",
                    "/sys/fs/cgroup/memory.current",
                    "/sys/fs/cgroup/memory.stat",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if probe.returncode:
                result["sample_failures"] += 1
            else:
                lines = probe.stdout.splitlines()
                stats = {key: int(value) for key, value in (s.split() for s in lines[1:])}
                row = {
                    "unix_time": time.time(),
                    "current_bytes": int(lines[0]),
                    "anon_bytes": stats["anon"],
                    "file_bytes": stats["file"],
                }
                result["samples"].append(row)
        except (OSError, ValueError, KeyError, subprocess.TimeoutExpired):
            result["sample_failures"] += 1
        stopped.wait(0.5)


def collection_disk_bytes(collection: str) -> int | None:
    docker = shutil.which("docker")
    if docker is None or not re.fullmatch(r"[a-zA-Z0-9_]+", collection):
        return None
    try:
        probe = subprocess.run(  # noqa: S603 - validated generated collection, no shell
            [
                docker,
                "exec",
                "supportpilot-qdrant",
                "du",
                "-sb",
                "/qdrant/storage/collections/" + collection,
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return int(probe.stdout.split()[0]) if probe.returncode == 0 else None
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "sequential_qps": len(values) / (sum(values) / 1000),
    }


def latency_slices(queries: list[RetrievalQuery], rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected: dict[str, list[float]] = {}
    for query, row in zip(queries, rows, strict=True):
        if query.query_id != row["query_id"]:
            raise ValueError("performance observation/query alignment changed")
        for scope in slices(query):
            if scope in ("all", "ru", "kk", "en", "cross_lingual", "code_switch"):
                selected.setdefault(scope, []).append(row["total_ms"])
    return {
        scope: {"queries": len(values), **latency_summary(values)}
        for scope, values in selected.items()
    }


def uncached_request(
    artifact: IndexArtifact,
    collection: str,
    client: Any,
    encoder: Transformer,
    reranker: Transformer,
    tokenizer: Any,
    query: RetrievalQuery,
    options: SearchOptions,
    filters: SearchFilters,
    budget: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    vector = None if options.mode == "sparse" else encoder.encode([query.query], query=True)[0]
    encoded = time.perf_counter()
    dense, sparse = candidate_lists(
        client, collection, artifact, query.query, vector, filters, options.candidate_k
    )
    rows = scored_chunks(artifact, dense, sparse, options)
    searched = time.perf_counter()
    if options.rerank_k:
        rows = rerank(query.query, rows, reranker, options)
    rows = evidence_gate(rows, options.evidence_threshold)[: options.k]
    reranked = time.perf_counter()
    context = assemble_context(
        rows,
        tokenizer,
        budget,
        kb_version=artifact.kb_version,
        index_version=artifact.manifest.dataset_version,
    )
    end = time.perf_counter()
    return {
        "query_id": query.query_id,
        "ranking": [row.chunk_id for row in rows],
        "packed": [row.chunk_id for row in context.evidence],
        "tokens": context.token_count,
        "encode_ms": (encoded - start) * 1000,
        "search_ms": (searched - encoded) * 1000,
        "rerank_ms": (reranked - searched) * 1000,
        "packing_ms": (end - reranked) * 1000,
        "total_ms": (end - start) * 1000,
    }


def fixed_work_probe(
    settings: Settings,
    scoped: Settings,
    client: Any,
    baseline: IndexArtifact,
    large: IndexArtifact,
    encoder: Transformer,
    reranker: Transformer,
    queries: list[RetrievalQuery],
    seed: int,
) -> dict[str, Any]:
    """Separate corpus-size search cost from fixed top-20 neural pair work."""
    chosen = np.random.default_rng(seed).choice(len(queries), 60, replace=False)
    texts = [queries[int(index)].query for index in chosen]
    vectors = encoder.encode(texts, query=True)
    workload_root = settings.resolve(settings.paths.data_dir) / "load/retrieval_scale"
    manifests = [json.loads(p.read_text()) for p in sorted(workload_root.glob("*/workload.json"))]
    load = next(
        (
            m
            for m in manifests
            if m["config"]["count"] == 100000
            and m["config"]["dimensions"] == 1024
            and client.collection_exists("sp_scale_load_" + m["identity"])
        ),
        None,
    )
    if load is None:
        raise ValueError("verified 100k workload is required for the size probe")
    load_name = "sp_scale_load_" + load["identity"]
    for name, checksum in load["sha256"].items():
        if sha256_file(workload_root / load["identity"] / name) != checksum:
            raise ValueError("100k source checksum changed")
    if client.count(load_name, exact=True).count != 100000:
        raise ValueError("100k collection inventory changed")
    # The exact same fixed 20 source texts are scored for all corpus sizes.
    # Random-load neighbours cannot carry business evidence and are not reranked.
    fixed_documents = [f"{c.title}\n{c.text}" for c in baseline.chunks[:20]]
    conditions = [
        ("baseline", require_ready(settings, baseline), "dense", baseline.kb_version),
        ("expanded", require_ready(scoped, large), "dense", large.kb_version),
        ("synthetic100k", load_name, None, None),
    ]
    result = {}
    for label, collection, using, kb in conditions:
        search_ms, neural_ms = [], []
        with measure() as resources:
            for text, vector in zip(texts, vectors, strict=True):
                start = time.perf_counter()
                qdrant_call(
                    client.query_points,
                    collection_name=collection,
                    using=using,
                    query=vector.tolist(),
                    limit=20,
                    with_payload=False,
                    query_filter=qm.Filter(
                        must=[qm.FieldCondition(key="kb_version", match=qm.MatchValue(value=kb))]
                    )
                    if kb
                    else None,
                    search_params=qm.SearchParams(exact=True),
                )
                searched = time.perf_counter()
                reranker.score([(text, document) for document in fixed_documents])
                end = time.perf_counter()
                search_ms.append((searched - start) * 1000)
                neural_ms.append((end - searched) * 1000)
        result[label] = {
            "collection": collection,
            "queries": len(texts),
            "search": latency_summary(search_ms),
            "fixed20_rerank": latency_summary(neural_ms),
            "resources": vars(resources),
        }
    return {
        "conditions": result,
        "query_ids": [queries[int(i)].query_id for i in chosen],
        "load_identity": load["identity"],
        "load_manifest": load,
        "scope": (
            "Exact dense search, precomputed queries, no payload. Identical 20 reranker documents "
            "in every condition; synthetic neighbours are not business evidence. This isolates "
            "size cost, not full 100k hybrid or semantic quality."
        ),
    }


def encoding_probe(settings: Settings, backends: dict[str, Any], out: Path) -> dict[str, Any]:
    """Measure complete warm neural encoding without the document-vector cache."""
    results = {}
    for name, receipt in backends.items():
        artifact = read_index(Path(receipt["index_path"]))
        model = Transformer(artifact.recipe.encoder, settings.retrieval.device)
        try:
            print(f"SCALE UNCACHED ENCODING {name} {len(artifact.units)}", flush=True)
            model.load()
            with measure() as resources:
                vectors = model.encode([unit.text for unit in artifact.units])
            cosine = np.sum(vectors * artifact.vectors, axis=1) / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(artifact.vectors, axis=1)
            )
            results[name] = {
                "model": artifact.recipe.encoder.model_dump(mode="json"),
                "index_version": artifact.manifest.dataset_version,
                "units": len(artifact.units),
                "model_load_seconds": model.load_seconds,
                "resources": vars(resources),
                "units_per_second": len(artifact.units) / resources.seconds,
                "maximum_absolute_delta": float(np.max(np.abs(vectors - artifact.vectors))),
                "minimum_cosine_to_stored_vectors": float(np.min(cosine)),
                "mean_cosine_to_stored_vectors": float(np.mean(cosine)),
                "scope": (
                    "Warm pinned model, full document cohort, no embedding cache. Batch padding "
                    "may differ from incremental cache fill; numerical drift is reported."
                ),
            }
            write_json(out / "encoding_probe.json", results)
            if not np.isfinite(vectors).all() or float(np.min(cosine)) < 0.999:
                raise ValueError("full neural encoding reproduction has excessive drift")
        finally:
            model.close()
    return results


def selected_online_probe(
    settings: Settings,
    frozen: Path,
    selection: dict[str, Any],
    backends: dict[str, Any],
    client: Any,
    tokenizer: Any,
    out: Path,
) -> dict[str, Any]:
    """Balanced development-only performance sample; never use it to retune selection."""
    chosen = selection["configurations"][selection["selected"]]
    artifact = read_index(Path(backends[chosen["encoder"]]["index_path"]))
    metadata = json.loads((frozen / "metadata.json").read_text())
    queries = [
        RetrievalQuery.model_validate_json(line)
        for line in (frozen / "queries.jsonl").read_text("utf-8").splitlines()
    ]
    queries = [q for q in queries if metadata[q.query_id]["phase"] == "development"]
    seed = json.loads((frozen / "protocol.json").read_text())["seed"]
    rng = np.random.default_rng(seed)
    sample: list[RetrievalQuery] = []
    for scope, count in (("ru", 20), ("kk", 20), ("en", 20), ("cross", 10), ("mixed", 10)):
        cohort = [
            q
            for q in queries
            if (
                q.cross_lingual
                if scope == "cross"
                else q.code_switch
                if scope == "mixed"
                else q.lang == scope and not q.cross_lingual and not q.code_switch
            )
        ]
        sample.extend(cohort[int(i)] for i in rng.choice(len(cohort), count, replace=False))
    scoped = settings.model_copy(
        update={
            "retrieval": settings.retrieval.model_copy(update={"namespace": "scale_" + frozen.name})
        }
    )
    collection = require_ready(scoped, artifact)
    options = SearchOptions.model_validate(chosen["options"])
    encoder = Transformer(artifact.recipe.encoder, settings.retrieval.device)
    reranker = Transformer(
        load_models(settings.resolve(settings.retrieval.models))["bge_reranker"],
        settings.retrieval.device,
    )
    try:

        def request(query: RetrievalQuery) -> dict[str, Any]:
            meta = metadata[query.query_id]
            return uncached_request(
                artifact,
                collection,
                client,
                encoder,
                reranker,
                tokenizer,
                query,
                options,
                SearchFilters(
                    corpus_lang=query.corpus_lang,
                    as_of=meta["as_of"],
                    all_versions=meta["all_versions"],
                ),
                chosen["budget"],
            )

        print(f"SCALE SELECTED ONLINE {selection['selected']} {len(sample)}", flush=True)
        with measure() as resources:
            first = request(sample[0])
            rows = [request(query) for query in sample]
        result = {
            "selected": selection["selected"],
            "configuration": chosen,
            "query_ids": [q.query_id for q in sample],
            "observations": rows,
            "first_request_ms": first["total_ms"],
            "resources": vars(resources),
            "model_load_seconds": encoder.load_seconds + reranker.load_seconds,
            **latency_summary([row["total_ms"] for row in rows]),
            "latency_by_slice": latency_slices(sample, rows),
            "component_p50_ms": {
                key: float(np.median([row[key] for row in rows]))
                for key in ("encode_ms", "search_ms", "rerank_ms", "packing_ms")
            },
            "scope": (
                "80 fixed-seed development variants, balanced RU/KK/EN plus cross/mixed. "
                "No query/pair caches; performance only, no selection or holdout retuning. "
                "First request loads weights in this already running Python process, "
                "not an OS-cold machine."
            ),
        }
        write_json(out / "selected_online.json", result)
        return result
    finally:
        encoder.close()
        reranker.close()


def profile_scale(settings: Settings, frozen: Path) -> dict[str, Any]:
    """Never select with these observations; require the sealed holdout result first."""
    import mlflow

    verify_freeze(frozen)
    protocol = json.loads((frozen / "protocol.json").read_text())
    root = settings.resolve(Path(protocol["output"])) / frozen.name
    selection = json.loads((root / "selection.json").read_text())
    holdout = json.loads((root / "holdout/results.json").read_text())
    if selection["freeze_sha256"] != sha256_file(frozen / "freeze.json") or selection[
        "development_sha256"
    ] != sha256_file(root / "development/results.json"):
        raise ValueError("sealed development/selection changed")
    if json.loads((root / "holdout/attempt.json").read_text())["status"] != "FINISHED":
        raise ValueError("complete holdout before online profiling")
    if holdout["kb_version"] != json.loads((frozen / "audit.json").read_text())["kb_version"]:
        raise ValueError("holdout KB mismatch")
    out = root / "profile"
    if (out / "receipt.json").exists():
        raise ValueError("completed profile is immutable")
    out.mkdir(exist_ok=True)
    development = json.loads((root / "development/results.json").read_text())
    baseline_settings = settings.model_copy(
        update={
            "retrieval": settings.retrieval.model_copy(
                update={"namespace": "", "encoder": "bge_m3"}
            )
        }
    )
    baseline_root = index_root(baseline_settings, default_recipe(baseline_settings))
    state = json.loads((baseline_root / "state.json").read_text())
    baseline = read_index(baseline_root / "versions" / state["index_version"])
    if baseline.kb_version != protocol["baseline_kb"]:
        raise ValueError("baseline serving artifact changed")
    large = read_index(Path(development["backends"]["bge_m3"]["index_path"]))
    scoped = settings.model_copy(
        update={
            "retrieval": settings.retrieval.model_copy(update={"namespace": "scale_" + frozen.name})
        }
    )
    _, old_queries = load_benchmark(
        settings.resolve(settings.paths.data_dir)
        / "benchmarks/retrieval"
        / protocol["baseline_benchmark"]
    )
    options = SearchOptions.model_validate(
        selection["configurations"][selection["control"]]["options"]
    )
    tokenizer = load_tokenizer(settings.eda.tokenizers[0])
    models = load_models(settings.resolve(settings.retrieval.models))
    encoder = Transformer(baseline.recipe.encoder, settings.retrieval.device)
    reranker = Transformer(models["bge_reranker"], settings.retrieval.device)
    client = make_qdrant_client(settings.qdrant, settings)
    stopped = threading.Event()
    server: dict[str, Any] = {
        "samples": [],
        "sample_failures": 0,
        "scope": (
            "Sampled Qdrant container cgroup memory including file cache, separate from Python RSS "
            "and whole Docker VM. Existing historical collections remain present."
        ),
    }
    sampler = threading.Thread(target=sample_server, args=(stopped, server), daemon=True)
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"{frozen.name}_uncached_growth",
        hypothesis="Measure corpus growth with identical historical queries and uncached control",
        dataset_version=frozen.name,
        seed=protocol["seed"],
        base_model=baseline.recipe.encoder.repo_id + "@" + baseline.recipe.encoder.revision,
        params={
            "baseline_kb": baseline.kb_version,
            "large_kb": large.kb_version,
            "queries": len(old_queries),
            "freeze_sha256": sha256_file(frozen / "freeze.json"),
            "query_cache": False,
            "rerank_cache": False,
            "selection_use": False,
            "old_holdout_previously_exposed": True,
            "profile_code_sha256": sha256_file(Path(__file__)),
        },
    )
    try:
        with start_run(spec, settings) as run:
            sampler.start()
            outputs, ranked_observations = {}, {}
            for label, artifact, cfg in (
                ("baseline", baseline, baseline_settings),
                ("expanded", large, scoped),
            ):
                name = require_ready(cfg, artifact)
                print(f"SCALE ONLINE {label} {len(old_queries)}", flush=True)
                with measure() as resources:
                    first = uncached_request(
                        artifact,
                        name,
                        client,
                        encoder,
                        reranker,
                        tokenizer,
                        old_queries[0],
                        options,
                        SearchFilters(
                            corpus_lang=old_queries[0].corpus_lang,
                            as_of=protocol["evaluation_date"],
                        ),
                        1024,
                    )
                    rows = []
                    for index, query in enumerate(old_queries):
                        rows.append(
                            uncached_request(
                                artifact,
                                name,
                                client,
                                encoder,
                                reranker,
                                tokenizer,
                                query,
                                options,
                                SearchFilters(
                                    corpus_lang=query.corpus_lang, as_of=protocol["evaluation_date"]
                                ),
                                1024,
                            )
                        )
                        if index % 100 == 0:
                            print(f"SCALE ONLINE {label} {index}/{len(old_queries)}", flush=True)
                ranked = {row["query_id"]: row["ranking"] for row in rows}
                packed = {row["query_id"]: row["packed"] for row in rows}
                ranked_observations[label] = observations(
                    old_queries, ranked, seed=protocol["seed"]
                )
                outputs[label] = {
                    "kb_version": artifact.kb_version,
                    "chunks": len(artifact.chunks),
                    "index_version": artifact.manifest.dataset_version,
                    "vector_bytes": artifact.vectors.nbytes,
                    "artifact_bytes": sum(
                        p.stat().st_size for p in artifact.path.rglob("*") if p.is_file()
                    ),
                    "server_collection_disk_bytes": collection_disk_bytes(name),
                    "first_request_ms": first["total_ms"],
                    "model_load_seconds_cumulative": encoder.load_seconds + reranker.load_seconds,
                    "resources": vars(resources),
                    **latency_summary([row["total_ms"] for row in rows]),
                    "latency_by_slice": latency_slices(old_queries, rows),
                    "component_p50_ms": {
                        key: float(np.median([row[key] for row in rows]))
                        for key in ("encode_ms", "search_ms", "rerank_ms", "packing_ms")
                    },
                    "metrics": summarise(
                        observations(old_queries, ranked, seed=protocol["seed"]),
                        seed=protocol["seed"],
                    ),
                    "packed_metrics": summarise(
                        observations(old_queries, packed, seed=protocol["seed"]),
                        seed=protocol["seed"],
                    ),
                    "observations": rows,
                }
                write_json(out / f"{label}.json", outputs[label])
            fixed_probe = fixed_work_probe(
                baseline_settings,
                scoped,
                client,
                baseline,
                large,
                encoder,
                reranker,
                old_queries,
                protocol["seed"],
            )
            write_json(out / "fixed_work.json", fixed_probe)
            encoder.close()
            reranker.close()
            encoding = encoding_probe(settings, development["backends"], out)
            selected_online = selected_online_probe(
                settings,
                frozen,
                selection,
                development["backends"],
                client,
                tokenizer,
                out,
            )
            stopped.set()
            sampler.join()
            server["peak_current_bytes"] = max(
                (r["current_bytes"] for r in server["samples"]),
                default=None,
            )
            server["peak_anon_bytes"] = max(
                (r["anon_bytes"] for r in server["samples"]),
                default=None,
            )
            server["peak_file_bytes"] = max(
                (r["file_bytes"] for r in server["samples"]),
                default=None,
            )
            write_json(out / "server_memory.json", server)
            result = {
                "run_id": run.info.run_id,
                "cohort": protocol["baseline_benchmark"],
                "query_ids": [q.query_id for q in old_queries],
                "results": outputs,
                "scope": (
                    "Historical exposed benchmark; matched uncached control, no retuning. Models "
                    "stay warm between corpora. First request is not OS-cold."
                ),
                "quality_claim": (
                    "Growth diagnostic only; new frozen holdout is reported separately."
                ),
                "fixed_work": fixed_probe,
                "uncached_encoding": encoding,
                "selected_online": selected_online,
                "server_memory": {k: v for k, v in server.items() if k != "samples"},
                "server_memory_samples": len(server["samples"]),
                "load_collection_disk_bytes": collection_disk_bytes(
                    fixed_probe["conditions"]["synthetic100k"]["collection"]
                ),
                "paired_growth": {
                    scope: paired_comparison(
                        ranked_observations["baseline"],
                        ranked_observations["expanded"],
                        seed=protocol["seed"],
                        subset="full",
                        scope=scope,
                    )
                    for scope in ("all", "ru", "kk", "en", "cross_lingual", "code_switch")
                },
            }
            write_json(out / "receipt.json", result)
            mlflow.log_artifact(str(out / "receipt.json"))
            mlflow.log_artifact(str(out / "server_memory.json"))
            for filename in ("encoding_probe.json", "selected_online.json", "fixed_work.json"):
                mlflow.log_artifact(str(out / filename))
            for label, values in outputs.items():
                mlflow.log_metric(f"{label}_p95_ms", values["p95_ms"])
            for scope, values in selected_online["latency_by_slice"].items():
                mlflow.log_metric(f"selected_{scope}_p95_ms", values["p95_ms"])
            mlflow.set_tags({"decision": "diagnostic_only", "cost_usd": "0"})
            return {"path": str(out), "run_id": run.info.run_id}
    finally:
        stopped.set()
        if sampler.ident is not None:
            sampler.join()
        encoder.close()
        reranker.close()
        client.close()
