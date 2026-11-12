"""Frozen Stage 6 experiment matrix, cached inference, MLflow and inspectable results."""

from __future__ import annotations

import json
import platform
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.eda import load_tokenizer
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.encode import Transformer, VectorCache
from supportpilot.retrieval.evaluate import (
    load_benchmark,
    observations,
    paired_comparison,
    summarise,
)
from supportpilot.retrieval.index import (
    IndexArtifact,
    build_artifact,
    read_index,
    require_ready,
    sync_index,
)
from supportpilot.retrieval.models import Recipe, SearchFilters, SearchOptions, load_models
from supportpilot.retrieval.search import (
    candidate_lists,
    scored_chunks,
)
from supportpilot.retrieval.sparse import tokenize


@dataclass
class Resources:
    rss: int = 0
    available_ram_min: int = field(default_factory=lambda: psutil.virtual_memory().available)
    seconds: float = 0
    peak_vram_allocated: int = 0
    peak_vram_reserved: int = 0


@contextmanager
def measure() -> Iterator[Resources]:
    import torch

    result = Resources()
    stopped = threading.Event()
    process = psutil.Process()

    def sample() -> None:
        while not stopped.wait(0.05):
            result.rss = max(result.rss, process.memory_info().rss)
            result.available_ram_min = min(
                result.available_ram_min, psutil.virtual_memory().available
            )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    worker = threading.Thread(target=sample, daemon=True)
    worker.start()
    started = time.perf_counter()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - started
        result.rss = max(result.rss, process.memory_info().rss)
        if torch.cuda.is_available():
            result.peak_vram_allocated = torch.cuda.max_memory_allocated()
            result.peak_vram_reserved = torch.cuda.max_memory_reserved()
        stopped.set()
        worker.join()


def report_root(settings: Settings) -> Path:
    root = settings.resolve(settings.paths.reports_dir) / "retrieval"
    root.mkdir(parents=True, exist_ok=True)
    return root


def benchmark(settings: Settings) -> tuple[str, list[RetrievalQuery]]:
    root = settings.resolve(settings.paths.data_dir) / "benchmarks/retrieval"
    version = json.loads((root / "current.json").read_text())["benchmark_version"]
    manifest, queries = load_benchmark(root / version)
    return manifest.dataset_version, queries


def recipes(settings: Settings) -> dict[str, Recipe]:
    models = load_models(settings.resolve(settings.retrieval.models))
    result = {
        name: Recipe(encoder=models[name], tokenizer_sha256=settings.eda.tokenizers[0].sha256)
        for name in ("e5_small", "e5_base", "bge_m3", "qwen_embedding")
    }
    base = result["e5_base"]
    result["prefix"] = base.model_copy(update={"sparse": "prefix"})
    result["window128"] = base.model_copy(
        update={"view": "window", "window_tokens": 128, "overlap_tokens": 0}
    )
    result["overlap128"] = base.model_copy(
        update={"view": "window", "window_tokens": 128, "overlap_tokens": 32}
    )
    result["overlap256"] = base.model_copy(
        update={"view": "window", "window_tokens": 256, "overlap_tokens": 64}
    )
    result["adjacent"] = base.model_copy(update={"view": "adjacent"})
    return result


def log_result(
    settings: Settings,
    name: str,
    params: dict[str, Any],
    summary: dict[str, Any],
    artifact_path: Path,
    *,
    control: str | None = None,
    operational_metrics: dict[str, float] | None = None,
) -> str:
    import mlflow
    import torch

    version, _ = benchmark(settings)
    spec = RunSpec(
        experiment="retrieval",
        run_name=name,
        hypothesis=(
            "Compare retrieval quality and resource cost under a frozen "
            "independent KB-bound benchmark"
        ),
        control_run_id=control,
        dataset_version=version,
        base_model=str(params.get("model", "none")),
        seed=settings.retrieval.seed,
        params={
            **{k: str(v) for k, v in params.items()},
            "lock_sha256": sha256_file(settings.root / "uv.lock"),
            "torch": torch.__version__,
            "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "cost_usd": 0,
            "python": platform.python_version(),
        },
    )
    with start_run(spec, settings) as run:
        if operational_metrics:
            mlflow.log_metrics(operational_metrics)
        mlflow.set_tag("evaluation_date", settings.retrieval.evaluation_date.isoformat())
        mlflow.set_tag("bootstrap_unit", "scenario_id")
        mlflow.set_tag(
            "model_registry_sha256", sha256_file(settings.resolve(settings.retrieval.models))
        )
        for scope in (
            "all",
            "ru",
            "kk",
            "en",
            "cross_lingual",
            "code_switch",
            "multi_fragment",
            "semantic_near_sft",
        ):
            for metric in ("recall5", "mrr10", "ndcg10"):
                entry = summary.get("full", {}).get(scope, {}).get(metric, {})
                for key in ("mean", "low", "high"):
                    value = entry.get(key)
                    if value is not None:
                        mlflow.log_metric(f"{scope}.{metric}.{key}", value)
        mlflow.set_tag(
            "conclusion",
            "Measured comparison; selection uses development scenarios only; "
            "holdout is exploratory after the confidence-protocol amendment",
        )
        if artifact_path.is_file():
            mlflow.log_artifact(str(artifact_path))
        else:
            mlflow.log_artifacts(str(artifact_path))
        return str(run.info.run_id)


def record_failure(settings: Settings, name: str, failure_path: Path) -> None:
    import mlflow

    details = json.loads(failure_path.read_text("utf-8"))
    spec = RunSpec(
        experiment="retrieval",
        run_name=name,
        hypothesis="Check local retrieval setup and hardware compatibility",
        dataset_version=details.get("benchmark", "retrieval-20261112-ecc28bf4"),
        seed=settings.retrieval.seed,
        params={
            "status": "failed",
            "metrics": "UNKNOWN",
            "cost_usd": 0,
            "lock_sha256": sha256_file(settings.root / "uv.lock"),
        },
    )
    try:
        with start_run(spec, settings):
            mlflow.set_tag("conclusion", "Failed setup/inference attempt; no quality result")
            mlflow.log_artifact(str(failure_path))
            raise RuntimeError("Record failed experiment status")
    except RuntimeError:
        pass


def prepare(settings: Settings) -> dict[str, Any]:
    root = report_root(settings)
    version, queries = benchmark(settings)
    client = make_qdrant_client(settings.qdrant, settings)
    results: dict[str, Any] = {}
    tokenizer = load_tokenizer(settings.eda.tokenizers[0])
    try:
        for name, recipe in recipes(settings).items():
            path = root / "prepared" / name
            path.mkdir(parents=True, exist_ok=True)
            completed = path / "manifest.json"
            stored: dict[str, Any] = {}
            if completed.exists():
                stored = json.loads(completed.read_text())
                if (
                    stored["benchmark"] == version
                    and stored["recipe"] == recipe.model_dump(mode="json")
                    and stored.get("query_protocol") == "kb_scoped_idf/1"
                ):
                    if sha256_file(path / "candidates.json") != stored["candidate_sha256"]:
                        raise ValueError("prepared candidate checksum mismatch")
                    results[name] = stored
                    continue
            print(f"PREPARE {name}", flush=True)
            encoder = Transformer(recipe.encoder, settings.retrieval.device)
            try:
                with measure() as resources:
                    artifact = build_artifact(
                        settings, recipe, encoder=encoder, tokenizer=tokenizer
                    )
                    sync = sync_index(settings, artifact, client)
                    if any(q.kb_version != artifact.kb_version for q in queries):
                        raise ValueError("benchmark and index KB versions differ")
                    cache = VectorCache(
                        settings.resolve(settings.paths.data_dir) / "indexes/embedding_cache",
                        recipe.encoder,
                        encoder,
                    )
                    vectors = cache.encode([q.query for q in queries], query=True)
                    candidates = {}
                    name_on_server = require_ready(settings, artifact)
                    search_times = []
                    for q, vector in zip(queries, vectors, strict=True):
                        started = time.perf_counter()
                        dense, sparse = candidate_lists(
                            client,
                            name_on_server,
                            artifact,
                            q.query,
                            vector,
                            SearchFilters(
                                corpus_lang=q.corpus_lang, as_of=settings.retrieval.evaluation_date
                            ),
                            40,
                        )
                        candidates[q.query_id] = {"dense": dense, "sparse": sparse}
                        search_times.append((time.perf_counter() - started) * 1000)
                    write_json(path / "candidates.json", candidates)
                result = {
                    "benchmark": version,
                    "as_of": settings.retrieval.evaluation_date.isoformat(),
                    "kb_version": artifact.kb_version,
                    "recipe": recipe.model_dump(mode="json"),
                    "index_path": str(artifact.path.relative_to(settings.root)),
                    "index_version": artifact.manifest.dataset_version,
                    "units": len(artifact.units),
                    "sync": sync,
                    "resources": vars(resources),
                    "initial_resources": stored.get(
                        "initial_resources", stored.get("resources", vars(resources))
                    ),
                    "query_protocol": "kb_scoped_idf/1",
                    "index_bytes": sum(f.bytes for f in artifact.manifest.files),
                    "query_cache_computed": cache.computed,
                    "query_cache_hits": cache.hits,
                    "qdrant_search_p50_ms": float(np.quantile(search_times, 0.5)),
                    "qdrant_search_p95_ms": float(np.quantile(search_times, 0.95)),
                    "candidate_sha256": sha256_file(path / "candidates.json"),
                }
                write_json(completed, result)
                log_result(
                    settings,
                    f"prepare_{name}",
                    {
                        "model": recipe.encoder.repo_id + "@" + recipe.encoder.revision,
                        "kb_version": artifact.kb_version,
                        "recipe": recipe.identity,
                        "seconds": resources.seconds,
                    },
                    {},
                    path,
                )
                results[name] = result
            except Exception as exc:
                write_json(
                    path / "failure.json",
                    {
                        "error": str(exc),
                        "type": type(exc).__name__,
                        "benchmark": version,
                        "recipe": recipe.model_dump(mode="json"),
                    },
                )
                record_failure(settings, f"prepare_{name}_failed", path / "failure.json")
                raise
            finally:
                encoder.close()
            print(f"READY {name} {resources.seconds:.1f}s", flush=True)
    finally:
        client.close()
    return results


def prepared(settings: Settings, name: str) -> tuple[IndexArtifact, dict[str, Any]]:
    path = report_root(settings) / "prepared" / name
    manifest = json.loads((path / "manifest.json").read_text())
    if sha256_file(path / "candidates.json") != manifest["candidate_sha256"]:
        raise ValueError("candidate cache checksum mismatch")
    version, _ = benchmark(settings)
    if version != manifest["benchmark"]:
        raise ValueError("stale benchmark candidate cache")
    if manifest.get("as_of", "2026-11-12") != settings.retrieval.evaluation_date.isoformat():
        raise ValueError("stale evaluation-date candidate cache")
    return read_index(settings.root / manifest["index_path"]), json.loads(
        (path / "candidates.json").read_text()
    )


def store_evaluation(
    settings: Settings,
    name: str,
    params: dict[str, Any],
    queries: list[RetrievalQuery],
    rankings: dict[str, list[str]],
    *,
    control: str | None = None,
) -> dict[str, Any]:
    path = report_root(settings) / "comparisons" / name
    path.mkdir(parents=True, exist_ok=True)
    rows = observations(queries, rankings, seed=settings.retrieval.seed)
    summary = summarise(rows, seed=settings.retrieval.seed)
    write_json(path / "observations.json", rows)
    write_json(path / "metrics.json", summary)
    write_json(path / "params.json", params)
    run_id = log_result(settings, name, params, summary, path, control=control)
    result = {"name": name, "params": params, "summary": summary, "run_id": run_id}
    write_json(path / "receipt.json", {"run_id": run_id, "params": params})
    return result


def compare(settings: Settings) -> list[dict[str, Any]]:
    import bm25s

    _, queries = benchmark(settings)
    results = []
    for name in recipes(settings):
        artifact, candidates = prepared(settings, name)
        variants = []
        if name in ("e5_small", "e5_base", "bge_m3", "qwen_embedding"):
            for k in (10, 20, 40):
                variants.extend(
                    [
                        (f"dense_k{k}", SearchOptions(mode="dense", candidate_k=k)),
                        (f"rrf_k{k}", SearchOptions(candidate_k=k)),
                    ]
                )
                for weight in (0.25, 0.5, 0.75):
                    variants.append(
                        (
                            f"weighted{weight}_k{k}",
                            SearchOptions(fusion="weighted", dense_weight=weight, candidate_k=k),
                        )
                    )
        else:
            variants.extend([("dense", SearchOptions(mode="dense")), ("rrf", SearchOptions())])
        if name in ("e5_base", "prefix"):
            variants.append(("sparse", SearchOptions(mode="sparse")))
        for variant, options in variants:
            label = f"{name}_{variant}"
            print(f"COMPARE {label}", flush=True)
            rankings = {}
            for q in queries:
                rows = scored_chunks(
                    artifact,
                    [(x, float(y)) for x, y in candidates[q.query_id]["dense"]],
                    [(x, float(y)) for x, y in candidates[q.query_id]["sparse"]],
                    options,
                )
                rankings[q.query_id] = [r.chunk_id for r in rows[:10]]
            results.append(
                store_evaluation(
                    settings,
                    label,
                    {
                        "recipe": name,
                        "options": options.model_dump(),
                        "model": artifact.recipe.encoder.repo_id
                        + "@"
                        + artifact.recipe.encoder.revision,
                        "kb_version": artifact.kb_version,
                    },
                    queries,
                    rankings,
                )
            )
        if name in ("e5_base", "prefix"):
            model = bm25s.BM25(method="lucene")
            model.index(
                [tokenize(u.text, artifact.recipe.sparse) for u in artifact.units],
                show_progress=False,
            )
            rankings = {}
            for q in queries:
                scores = model.get_scores(tokenize(q.query, artifact.recipe.sparse))
                sparse_rows = [
                    (u.unit_id, float(scores[i]))
                    for i, u in enumerate(artifact.units)
                    if u.lang == q.corpus_lang and scores[i] > 0
                ]
                sparse_rows.sort(key=lambda p: (-p[1], p[0]))
                chunks = scored_chunks(
                    artifact, [], sparse_rows[:40], SearchOptions(mode="sparse", candidate_k=40)
                )
                rankings[q.query_id] = [r.chunk_id for r in chunks[:10]]
            results.append(
                store_evaluation(
                    settings,
                    f"{name}_bm25s",
                    {
                        "recipe": name,
                        "model": "bm25s.lucene",
                        "options": SearchOptions(mode="sparse").model_dump(),
                    },
                    queries,
                    rankings,
                )
            )

    # Use development-only metrics; no gold questions or holdout scores in the selection key.
    def key(result: dict[str, Any]) -> tuple[float, float, float]:
        dev = result["summary"]["development"]
        recalls = [dev[lang]["recall5"]["mean"] for lang in ("ru", "kk", "en")]
        return min(recalls), sum(recalls) / 3, dev["all"]["ndcg10"]["mean"]

    chosen = max(results, key=key)
    write_json(
        report_root(settings) / "selected_base.json",
        {k: v for k, v in chosen.items() if k != "summary"},
    )
    write_json(report_root(settings) / "comparison_summary.json", results)
    return results


def score_pairs(
    settings: Settings, name: str, pairs: list[tuple[str, str]], path: Path
) -> tuple[list[float], dict[str, Any]]:
    spec = load_models(settings.resolve(settings.retrieval.models))[name]
    cache_path = (
        settings.resolve(settings.paths.data_dir)
        / "indexes"
        / f"rerank-{sha256_text(spec.model_dump_json())[:16]}.sqlite"
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    model = Transformer(spec, settings.retrieval.device)
    keys = [sha256_text(json.dumps(pair, ensure_ascii=False)) for pair in pairs]
    found: dict[str, float] = {}
    pending: dict[str, tuple[str, str]] = {}
    with sqlite3.connect(cache_path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, score REAL NOT NULL)"
        )
        for key, pair in zip(keys, pairs, strict=True):
            row = conn.execute("SELECT score FROM scores WHERE key=?", (key,)).fetchone()
            if row:
                found[key] = float(row[0])
            else:
                pending[key] = pair
        try:
            with measure() as resources:
                items = list(pending.items())
                for start in range(0, len(items), 32):
                    batch = items[start : start + 32]
                    values = model.score([pair for _, pair in batch])
                    for (key, _), score in zip(batch, values, strict=True):
                        conn.execute("INSERT OR REPLACE INTO scores VALUES (?,?)", (key, score))
                        found[key] = score
                    conn.commit()
                    if start % 512 == 0:
                        print(f"RERANK {name} {start}/{len(items)}", flush=True)
        except Exception as exc:
            write_json(
                path / "failure.json",
                {"error": str(exc), "completed_pairs": len(found), "total_pairs": len(pairs)},
            )
            raise
        finally:
            model.close()
    return [found[k] for k in keys], {
        **vars(resources),
        "computed_pairs": len(pending),
        "cached_pairs": len(pairs) - len(pending),
        "total_pairs": len(pairs),
    }


def rerank_comparison(settings: Settings) -> list[dict[str, Any]]:
    _, queries = benchmark(settings)
    root = report_root(settings)
    selected = json.loads((root / "selected_base.json").read_text())
    params = selected["params"]
    artifact, candidates = prepared(settings, params["recipe"])
    base_options = SearchOptions.model_validate(params["options"])
    # Fixed top-20 union is independent of reranker labels and shared by both models.
    base_options = base_options.model_copy(update={"candidate_k": 20})
    base_rows = {
        q.query_id: scored_chunks(
            artifact,
            candidates[q.query_id]["dense"],
            candidates[q.query_id]["sparse"],
            base_options,
        )[:20]
        for q in queries
    }
    pairs = [
        (q.query, f"{row.chunk.title}\n{row.chunk.text}")
        for q in queries
        for row in base_rows[q.query_id]
    ]
    params = {**params, "options": base_options.model_dump()}
    matched_control = store_evaluation(
        settings,
        "rerank_control_k20",
        params,
        queries,
        {qid: [row.chunk_id for row in rows[:10]] for qid, rows in base_rows.items()},
        control=selected["run_id"],
    )
    selected["run_id"] = matched_control["run_id"]
    results = [matched_control]
    for name in ("bge_reranker", "qwen_reranker"):
        path = root / "rerank" / name
        path.mkdir(parents=True, exist_ok=True)
        scores, resources = score_pairs(settings, name, pairs, path)
        if not (path / "inference_resources.json").exists():
            original = (
                json.loads((path / "resources.json").read_text())
                if (path / "resources.json").exists()
                else resources
            )
            write_json(path / "inference_resources.json", original)
        write_json(path / "resources.json", resources)
        scored: dict[str, list[tuple[str, float]]] = {}
        offset = 0
        for q in queries:
            rows = base_rows[q.query_id]
            scored[q.query_id] = [
                (r.chunk_id, s)
                for r, s in zip(rows, scores[offset : offset + len(rows)], strict=True)
            ]
            offset += len(rows)
        write_json(path / "scores.json", scored)
        for k in (10, 20):
            rankings = {
                qid: [cid for cid, _ in sorted(values[:k], key=lambda p: (-p[1], p[0]))][:10]
                for qid, values in scored.items()
            }
            results.append(
                store_evaluation(
                    settings,
                    f"{name}_k{k}",
                    {**params, "reranker": name, "rerank_k": k},
                    queries,
                    rankings,
                    control=selected["run_id"],
                )
            )
        # Tune confidence only on development scenarios; publish untouched holdout results.
        development = [
            q
            for q in queries
            if observations([q], {q.query_id: []}, seed=settings.retrieval.seed)[0]["phase"]
            == "development"
        ]
        thresholds = (0.1, 0.3, 0.5, 0.7, 0.9)
        threshold_scores = {}
        for threshold in thresholds:
            correct = sum(
                (max((s for _, s in scored[q.query_id]), default=0) >= threshold) == q.answerable
                for q in development
            )
            threshold_scores[threshold] = correct / len(development)
        threshold = max(thresholds, key=lambda t: (threshold_scores[t], -t))
        rankings = {
            qid: [cid for cid, s in sorted(values, key=lambda p: (-p[1], p[0])) if s >= threshold][
                :10
            ]
            for qid, values in scored.items()
        }
        results.append(
            store_evaluation(
                settings,
                f"{name}_threshold",
                {
                    **params,
                    "reranker": name,
                    "rerank_k": 20,
                    "threshold": threshold,
                    "threshold_development_accuracy": threshold_scores[threshold],
                },
                queries,
                rankings,
                control=selected["run_id"],
            )
        )
    write_json(root / "rerank_summary.json", results)
    return results


def finish_report(settings: Settings) -> dict[str, Any]:
    root = report_root(settings)
    _, queries = benchmark(settings)
    results = json.loads((root / "comparison_summary.json").read_text())
    reranked = json.loads((root / "rerank_summary.json").read_text())
    candidates = [*results, *reranked]

    def key(result: dict[str, Any]) -> tuple[float, float, float]:
        dev = result["summary"]["development"]
        recalls = [dev[lang]["recall5"]["mean"] for lang in ("ru", "kk", "en")]
        return min(recalls), sum(recalls) / 3, dev["all"]["ndcg10"]["mean"]

    base = max(results, key=key)
    base_rows = json.loads((root / "comparisons" / base["name"] / "observations.json").read_text())
    justified = [base]
    decisions = []
    for candidate in reranked:
        if not candidate["params"].get("reranker"):
            continue
        candidate_rows = json.loads(
            (root / "comparisons" / candidate["name"] / "observations.json").read_text()
        )
        delta = paired_comparison(
            base_rows, candidate_rows, seed=settings.retrieval.seed, subset="development"
        )
        recall_safe = all(
            candidate["summary"]["development"][lang]["recall5"]["mean"]
            >= base["summary"]["development"][lang]["recall5"]["mean"] - 0.005
            for lang in ("ru", "kk", "en")
        )
        supported = (delta["recall5"]["low"] > 0 or delta["ndcg10"]["low"] > 0) and recall_safe
        decisions.append(
            {
                "name": candidate["name"],
                "paired_development_delta": delta,
                "recall_noninferior_margin": 0.005,
                "adoptable": supported,
            }
        )
        if supported:
            justified.append(candidate)
    chosen = max(justified, key=key)
    write_json(
        root / "selection_evidence.json",
        {
            "base": base["name"],
            "selected": chosen["name"],
            "rule": (
                "Development-only paired CI excludes zero for recall or nDCG; "
                "no language loses more than 0.005 recall; prefer the cheap base otherwise"
            ),
            "reranker_comparisons": decisions,
        },
    )
    from supportpilot.retrieval.abstain import calibrate

    ranking_choice = chosen["name"]
    chosen, gate_results = calibrate(settings, chosen, queries, root)
    candidates.extend(gate_results)
    evidence = json.loads((root / "selection_evidence.json").read_text())
    evidence.update(
        ranking_selected=ranking_choice,
        selected=chosen["name"],
        confidence_selection="abstention_manifest.json",
    )
    write_json(root / "selection_evidence.json", evidence)
    write_json(root / "selected.json", {k: v for k, v in chosen.items() if k != "summary"})
    control = json.loads((root / "comparisons/e5_small_dense_k20/observations.json").read_text())
    rows = json.loads((root / "comparisons" / chosen["name"] / "observations.json").read_text())
    comparisons = {
        scope: paired_comparison(control, rows, seed=settings.retrieval.seed, scope=scope)
        for scope in (
            "all",
            "ru",
            "kk",
            "en",
            "cross_lingual",
            "code_switch",
            "multi_fragment",
            "semantic_near_sft",
        )
    }
    write_json(root / "paired_comparisons.json", comparisons)
    by_id = {q.query_id: q for q in queries}
    errors = {}
    for lang in ("ru", "kk", "en"):
        failures = [r for r in rows if r["answerable"] and r["lang"] == lang and r["recall5"] < 1]
        failed = []
        seen_needs: set[str] = set()
        for row in sorted(failures, key=lambda r: (r["recall5"], r["query_id"])):
            if row["scenario_id"] not in seen_needs:
                failed.append(row)
                seen_needs.add(row["scenario_id"])
        errors[lang] = [
            {
                "query_id": r["query_id"],
                "query": by_id[r["query_id"]].query,
                "category": by_id[r["query_id"]].category,
                "corpus_lang": by_id[r["query_id"]].corpus_lang,
                "phase": r["phase"],
                "diagnostic": (
                    "query_gate_rejection"
                    if not r["ranking"]
                    else "background_only_missed"
                    if all(
                        rel.grade < 2
                        for rel in by_id[r["query_id"]].relevant
                        if rel.grade > 0 and rel.chunk_id not in r["ranking"][:5]
                    )
                    else "positive_not_in_top5"
                ),
                "recall5": r["recall5"],
                "missed": [
                    rel.model_dump()
                    for rel in by_id[r["query_id"]].relevant
                    if rel.grade > 0 and rel.chunk_id not in r["ranking"][:5]
                ],
                "retrieved": r["ranking"][:5],
                "review": "diagnostic; qrels unchanged",
            }
            for r in failed[:50]
        ]
    write_json(root / "errors.json", errors)
    table = [
        "# Stage 6 retrieval comparison",
        "",
        "Development-only selection; full metrics below are descriptive. All qrels remain draft.",
        "",
        "| configuration | RU R@5 | KK R@5 | EN R@5 | cross R@5 | all nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in candidates:
        s = result["summary"]["full"]
        table.append(
            "| "
            + result["name"]
            + " | "
            + " | ".join(
                f"{s[lang]['recall5']['mean']:.4f}" for lang in ("ru", "kk", "en", "cross_lingual")
            )
            + f" | {s['all']['ndcg10']['mean']:.4f} |"
        )
    table += [
        "",
        f"Selected: `{chosen['name']}`. Full per-slice bootstrap CIs: "
        "comparisons/<configuration>/metrics.json.",
        "Exploratory paired holdout CIs: paired_comparisons.json. Error samples: errors.json.",
        "Confidence protocol was amended after initial holdout inspection; "
        "see gate_selection_history.json. No untouched confirmation claim.",
        "Gold is not an input. No LLM training or end-to-end RAG generation was run.",
    ]
    (root / "comparison.md").write_text("\n".join(table) + "\n", "utf-8")
    result = {
        "selected": chosen["name"],
        "params": chosen["params"],
        "metrics": chosen["summary"],
        "paired": comparisons,
        "error_counts": {lang: len(v) for lang, v in errors.items()},
        "configurations": len(candidates),
    }
    write_json(root / "summary.json", result)
    return result


def run_experiments(settings: Settings, *, stage: str = "all") -> dict[str, Any]:
    if stage in ("prepare", "all"):
        prepare(settings)
    if stage in ("compare", "all"):
        compare(settings)
    if stage in ("batch", "all"):
        from supportpilot.retrieval.profile import probe_batches

        probe_batches(settings)
    if stage in ("rerank", "all"):
        rerank_comparison(settings)
    result: dict[str, Any] = {"stage": stage, "report_root": str(report_root(settings))}
    if stage in ("report", "all"):
        result["summary"] = finish_report(settings)
    if stage in ("profile", "all"):
        from supportpilot.retrieval.profile import profile

        result["profile"] = profile(settings)
    if stage in ("drill", "all"):
        from supportpilot.retrieval.profile import index_drill

        result["index_drill"] = index_drill(settings)
    return result
