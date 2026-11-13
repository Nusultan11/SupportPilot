"""Isolated Qdrant scale workload. Random vectors never count as semantic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from qdrant_client import models as qm

from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.experiments import measure
from supportpilot.retrieval.index import qdrant_call, writer_lock
from supportpilot.retrieval.scale_data import load_protocol


@contextmanager
def final_receipt(path: Path, result: dict[str, Any]) -> Iterator[None]:
    """Run after measurement exits, while MLflow is still active."""
    import mlflow

    try:
        yield
    finally:
        write_json(path, result)
        mlflow.log_artifact(str(path))


def workload(root: Path, *, count: int, dimensions: int, batch: int, seed: int) -> dict[str, Any]:
    """Build bounded batches into a memmap; persist an immutable source manifest."""
    if min(count, dimensions, batch) <= 0:
        raise ValueError("count, dimensions and batch must be positive")
    text_template = "Synthetic load record {point}; no semantic evidence."
    config = {
        "count": count,
        "dimensions": dimensions,
        "batch": batch,
        "seed": seed,
        "generator": "independent_normalised_random/2",
        "text_template": text_template,
    }
    identity = sha256_text(json.dumps(config, sort_keys=True))[:20]
    out = root / identity
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "workload.json"
    if manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored["config"] != config or stored["identity"] != identity:
            raise ValueError("synthetic workload identity mismatch")
        for key in ("vectors.npy", "chunks.jsonl"):
            if sha256_file(out / key) != stored["sha256"][key]:
                raise ValueError("synthetic workload checksum mismatch")
        return {**stored, "path": str(out)}
    vectors = np.lib.format.open_memmap(
        out / "vectors.npy", mode="w+", dtype=np.float32, shape=(count, dimensions)
    )
    with (out / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for offset in range(0, count, batch):
            size = min(batch, count - offset)
            rng = np.random.default_rng(np.random.SeedSequence([seed, offset]))
            rows = rng.standard_normal((size, dimensions), dtype=np.float32)
            rows /= np.linalg.norm(rows, axis=1, keepdims=True)
            vectors[offset : offset + size] = rows
            for point in range(offset, offset + size):
                stream.write(
                    json.dumps(
                        {
                            "chunk_id": f"load_{point}",
                            "doc_id": f"load_doc_{point // 13}",
                            "lang": ("ru", "kk", "en")[point % 3],
                            "synthetic_only": True,
                            "text": text_template.format(point=point),
                            "locator": {"line_range": [point + 1, point + 1]},
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    vectors.flush()
    del vectors
    stored = {
        "schema_version": "retrieval_load_corpus/1",
        "identity": identity,
        "config": config,
        "semantic_quality": "NOT_EVALUATED",
        "sha256": {key: sha256_file(out / key) for key in ("vectors.npy", "chunks.jsonl")},
    }
    write_json(manifest, stored)
    return {**stored, "path": str(out)}


def inventory_digest(client: Any, name: str) -> dict[str, Any]:
    """Hash the complete sorted live payload/vector inventory in bounded pages."""
    digest = hashlib.sha256()
    count, cursor = 0, None
    while True:
        points, cursor = qdrant_call(
            client.scroll, name, limit=128, offset=cursor, with_payload=True, with_vectors=True
        )
        for point in points:
            digest.update(str(point.id).encode())
            digest.update(json.dumps(point.payload, sort_keys=True).encode())
            digest.update(np.array(point.vector, dtype=np.float32).tobytes())
            count += 1
        if cursor is None:
            break
    return {"points": count, "sha256": digest.hexdigest()}


def upload_batches(
    client: Any,
    name: str,
    vectors: Any,
    *,
    batch: int,
    offset: int = 0,
    stop_after: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    for calls, start in enumerate(range(offset, len(vectors), batch), 1):
        end = min(start + batch, len(vectors))
        points = [
            qm.PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={
                    "synthetic_only": True,
                    "chunk_id": f"load_{i}",
                    "lang": ("ru", "kk", "en")[i % 3],
                    "revision": 0,
                },
            )
            for i in range(start, end)
        ]
        qdrant_call(client.upsert, name, points=points, wait=True)
        if calls % 50 == 0:
            print(f"LOAD {name} {end}/{len(vectors)}", flush=True)
        if stop_after is not None and calls >= stop_after:
            raise InterruptedError(f"intentional committed-batch interruption at {end}")
    seconds = time.perf_counter() - started
    return {
        "vectors": len(vectors) - offset,
        "seconds": seconds,
        "vectors_per_second": (len(vectors) - offset) / seconds,
        "scope": "precomputed_dense_vector_upload_not_encoding",
    }


def search_profile(client: Any, name: str, queries: Any, *, exact: bool, ef: int) -> dict[str, Any]:
    rankings, latencies = [], []
    # Warm server paths before the measured cohort; keep first-call latency separate.
    started = time.perf_counter()
    qdrant_call(
        client.query_points,
        name,
        query=queries[0].tolist(),
        limit=10,
        search_params=qm.SearchParams(exact=exact, hnsw_ef=ef),
        with_payload=False,
    )
    first = time.perf_counter() - started
    for vector in queries:
        started = time.perf_counter()
        result = qdrant_call(
            client.query_points,
            name,
            query=vector.tolist(),
            limit=10,
            search_params=qm.SearchParams(exact=exact, hnsw_ef=ef),
            with_payload=False,
        )
        latencies.append(time.perf_counter() - started)
        rankings.append([point.id for point in result.points])
    return {
        "rankings": rankings,
        "latencies_seconds": latencies,
        "first_call_seconds": first,
        "p50_ms": float(np.quantile(latencies, 0.5) * 1000),
        "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        "sequential_requests_per_second": len(queries) / sum(latencies),
        "exact": exact,
        "hnsw_ef": ef,
        "queries": len(queries),
    }


def wait_indexed(client: Any, name: str, *, timeout: float = 1800) -> dict[str, Any]:
    started = time.perf_counter()
    last = 0.0
    while True:
        info = qdrant_call(client.get_collection, name)
        if str(info.optimizer_status) != "ok":
            raise ValueError(f"Qdrant optimizer failure: {info.optimizer_status}")
        indexed = info.indexed_vectors_count or 0
        # Qdrant can retain a small appendable, unindexed segment below its threshold.
        if str(info.status) == "green" and indexed >= 0.9 * (info.points_count or 0):
            return {
                "seconds": time.perf_counter() - started,
                "indexed_vectors": indexed,
                "points": info.points_count,
                "status": str(info.status),
            }
        elapsed = time.perf_counter() - started
        if elapsed > timeout:
            raise TimeoutError(f"HNSW not ready after {elapsed:.1f}s, {indexed} indexed")
        if elapsed - last >= 15:
            print(f"HNSW {name}: {indexed}/{info.points_count}, {elapsed:.0f}s", flush=True)
            last = elapsed
        time.sleep(1)


def restore_snapshot(
    settings: Settings,
    client: Any,
    name: str,
    target: str,
    root: Path,
) -> dict[str, Any]:
    snapshot = qdrant_call(client.create_snapshot, name, wait=True)
    path = root / snapshot.name
    base = settings.qdrant.url.rstrip("/")
    headers = (
        {"api-key": settings.qdrant.api_key.get_secret_value()} if settings.qdrant.api_key else {}
    )
    with httpx.Client(timeout=600, headers=headers) as http:
        with http.stream("GET", f"{base}/collections/{name}/snapshots/{snapshot.name}") as response:
            response.raise_for_status()
            with path.open("wb") as stream:
                for block in response.iter_bytes(1024 * 1024):
                    stream.write(block)
        if client.collection_exists(target):
            raise ValueError(
                "restore target already exists; never overwrite an unverified collection"
            )
        with path.open("rb") as stream:
            response = http.post(
                f"{base}/collections/{target}/snapshots/upload",
                params={"priority": "snapshot", "wait": "true"},
                files={"snapshot": (snapshot.name, stream, "application/octet-stream")},
            )
        response.raise_for_status()
    before, after = inventory_digest(client, name), inventory_digest(client, target)
    if before != after:
        raise ValueError("restored snapshot differs from full source inventory")
    return {
        "source": name,
        "target": target,
        "snapshot": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "inventory": after,
    }


def vector_update_drill(settings: Settings) -> dict[str, Any]:
    """Mutate real vectors in a verified restore target, then restore its full inventory."""
    import mlflow

    root = settings.resolve(settings.paths.reports_dir) / "retrieval_scale/load"
    receipts = [json.loads(p.read_text("utf-8")) for p in sorted(root.glob("*/receipt.json"))]
    original = next(
        r
        for r in receipts
        if r.get("status") == "FINISHED"
        and r.get("source", {}).get("config", {}).get("count") == 100000
        and "restore" in r
    )
    target = original["restore"]["target"]
    out = root / f"vector_update-{time.time_ns()}"
    out.mkdir()
    client = make_qdrant_client(settings.qdrant, settings)
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name="100k_vector_update_restore",
        hypothesis="Real dense-vector updates affect exact search and can be fully rolled back",
        dataset_version=original["source"]["identity"],
        seed=original["source"]["config"]["seed"],
        base_model="none_precomputed_random_vectors",
        control_run_id=original["run_id"],
        params={
            "target": target,
            "updated_points": 128,
            "code_sha256": sha256_file(Path(__file__)),
        },
    )
    result: dict[str, Any] = {"target": target, "status": "RUNNING"}
    try:
        with writer_lock(root / "load.lock"), start_run(spec, settings) as run:
            result["run_id"] = run.info.run_id
            try:
                before = inventory_digest(client, target)
                if before != original["restore"]["inventory"]:
                    raise ValueError("restore target differs from verified snapshot inventory")
                records = client.retrieve(
                    target, ids=list(range(128)), with_payload=True, with_vectors=True
                )
                if {p.id for p in records} != set(range(128)):
                    raise ValueError("update target points missing")
                saved, altered = [], []
                expected_vectors: dict[str | int, list[float]] = {}
                for point in records:
                    if not isinstance(point.vector, list):
                        raise ValueError("expected unnamed synthetic dense vector")
                    saved.append(
                        qm.PointStruct(id=point.id, vector=point.vector, payload=point.payload)
                    )
                    expected_vectors[point.id] = (
                        -np.asarray(point.vector, dtype=np.float32)
                    ).tolist()
                    altered.append(
                        qm.PointStruct(
                            id=point.id,
                            vector=expected_vectors[point.id],
                            payload={**(point.payload or {}), "vector_update_drill": True},
                        )
                    )
                started = time.perf_counter()
                try:
                    qdrant_call(client.upsert, target, points=altered, wait=True)
                    result["update_seconds"] = time.perf_counter() - started
                    changed = client.retrieve(
                        target, ids=list(range(128)), with_vectors=True, with_payload=True
                    )
                    expected = {p.id: p for p in altered}
                    if len(changed) != 128 or any(
                        not np.allclose(p.vector, expected_vectors[p.id], atol=1e-6)
                        or p.payload != expected[p.id].payload
                        for p in changed
                    ):
                        raise ValueError("dense vector/payload update verification failed")
                    top = qdrant_call(
                        client.query_points,
                        target,
                        query=altered[0].vector,
                        limit=1,
                        search_params=qm.SearchParams(exact=True),
                    ).points
                    if len(top) != 1 or top[0].id != altered[0].id or top[0].score < 0.99999:
                        raise ValueError(
                            "updated dense vector is not its own exact nearest neighbour"
                        )
                    result["updated_vectors_verified"] = 128
                    result["updated_exact_top1"] = {"id": top[0].id, "score": top[0].score}
                finally:
                    restored_at = time.perf_counter()
                    qdrant_call(client.upsert, target, points=saved, wait=True)
                    result["rollback_seconds"] = time.perf_counter() - restored_at
                after = inventory_digest(client, target)
                if before != after:
                    raise ValueError("full vector/payload inventory differs after rollback")
                result.update(
                    status="FINISHED",
                    inventory_before=before,
                    inventory_after=after,
                    scope=(
                        "128 actual vector+payload updates in a 100k restore collection; "
                        "original source collection unchanged"
                    ),
                )
                mlflow.log_metric("update_seconds", result["update_seconds"])
                mlflow.log_metric("rollback_seconds", result["rollback_seconds"])
                mlflow.set_tags({"decision": "verified_vector_roundtrip", "cost_usd": "0"})
            except BaseException as exc:
                result.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
                raise
            finally:
                write_json(out / "receipt.json", result)
                mlflow.log_artifact(str(out / "receipt.json"))
    finally:
        client.close()
    return {"path": str(out), **result}


def run_stress(settings: Settings, *, count: int | None = None) -> dict[str, Any]:
    """Real server run, retaining collections, snapshot and all failed MLflow attempts."""
    import mlflow

    protocol = load_protocol(settings)
    cfg = dict(protocol["stress"])
    if count is not None:
        cfg["chunks"] = count
    if settings.qdrant.mode != "server":
        raise ValueError("scale measurements require the real Qdrant server")
    root = settings.resolve(settings.paths.reports_dir) / "retrieval_scale/load"
    root.mkdir(parents=True, exist_ok=True)
    source = workload(
        settings.resolve(settings.paths.data_dir) / "load/retrieval_scale",
        count=cfg["chunks"],
        dimensions=cfg["dimensions"],
        batch=cfg["batch_size"],
        seed=protocol["seed"],
    )
    name = cfg["collection_prefix"] + source["identity"]
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"load_{source['identity']}",
        hypothesis="Measure isolated 100k random-vector indexing, ANN fidelity and recovery",
        dataset_version=source["identity"],
        seed=protocol["seed"],
        base_model="none_precomputed_random_vectors",
        params={
            "chunks": cfg["chunks"],
            "dimensions": cfg["dimensions"],
            "on_disk": cfg["on_disk"],
            "semantic_quality": "NOT_EVALUATED",
            "uv_lock_sha256": sha256_file(settings.root / "uv.lock"),
            "protocol_sha256": sha256_file(settings.root / "configs/retrieval_scale.yaml"),
        },
    )
    client = make_qdrant_client(settings.qdrant, settings)
    attempt = root / f"{source['identity']}-{time.time_ns()}"
    attempt.mkdir()
    result: dict[str, Any] = {}
    try:
        with (
            writer_lock(root / "load.lock"),
            start_run(spec, settings) as run,
            final_receipt(attempt / "receipt.json", result),
            measure() as resources,
        ):
            result.update(
                {
                    "source": source,
                    "collection": name,
                    "run_id": run.info.run_id,
                    "status": "RUNNING",
                    "semantic_quality": "NOT_EVALUATED",
                }
            )
            write_json(attempt / "receipt.json", result)
            try:
                vectors = np.load(Path(source["path"]) / "vectors.npy", mmap_mode="r")
                if not client.collection_exists(name):
                    qdrant_call(
                        client.create_collection,
                        name,
                        vectors_config=qm.VectorParams(
                            size=cfg["dimensions"],
                            distance=qm.Distance.COSINE,
                            on_disk=cfg["on_disk"],
                        ),
                        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=0),
                        hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=100, on_disk=True),
                    )
                # Commit a partial upload, inject interruption, replay whole input idempotently.
                initial_points = client.count(name, exact=True).count
                result["initial_live_points"] = initial_points
                result["indexing_scope"] = "initial" if initial_points == 0 else "replayed_upsert"
                if initial_points == 0:
                    try:
                        upload_batches(client, name, vectors, batch=cfg["batch_size"], stop_after=2)
                    except InterruptedError as exc:
                        result["interruption"] = {
                            "error": str(exc),
                            "committed_points": client.count(name, exact=True).count,
                        }
                        write_json(attempt / "interruption.json", result["interruption"])
                result["upload"] = upload_batches(client, name, vectors, batch=cfg["batch_size"])
                if client.count(name, exact=True).count != len(vectors):
                    raise ValueError("post-resume point count differs")
                qdrant_call(
                    client.update_collection,
                    name,
                    optimizers_config=qm.OptimizersConfigDiff(
                        indexing_threshold=10000 if len(vectors) >= 10000 else 1
                    ),
                )
                result["hnsw_build"] = wait_indexed(client, name)
                rng = np.random.default_rng(protocol["seed"] + 1)
                queries = rng.standard_normal((cfg["queries"], cfg["dimensions"]), dtype=np.float32)
                queries /= np.linalg.norm(queries, axis=1, keepdims=True)
                np.save(attempt / "query_vectors.npy", queries, allow_pickle=False)
                result["search"] = {
                    "exact": search_profile(client, name, queries, exact=True, ef=128)
                }
                for ef in cfg["hnsw_ef"]:
                    measured = search_profile(client, name, queries, exact=False, ef=ef)
                    measured["ann_recall10_against_exact"] = float(
                        np.mean(
                            [
                                len(set(a) & set(b)) / len(b)
                                for a, b in zip(
                                    measured["rankings"],
                                    result["search"]["exact"]["rankings"],
                                    strict=True,
                                )
                            ]
                        )
                    )
                    result["search"][f"hnsw_{ef}"] = measured
                # Exercise add/change/delete on a deterministic subset before snapshot recovery.
                started = time.perf_counter()
                ids = list(range(min(128, len(vectors))))
                qdrant_call(
                    client.delete, name, points_selector=qm.PointIdsList(points=ids), wait=True
                )
                if client.count(name, exact=True).count != len(vectors) - len(ids):
                    raise ValueError("delete did not change live inventory")
                qdrant_call(
                    client.upsert,
                    name,
                    points=[
                        qm.PointStruct(
                            id=i,
                            vector=vectors[i].tolist(),
                            payload={
                                "synthetic_only": True,
                                "chunk_id": f"load_{i}",
                                "lang": ("ru", "kk", "en")[i % 3],
                                "revision": 1,
                            },
                        )
                        for i in ids
                    ],
                    wait=True,
                )
                changed = client.retrieve(name, ids=ids, with_payload=True)
                if len(changed) != len(ids) or any(p.payload["revision"] != 1 for p in changed):
                    raise ValueError("partial update failed")
                # Additional points exercise add/delete without mutating the frozen load corpus.
                extra = [len(vectors) + i for i in ids]
                qdrant_call(
                    client.upsert,
                    name,
                    points=[
                        qm.PointStruct(
                            id=i,
                            vector=vectors[i - len(vectors)].tolist(),
                            payload={"synthetic_only": True},
                        )
                        for i in extra
                    ],
                    wait=True,
                )
                if client.count(name, exact=True).count != len(vectors) + len(extra):
                    raise ValueError("incremental add failed")
                qdrant_call(
                    client.delete, name, points_selector=qm.PointIdsList(points=extra), wait=True
                )
                result["incremental"] = {
                    "seconds": time.perf_counter() - started,
                    "added": len(extra),
                    "deleted": len(extra) + len(ids),
                    "changed": len(ids),
                }
                result["restore"] = restore_snapshot(
                    settings, client, name, name + "_restore_" + str(time.time_ns()), attempt
                )
                restored = search_profile(
                    client, result["restore"]["target"], queries[:6], exact=True, ef=128
                )
                current = search_profile(client, name, queries[:6], exact=True, ef=128)
                if restored["rankings"] != current["rankings"]:
                    raise ValueError("snapshot search rankings differ")
                result["restore"]["exact_queries_verified"] = 6
                result["resources"] = vars(resources)
                result["vector_bytes"] = cfg["chunks"] * cfg["dimensions"] * 4
                result["portable_bytes"] = sum(
                    p.stat().st_size for p in Path(source["path"]).iterdir()
                )
                result["limitations"] = [
                    "random vectors are not encoder output or semantic-quality evidence",
                    "RSS excludes Docker; server resident RAM and VRAM are not inferred",
                    "sequential throughput, not concurrent serving capacity",
                    "snapshot bytes are storage evidence, not full Docker volume bytes",
                    "injected committed-batch interruption, not an OS process kill",
                    "reranking and model encoding are NOT_MEASURED by this workload",
                ]
                result["status"] = "FINISHED"
                mlflow.set_tags(
                    {
                        "decision": "follow-up",
                        "conclusion": "isolated scale measurements only",
                        "cost_usd": "0",
                        "gpu_hours": "0_no_model_inference",
                    }
                )
                mlflow.log_metrics(
                    {
                        "upload_vectors_per_second": result["upload"]["vectors_per_second"],
                        "exact_p95_ms": result["search"]["exact"]["p95_ms"],
                    }
                )
            except BaseException as exc:
                result.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
                write_json(attempt / "receipt.json", result)
                raise
            finally:
                write_json(attempt / "receipt.json", result)
        result["resources"] = vars(resources)
        write_json(attempt / "receipt.json", result)
        return {"path": str(attempt), **result}
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, help="smoke size; default is protocol's 100k")
    args = parser.parse_args()
    print(json.dumps(run_stress(load_settings(), count=args.count), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
