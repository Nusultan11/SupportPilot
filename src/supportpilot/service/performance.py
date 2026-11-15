"""Matched local performance studies; synthetic load never measures semantic quality."""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import uvicorn
from pydantic import SecretStr, ValidationError

from supportpilot.api import create_app
from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.schemas import AssistantResponse, LanguageCode
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.llm.backend import TransformersBackend
from supportpilot.llm.generation import build_messages
from supportpilot.llm.models import InferenceConfig
from supportpilot.retrieval.experiments import measure
from supportpilot.retrieval.scale_profile import sample_server
from supportpilot.retrieval.search import Retriever, assemble_context
from supportpilot.routing.router import EncoderRouter
from supportpilot.service.evaluation import SCOPES, Need, protocol, root, verify_freeze
from supportpilot.service.orchestrator import SupportService
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store


def latency(values: list[float]) -> dict[str, Any]:
    return {
        "calls": len(values),
        "first_seconds": values[0] if values else None,
        "warm_p50_seconds": float(np.quantile(values[1:], 0.5)) if len(values) > 1 else None,
        "warm_p95_seconds": float(np.quantile(values[1:], 0.95)) if len(values) > 1 else None,
        "requests_per_second": len(values) / sum(values) if values and sum(values) else None,
    }


def http_load(settings: Settings, out: Path) -> dict[str, Any]:
    """Real loopback TCP/HTTP, not ASGI in-process transport; no automatic retries."""
    token = secrets.token_urlsafe(32)
    cfg = settings.serving.model_copy(
        update={
            "database": out / "load.sqlite",
            "generation_enabled": False,
            "tokens": {"demo_alice": SecretStr(token)},
            "max_sessions": 10000,
        }
    )
    local = settings.model_copy(update={"serving": cfg})
    Store(cfg.database).seed_demo()
    reports: dict[str, Any] = {}
    with open_service(local) as service, socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        server = uvicorn.Server(
            uvicorn.Config(create_app(local, service), log_level="error", access_log=False)
        )
        worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if not worker.is_alive() or time.monotonic() >= deadline:
                    raise TimeoutError("HTTP server did not start")
                time.sleep(0.01)
            url = "http://127.0.0.1:" + str(listener.getsockname()[1])
            with httpx.Client(
                base_url=url, headers={"Authorization": "Bearer " + token}, timeout=65
            ) as client:
                # Warm LID and SQLite, recording the first call separately.
                tick = time.perf_counter()
                first = client.post(
                    "/chat",
                    json={"message": "Stock TV-5501", "language": "en", "request_id": str(uuid4())},
                )
                reports["cold_http_seconds"] = time.perf_counter() - tick
                reports["cold_http_status"] = first.status_code

                def request(i: int) -> dict[str, Any]:
                    message = ("Stock TV-5501", "Track order SP-1001", "Operator for a complaint")[
                        i % 3
                    ]
                    started = time.perf_counter()
                    try:
                        response = client.post(
                            "/chat",
                            json={"message": message, "language": "en", "request_id": str(uuid4())},
                        )
                        return {
                            "seconds": time.perf_counter() - started,
                            "status": response.status_code,
                            "result": response.json(),
                            "kind": i % 3,
                        }
                    except httpx.HTTPError as exc:
                        return {
                            "seconds": time.perf_counter() - started,
                            "status": "transport_error",
                            "error": type(exc).__name__,
                            "kind": i % 3,
                        }

                for concurrency in protocol(settings)["http_concurrency"]:
                    started = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(request, range(protocol(settings)["http_requests"])))
                    seconds = time.perf_counter() - started
                    successful = [r for r in rows if r["status"] == 200]
                    queue_ids = [r["result"]["escalation_id"] for r in successful if r["kind"] == 2]
                    with service.store.transaction() as db:
                        verified = sum(
                            bool(
                                db.execute(
                                    "SELECT 1 FROM escalations WHERE id=?", (qid,)
                                ).fetchone()
                            )
                            for qid in queue_ids
                        )
                    reports[str(concurrency)] = {
                        "rows": rows,
                        "wall_seconds": seconds,
                        "attempts_per_second": len(rows) / seconds,
                        "successful_requests_per_second": len(successful) / seconds,
                        "status_counts": dict(Counter(str(r["status"]) for r in rows)),
                        "all_attempts_latency": latency([r["seconds"] for r in rows]),
                        "successful_latency": latency([r["seconds"] for r in successful]),
                        "acknowledged_queues": len(queue_ids),
                        "verified_queues": verified,
                        "ready_after": client.get("/ready").status_code,
                        "recovery_status": client.post(
                            "/chat",
                            json={
                                "message": "Stock TV-5501",
                                "language": "en",
                                "request_id": str(uuid4()),
                            },
                        ).status_code,
                    }
                    write_json(out / "http.json", reports)
                reports["unauthorized_status"] = client.get(
                    "/escalations", headers={"Authorization": "Bearer invalid"}
                ).status_code
        finally:
            server.should_exit = True
            worker.join(30)
            if worker.is_alive():
                raise TimeoutError("task-owned HTTP server did not stop")
    return reports


def retrieval_and_encoder(settings: Settings, frozen: Path, out: Path) -> dict[str, Any]:
    data = verify_freeze(frozen)
    needs = [Need.model_validate(n) for n in data["needs"] if n["phase"] == "development"]
    # No real business state, responses or cross-principal tool values enter this cache study.
    texts = [text for n in needs for text in n.texts]
    result: dict[str, Any] = {"semantic_quality": "NOT_EVALUATED", "encoder": {}, "retrieval": []}
    prompts: list[dict[str, Any]] = []
    local = settings.model_copy(
        update={
            "serving": settings.serving.model_copy(
                update={
                    "database": out / "profile.sqlite",
                    "generation_enabled": False,
                }
            )
        }
    )
    with open_service(local) as service:
        if not isinstance(service.router, EncoderRouter):
            raise TypeError("performance study requires actual encoder router")
        encoder = service.router.encoder
        encoder.load()
        for batch in protocol(settings)["encoder_batches"]:
            encoder.spec = encoder.spec.model_copy(update={"batch_size": batch})
            times, arrays = [], []
            for _ in range(3):
                started = time.perf_counter()
                arrays.append(encoder.encode(texts, query=True))
                times.append(time.perf_counter() - started)
            np.save(out / f"encoder_batch{batch}.npy", arrays[0], allow_pickle=False)
            result["encoder"][str(batch)] = {
                "seconds": times,
                "texts": len(texts),
                "texts_per_second": len(texts) / float(np.mean(times)),
                "repeat_max_error": float(max(np.max(np.abs(a - arrays[0])) for a in arrays)),
            }
        result["encoder"]["batch_max_error"] = float(
            np.max(
                np.abs(np.load(out / "encoder_batch1.npy") - np.load(out / "encoder_batch8.npy"))
            )
        )
        encoder.spec = encoder.spec.model_copy(update={"batch_size": 1})
        cache: OrderedDict[str, Any] = OrderedDict()
        cache_times = []
        for _ in range(3):
            started = time.perf_counter()
            for text in texts:
                if text not in cache:
                    cache[text] = encoder.encode([text], query=True)[0].copy()
                cache.move_to_end(text)
                if len(cache) > 128:
                    cache.popitem(last=False)
            cache_times.append(time.perf_counter() - started)
        result["encoder"]["exact_cache"] = {
            "seconds": cache_times,
            "entries": len(cache),
            "vector_bytes": sum(v.nbytes for v in cache.values()),
            "scope": "bounded model-local exact-text encoder cache; no tools/responses",
        }
        retriever = service.retriever
        if not isinstance(retriever, Retriever):
            raise ValueError("retrieval required")
        for n in needs:
            if n.need_id not in {"d09", "d10"}:
                continue
            document = "returns_policy" if n.need_id == "d09" else "payment_policy"
            for index, scope in enumerate(SCOPES):
                lang: LanguageCode = "kk" if scope == "mixed" else scope  # type: ignore[assignment]
                started = time.perf_counter()
                ranked = retriever.retrieve(n.texts[index], lang, k=10)
                seconds = time.perf_counter() - started
                for budget in protocol(settings)["context_budgets"]:
                    context = assemble_context(
                        ranked,
                        retriever.tokenizer,
                        budget,
                        kb_version=retriever.artifact.kb_version,
                        index_version=retriever.artifact.manifest.dataset_version,
                    )
                    result["retrieval"].append(
                        {
                            "need_id": n.need_id,
                            "scope": scope,
                            "budget": budget,
                            "ranking": [r.chunk_id for r in ranked],
                            "packed": [r.chunk_id for r in context.evidence],
                            "token_count": context.token_count,
                            "seconds": seconds,
                            "draft_document_hit5": any(
                                r.chunk.doc_id == document for r in ranked[:5]
                            ),
                            "draft_packed_document_hit": any(
                                r.chunk.doc_id == document for r in context.evidence
                            ),
                            "semantic_sufficiency": "NOT_EVALUATED",
                        }
                    )
                    if n.need_id == "d09" and budget == 1024:
                        prompts.append(
                            {
                                "scope": scope,
                                "need_id": n.need_id,
                                "messages": build_messages(
                                    n.texts[index], lang, context, prompt_mode="schema"
                                ),
                            }
                        )
    write_json(out / "captured_prompts.json", prompts)
    result["prompts_sha256"] = sha256_file(out / "captured_prompts.json")
    return result


def precision(
    settings: Settings,
    prompts_path: Path,
    out: Path,
    quantization: str,
    *,
    service: SupportService | None = None,
) -> dict[str, Any]:
    seal = read_json(settings.root / "artifacts/reports/llm/selection.json")
    base = InferenceConfig.model_validate(seal["configurations"]["adapter"])
    cfg = base.model_copy(
        update={
            "quantization": quantization,
            "max_new_tokens": protocol(settings)["generation_max_new_tokens"],
            "max_seconds": protocol(settings)["generation_max_seconds"],
        }
    )
    result: dict[str, Any] = {
        "config": cfg.model_dump(mode="json"),
        "prompts_sha256": sha256_file(prompts_path),
        "scope": "joint" if service is not None else "isolated",
        "calls": [],
        "semantic_quality": "NOT_EVALUATED",
        "delivered_factual_answers": 0,
    }
    backend = TransformersBackend(cfg)
    try:
        result["load_seconds"] = backend.load_seconds
        for repeat in range(3):
            for prompt in read_json(prompts_path):
                row: dict[str, Any] = {
                    "repeat": repeat,
                    "scope": prompt["scope"],
                    "messages": prompt["messages"],
                }
                try:
                    if service is not None:
                        lang: LanguageCode = "kk" if prompt["scope"] == "mixed" else prompt["scope"]
                        started = time.perf_counter()
                        context = service.context("Return policy conditions", lang)
                        row["joint_retrieval"] = {
                            "seconds": time.perf_counter() - started,
                            "kb_version": context.kb_version,
                            "index_version": context.index_version,
                            "packed_ids": [r.chunk_id for r in context.evidence],
                        }
                    raw = backend.generate(prompt["messages"])
                    row.update(raw=raw, resources=dict(backend.last_observation))
                    try:
                        AssistantResponse.model_validate_json(raw, strict=True)
                        row["valid_schema"] = True
                    except ValidationError:
                        row["valid_schema"] = False
                    row["semantic_support"] = "unknown"
                except Exception as exc:
                    row.update(error=type(exc).__name__, detail=str(exc), valid_schema=False)
                result["calls"].append(row)
                write_json(out / "precision.json", result)
                print(f"PRECISION {quantization} {repeat} {prompt['scope']}", flush=True)
        result["latency"] = {
            s: latency(
                [
                    r["resources"]["seconds"]
                    for r in result["calls"]
                    if r["scope"] == s and "resources" in r
                ]
            )
            for s in SCOPES
        }
    finally:
        backend.close()
    return result


def run(
    settings: Settings,
    action: str,
    frozen: Path,
    *,
    prompts: Path | None = None,
    quantization: str = "nf4",
) -> Path:
    import mlflow

    data = verify_freeze(frozen)
    out = root(settings) / "performance" / uuid4().hex
    out.mkdir(parents=True)
    result: dict[str, Any] = {
        "status": "RUNNING",
        "action": action,
        "freeze_identity": data["identity"],
        "paid_cost_usd": 0,
        "source_hashes": {
            str(p.relative_to(settings.root)): sha256_file(p)
            for p in (settings.root / "src/supportpilot").rglob("*.py")
        },
        "lock_sha256": sha256_file(settings.root / "uv.lock"),
    }
    spec = RunSpec(
        experiment="system",
        run_name=action + "_" + quantization,
        hypothesis="Measure matched local trade-offs without semantic acceptance",
        dataset_version=data["identity"],
        seed=20261115,
        params={"action": action, "quantization": quantization},
    )
    stopped = threading.Event()
    server_resources: dict[str, Any] = {"samples": [], "sample_failures": 0}
    sampler = threading.Thread(target=sample_server, args=(stopped, server_resources), daemon=True)
    sampler.start()
    try:
        with start_run(spec, settings) as active, measure() as resources:
            result["run_id"] = active.info.run_id
            write_json(out / "report.json", result)
            if action == "http":
                result["result"] = http_load(settings, out)
            elif action == "retrieval":
                result["result"] = retrieval_and_encoder(settings, frozen, out)
            elif action == "precision_joint" and prompts is not None:
                local = settings.model_copy(
                    update={
                        "serving": settings.serving.model_copy(
                            update={
                                "database": out / "joint.sqlite",
                                "generation_enabled": False,
                            }
                        )
                    }
                )
                with open_service(local) as service:
                    service.context("Return policy conditions", "en")
                    result["result"] = precision(
                        settings, prompts, out, quantization, service=service
                    )
            elif prompts is not None:
                result["result"] = precision(settings, prompts, out, quantization)
            else:
                raise ValueError("precision requires --prompts")
            result["status"] = "FINISHED"
            mlflow.set_tags({"decision": "measurement_only", "semantic_quality": "NOT_EVALUATED"})
    except BaseException as exc:
        result.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        stopped.set()
        sampler.join(10)
        result["qdrant_cgroup"] = server_resources
        if "resources" in locals():
            result["resources"] = vars(resources)
        write_json(out / "report.json", result)
        if result.get("run_id"):
            mlflow.MlflowClient().log_artifacts(result["run_id"], str(out))
    return out / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["http", "retrieval", "precision", "precision_joint"])
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--quantization", choices=["none", "nf4"], default="nf4")
    args = parser.parse_args()
    settings = load_settings()
    frozen = Path(read_json(root(settings) / "freeze_pointer.json")["path"])
    print(
        json.dumps(
            {
                "path": str(
                    run(
                        settings,
                        args.action,
                        frozen,
                        prompts=args.prompts,
                        quantization=args.quantization,
                    )
                )
            }
        )
    )


if __name__ == "__main__":
    main()
