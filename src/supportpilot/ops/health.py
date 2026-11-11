"""Round-trip health checks for MLflow and Qdrant (`supportpilot check ...`)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel

from supportpilot.common.config import QdrantSettings, Settings
from supportpilot.common.tracking import RunSpec, start_run, tracking_uri


class CheckResult(BaseModel):
    name: str
    ok: bool
    details: dict[str, Any]
    problems: list[str]


def check_mlflow(settings: Settings) -> CheckResult:
    """Log a run with params, a metric series and tags; read it back and compare."""
    import mlflow

    t0 = time.perf_counter()
    spec = RunSpec(
        experiment="healthcheck",
        run_name=f"healthcheck-{uuid.uuid4().hex[:6]}",
        hypothesis="tracking backend stores and returns params/metrics/tags",
        dataset_version="none",
        seed=0,
        params={"lr": 2e-4, "rank": 16, "packing": False},
    )
    losses = [1.0, 0.7, 0.55]
    with start_run(spec, settings) as run:
        for step, loss in enumerate(losses):
            mlflow.log_metric("loss", loss, step=step)
        mlflow.log_metric("recall_at_5_kk", 0.81)
        run_id = run.info.run_id

    client = mlflow.MlflowClient(tracking_uri=tracking_uri(settings))
    got = client.get_run(run_id)
    history = [m.value for m in client.get_metric_history(run_id, "loss")]
    problems: list[str] = []
    if got.data.params != {"lr": "0.0002", "rank": "16", "packing": "False"}:
        problems.append(f"params mismatch: {got.data.params}")
    if history != losses:
        problems.append(f"loss history mismatch: {history}")
    if got.data.metrics.get("recall_at_5_kk") != 0.81:
        problems.append("metric recall_at_5_kk mismatch")
    if got.data.tags.get("hypothesis") != spec.hypothesis:
        problems.append("hypothesis tag missing")
    if got.info.status != "FINISHED":
        problems.append(f"run status {got.info.status}")
    return CheckResult(
        name="mlflow",
        ok=not problems,
        details={
            "tracking_uri": tracking_uri(settings),
            "run_id": run_id,
            "mlflow": mlflow.__version__,
            "code_commit": got.data.tags.get("code_commit"),
            "seconds": round(time.perf_counter() - t0, 2),
        },
        problems=problems,
    )


def make_qdrant_client(cfg: QdrantSettings, settings: Settings) -> Any:
    from qdrant_client import QdrantClient

    if cfg.mode == "memory":
        return QdrantClient(location=":memory:")
    if cfg.mode == "embedded":
        path = settings.resolve(cfg.path)
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    api_key = cfg.api_key.get_secret_value() if cfg.api_key else None
    return QdrantClient(url=cfg.url, api_key=api_key, timeout=int(cfg.timeout_s))


def check_qdrant(settings: Settings, cfg: QdrantSettings | None = None) -> CheckResult:
    """Exercise what Stage 5-6 rely on: named dense + IDF sparse vectors, payload filter,
    prefetch + RRF fusion, delete-by-filter (incremental reindex), then drop the collection."""
    from qdrant_client import models as qm

    cfg = cfg or settings.qdrant
    t0 = time.perf_counter()
    client = make_qdrant_client(cfg, settings)
    name = f"sp_healthcheck_{uuid.uuid4().hex[:8]}"
    problems: list[str] = []
    timings: dict[str, float] = {}
    details: dict[str, Any] = {"mode": cfg.mode, "collection": name}

    def lap(key: str, start: float) -> None:
        timings[key] = round((time.perf_counter() - start) * 1000, 1)

    try:
        if cfg.mode == "server":
            details["server_version"] = client.info().version
        s = time.perf_counter()
        client.create_collection(
            name,
            vectors_config={"dense": qm.VectorParams(size=4, distance=qm.Distance.COSINE)},
            sparse_vectors_config={"sparse": qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        lap("create_ms", s)

        docs = [
            ("returns_policy", "kk", [1.0, 0.0, 0.0, 0.0], ([1, 2], [1.0, 1.0])),
            ("returns_policy", "ru", [0.9, 0.1, 0.0, 0.0], ([1, 3], [1.0, 1.0])),
            ("delivery_faq", "kk", [0.0, 1.0, 0.0, 0.0], ([4, 5], [1.0, 1.0])),
        ]
        points = [
            qm.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc}:{lang}:{i}")),
                vector={"dense": dense, "sparse": qm.SparseVector(indices=idx, values=val)},
                payload={"doc_id": doc, "lang": lang},
            )
            for i, (doc, lang, dense, (idx, val)) in enumerate(docs)
        ]
        s = time.perf_counter()
        client.upsert(name, points=points, wait=True)
        lap("upsert_ms", s)

        s = time.perf_counter()
        res = client.query_points(
            name,
            prefetch=[
                qm.Prefetch(query=[1.0, 0.05, 0.0, 0.0], using="dense", limit=10),
                qm.Prefetch(
                    query=qm.SparseVector(indices=[1, 2], values=[1.0, 1.0]),
                    using="sparse",
                    limit=10,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            query_filter=qm.Filter(
                must=[qm.FieldCondition(key="lang", match=qm.MatchValue(value="kk"))]
            ),
            limit=3,
            with_payload=True,
        ).points
        lap("hybrid_query_ms", s)
        top = [(p.payload or {}).get("doc_id") for p in res]
        details["hybrid_top"] = top
        if not top or top[0] != "returns_policy" or len(top) != 2:
            problems.append(f"unexpected hybrid result {top}")

        s = time.perf_counter()
        client.delete(
            name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[
                        qm.FieldCondition(key="doc_id", match=qm.MatchValue(value="returns_policy"))
                    ]
                )
            ),
            wait=True,
        )
        lap("delete_by_filter_ms", s)
        remaining = client.count(name, exact=True).count
        if remaining != 1:
            problems.append(f"expected 1 point after delete, got {remaining}")
    except Exception as exc:
        problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        try:
            if client.collection_exists(name):
                client.delete_collection(name)
        except Exception as exc:
            problems.append(f"cleanup failed: {type(exc).__name__}: {exc}")
        client.close()

    details["timings"] = timings
    details["seconds"] = round(time.perf_counter() - t0, 3)
    return CheckResult(name="qdrant", ok=not problems, details=details, problems=problems)
