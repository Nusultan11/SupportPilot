"""Declared guarded HTTP workload, error-inclusive accounting and actual process restart."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import numpy as np
import uvicorn
import yaml
from pydantic import SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import Settings, load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.retrieval.experiments import measure
from supportpilot.retrieval.scale_profile import sample_server
from supportpilot.service.quality_evaluation import runtime_hashes
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store

# Fixed operational fixtures. A repeated row never adds independent language evidence.
TEXTS = {
    "stock": ("Stock TV-5501", "Наличие TV-5501", "Қор TV-5501", "Наличие TV-5501, қорда бар ма"),
    "order": (
        "Track SP-1001",
        "Где заказ SP-1001",
        "Тапсырыс SP-1001 қайда",
        "Где заказ SP-1001, қай кезеңде",
    ),
    "payment": ("Payment SP-1001", "Оплата SP-1001", "Төлем SP-1001", "Оплата SP-1001, төлем күйі"),
    "cancellation": (
        "Cancel SP-1001",
        "Отмени SP-1001",
        "SP-1001 болдырма",
        "Отмени SP-1001, бас тарту",
    ),
    "operator": (
        "Connect me to a human agent",
        "Соедините с оператором",
        "Адаммен сөйлескім келеді",
        "Нужен оператор, адаммен сөйлескім келеді",
    ),
    "kb": (
        "Explain return policy conditions",
        "Объясните правила возврата",
        "Қайтару ережелерін түсіндіріңіз",
        "Правила возврата, қайтару шарттары",
    ),
}


def percentiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    times = [r["seconds"] for r in rows]
    return {
        "denominator": len(rows),
        "p50_seconds": float(np.quantile(times, 0.5)) if times else None,
        "p95_seconds": float(np.quantile(times, 0.95)) if times else None,
    }


def summarise(rows: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    successes = [r for r in rows if r["status"] == 200 and r["outcome_correct"]]
    return {
        "attempts": len(rows),
        "wall_seconds": seconds,
        "status_counts": dict(Counter(str(r["status"]) for r in rows)),
        "http200": sum(r["status"] == 200 for r in rows),
        "correct_service_deliveries": len(successes),
        "expected_auth_denials": sum(
            r["kind"] == "authentication" and r["status"] == 401 for r in rows
        ),
        "incorrect_outcomes": sum(not r["outcome_correct"] for r in rows),
        "successful_requests_per_second": len(successes) / seconds,
        "attempts_per_second": len(rows) / seconds,
        "all_attempts_latency": percentiles(rows),
        "successful_latency": percentiles(successes),
        "by_scope": {
            s: percentiles([r for r in rows if r["scope"] == s])
            for s in sorted({r["scope"] for r in rows})
        },
        "by_kind": {
            k: {
                "latency": percentiles([r for r in rows if r["kind"] == k]),
                "status_counts": dict(Counter(str(r["status"]) for r in rows if r["kind"] == k)),
            }
            for k in sorted({r["kind"] for r in rows})
        },
    }


@contextmanager
def serving(settings: Settings) -> Any:
    with open_service(settings) as service, socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        server = uvicorn.Server(
            uvicorn.Config(create_app(settings, service), log_level="error", access_log=False)
        )
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 30
            while not server.started:
                if not thread.is_alive() or time.monotonic() >= deadline:
                    raise TimeoutError("task-owned HTTP server did not start")
                time.sleep(0.01)
            yield service, "http://127.0.0.1:" + str(listener.getsockname()[1])
        finally:
            server.should_exit = True
            thread.join(30)
            if thread.is_alive():
                raise TimeoutError("task-owned HTTP server did not stop")


def request(client: httpx.Client, kind: str, scope: str) -> dict[str, Any]:
    scopes = ("en", "ru", "kk", "mixed")
    text = TEXTS["stock" if kind == "authentication" else kind][scopes.index(scope)]
    payload = {
        "message": text,
        "language": "kk" if scope == "mixed" else scope,
        "request_id": str(uuid4()),
        "confirm_cancellation": kind == "cancellation",
    }
    row: dict[str, Any] = {
        "kind": kind,
        "scope": scope,
        "request": payload,
        "outcome_correct": False,
    }
    tick = time.perf_counter()
    try:
        headers = {"Authorization": "Bearer invalid"} if kind == "authentication" else {}
        response = client.post("/chat", json=payload, headers=headers)
        body = response.json()
        row.update(status=response.status_code, result=body)
        expected = "queued" if kind in {"operator", "kb"} else "tool"
        facts = {
            "stock": ["sku=TV-5501", "quantity=7"],
            "order": ["order_id=SP-1001", "order_status=processing"],
            "payment": ["order_id=SP-1001", "payment_status=paid"],
            "cancellation": ["order_id=SP-1001", "cancellation_request=recorded"],
        }.get(kind, [])
        row["outcome_correct"] = (
            response.status_code == 401
            if kind == "authentication"
            else response.status_code == 200
            and body.get("delivery") == expected
            and body.get("response", {}).get("language") == payload["language"]
            and all(fact in body.get("response", {}).get("answer", "") for fact in facts)
        )
    except (httpx.HTTPError, ValueError) as exc:
        row.update(status="transport_or_decode_error", error=type(exc).__name__)
    row["seconds"] = time.perf_counter() - tick
    return row


def verify_queues(store: Store, rows: list[dict[str, Any]]) -> dict[str, int]:
    acknowledged = [
        r["result"] for r in rows if r["status"] == 200 and r.get("result", {}).get("escalation_id")
    ]
    with store.transaction() as db:
        verified = sum(
            bool(
                db.execute(
                    "SELECT 1 FROM escalations "
                    "WHERE id=? AND customer=? AND session=? AND reason=?",
                    (
                        r["escalation_id"],
                        "demo_alice",
                        r["session_id"],
                        r["response"]["escalation_reason"],
                    ),
                ).fetchone()
            )
            for r in acknowledged
        )
    return {"acknowledged": len(acknowledged), "verified": verified}


def restart(
    settings: Settings, token: str, out: Path, receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Start a new OS process on the retained DB and replay committed results exactly."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "SUPPORTPILOT_SERVING__DATABASE": str(settings.serving.database),
        "SUPPORTPILOT_SERVING__TOKENS": json.dumps({"demo_alice": token}),
        "SUPPORTPILOT_SERVING__GENERATION_ENABLED": "false",
    }
    started = time.perf_counter()
    with (out / "restart.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(  # noqa: S603 - fixed task-owned command, no shell
            [
                sys.executable,
                "-m",
                "uvicorn",
                "supportpilot.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "error",
                "--no-access-log",
            ],
            cwd=settings.root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}",
                headers={"Authorization": "Bearer " + token},
                timeout=5,
            ) as client:
                deadline = time.monotonic() + 60
                while True:
                    if child.poll() is not None or time.monotonic() >= deadline:
                        raise TimeoutError("task-owned restarted process did not become ready")
                    try:
                        if client.get("/ready").status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                ready_seconds = time.perf_counter() - started
                replay = [
                    client.post("/chat", json=r["request"]).json() == r["result"] for r in receipts
                ]
                old_session = client.post(
                    "/chat",
                    json={
                        **receipts[0]["request"],
                        "request_id": str(uuid4()),
                        "session_id": receipts[0]["result"]["session_id"],
                    },
                ).status_code
                queues = client.get("/escalations").json()
                verified = all(
                    any(q["escalation_id"] == r["result"]["escalation_id"] for q in queues)
                    for r in receipts
                    if r["result"].get("escalation_id")
                )
                with Store(settings.serving.database).transaction() as db:
                    integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
                    cancellation = db.execute(
                        "SELECT cancellation_requested FROM orders WHERE id='SP-1001'"
                    ).fetchone()[0]
                if not (
                    all(replay)
                    and old_session == 404
                    and verified
                    and integrity == "ok"
                    and cancellation == 1
                ):
                    raise ValueError("restart/replay/state verification failed")
                return {
                    "new_process_pid": child.pid,
                    "ready_seconds": ready_seconds,
                    "replay_equal": replay,
                    "expired_session_http": old_session,
                    "queues_retained": verified,
                    "sqlite_integrity": integrity,
                    "cancellation_requested": cancellation,
                    "scope": (
                        "Gracefully closed original service then new OS process; "
                        "not a crash/power-loss drill. Child outside parent Torch/RSS scope."
                    ),
                }
        finally:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)


def run(settings: Settings, *, smoke: bool = False) -> Path:
    import mlflow

    protocol = settings.root / "configs/system_workload.yaml"
    cfg = yaml.safe_load(protocol.read_text("utf-8"))
    out = settings.resolve(settings.paths.reports_dir) / "system/workload" / uuid4().hex
    out.mkdir(parents=True)
    token = secrets.token_urlsafe(32)
    local = settings.model_copy(
        update={
            "serving": settings.serving.model_copy(
                update={
                    "database": out / "service.sqlite",
                    "generation_enabled": False,
                    "max_sessions": 10000,
                    "tokens": {"demo_alice": SecretStr(token)},
                }
            )
        }
    )
    Store(local.serving.database).seed_demo()
    report: dict[str, Any] = {
        "status": "RUNNING",
        "smoke": smoke,
        "protocol_sha256": sha256_file(protocol),
        "runtime": runtime_hashes(settings),
        "paid_cost_usd": 0,
        "electricity_cost": "UNKNOWN",
        "workload": cfg,
        "levels": {},
        "generation": "off; unqualified; original precision/HTTP probes retained separately",
    }
    write_json(out / "report.json", report)
    stopped = threading.Event()
    server_resources: dict[str, Any] = {"samples": [], "sample_failures": 0}
    sampler = threading.Thread(target=sample_server, args=(stopped, server_resources), daemon=True)
    sampler.start()
    spec = RunSpec(
        experiment="system",
        run_name="guarded_workload_smoke" if smoke else "guarded_workload",
        hypothesis="Measure declared guarded workload and recovery before capacity changes",
        dataset_version=report["protocol_sha256"],
        seed=cfg["seed"],
        params={
            "generation": False,
            "smoke": smoke,
            "control": "serialized guarded service; prior HTTP study has a different request mix",
        },
    )
    try:
        with start_run(spec, settings) as active, measure() as resources:
            report["run_id"] = active.info.run_id
            tick = time.perf_counter()
            with (
                serving(local) as (service, url),
                httpx.Client(
                    base_url=url,
                    headers={"Authorization": "Bearer " + token},
                    timeout=cfg["client_timeout_s"],
                ) as client,
            ):
                report["composition_and_server_seconds"] = time.perf_counter() - tick
                cases = [(kind, scope) for kind in cfg["kinds"] for scope in cfg["scopes"]]
                if smoke:
                    cases = [("stock", "en")]
                report["first_calls"] = [request(client, kind, scope) for kind, scope in cases]
                levels = [] if smoke else cfg["concurrency"]
                for concurrency in levels:
                    tick = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(
                            pool.map(
                                lambda case: request(client, *case),
                                cases * (1 if smoke else cfg["repeats_per_level"]),
                            )
                        )
                    seconds = time.perf_counter() - tick
                    report["levels"][str(concurrency)] = {
                        **summarise(rows, seconds),
                        "rows": rows,
                        "queues": verify_queues(service.store, rows),
                        "ready_after": client.get("/ready").status_code,
                        "recovery": request(client, "stock", "en"),
                    }
                    write_json(out / "report.json", report)
                receipts = (
                    []
                    if smoke
                    else [request(client, kind, "en") for kind in ("cancellation", "operator")]
                )
                if not all(r["outcome_correct"] for r in receipts):
                    raise ValueError("restart preparation failed")
                report["restart_inputs"] = receipts
            if not smoke:
                report["restart"] = restart(local, token, out, receipts)
            report["status"] = "FINISHED"
            mlflow.set_tags(
                {
                    "decision": "measurement_only_no_capacity_change",
                    "semantic_quality": "NOT_EVALUATED",
                }
            )
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        stopped.set()
        sampler.join(10)
        report["qdrant_cgroup"] = server_resources
        if "resources" in locals():
            report["resources"] = vars(resources)
        write_json(out / "report.json", report)
        if report.get("run_id"):
            mlflow.MlflowClient().log_artifacts(report["run_id"], str(out))
    return out / "report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps({"path": str(run(load_settings(), smoke=args.smoke))}))


if __name__ == "__main__":
    main()
