"""Reproducible local service initialization and actual HTTP-boundary resource probe."""

from __future__ import annotations

import argparse
import json
import platform
import secrets
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import psutil
from fastapi.testclient import TestClient
from pydantic import SecretStr

from supportpilot.api import create_app
from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.provenance import code_commit
from supportpilot.common.schemas import LanguageCode
from supportpilot.llm.generation import Backend
from supportpilot.service.runtime import open_service
from supportpilot.service.store import Store


class ProbeBackend:
    """Retain actual prompt/completion receipts only for the explicitly synthetic probe."""

    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.calls: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        raw = self.backend.generate(messages)
        self.calls.append(
            {
                "messages": messages,
                "raw": raw,
                "resources": dict(getattr(self.backend, "last_observation", {})),
            }
        )
        return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["seed", "probe", "e2e"])
    parser.add_argument(
        "--generation", action="store_true", help="Load frozen NF4 research adapter"
    )
    args = parser.parse_args()
    settings = load_settings()
    if args.action == "seed":
        Store(settings.resolve(settings.serving.database)).seed_demo()
        print("Synthetic demo state initialized without overwriting existing state.")
        return
    identity = uuid4().hex
    report_dir = settings.resolve(settings.paths.reports_dir) / "service" / "attempts" / identity
    report_dir.mkdir(parents=True)
    # A probe never mutates the live demo database or relies on persistent credentials.
    cfg = settings.serving.model_copy(
        update={
            "database": report_dir / "probe.sqlite",
            "generation_enabled": args.generation,
            "tokens": {"demo_alice": SecretStr(secrets.token_urlsafe(32))},
        }
    )
    settings = settings.model_copy(update={"serving": cfg})
    Store(cfg.database).seed_demo()
    report: dict[str, object] = {
        "schema_version": "service_probe/1",
        "attempt": identity,
        "status": "RUNNING",
        "semantic_quality": "NOT_EVALUATED",
        "generation_enabled": args.generation,
        "source_hashes": {
            str(p.relative_to(settings.root)): sha256_file(p)
            for p in (settings.root / "src/supportpilot/service").glob("*.py")
        },
        "observations": [],
        "paid_cost_usd": 0,
    }
    source_files = [
        *(settings.root / "src/supportpilot").rglob("*.py"),
        settings.root / "apps/ui.py",
        settings.root / "uv.lock",
        settings.root / "configs/base.yaml",
        settings.root / "configs/taxonomy.yaml",
        settings.root / "tests/integration/test_service.py",
    ]
    report["source_hashes"] = {
        str(p.relative_to(settings.root)): sha256_file(p) for p in source_files
    }
    report["python"] = platform.python_version()
    report["platform"] = platform.platform()
    report["code_commit"] = code_commit(settings.root)
    report["serving_config"] = cfg.model_dump(mode="json", exclude={"tokens"})
    report["retrieval_config"] = settings.retrieval.model_dump(mode="json")
    write_json(report_dir / "report.json", report)
    observations: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        if args.action == "e2e":
            tests_started = time.perf_counter()
            with (report_dir / "e2e.log").open("w", encoding="utf-8") as log:
                subprocess.run(
                    [sys.executable, "-m", "pytest", "tests/integration/test_service.py", "-q"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=settings.root,
                    check=True,
                    timeout=600,
                )
            report["controlled_e2e_log_sha256"] = sha256_file(report_dir / "e2e.log")
            report["controlled_e2e_seconds"] = time.perf_counter() - tests_started
        load_started = time.perf_counter()
        with open_service(settings) as service, TestClient(create_app(settings, service)) as client:
            inference_config = getattr(service.backend, "config", None)
            report["inference_config"] = (
                inference_config.model_dump(mode="json") if inference_config is not None else None
            )
            artifact = getattr(service.retriever, "artifact", None)
            report["retrieval_identity"] = (
                {
                    "kb_version": artifact.kb_version,
                    "index_version": artifact.manifest.dataset_version,
                }
                if artifact
                else None
            )
            capture = ProbeBackend(service.backend) if service.backend else None
            if capture:
                service.backend = capture
            client.headers["Authorization"] = (
                "Bearer " + cfg.tokens["demo_alice"].get_secret_value()
            )
            loaded = time.perf_counter()
            report["load_seconds"] = loaded - load_started
            questions: list[tuple[LanguageCode, str]] = [
                ("en", "What is your return policy?"),
                ("ru", "Какие сроки возврата товара?"),
                ("kk", "Тауарды қайтару шарттары қандай?"),
                ("en", "Track order SP-1001"),
                ("ru", "Наличие TV-5501"),
                ("kk", "Оператормен сөйлескім келеді"),
            ]
            for lang, question in questions:
                call_count = len(capture.calls) if capture else 0
                call_start = time.perf_counter()
                response = client.post(
                    "/chat",
                    json={"message": question, "language": lang, "request_id": str(uuid4())},
                )
                observation: dict[str, object] = {
                    "language": lang,
                    "query": question,
                    "http_status": response.status_code,
                    "seconds": time.perf_counter() - call_start,
                    "rss_bytes": psutil.Process().memory_info().rss,
                    "result": response.json(),
                }
                observation["model_calls"] = capture.calls[call_count:] if capture else []
                observations.append(observation)
                report["observations"] = observations
                write_json(report_dir / "report.json", report)
                if response.status_code != 200:
                    raise RuntimeError("HTTP probe failed")
            report["ready"] = client.get("/ready").json()
            report["queue_count"] = len(client.get("/escalations").json())
            report["status"] = "COMPLETE"
    except Exception as exc:
        report["status"] = "FAILED"
        report["error_type"] = type(exc).__name__
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        write_json(report_dir / "report.json", report)
        write_json(
            report_dir.parent.parent / "latest_probe.json",
            {
                "report": str(report_dir / "report.json"),
                "sha256": sha256_file(report_dir / "report.json"),
                "status": report["status"],
            },
        )
        print(json.dumps({"report": str(report_dir / "report.json"), "status": report["status"]}))


if __name__ == "__main__":
    main()
