"""Stage 9 million-record driver around the retained scale/recovery engine."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from supportpilot.common.config import load_settings
from supportpilot.common.io import write_json
from supportpilot.retrieval.scale_profile import collection_disk_bytes, sample_server
from supportpilot.retrieval.stress import run_stress
from supportpilot.service.evaluation import root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1_000_000)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--timeout", type=float, default=10)
    args = parser.parse_args()
    settings = load_settings()
    settings = settings.model_copy(
        update={"qdrant": settings.qdrant.model_copy(update={"timeout_s": args.timeout})}
    )
    stopped = threading.Event()
    resources: dict[str, Any] = {"samples": [], "sample_failures": 0}
    sampler = threading.Thread(target=sample_server, args=(stopped, resources), daemon=True)
    report: dict[str, Any] = {"status": "RUNNING", "semantic_quality": "NOT_EVALUATED"}
    directory = root(settings) / "stress" / uuid4().hex
    directory.mkdir(parents=True)
    path = directory / "report.json"
    sampler.start()
    try:
        result = run_stress(settings, count=args.count, resume=args.resume)
        report.update(
            status="FINISHED",
            receipt=result,
            collection_disk_bytes=collection_disk_bytes(result["collection"]),
        )
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        stopped.set()
        sampler.join(10)
        samples = directory / "cgroup_samples.json"
        write_json(samples, resources)
        report["cgroup_samples"] = str(samples)
        report["cgroup_peak_bytes"] = max(
            (r["current_bytes"] for r in resources["samples"]), default=None
        )
        report["cgroup_sample_failures"] = resources["sample_failures"]
        write_json(path, report)
        write_json(root(settings) / "stress_latest.json", {"path": str(path), "count": args.count})
    print(json.dumps({"path": str(path)}))


if __name__ == "__main__":
    main()
