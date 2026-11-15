"""Track a real Stage 9 HTTP-generation probe using the existing Stage 8 probe engine."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.service.evaluation import root


def main() -> None:
    import mlflow

    settings = load_settings()
    out = root(settings) / "live_generation" / uuid4().hex
    out.mkdir(parents=True)
    environment = {**os.environ, "SUPPORTPILOT_PATHS__REPORTS_DIR": str(out)}
    spec = RunSpec(
        experiment="system",
        run_name="actual_http_generation_probe",
        hypothesis="Unverified real model candidates are withheld with actual queue delivery",
        dataset_version="stage8_operational_probe_questions",
        seed=20261113,
        params={"scope": "engineering_compatibility_not_independent_quality", "paid_cost_usd": 0},
    )
    receipt: dict[str, object] = {"status": "RUNNING"}
    with start_run(spec, settings) as run:
        receipt["run_id"] = run.info.run_id
        try:
            with (out / "execution.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [sys.executable, "-m", "supportpilot.service.demo", "probe", "--generation"],
                    cwd=settings.root,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    check=False,
                )
            receipt["exit_code"] = completed.returncode
            latest = read_json(out / "service/latest_probe.json")
            report_path = Path(latest["report"])
            if sha256_file(report_path) != latest["sha256"]:
                raise ValueError("probe report identity mismatch")
            actual = read_json(report_path)
            receipt["execution_report"] = str(report_path)
            receipt["report_sha256"] = latest["sha256"]
            receipt["actual_model_calls"] = sum(
                len(r["model_calls"]) for r in actual["observations"]
            )
            if completed.returncode or actual["status"] != "COMPLETE":
                raise RuntimeError("actual HTTP-generation probe did not complete")
            receipt["status"] = "FINISHED"
            mlflow.set_tags({"decision": "compatibility_only", "semantic_quality": "NOT_EVALUATED"})
        except BaseException as exc:
            receipt.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
            raise
        finally:
            write_json(out / "receipt.json", receipt)
            mlflow.log_artifacts(str(out))
    print(json.dumps({"path": str(out / "receipt.json")}))


if __name__ == "__main__":
    main()
