"""Round-trip health check for MLflow (`supportpilot check mlflow`)."""

from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel

from supportpilot.common.config import Settings
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
