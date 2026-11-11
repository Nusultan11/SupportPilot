"""MLflow experiment tracking with the mandatory run metadata from EVALUATION §4.

Backend: local SQLite (`mlflow.db_path`) + local artifact directory. Nothing leaves the machine.
"""

from __future__ import annotations

import platform
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, Field

from supportpilot.common.config import Settings
from supportpilot.common.provenance import code_commit

ParamValue = str | int | float | bool


class RunSpec(BaseModel):
    """Required metadata of every experiment run."""

    experiment: str = Field(min_length=1, description="suffix after the configured prefix")
    run_name: str = Field(min_length=1)
    hypothesis: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    seed: int
    base_model: str | None = None
    control_run_id: str | None = None
    params: dict[str, ParamValue] = Field(default_factory=dict)


def tracking_uri(settings: Settings) -> str:
    db = settings.resolve(settings.mlflow.db_path)
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db.as_posix()}"


def _experiment_id(client: Any, name: str, settings: Settings) -> str:
    exp = client.get_experiment_by_name(name)
    if exp is not None:
        return str(exp.experiment_id)
    artifacts = settings.resolve(settings.mlflow.artifact_root) / name
    artifacts.mkdir(parents=True, exist_ok=True)
    return str(client.create_experiment(name, artifact_location=artifacts.as_uri()))


@contextmanager
def start_run(spec: RunSpec, settings: Settings) -> Iterator[Any]:
    """Open an MLflow run with provenance tags; yields the active `mlflow.ActiveRun`."""
    import mlflow  # heavy import, kept local

    uri = tracking_uri(settings)
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient(tracking_uri=uri)
    exp_name = f"{settings.mlflow.experiment_prefix}/{spec.experiment}"
    exp_id = _experiment_id(client, exp_name, settings)
    tags = {
        "hypothesis": spec.hypothesis,
        "dataset_version": spec.dataset_version,
        "seed": str(spec.seed),
        "base_model": spec.base_model or "",
        "control_run_id": spec.control_run_id or "",
        "code_commit": code_commit(settings.root) or "unknown",
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    with mlflow.start_run(experiment_id=exp_id, run_name=spec.run_name, tags=tags) as run:
        if spec.params:
            mlflow.log_params(spec.params)
        yield run
