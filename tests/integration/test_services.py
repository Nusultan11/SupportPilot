"""MLflow and Qdrant round trips (the same code paths as `supportpilot check ...`)."""

from __future__ import annotations

import os
import urllib.request

import pytest

from supportpilot.common.config import QdrantSettings, Settings
from supportpilot.ops.health import check_mlflow, check_qdrant


@pytest.mark.slow
def test_mlflow_round_trip(settings: Settings) -> None:
    result = check_mlflow(settings)
    assert result.ok, result.problems
    assert (settings.root / "mlruns" / "mlflow.db").is_file()


def test_qdrant_memory_mode(settings: Settings) -> None:
    result = check_qdrant(settings, QdrantSettings(mode="memory"))
    assert result.ok, result.problems
    assert result.details["hybrid_top"][0] == "returns_policy"


def test_qdrant_embedded_mode(settings: Settings) -> None:
    result = check_qdrant(settings, QdrantSettings(mode="embedded", path=settings.root / "qd"))
    assert result.ok, result.problems
    assert (settings.root / "qd").is_dir()


def _server_url() -> str | None:
    url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    try:
        with urllib.request.urlopen(f"{url}/readyz", timeout=2) as resp:
            return url if resp.status == 200 else None
    except OSError:
        return None


@pytest.mark.qdrant_server
def test_qdrant_server_mode(settings: Settings) -> None:
    url = _server_url()
    if url is None:
        pytest.skip("no Qdrant server reachable (docker compose up -d qdrant)")
    result = check_qdrant(settings, QdrantSettings(mode="server", url=url))
    assert result.ok, result.problems
    assert result.details["server_version"]
