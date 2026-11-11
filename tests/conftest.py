from __future__ import annotations

from pathlib import Path

import pytest

from supportpilot.common.config import (
    BackupSettings,
    MLflowSettings,
    QdrantSettings,
    Settings,
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never see the developer's SUPPORTPILOT_* variables."""
    import os

    for key in list(os.environ):
        if key.startswith("SUPPORTPILOT_"):
            monkeypatch.delenv(key)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    root = tmp_path / "repo"
    root.mkdir()
    return Settings(
        env="test",
        root=root,
        backup=BackupSettings(root=tmp_path / "backups"),
        mlflow=MLflowSettings(db_path=Path("mlruns/mlflow.db"), artifact_root=Path("mlruns/art")),
        qdrant=QdrantSettings(mode="memory"),
    )
