from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from supportpilot.common.config import Settings, find_project_root, load_settings

REPO = find_project_root(Path(__file__).parent)


def _yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_base_yaml_loads_and_matches_defaults() -> None:
    s = load_settings(REPO / "configs" / "base.yaml")
    assert s.qdrant.mode == "server"
    assert s.logging.log_text is False
    assert s.backup.keep_per_name == 3
    assert s.root == REPO


def test_yaml_values_applied(tmp_path: Path) -> None:
    s = load_settings(_yaml(tmp_path, "qdrant:\n  mode: embedded\nlogging:\n  level: DEBUG\n"))
    assert s.qdrant.mode == "embedded"
    assert s.qdrant.url == "http://127.0.0.1:6333"  # untouched default
    assert s.logging.level == "DEBUG"


def test_env_overrides_yaml_and_init_overrides_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _yaml(tmp_path, "qdrant:\n  mode: embedded\n  timeout_s: 3\n")
    monkeypatch.setenv("SUPPORTPILOT_QDRANT__MODE", "memory")
    s = load_settings(cfg)
    assert s.qdrant.mode == "memory"
    assert s.qdrant.timeout_s == 3  # yaml value of the same section survives (deep merge)
    s2 = load_settings(cfg, env="prod")
    assert s2.env == "prod"


def test_config_path_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORTPILOT_CONFIG", str(_yaml(tmp_path, "env: test\n")))
    assert load_settings().env == "test"


def test_missing_explicit_config_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


@pytest.mark.parametrize(
    "body",
    [
        "unknown_section: 1\n",
        "qdrant:\n  mode: cloud\n",
        "qdrant:\n  typo_key: 1\n",
        "backup:\n  keep_per_name: 0\n",
        "logging:\n  level: verbose\n",
    ],
)
def test_invalid_config_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(ValidationError):
        load_settings(_yaml(tmp_path, body))


def test_secret_never_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPPORTPILOT_QDRANT__API_KEY", "super-secret-value")
    s = load_settings(REPO / "configs" / "base.yaml")
    assert s.qdrant.api_key is not None
    assert s.qdrant.api_key.get_secret_value() == "super-secret-value"
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in s.model_dump_json()


def test_resolve_relative_paths(tmp_path: Path) -> None:
    s = Settings(root=tmp_path)
    assert s.resolve(Path("data")) == (tmp_path / "data").resolve()
    assert s.resolve(tmp_path / "x") == tmp_path / "x"


def test_settings_are_immutable() -> None:
    s = Settings()
    with pytest.raises(ValidationError):
        s.env = "prod"  # type: ignore[misc]
