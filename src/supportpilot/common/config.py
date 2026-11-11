"""Typed settings: defaults < YAML file < .env < environment variables < explicit overrides.

YAML file: `SUPPORTPILOT_CONFIG` or `<root>/configs/base.yaml`.
Env vars: prefix `SUPPORTPILOT_`, nested with `__`, e.g. `SUPPORTPILOT_QDRANT__MODE=memory`.
Relative paths are resolved against the project root (`Settings.resolve`).
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

ENV_PREFIX = "SUPPORTPILOT_"
_yaml_file: ContextVar[Path | None] = ContextVar("supportpilot_yaml_file", default=None)


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor with `pyproject.toml` + `src/supportpilot`; else the start dir."""
    here = (start or Path.cwd()).resolve()
    candidates = [here, *here.parents, *Path(__file__).resolve().parents]
    for cand in candidates:
        if (cand / "pyproject.toml").is_file() and (cand / "src" / "supportpilot").is_dir():
            return cand
    return here


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsSettings(_Section):
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    reports_dir: Path = Path("artifacts/reports")


class DataSettings(_Section):
    registry: Path = Path("configs/sources.yaml")
    max_reject_ratio: float = Field(
        default=0.01, ge=0.0, le=1.0, description="conversion fails above this share of bad rows"
    )
    download_retries: int = Field(default=3, ge=1)


class CleanSettings(_Section):
    """Stage 3 cleaning, language ID, dedup and quality gates (ADR-020..022)."""

    intent_map: Path = Path("configs/intent_map.yaml")
    placeholders: Path = Path("configs/placeholders.yaml")
    # Language ID: a non-supported language wins only when lingua is this confident.
    lid_other_min_conf: float = Field(default=0.9, gt=0.0, le=1.0)
    lid_low_conf: float = Field(default=0.5, ge=0.0, le=1.0, description="flag below this")
    lid_relabel_min_conf: float = Field(
        default=0.8, ge=0.0, le=1.0, description="detected != declared overrides only above this"
    )
    lid_min_letters: int = Field(default=3, ge=1, description="fewer letters -> undetermined")
    # Near-duplicate detection: char shingles + MinHash/LSH, verified by exact Jaccard.
    shingle_size: int = Field(default=5, ge=2)
    minhash_perms: int = Field(default=128, ge=16)
    lsh_bands: int = Field(default=32, ge=1)
    family_threshold: float = Field(default=0.8, gt=0.0, le=1.0)
    pair_report_min: float = Field(
        default=0.5, gt=0.0, le=1.0, description="verified pairs >= this are kept for EDA"
    )
    sim_text_max_chars: int = Field(default=4000, ge=100)
    seed: int = 20261111
    # Gates (DATA_CONTRACTS §10)
    min_validity: float = Field(default=0.995, ge=0.0, le=1.0)
    min_lid_agreement: float = Field(default=0.99, ge=0.0, le=1.0)


class TokenizerSpec(_Section):
    name: str
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$", description="of tokenizer.json")


class EdaSettings(_Section):
    review_random: int = Field(default=100, ge=0)
    review_flagged: int = Field(default=100, ge=0)
    seed: int = 20261111
    review_annotations: Path = Path("docs/reviews/stage3_review.yaml")
    tokenizers: list[TokenizerSpec] = Field(default_factory=list)


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "json"
    log_text: bool = Field(
        default=False,
        description="Log customer/model text verbatim. Keep False outside local debugging (PII).",
    )


class BackupSettings(_Section):
    root: Path = Path("backups")  # project root; must be git-ignored
    keep_per_name: int = Field(default=3, ge=1)


class MLflowSettings(_Section):
    db_path: Path = Path("mlruns/mlflow.db")
    artifact_root: Path = Path("mlruns/artifacts")
    experiment_prefix: str = "supportpilot"


class QdrantSettings(_Section):
    mode: Literal["server", "embedded", "memory"] = "server"
    # 127.0.0.1, not "localhost": on Windows the IPv6-first lookup adds ~2 s to every call.
    url: str = "http://127.0.0.1:6333"
    path: Path = Path("data/qdrant_embedded")
    api_key: SecretStr | None = None
    timeout_s: float = Field(default=10.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    env: Literal["dev", "test", "prod"] = "dev"
    root: Path = Field(default_factory=find_project_root)
    paths: PathsSettings = PathsSettings()
    data: DataSettings = DataSettings()
    clean: CleanSettings = CleanSettings()
    eda: EdaSettings = EdaSettings()
    logging: LoggingSettings = LoggingSettings()
    backup: BackupSettings = BackupSettings()
    mlflow: MLflowSettings = MLflowSettings()
    qdrant: QdrantSettings = QdrantSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings, dotenv_settings]
        yaml_path = _yaml_file.get()
        if yaml_path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=yaml_path))
        return tuple(sources)

    def resolve(self, path: Path) -> Path:
        """Absolute path; relative paths are taken from the project root."""
        return path if path.is_absolute() else (self.root / path).resolve()


def load_settings(config_path: Path | None = None, **overrides: Any) -> Settings:
    """Load settings. A missing explicit config file is an error; a missing default one is not."""
    root = find_project_root()
    env_cfg = os.environ.get(f"{ENV_PREFIX}CONFIG")
    if config_path is None and env_cfg:
        config_path = Path(env_cfg)
    if config_path is not None:
        if not config_path.is_file():
            raise FileNotFoundError(f"config file not found: {config_path}")
    elif (root / "configs" / "base.yaml").is_file():
        config_path = root / "configs" / "base.yaml"

    token = _yaml_file.set(config_path)
    try:
        dotenv = root / ".env"
        return Settings(_env_file=dotenv if dotenv.is_file() else None, **overrides)
    finally:
        _yaml_file.reset(token)
