"""Stage 6 boundary contracts; original KB contracts are unchanged."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.hashing import sha256_text
from supportpilot.common.schemas import KBChunk, LanguageCode


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Contract):
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    licence: Literal["mit", "apache-2.0"]
    kind: Literal["embedding", "reranker"]
    pooling: Literal["mean", "cls", "last"] = "mean"
    dimensions: int = Field(default=1, ge=1)
    protocol: Literal["e5", "plain", "qwen", "classification"]
    batch_size: int = Field(default=4, ge=1)
    max_length: int = Field(default=512, ge=64)


class Registry(Contract):
    schema_version: Literal["retrieval_models/1"]
    reviewed_at: date
    models: dict[str, ModelSpec]
    references: list[str] = Field(default_factory=list)


def load_models(path: Path) -> dict[str, ModelSpec]:
    return Registry.model_validate(yaml.safe_load(path.read_text("utf-8"))).models


class Recipe(Contract):
    schema_version: Literal["retrieval_recipe/1"] = "retrieval_recipe/1"
    encoder: ModelSpec
    view: Literal["canonical", "window", "adjacent"] = "canonical"
    window_tokens: int = Field(default=128, ge=16)
    overlap_tokens: int = Field(default=32, ge=0)
    sparse: Literal["word", "prefix"] = "word"
    bm25_k1: float = Field(default=1.2, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    length_pivot: float = Field(default=64.0, gt=0)
    tokenizer_sha256: str = ""

    @model_validator(mode="after")
    def valid(self) -> Recipe:
        if self.overlap_tokens >= self.window_tokens:
            raise ValueError("overlap must be smaller than the window")
        if self.encoder.kind != "embedding":
            raise ValueError("index encoder must be an embedding model")
        return self

    @property
    def identity(self) -> str:
        return sha256_text(json.dumps(self.model_dump(mode="json"), sort_keys=True))[:20]


class SearchUnit(Contract):
    unit_id: str
    chunk_ids: list[str] = Field(min_length=1)
    text: str = Field(min_length=1)
    lang: LanguageCode
    doc_id: str
    doc_type: str
    effective_from: date
    effective_to: date | None


class SearchFilters(Contract):
    corpus_lang: LanguageCode | None = None
    doc_ids: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    as_of: date = Field(default_factory=date.today)


class SearchOptions(Contract):
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    fusion: Literal["rrf", "weighted"] = "rrf"
    dense_weight: float = Field(default=0.5, ge=0, le=1)
    rrf_k: int = Field(default=60, ge=1)
    candidate_k: int = Field(default=20, ge=1, le=1000)
    rerank_k: int | None = Field(default=None, ge=1, le=1000)
    k: int = Field(default=10, ge=1, le=1000)
    min_dense_score: float | None = None
    min_rerank_score: float | None = None
    evidence_threshold: float | None = Field(default=None, ge=0, le=1)


class ScoredChunk(Contract):
    chunk_id: str
    kb_version: str
    index_version: str
    chunk: KBChunk
    score: float
    dense_score: float | None = None
    sparse_score: float | None = None
    rerank_score: float | None = None
    matched_unit_ids: list[str]


class RetrievalContext(Contract):
    schema_version: Literal["retrieval_context/1"] = "retrieval_context/1"
    kb_version: str
    index_version: str
    evidence: list[ScoredChunk]
    text: str
    token_count: int = Field(ge=0)
    budget: int = Field(ge=0)
    truncated: bool
    abstain: bool
    reason: str | None = None
