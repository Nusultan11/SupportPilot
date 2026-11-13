"""Pinned experiment and runtime boundaries for Stage 7."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.hashing import sha256_text


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelSpec(Contract):
    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    licence: Literal["apache-2.0", "mit"]
    architecture: Literal["causal", "conditional"] = "causal"


class InferenceConfig(Contract):
    model: ModelSpec
    device: str = "cuda:0"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    quantization: Literal["none", "nf4"] = "none"
    max_input_tokens: int = Field(default=4096, ge=128)
    max_new_tokens: int = Field(default=512, ge=16)
    max_seconds: float = Field(default=90, gt=0)
    constrained: bool = False
    adapter: Path | None = None
    seed: int = 20261113


class TrainingConfig(Contract):
    inference: InferenceConfig
    rank: int = Field(default=16, ge=1)
    alpha: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    max_length: int = Field(default=2048, ge=128)
    packing: bool = False
    batch_size: int = Field(default=1, ge=1)
    gradient_accumulation: int = Field(default=8, ge=1)
    max_steps: int = Field(default=100, ge=1)
    save_steps: int = Field(default=25, ge=1)
    train_limit: int | None = Field(default=None, ge=1)
    mt_weight: float = Field(default=0.5, gt=0, le=1)
    prompt_mode: Literal["legacy", "schema", "fewshot"] = "schema"

    @property
    def identity(self) -> str:
        return sha256_text(self.model_dump_json())[:20]


def load_models(path: Path) -> dict[str, ModelSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {key: ModelSpec.model_validate(value) for key, value in raw["models"].items()}
