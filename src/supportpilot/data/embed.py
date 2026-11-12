"""Multilingual sentence embeddings for translation QE and semantic leakage checks.

Model: intfloat/multilingual-e5-small (MIT, 118M parameters, XLM-R vocabulary covering
en/ru/kk), pinned by revision in configs/base.yaml (`stage4.embedder`). Mean pooling over the
last hidden state, L2-normalised; texts get the "query: " prefix as the model card asks for
symmetric tasks. torch/transformers are imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field


class EmbedderSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repo_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    licence: str
    batch_size: int = Field(default=128, ge=1)
    max_length: int = Field(default=128, ge=16)


class Embedder:
    def __init__(self, spec: EmbedderSpec) -> None:
        self.spec = spec
        self._model: Any = None
        self._tok: Any = None

    def _load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        path = snapshot_download(
            self.spec.repo_id,
            revision=self.spec.revision,
            allow_patterns=["*.json", "model.safetensors", "sentencepiece.bpe.model"],
        )
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tok = AutoTokenizer.from_pretrained(path)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModel.from_pretrained(path, dtype=dtype).to(self._device).eval()

    def encode(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        import torch

        if self._model is None:
            self._load()
        out: list[npt.NDArray[np.float32]] = []
        for i in range(0, len(texts), self.spec.batch_size):
            batch = [f"query: {t}" for t in texts[i : i + self.spec.batch_size]]
            enc = self._tok(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
            ).to(self._device)
            with torch.no_grad():
                hidden = self._model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            out.append(pooled.cpu().numpy().astype(np.float32))
        if not out:
            return np.zeros((0, 384), dtype=np.float32)
        return np.concatenate(out)


def max_cosine(
    queries: npt.NDArray[np.float32], corpus: npt.NDArray[np.float32], chunk: int = 4096
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """For each query row: the highest cosine to any corpus row and its index (both L2-normed)."""
    best = np.full(len(queries), -1.0, dtype=np.float32)
    arg = np.full(len(queries), -1, dtype=np.int64)
    for j in range(0, len(corpus), chunk):
        sims = queries @ corpus[j : j + chunk].T
        idx = sims.argmax(axis=1)
        val = sims[np.arange(len(queries)), idx]
        better = val > best
        best[better] = val[better]
        arg[better] = idx[better] + j
    return best, arg
