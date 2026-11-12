"""Pinned transformers inference, bounded batches, verified resumable vector cache."""

from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from supportpilot.common.hashing import sha256_text
from supportpilot.retrieval.models import ModelSpec

INSTRUCTION = "Given a customer support question, retrieve relevant store documents that answer it"
QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Encoder(Protocol):
    def encode(self, texts: Sequence[str], *, query: bool = False) -> npt.NDArray[np.float32]: ...


class Transformer:
    def __init__(self, spec: ModelSpec, device: str = "auto") -> None:
        self.spec = spec
        self.device = device
        self.model: Any = None
        self.tokenizer: Any = None
        self.load_seconds = 0.0

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
        from transformers import (
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        started = time.perf_counter()
        torch.set_num_threads(4)
        torch.manual_seed(20261112)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        cached = [
            try_to_load_from_cache(self.spec.repo_id, filename, revision=self.spec.revision)
            for filename in ("model.safetensors", "pytorch_model.bin")
        ]
        weight_path = next((p for p in cached if isinstance(p, str) and Path(p).is_file()), None)
        if weight_path:
            path = str(Path(weight_path).parent)
        else:
            files = HfApi().list_repo_files(self.spec.repo_id, revision=self.spec.revision)
            weight = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
            path = snapshot_download(
                self.spec.repo_id,
                revision=self.spec.revision,
                allow_patterns=[
                    "*.json",
                    weight,
                    "sentencepiece.bpe.model",
                    "tokenizer.model",
                    "vocab.txt",
                    "merges.txt",
                ],
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            path,
            padding_side="left" if self.spec.protocol == "qwen" else "right",
            trust_remote_code=False,
        )
        cls: Any = AutoModel
        if self.spec.kind == "reranker":
            cls = (
                AutoModelForCausalLM
                if self.spec.protocol == "qwen"
                else AutoModelForSequenceClassification
            )
        self.model = (
            cls.from_pretrained(
                path,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=False,
            )
            .to(self.device)
            .eval()
        )
        self.load_seconds = time.perf_counter() - started

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode(self, texts: Sequence[str], *, query: bool = False) -> npt.NDArray[np.float32]:
        import torch

        if not texts:
            return np.zeros((0, self.spec.dimensions), dtype=np.float32)
        self.load()
        outputs = []
        for start in range(0, len(texts), self.spec.batch_size):
            batch = list(texts[start : start + self.spec.batch_size])
            if self.spec.protocol == "e5":
                batch = [("query: " if query else "passage: ") + t for t in batch]
            elif self.spec.protocol == "qwen" and query:
                batch = [f"Instruct: {INSTRUCTION}\nQuery: {t}" for t in batch]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
                if self.spec.pooling == "mean":
                    mask = enc.attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                elif self.spec.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    # Qwen uses left padding, so the last position is always real.
                    pooled = hidden[:, -1]
                pooled = torch.nn.functional.normalize(pooled.float(), dim=-1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs)

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        import torch

        if not pairs:
            return []
        self.load()
        result: list[float] = []
        for start in range(0, len(pairs), self.spec.batch_size):
            batch = pairs[start : start + self.spec.batch_size]
            if self.spec.protocol == "qwen":
                prefix = self.tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
                suffix = self.tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
                texts = [
                    f"<Instruct>: {INSTRUCTION}\n<Query>: {q}\n<Document>: {d}" for q, d in batch
                ]
                ids = self.tokenizer(
                    texts,
                    padding=False,
                    truncation=True,
                    max_length=self.spec.max_length - len(prefix) - len(suffix),
                    add_special_tokens=False,
                )["input_ids"]
                enc = self.tokenizer.pad(
                    {"input_ids": [prefix + v + suffix for v in ids]},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    hidden = self.model.model(**enc, use_cache=False).last_hidden_state[:, -1]
                    indices = [self.tokenizer.convert_tokens_to_ids(t) for t in ("no", "yes")]
                    logits = torch.nn.functional.linear(
                        hidden, self.model.lm_head.weight[indices]
                    ).float()
                    scores = logits.softmax(-1)[:, 1]
            else:
                enc = self.tokenizer(
                    [list(p) for p in batch],
                    padding=True,
                    truncation=True,
                    max_length=self.spec.max_length,
                    return_tensors="pt",
                ).to(self.device)
                with torch.inference_mode():
                    scores = self.model(**enc).logits.view(-1).float().sigmoid()
            result.extend(float(s) for s in scores.cpu())
        return result


class VectorCache:
    """SQLite commits bounded batches; corrupt blobs fail rather than silently changing data."""

    def __init__(self, root: Path, spec: ModelSpec, encoder: Encoder) -> None:
        self.encoder = encoder
        self.spec = spec
        identity = sha256_text(json.dumps(spec.model_dump(), sort_keys=True))[:20]
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{identity}.sqlite"
        self.computed = 0
        self.hits = 0
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS vectors "
                "(key TEXT PRIMARY KEY, data BLOB NOT NULL, sha TEXT NOT NULL)"
            )

    def encode(self, texts: Sequence[str], *, query: bool = False) -> npt.NDArray[np.float32]:
        keys = [sha256_text(f"{query}|{t}") for t in texts]
        found: dict[str, npt.NDArray[np.float32]] = {}
        missing: dict[str, str] = {}
        with sqlite3.connect(self.path) as conn:
            for key, text in zip(keys, texts, strict=True):
                row = conn.execute("SELECT data, sha FROM vectors WHERE key=?", (key,)).fetchone()
                if row:
                    blob, checksum = row
                    if hashlib.sha256(blob).hexdigest() != checksum:
                        raise ValueError("embedding cache checksum mismatch")
                    vector = np.frombuffer(blob, dtype=np.float32)
                    if vector.shape != (self.spec.dimensions,) or not np.isfinite(vector).all():
                        raise ValueError("embedding cache invalid vector")
                    found[key] = vector
                    self.hits += 1
                else:
                    missing[key] = text
            pending = list(missing.items())
            for start in range(0, len(pending), 32):
                batch = pending[start : start + 32]
                vectors = self.encoder.encode([v for _, v in batch], query=query)
                if (
                    vectors.shape != (len(batch), self.spec.dimensions)
                    or not np.isfinite(vectors).all()
                ):
                    raise ValueError("encoder returned invalid vectors")
                for (key, _), vec in zip(batch, vectors, strict=True):
                    blob = vec.astype(np.float32).tobytes()
                    conn.execute(
                        "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?)",
                        (key, blob, hashlib.sha256(blob).hexdigest()),
                    )
                    found[key] = vec
                conn.commit()
                self.computed += len(batch)
        return (
            np.stack([found[k] for k in keys])
            if keys
            else np.zeros((0, self.spec.dimensions), dtype=np.float32)
        )
