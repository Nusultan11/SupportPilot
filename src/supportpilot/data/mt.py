"""Machine translation with a resumable on-disk cache (Stage 4, ADR-024).

Systems are pinned Hugging Face seq2seq models (configs/base.yaml `stage4.mt_systems`):
Marian (Opus-MT), M2M100 and MADLAD-400 (T5). A `pivot` system chains two systems
(e.g. en->ru->kk). Every translation is cached in Parquet shards under
`data/interim/stage4/mt/<system>/<src>-<tgt>/`, keyed by sha256(system|src|tgt|text), so an
interrupted run resumes where it stopped and a re-run translates nothing.

torch/transformers are imported lazily: building datasets from an existing cache does not
need the `ml`/`gpu` dependency groups.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.hashing import sha256_text
from supportpilot.common.logging import get_logger

log = get_logger(__name__)
SHARD_ROWS = 512
CACHE_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("system", pa.string()),
        ("src_lang", pa.string()),
        ("tgt_lang", pa.string()),
        ("src_text", pa.string()),
        ("mt_text", pa.string()),
    ]
)


class MTSystem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    arch: Literal["marian", "m2m100", "madlad", "pivot"]
    repo_id: str | None = None
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    licence: str
    # src->tgt pairs this system serves, e.g. ["en-ru"]; pivot: [first_system, second_system]
    pairs: list[str] = Field(default_factory=list)
    via: list[str] = Field(default_factory=list)
    target_prefix: dict[str, str] = Field(default_factory=dict)  # Marian multi-target tokens
    quant: Literal["none", "nf4"] = "none"
    beams: int = Field(default=4, ge=1)
    batch_size: int = Field(default=32, ge=1)
    max_new_tokens: int = Field(default=160, ge=8)
    training_use: bool = Field(
        description="false for NC-licensed systems: outputs may be compared, never trained on"
    )


def cache_key(system: str, src: str, tgt: str, text: str) -> str:
    return sha256_text(f"{system}|{src}|{tgt}|{text}")[:32]


def cache_dir(data_dir: Path, system: str, src: str, tgt: str) -> Path:
    return data_dir / "interim" / "stage4" / "mt" / system / f"{src}-{tgt}"


def read_cache(path: Path) -> dict[str, str]:
    if not path.is_dir() or not any(path.glob("shard-*.parquet")):
        return {}
    df = pl.read_parquet(sorted(path.glob("shard-*.parquet")), columns=["key", "mt_text"])
    return dict(zip(df["key"].to_list(), df["mt_text"].to_list(), strict=True))


def _write_shard(path: Path, rows: list[dict[str, str]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    tmp = path / f".tmp-{uuid.uuid4().hex[:8]}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=CACHE_SCHEMA), tmp, compression="zstd")
    n = len(list(path.glob("shard-*.parquet")))
    tmp.replace(path / f"shard-{n:05d}-{uuid.uuid4().hex[:6]}.parquet")


Backend = Callable[[Sequence[str], str, str], list[str]]


class HFTranslator:
    """Lazy-loading wrapper around one pinned HF seq2seq model."""

    def __init__(self, spec: MTSystem) -> None:
        if spec.arch == "pivot" or not spec.repo_id or not spec.revision:
            raise ValueError(f"{spec.id}: not a single-model system")
        self.spec = spec
        self._model: Any = None
        self._tok: Any = None

    def _load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

        assert self.spec.repo_id
        assert self.spec.revision
        path = snapshot_download(
            self.spec.repo_id,
            revision=self.spec.revision,
            allow_patterns=["*.json", "*.spm", "*.model", "pytorch_model.bin", "model.safetensors"],
        )
        cuda = torch.cuda.is_available()
        kw: dict[str, Any] = {}
        if self.spec.quant == "nf4":
            if not cuda:
                raise RuntimeError(f"{self.spec.id}: nf4 quantisation needs CUDA")
            kw["quantization_config"] = BitsAndBytesConfig(  # type: ignore[no-untyped-call]
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            kw["device_map"] = "cuda:0"
            kw["dtype"] = torch.bfloat16
        elif cuda:
            kw["dtype"] = torch.float16
        self._tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSeq2SeqLM.from_pretrained(path, **kw)
        if cuda and self.spec.quant == "none":
            model = model.cuda()
        self._model = model.eval()
        self._device = "cuda" if cuda else "cpu"

    def __call__(self, texts: Sequence[str], src: str, tgt: str) -> list[str]:
        import torch

        if self._model is None:
            self._load()
        tok, spec = self._tok, self.spec
        gen: dict[str, Any] = {"num_beams": spec.beams, "max_new_tokens": spec.max_new_tokens}
        if spec.arch == "madlad":
            inputs = [f"<2{tgt}> {t}" for t in texts]
        elif spec.arch == "m2m100":
            tok.src_lang = src
            gen["forced_bos_token_id"] = tok.get_lang_id(tgt)
            inputs = list(texts)
        else:
            prefix = spec.target_prefix.get(tgt, "")
            inputs = [f"{prefix}{t}" for t in texts]
        out: list[str] = []
        for i in range(0, len(inputs), spec.batch_size):
            batch = inputs[i : i + spec.batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
            enc = enc.to(self._device)
            # Short support messages: cap the output at ~2.5x the input so a degenerate
            # loop cannot run to max_new_tokens (and spill the 4 GB GPU into shared memory).
            cap = int(enc["input_ids"].shape[1] * 2.5) + 16
            with torch.no_grad():
                ids = self._model.generate(
                    **enc, **{**gen, "max_new_tokens": min(cap, spec.max_new_tokens)}
                )
            out.extend(s.strip() for s in tok.batch_decode(ids, skip_special_tokens=True))
        return out

    def close(self) -> None:
        import gc

        import torch

        self._model = self._tok = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class TranslateResult(BaseModel):
    system: str
    pair: str
    requested: int
    cached: int
    translated: int
    seconds: float


def translate_cached(
    texts: Sequence[str],
    *,
    system: MTSystem,
    src: str,
    tgt: str,
    data_dir: Path,
    backend: Backend,
    limit: int | None = None,
) -> TranslateResult:
    """Translate the texts missing from the cache, writing one shard per `SHARD_ROWS` rows."""
    start = time.perf_counter()
    out = cache_dir(data_dir, system.id, src, tgt)
    for stale in out.glob(".tmp-*.parquet") if out.is_dir() else []:
        stale.unlink(missing_ok=True)  # left by a killed run
    have = read_cache(out)
    unique = list(dict.fromkeys(texts))
    todo = [t for t in unique if cache_key(system.id, src, tgt, t) not in have]
    n_missing = len(todo)
    if limit is not None:
        todo = todo[:limit]
    # Length-sorted batches waste less padding; the cache is keyed, so order does not matter.
    todo.sort(key=len)
    done = 0
    for i in range(0, len(todo), SHARD_ROWS):
        chunk = todo[i : i + SHARD_ROWS]
        hyps = backend(chunk, src, tgt)
        rows = [
            {
                "key": cache_key(system.id, src, tgt, s),
                "system": system.id,
                "src_lang": src,
                "tgt_lang": tgt,
                "src_text": s,
                "mt_text": h,
            }
            for s, h in zip(chunk, hyps, strict=True)
        ]
        _write_shard(out, rows)
        done += len(chunk)
        log.info(
            "mt shard written",
            extra={"system": system.id, "pair": f"{src}-{tgt}", "done": done, "todo": len(todo)},
        )
    return TranslateResult(
        system=system.id,
        pair=f"{src}-{tgt}",
        requested=len(unique),
        cached=len(unique) - n_missing,
        translated=done,
        seconds=round(time.perf_counter() - start, 2),
    )


def lookup(
    data_dir: Path, system: str, src: str, tgt: str, texts: Sequence[str]
) -> list[str | None]:
    have = read_cache(cache_dir(data_dir, system, src, tgt))
    return [have.get(cache_key(system, src, tgt, t)) for t in texts]
