from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_chunk_id
from supportpilot.common.schemas import KBChunk, Locator
from supportpilot.retrieval.corpus import search_units
from supportpilot.retrieval.encode import Transformer, VectorCache
from supportpilot.retrieval.evaluate import clustered_ci, paired_comparison, phase, query_metrics
from supportpilot.retrieval.index import qdrant_call, writer_lock
from supportpilot.retrieval.models import ModelSpec, Recipe, ScoredChunk, SearchOptions
from supportpilot.retrieval.search import assemble_context, fuse, rerank
from supportpilot.retrieval.sparse import sparse_vector, tokenize


def spec() -> ModelSpec:
    return ModelSpec(
        repo_id="synthetic/encoder",
        revision="0" * 40,
        licence="mit",
        kind="embedding",
        dimensions=3,
        protocol="plain",
    )


def tokenizer() -> Tokenizer:
    tok = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tok


def chunk(
    text: str = "Доставка в Алматы занимает три рабочих дня.",
    *,
    lang: str = "ru",
    doc_id: str = "delivery",
    position: int = 0,
    version: str | None = None,
    effective_to: date | None = None,
) -> KBChunk:
    version = version or sha256_text(text)
    return KBChunk(
        chunk_id=make_chunk_id(doc_id, version, position),
        doc_id=doc_id,
        version=version,
        lang=lang,
        position=position,
        section_id="terms",
        section_path=["Delivery"],
        title="Store delivery",
        text=text,
        locator=Locator(line_range=(1, 2)),
        doc_type="policy",
        format="txt",
        source_path=f"sources/{lang}/{doc_id}.txt",
        effective_from=date(2026, 1, 1),
        effective_to=effective_to,
        token_count=20,
        translation_group=f"{doc_id}.terms",
        checksum=sha256_text(text),
    )


class FakeEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: Any, *, query: bool = False) -> npt.NDArray[np.float32]:
        self.calls += len(texts)
        rows = np.array(
            [[int(sha256_text(t)[i : i + 4], 16) + 1 for i in (0, 4, 8)] for t in texts],
            dtype=np.float32,
        )
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_metrics_with_graded_relevance_and_no_duplicates() -> None:
    result = query_metrics(["irrelevant", "b", "a"], {"a": 3, "b": 1, "c": 2, "bad": 0})
    assert result["recall5"] == pytest.approx(2 / 3)
    assert result["mrr10"] == 0.5
    assert 0 < result["ndcg10"] < 1
    assert query_metrics(["a", "b"], {"a": 3, "b": 1}, minimum_grade=2)["recall5"] == 1
    with pytest.raises(ValueError, match="unique"):
        query_metrics(["a", "a"], {"a": 3})
    with pytest.raises(ValueError, match="undefined"):
        query_metrics([], {})


def test_cluster_bootstrap_keeps_translations_together() -> None:
    rows = [{"scenario_id": "one", "recall5": 1.0}] * 30
    rows += [{"scenario_id": "two", "recall5": 0.0}] * 30
    ci = clustered_ci(rows, "recall5", seed=42)
    assert ci == clustered_ci(rows, "recall5", seed=42)
    assert ci["mean"] == 0.5
    assert (ci["low"], ci["high"], ci["scenarios"]) == (0, 1, 2)
    assert phase("same-family", 42) == phase("same-family", 42)


def test_paired_comparison_requires_matched_queries() -> None:
    base = [
        {
            "query_id": "q",
            "scenario_id": "s",
            "answerable": True,
            "phase": "holdout",
            "slices": ["all"],
            "recall5": 0,
            "mrr10": 0,
            "ndcg10": 0,
        }
    ]
    better = [{**base[0], "recall5": 1, "mrr10": 1, "ndcg10": 1}]
    assert paired_comparison(base, better, seed=0)["recall5"]["low"] == 1
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(base, [], seed=0)


def test_kazakh_sparse_keeps_letters_and_tests_morphology() -> None:
    assert tokenize("ҚҰЛАҚҚАП, Өлшемі, төлем!") == ["құлаққап", "өлшемі", "төлем"]
    assert set(tokenize("тапсырысыңызды", "prefix")) & set(tokenize("тапсырыс", "prefix"))
    recipe = Recipe(encoder=spec())
    ids, values = sparse_vector("төлем төлем", recipe)
    assert len(ids) == 1
    assert values[0] > 0
    assert sparse_vector("", recipe) == ([], [])
    assert sparse_vector("төлем", recipe, query=True)[1] == [1]


def test_fusion_ties_constant_scores_and_weighted_extremes() -> None:
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("b", 2.0), ("c", 2.0)]
    assert fuse(dense, sparse, SearchOptions())[0][0] == "b"
    result = fuse(dense, sparse, SearchOptions(fusion="weighted", dense_weight=1))
    assert result[0] == ("a", 1)
    assert fuse([], [], SearchOptions()) == []


def test_cache_resumes_and_rejects_corruption(tmp_path: Path) -> None:
    encoder = FakeEncoder()
    cache = VectorCache(tmp_path, spec(), encoder)
    first = cache.encode(["a", "b", "a"])
    assert encoder.calls == 2
    assert np.array_equal(first, cache.encode(["a", "b", "a"]))
    assert encoder.calls == 2
    cache.encode(["a"], query=True)
    assert encoder.calls == 3
    with sqlite3.connect(cache.path) as conn:
        conn.execute("UPDATE vectors SET sha='broken'")
    with pytest.raises(ValueError, match="checksum"):
        cache.encode(["a"])


def test_search_windows_preserve_ids_and_source_boundaries() -> None:
    text = " ".join(f"word{i}" for i in range(40))
    source = chunk(text)
    recipe = Recipe(encoder=spec(), view="window", window_tokens=16, overlap_tokens=4)
    units = search_units([source], recipe, tokenizer())
    assert len(units) == 3
    assert all(u.chunk_ids == [source.chunk_id] for u in units)
    assert all("word" in u.text for u in units)
    assert units == search_units([source], recipe, tokenizer())
    with pytest.raises(ValueError, match="overlap"):
        Recipe(encoder=spec(), window_tokens=16, overlap_tokens=16)


def test_context_counts_metadata_and_never_cuts_evidence() -> None:
    tok = tokenizer()
    source = chunk()
    row = ScoredChunk(
        chunk_id=source.chunk_id,
        kb_version="kb",
        index_version="idx",
        chunk=source,
        score=1,
        matched_unit_ids=["u"],
    )
    full = assemble_context([row, row], tok, 1000, kb_version="kb", index_version="idx")
    assert len(full.evidence) == 1
    assert source.text in full.text
    assert '"locator"' in full.text
    assert full.token_count == len(tok.encode(full.text, add_special_tokens=False).ids)
    empty = assemble_context([row], tok, full.token_count - 1, kb_version="kb", index_version="idx")
    assert empty.abstain
    assert empty.text == ""
    assert empty.evidence == []
    assert empty.reason == "context_budget_too_small"
    assert (
        assemble_context([], tok, 100, kb_version="kb", index_version="idx").reason
        == "no_retrieval_evidence"
    )


def test_bounded_idempotent_io_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("supportpilot.retrieval.index.time.sleep", lambda seconds: None)
    calls = []

    def transient() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("synthetic connection loss")
        return "ok"

    assert qdrant_call(transient) == "ok"
    assert len(calls) == 3

    def unavailable() -> None:
        raise TimeoutError("synthetic timeout")

    with pytest.raises(ValueError, match="three bounded attempts"):
        qdrant_call(unavailable)


def test_writer_lock_is_released_on_failure(tmp_path: Path) -> None:
    lock = tmp_path / "writer.lock"

    def interrupted_writer() -> None:
        with writer_lock(lock):
            with pytest.raises(ValueError, match="busy"), writer_lock(lock):
                pytest.fail("concurrent writer entered")
            raise RuntimeError("synthetic interruption")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        interrupted_writer()
    with writer_lock(lock):
        pass
    # Empty lock files from a crash before the first write also remain recoverable.
    lock.write_bytes(b"")
    with writer_lock(lock):
        pass


def test_os_lock_releases_after_a_real_process_is_killed(tmp_path: Path) -> None:
    import subprocess
    import sys

    lock = tmp_path / "writer.lock"
    script = (
        "import os,sys,time; from pathlib import Path; "
        "from supportpilot.retrieval.index import writer_lock; "
        "lock=writer_lock(Path(sys.argv[1])); lock.__enter__(); "
        "print(os.getpid(),flush=True); time.sleep(30)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(lock)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        import psutil

        owner = psutil.Process(int(process.stdout.readline().strip()))
        with pytest.raises(ValueError, match="busy"), writer_lock(lock):
            pytest.fail("parent acquired child lock")
        owner.kill()
        owner.wait(timeout=5)
        process.wait(timeout=5)
        with writer_lock(lock):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_embedding_cache_preserves_committed_batches_after_interruption(tmp_path: Path) -> None:
    class Interrupted(FakeEncoder):
        def encode(self, texts: Any, *, query: bool = False) -> npt.NDArray[np.float32]:
            if self.calls:
                raise RuntimeError("synthetic encoder interruption")
            return super().encode(texts, query=query)

    texts = [f"synthetic document {i}" for i in range(35)]
    failed = VectorCache(tmp_path, spec(), Interrupted())
    with pytest.raises(RuntimeError, match="encoder interruption"):
        failed.encode(texts)
    resumed_encoder = FakeEncoder()
    resumed = VectorCache(tmp_path, spec(), resumed_encoder)
    assert resumed.encode(texts).shape == (35, 3)
    assert resumed_encoder.calls == 3
    assert resumed.hits == 32


def test_rerank_budget_is_distinct_from_candidate_and_output_budget() -> None:
    class FixedRanker(Transformer):
        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            assert len(pairs) == 2
            return [0.2, 0.9]

    sources = [chunk(f"text {i}", doc_id=f"document_{i}") for i in range(3)]
    rows = [
        ScoredChunk(
            chunk_id=c.chunk_id,
            kb_version="kb",
            index_version="idx",
            chunk=c,
            score=1,
            matched_unit_ids=["u"],
        )
        for c in sources
    ]
    result = rerank(
        "question", rows, FixedRanker(spec()), SearchOptions(candidate_k=3, rerank_k=2, k=1)
    )
    assert len(result) == 2
    assert result[0].chunk_id == sources[1].chunk_id
