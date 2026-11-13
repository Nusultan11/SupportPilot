"""Qdrant retrieval, comparable fusion, reranking and token-exact evidence packing."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from qdrant_client import models as qm

from supportpilot.common.config import Settings
from supportpilot.data.eda import load_tokenizer
from supportpilot.retrieval.abstain import evidence_gate
from supportpilot.retrieval.encode import Encoder, Transformer
from supportpilot.retrieval.index import IndexArtifact, qdrant_call, require_ready
from supportpilot.retrieval.models import (
    RetrievalContext,
    ScoredChunk,
    SearchFilters,
    SearchOptions,
)
from supportpilot.retrieval.sparse import sparse_vector


def qdrant_filter(filters: SearchFilters, kb_version: str) -> qm.Filter:
    must: list[Any] = [
        qm.FieldCondition(key="kb_version", match=qm.MatchValue(value=kb_version)),
    ]
    if not filters.all_versions:
        must.extend(
            [
                qm.FieldCondition(key="valid_from", range=qm.Range(lte=filters.as_of.toordinal())),
                qm.FieldCondition(key="valid_to", range=qm.Range(gte=filters.as_of.toordinal())),
            ]
        )
    if filters.corpus_lang:
        must.append(qm.FieldCondition(key="lang", match=qm.MatchValue(value=filters.corpus_lang)))
    for key, values in (("doc_id", filters.doc_ids), ("doc_type", filters.doc_types)):
        if values:
            must.append(qm.FieldCondition(key=key, match=qm.MatchAny(any=values)))
    return qm.Filter(must=must)


def fuse(
    dense: list[tuple[str, float]], sparse: list[tuple[str, float]], options: SearchOptions
) -> list[tuple[str, float]]:
    if options.mode == "dense":
        return dense
    if options.mode == "sparse":
        return sparse
    scores: dict[str, float] = {}
    for rows, weight in ((dense, options.dense_weight), (sparse, 1 - options.dense_weight)):
        if not rows:
            continue
        low = min(s for _, s in rows)
        high = max(s for _, s in rows)
        for rank, (uid, score) in enumerate(rows, 1):
            if options.fusion == "rrf":
                value = 1 / (options.rrf_k + rank)
            else:
                value = weight * ((score - low) / (high - low) if high > low else 1.0)
            scores[uid] = scores.get(uid, 0) + value
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


def candidate_lists(
    client: Any,
    name: str,
    artifact: IndexArtifact,
    query: str,
    vector: npt.NDArray[np.float32] | None,
    filters: SearchFilters,
    limit: int,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    filt = qdrant_filter(filters, artifact.kb_version)
    common = {
        "collection_name": name,
        "query_filter": filt,
        "limit": limit,
        "with_payload": ["unit_id"],
        "search_params": qm.SearchParams(exact=True),
    }
    dense = []
    if vector is not None:
        points = qdrant_call(
            client.query_points, query=vector.tolist(), using="dense", **common
        ).points
        dense = [(p.payload["unit_id"], float(p.score)) for p in points]
    ids, values = sparse_vector(query, artifact.recipe, query=True)
    sparse = []
    if ids:
        # Global IDF can include tombstoned segment statistics after updates.
        # Scope statistics to live members of this immutable KB version.
        sparse_common = {
            **common,
            "search_params": qm.SearchParams(
                exact=True,
                idf=qm.IdfCorpusParams(
                    corpus=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="kb_version", match=qm.MatchValue(value=artifact.kb_version)
                            )
                        ]
                    )
                ),
            ),
        }
        points = qdrant_call(
            client.query_points,
            query=qm.SparseVector(indices=ids, values=values),
            using="sparse",
            **sparse_common,
        ).points
        sparse = [(p.payload["unit_id"], float(p.score)) for p in points if p.score > 0]
    # Stable ties are needed for reproducible offline and online evaluation.
    return (sorted(dense, key=lambda p: (-p[1], p[0])), sorted(sparse, key=lambda p: (-p[1], p[0])))


def scored_chunks(
    artifact: IndexArtifact,
    dense: list[tuple[str, float]],
    sparse: list[tuple[str, float]],
    options: SearchOptions,
) -> list[ScoredChunk]:
    units = {u.unit_id: u for u in artifact.units}
    chunks = {c.chunk_id: c for c in artifact.chunks}
    ds, ss = dict(dense), dict(sparse)
    rows = fuse(dense[: options.candidate_k], sparse[: options.candidate_k], options)
    result: dict[str, ScoredChunk] = {}
    for uid, score in rows:
        if options.min_dense_score is not None and ds.get(uid, -1) < options.min_dense_score:
            continue
        for cid in units[uid].chunk_ids:
            if cid in result:
                previous = result[cid]
                result[cid] = previous.model_copy(
                    update={"matched_unit_ids": [*previous.matched_unit_ids, uid]}
                )
                continue
            result[cid] = ScoredChunk(
                chunk_id=cid,
                kb_version=artifact.kb_version,
                index_version=artifact.manifest.dataset_version,
                chunk=chunks[cid],
                score=score,
                dense_score=ds.get(uid),
                sparse_score=ss.get(uid),
                matched_unit_ids=[uid],
            )
    return list(result.values())


def rerank(
    query: str, rows: list[ScoredChunk], model: Transformer, options: SearchOptions
) -> list[ScoredChunk]:
    candidates = rows[: options.rerank_k or options.candidate_k]
    scores = model.score([(query, f"{r.chunk.title}\n{r.chunk.text}") for r in candidates])
    ranked = [
        r.model_copy(update={"rerank_score": s, "score": s})
        for r, s in zip(candidates, scores, strict=True)
        if options.min_rerank_score is None or s >= options.min_rerank_score
    ]
    return sorted(ranked, key=lambda r: (-r.score, r.chunk_id))


def assemble_context(
    rows: list[ScoredChunk],
    tokenizer: Any,
    budget: int,
    *,
    kb_version: str,
    index_version: str,
    diversify: bool = False,
) -> RetrievalContext:
    if budget < 0:
        raise ValueError("negative context budget")
    if diversify:
        first, rest, seen = [], [], set()
        for row in rows:
            key = (row.chunk.doc_id, row.chunk.section_id)
            if key in seen:
                rest.append(row)
            else:
                first.append(row)
                seen.add(key)
        rows = first + rest
    selected: list[ScoredChunk] = []
    blocks: list[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        if row.chunk_id in seen_ids:
            continue
        seen_ids.add(row.chunk_id)
        metadata = {
            "chunk_id": row.chunk_id,
            "doc_id": row.chunk.doc_id,
            "effective_from": row.chunk.effective_from.isoformat(),
            "effective_to": row.chunk.effective_to.isoformat() if row.chunk.effective_to else None,
            "source_path": row.chunk.source_path,
            "locator": row.chunk.locator.model_dump(exclude_none=True),
        }
        # JSON encoding keeps untrusted source text inside data, without interpreting instructions.
        block = json.dumps({"source": metadata, "text": row.chunk.text}, ensure_ascii=False)
        proposed = "\n".join([*blocks, block])
        if len(tokenizer.encode(proposed, add_special_tokens=False).ids) <= budget:
            blocks.append(block)
            selected.append(row)
    text = "\n".join(blocks)
    return RetrievalContext(
        kb_version=kb_version,
        index_version=index_version,
        evidence=selected,
        text=text,
        token_count=len(tokenizer.encode(text, add_special_tokens=False).ids),
        budget=budget,
        truncated=len(selected) < len(seen_ids),
        abstain=not selected,
        reason=("context_budget_too_small" if rows else "no_retrieval_evidence")
        if not selected
        else None,
    )


@contextmanager
def open_retriever(settings: Settings) -> Iterator[Retriever]:
    from supportpilot.ops.health import make_qdrant_client
    from supportpilot.retrieval.index import default_recipe, index_root, read_index
    from supportpilot.retrieval.models import load_models

    recipe = default_recipe(settings)
    root = index_root(settings, recipe)
    state = json.loads((root / "state.json").read_text())
    index_version = state["index_version"]
    if Path(index_version).name != index_version:
        raise ValueError("invalid index version")
    artifact = read_index(root / "versions" / index_version)
    if artifact.recipe.identity != recipe.identity:
        raise ValueError("active index uses a different retrieval recipe")
    require_ready(settings, artifact)
    client = make_qdrant_client(settings.qdrant, settings)
    encoder = Transformer(recipe.encoder, settings.retrieval.device)
    reranker = (
        Transformer(
            load_models(settings.resolve(settings.retrieval.models))[settings.retrieval.reranker],
            settings.retrieval.device,
        )
        if settings.retrieval.reranker
        else None
    )
    try:
        yield Retriever(settings, artifact, client, encoder=encoder, reranker=reranker)
    finally:
        encoder.close()
        if reranker:
            reranker.close()
        client.close()


class Retriever:
    def __init__(
        self,
        settings: Settings,
        artifact: IndexArtifact,
        client: Any,
        *,
        encoder: Encoder | None = None,
        reranker: Transformer | None = None,
        tokenizer: Any = None,
    ) -> None:
        self.settings, self.artifact, self.client = settings, artifact, client
        self.encoder = encoder or Transformer(artifact.recipe.encoder, settings.retrieval.device)
        self.reranker = reranker
        self.tokenizer = (
            tokenizer if tokenizer is not None else load_tokenizer(settings.eda.tokenizers[0])
        )
        self.last_timings: dict[str, float] = {}
        self.last_rejection_reason: str | None = None

    def _ready(self) -> str:
        active = (
            self.settings.resolve(self.settings.paths.data_dir)
            / "kb"
            / self.settings.kb.name
            / "current.json"
        )
        if json.loads(active.read_text())["kb_version"] != self.artifact.kb_version:
            raise ValueError("active KB changed; synchronise retrieval index before serving")
        return require_ready(self.settings, self.artifact)

    def retrieve(
        self,
        query: str,
        lang: str | None = None,
        filters: SearchFilters | None = None,
        k: int = 10,
        *,
        options: SearchOptions | None = None,
    ) -> list[ScoredChunk]:
        name = self._ready()
        cfg = self.settings.retrieval
        options = options or SearchOptions(
            mode=cfg.mode,
            fusion=cfg.fusion,
            dense_weight=cfg.dense_weight,
            candidate_k=cfg.candidate_k,
            rerank_k=cfg.rerank_k,
            k=k,
            min_dense_score=cfg.min_dense_score,
            min_rerank_score=cfg.min_rerank_score,
            evidence_threshold=cfg.evidence_threshold,
        )
        self.last_rejection_reason = None
        filters = filters or SearchFilters(corpus_lang=lang)
        if not query.strip():
            return []
        started = time.perf_counter()
        vector = None if options.mode == "sparse" else self.encoder.encode([query], query=True)[0]
        encoded = time.perf_counter()
        dense, sparse = candidate_lists(
            self.client, name, self.artifact, query, vector, filters, options.candidate_k
        )
        rows = scored_chunks(self.artifact, dense, sparse, options)
        searched = time.perf_counter()
        if self.reranker:
            rows = rerank(query, rows, self.reranker, options)
        gated = evidence_gate(rows, options.evidence_threshold)
        if rows and not gated:
            self.last_rejection_reason = "low_evidence_confidence"
        rows = gated
        self._ready()
        finished = time.perf_counter()
        self.last_timings = {
            "encode_ms": (encoded - started) * 1000,
            "search_ms": (searched - encoded) * 1000,
            "rerank_ms": (finished - searched) * 1000,
            "total_ms": (finished - started) * 1000,
        }
        return rows[: options.k]

    def context(
        self,
        query: str,
        lang: str | None = None,
        filters: SearchFilters | None = None,
        k: int = 10,
        *,
        budget: int | None = None,
        diversify: bool = False,
    ) -> RetrievalContext:
        rows = self.retrieve(query, lang, filters, k)
        result = assemble_context(
            rows,
            self.tokenizer,
            self.settings.retrieval.context_tokens if budget is None else budget,
            kb_version=self.artifact.kb_version,
            index_version=self.artifact.manifest.dataset_version,
            diversify=diversify,
        )
        return (
            result.model_copy(update={"reason": self.last_rejection_reason})
            if result.abstain and self.last_rejection_reason
            else result
        )
