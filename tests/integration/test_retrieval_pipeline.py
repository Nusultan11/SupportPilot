from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest
from qdrant_client import QdrantClient

from supportpilot.common.config import Settings
from supportpilot.common.io import write_json
from supportpilot.common.schemas import DocType
from supportpilot.retrieval import index
from supportpilot.retrieval.index import (
    build_artifact,
    index_root,
    inventory,
    require_ready,
    sync_index,
)
from supportpilot.retrieval.models import Recipe, SearchFilters, SearchOptions
from supportpilot.retrieval.search import Retriever, candidate_lists, scored_chunks
from tests.unit.test_retrieval import FakeEncoder, chunk, spec, tokenizer


def install_active(settings: Settings, version: str) -> None:
    write_json(
        settings.resolve(settings.paths.data_dir) / "kb" / settings.kb.name / "current.json",
        {"kb_version": version},
    )


def test_metadata_only_update_refreshes_filters_without_reencoding(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = Recipe(encoder=spec())
    source = chunk()
    current = {"version": "kb_demo_store-20261112-88888888", "chunks": [source]}
    monkeypatch.setattr(
        index, "load_kb", lambda *a, **kw: (current["version"], current["chunks"], {})
    )
    encoder = FakeEncoder()
    initial = build_artifact(settings, recipe, encoder=encoder, tokenizer=tokenizer())
    client = QdrantClient(":memory:")
    try:
        sync_index(settings, initial, client)
        calls = encoder.calls
        current["version"] = "kb_demo_store-20261112-99999999"
        current["chunks"] = [
            source.model_copy(
                update={
                    "effective_to": date(2026, 11, 11),
                    "doc_type": DocType.FAQ,
                }
            )
        ]
        updated = build_artifact(settings, recipe, encoder=encoder, tokenizer=tokenizer())
        assert encoder.calls == calls
        assert initial.units[0].unit_id == updated.units[0].unit_id
        result = sync_index(settings, updated, client)
        assert result["upserted"] == result["deleted"] == 0
        assert result["metadata_updates"] == result["unchanged_vectors"] == 1

        def sparse_rows(
            artifact: index.IndexArtifact, day: date, kind: str
        ) -> list[tuple[str, float]]:
            return candidate_lists(
                client,
                result["collection"],
                artifact,
                "Доставка",
                None,
                SearchFilters(as_of=day, doc_types=[kind]),
                10,
            )[1]

        assert sparse_rows(updated, date(2026, 11, 10), "faq")
        assert not sparse_rows(updated, date(2026, 11, 10), "policy")
        assert not sparse_rows(updated, date(2026, 11, 12), "faq")
        assert sync_index(settings, updated, client)["status"] == "skipped"
        sync_index(settings, initial, client)
        assert sparse_rows(initial, date(2026, 11, 12), "policy")
    finally:
        client.close()


def test_qdrant_resumption_incremental_rollback_and_version_guard(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = Recipe(encoder=spec())
    original = [
        chunk(),
        chunk("Ескі кепілдік", doc_id="warranty", effective_to=date(2026, 2, 1)),
        chunk("Delivery costs", lang="en", doc_id="delivery_en"),
    ]
    version = "kb_demo_store-20261112-11111111"
    current = {"version": version, "chunks": original}
    monkeypatch.setattr(
        index, "load_kb", lambda *a, **kw: (current["version"], current["chunks"], {"parent": None})
    )
    encoder = FakeEncoder()
    artifact = build_artifact(settings, recipe, encoder=encoder, tokenizer=tokenizer())
    install_active(settings, version)
    client = QdrantClient(":memory:")
    try:

        def interrupt() -> None:
            raise RuntimeError("simulated process loss after a committed batch")

        with pytest.raises(RuntimeError, match="process loss"):
            sync_index(settings, artifact, client, after_batch=interrupt)
        with pytest.raises(ValueError, match="incomplete"):
            require_ready(settings, artifact)
        with index.writer_lock(index_root(settings, recipe) / "writer.lock"):
            pass
        resumed = sync_index(settings, artifact, client)
        assert resumed["upserted"] == 0
        assert sync_index(settings, artifact, client)["status"] == "skipped"
        dense, sparse = candidate_lists(
            client,
            require_ready(settings, artifact),
            artifact,
            "Доставка",
            artifact.vectors[0],
            SearchFilters(corpus_lang="ru"),
            10,
        )
        rows = scored_chunks(artifact, dense, sparse, SearchOptions(mode="dense"))
        assert [r.chunk_id for r in rows] == [original[0].chunk_id]
        dense, sparse = candidate_lists(
            client,
            require_ready(settings, artifact),
            artifact,
            "Delivery",
            artifact.vectors[0],
            SearchFilters(corpus_lang="en", doc_ids=["missing"]),
            10,
        )
        assert dense == sparse == []
        current["version"] = "kb_demo_store-20261112-22222222"
        current["chunks"] = [original[0], chunk("A changed policy", lang="en", doc_id="changed")]
        replacement = build_artifact(settings, recipe, encoder=encoder, tokenizer=tokenizer())
        install_active(settings, str(current["version"]))
        retriever = Retriever(
            settings, artifact, client, encoder=AnyEncoder(encoder), tokenizer=tokenizer()
        )
        with pytest.raises(ValueError, match="active KB changed"):
            retriever.retrieve("Доставка", "ru")
        changed = sync_index(settings, replacement, client)
        assert (changed["deleted"], changed["upserted"], changed["unchanged_vectors"]) == (2, 1, 1)
        assert len(inventory(client, changed["collection"])) == 2
        current["version"], current["chunks"] = version, original
        install_active(settings, version)
        rollback = sync_index(settings, artifact, client)
        assert (rollback["deleted"], rollback["upserted"]) == (1, 2)
        assert all(
            p["kb_version"] == version for p in inventory(client, rollback["collection"]).values()
        )
        before = encoder.calls
        assert (
            build_artifact(settings, recipe, encoder=encoder, tokenizer=tokenizer()).manifest
            == artifact.manifest
        )
        assert encoder.calls == before
        (artifact.path / "vectors.npy").write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="checksum"):
            index.read_index(artifact.path)
    finally:
        client.close()


class AnyEncoder:
    """Duck-typed test encoder; no production model download in a pipeline test."""

    def __init__(self, encoder: FakeEncoder) -> None:
        self.encoder = encoder

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        return self.encoder.encode(*args, **kwargs)


def test_incomplete_index_and_expired_filter_return_no_context(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe = Recipe(encoder=spec())
    expired = chunk(effective_to=date(2026, 2, 1))
    version = "kb_demo_store-20261112-33333333"
    monkeypatch.setattr(index, "load_kb", lambda *a, **kw: (version, [expired], {}))
    artifact = build_artifact(settings, recipe, encoder=FakeEncoder(), tokenizer=tokenizer())
    install_active(settings, version)
    client = QdrantClient(":memory:")
    try:
        sync_index(settings, artifact, client)
        retriever = Retriever(settings, artifact, client, tokenizer=tokenizer())
        # Sparse mode should never load a neural model.
        cfg = settings.retrieval.model_copy(update={"mode": "sparse"})
        retriever.settings = settings.model_copy(update={"retrieval": cfg})
        context = retriever.context(
            "Доставка", filters=SearchFilters(corpus_lang="ru", as_of=date(2026, 3, 1))
        )
        assert context.abstain
        assert context.evidence == []
        historical = retriever.context(
            "Доставка",
            filters=SearchFilters(corpus_lang="ru", as_of=date(2026, 3, 1), all_versions=True),
        )
        assert not historical.abstain
        source_metadata = json.loads(historical.text.splitlines()[0])["source"]
        assert source_metadata["effective_to"] == "2026-02-01"
        state_path = index_root(settings, recipe) / "state.json"
        state = json.loads(state_path.read_text())
        state["ready"] = False
        write_json(state_path, state)
        with pytest.raises(ValueError, match="incomplete"):
            retriever.context("Доставка", "ru")
    finally:
        client.close()


def test_query_confidence_rejection_returns_explicit_empty_evidence(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    from collections.abc import Sequence

    from supportpilot.retrieval.encode import Transformer

    class LowConfidence(Transformer):
        def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
            return [0.2] * len(pairs)

    recipe = Recipe(encoder=spec())
    version = "kb_demo_store-20261112-55555555"
    monkeypatch.setattr(index, "load_kb", lambda *a, **kw: (version, [chunk()], {}))
    artifact = build_artifact(settings, recipe, encoder=FakeEncoder(), tokenizer=tokenizer())
    install_active(settings, version)
    cfg = settings.retrieval.model_copy(update={"mode": "sparse", "evidence_threshold": 0.5})
    configured = settings.model_copy(update={"retrieval": cfg})
    client = QdrantClient(":memory:")
    try:
        sync_index(configured, artifact, client)
        retriever = Retriever(
            configured, artifact, client, tokenizer=tokenizer(), reranker=LowConfidence(spec())
        )
        rejected = retriever.context("Доставка", "ru")
        assert rejected.abstain
        assert rejected.evidence == []
        assert rejected.reason == "low_evidence_confidence"
        retriever.settings = settings.model_copy(
            update={"retrieval": cfg.model_copy(update={"evidence_threshold": 0.1})}
        )
        assert not retriever.context("Доставка", "ru").abstain
    finally:
        client.close()


@pytest.mark.qdrant_server
def test_retrieval_on_real_qdrant_with_idf_and_document_filter(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"), timeout=5)
    try:
        client.get_collections()
    except Exception:
        client.close()
        pytest.skip("Qdrant server unreachable")
    recipe = Recipe(encoder=spec())
    sources = [chunk(), chunk("Оплата картой", doc_id="payment")]
    version = "kb_demo_store-20261112-44444444"
    monkeypatch.setattr(index, "load_kb", lambda *a, **kw: (version, sources, {}))
    artifact = build_artifact(settings, recipe, encoder=FakeEncoder(), tokenizer=tokenizer())
    name = index.collection_name(settings, recipe)
    try:
        sync_index(settings, artifact, client)
        dense, sparse = candidate_lists(
            client,
            name,
            artifact,
            "Доставка",
            None,
            SearchFilters(corpus_lang="ru", doc_types=["policy"]),
            10,
        )
        assert dense == []
        assert sparse[0][0] == artifact.units[0].unit_id
        assert sync_index(settings, artifact, client)["status"] == "skipped"
        pid = index.point_id(artifact.units[0].unit_id)
        client.set_payload(name, payload={"valid_to": 1}, points=[pid], wait=True)
        repaired = sync_index(settings, artifact, client)
        assert repaired["upserted"] == repaired["deleted"] == 0
        assert repaired["metadata_updates"] == 1
        assert inventory(client, name)[pid]["valid_to"] == 3652059
    finally:
        client.delete_collection(name)
        client.close()


@pytest.mark.qdrant_server
def test_kb_scoped_idf_is_stable_after_delete_and_reinsert(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    import numpy as np

    client = QdrantClient(url=os.environ.get("QDRANT_URL", "http://127.0.0.1:6333"), timeout=5)
    try:
        client.get_collections()
    except Exception:
        client.close()
        pytest.skip("Qdrant server unreachable")
    recipe = Recipe(encoder=spec())
    version = "kb_demo_store-20261112-66666666"
    sources = [
        chunk("Доставка товара в Алматы"),
        chunk("Доставка товара в Астану", doc_id="delivery_other"),
    ]
    current = {"version": version, "chunks": sources}
    monkeypatch.setattr(
        index, "load_kb", lambda *a, **kw: (current["version"], current["chunks"], {})
    )
    initial = build_artifact(settings, recipe, encoder=FakeEncoder(), tokenizer=tokenizer())
    name = index.collection_name(settings, recipe)
    try:
        sync_index(settings, initial, client)
        before = candidate_lists(
            client, name, initial, "доставка товара", None, SearchFilters(corpus_lang="ru"), 10
        )[1]
        current["version"] = "kb_demo_store-20261112-77777777"
        current["chunks"] = [chunk("Другой документ", doc_id="removed")]
        changed = build_artifact(settings, recipe, encoder=FakeEncoder(), tokenizer=tokenizer())
        sync_index(settings, changed, client)
        sync_index(settings, initial, client)
        after = candidate_lists(
            client, name, initial, "доставка товара", None, SearchFilters(corpus_lang="ru"), 10
        )[1]
        assert [uid for uid, _ in before] == [uid for uid, _ in after]
        assert np.allclose([score for _, score in before], [score for _, score in after])
    finally:
        client.delete_collection(name)
        client.close()


@pytest.mark.gpu
def test_pinned_retrieval_encoder_on_cuda() -> None:
    import torch

    from supportpilot.retrieval.encode import Transformer

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = Transformer(
        spec().model_copy(
            update={
                "repo_id": "intfloat/multilingual-e5-small",
                "revision": "614241f622f53c4eeff9890bdc4f31cfecc418b3",
                "dimensions": 384,
                "protocol": "e5",
                "batch_size": 2,
            }
        ),
        "cuda",
    )
    try:
        import numpy as np

        queries = model.encode(["Доставка в Алматы", "Алматыға жеткізу"], query=True)
        passages = model.encode(["Доставка в Алматы", "Алматыға жеткізу"])
        assert queries.shape == passages.shape == (2, 384)
        assert np.isfinite(queries).all()
        assert np.allclose(np.linalg.norm(queries, axis=1), 1, atol=1e-5)
        assert not np.allclose(queries, passages)
    finally:
        model.close()
