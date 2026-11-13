from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from supportpilot.common.io import write_json
from supportpilot.common.manifest import scan_files
from supportpilot.retrieval.scale_data import ScaleNeed, family_phase, verify_freeze
from supportpilot.retrieval.scale_metrics import (
    evidence_metrics,
    measure_query,
    paired_scale,
    summarise_scale,
)
from supportpilot.retrieval.scale_run import resume_candidates, save_candidates, select_development
from supportpilot.retrieval.stress import workload


def need(**overrides: Any) -> ScaleNeed:
    return ScaleNeed.model_validate(
        {
            "id": "repair_and_return",
            "family_id": "combined_policy",
            "need_key": "diagnostic_price_with_return_transport",
            "category": "multi_fragment",
            "en": "Which charges apply?",
            "ru": "Какие расходы предусмотрены?",
            "kk": "Қандай шығындар қарастырылған?",
            "relevant": [
                {"section": "repair.diagnostics", "grade": 3},
                {"section": "returns.transport", "grade": 3},
            ],
            "required": [
                {"fact": "diagnostic_cost", "sections": ["repair.diagnostics"]},
                {"fact": "return_shipping_cost", "sections": ["returns.transport"]},
            ],
            "rationale": "Both independent charges are required to answer the combined question.",
            **overrides,
        }
    )


def test_sufficiency_is_and_across_groups_or_across_alternatives() -> None:
    groups = [["a", "alternative_a"], ["b"]]
    result = evidence_metrics(["alternative_a", "irrelevant", "b"], ["alternative_a"], groups)
    assert result["full_evidence5"] == 1
    assert result["packed_coverage"] == 0.5
    assert result["packed_sufficiency"] == 0
    assert evidence_metrics(["a", "b"], ["a", "b"], groups)["packed_sufficiency"] == 1
    with pytest.raises(ValueError, match="from the ranking"):
        evidence_metrics(["a"], ["b"], groups)


def test_fpr_and_fnr_use_packed_context_with_separate_denominators() -> None:
    negative = measure_query(
        ["background"], ["background"], {"background": 0}, [], answerable=False
    )
    assert negative == {"abstention_accuracy": 0, "unanswerable_fpr": 1}
    positive = measure_query(["a"], [], {"a": 3}, [["a"]], answerable=True)
    assert positive["recall5"] == 1
    assert positive["answerable_fnr"] == 1
    assert positive["packed_sufficiency"] == 0
    assert "unanswerable_fpr" not in positive
    with pytest.raises(ValueError, match="positive proof"):
        measure_query([], [], {"a": 3}, [], answerable=False)


def test_source_rejects_false_answerability_and_background_proof() -> None:
    assert len(need().required) == 2
    with pytest.raises(ValidationError, match="answerability"):
        need(required=[])
    with pytest.raises(ValidationError, match="proof sections"):
        need(required=[{"fact": "amount", "sections": ["unjudged.section"]}])
    with pytest.raises(ValidationError, match="answerability"):
        need(negative_reason="live_api")
    assert need(category="unanswerable", relevant=[], required=[], negative_reason="live_api")


def test_family_split_keeps_related_needs_together() -> None:
    protocol = {"seed": 42, "split": {"holdout_modulus": 5, "holdout_bucket": 0}}
    assignments = [family_phase("same_template", protocol) for _ in range(100)]
    assert len(set(assignments)) == 1
    assert {family_phase(f"family_{i}", protocol) for i in range(100)} == {"development", "holdout"}


def test_candidate_resume_binds_inputs_and_detects_partial_result_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "checkpoint.json"
    assert resume_candidates(path, "execution") == {}
    candidates = {"query": [[["unit", 0.4]], []]}
    save_candidates(path, "execution", candidates)
    assert resume_candidates(path, "execution") == candidates
    with pytest.raises(ValueError, match="another execution"):
        resume_candidates(path, "other execution")
    data = json.loads(path.read_text())
    data["candidates"]["query"][0][0][1] = 0.9
    write_json(path, data)
    with pytest.raises(ValueError, match="checkpoint changed"):
        resume_candidates(path, "execution")


def test_restored_vector_verifier_detects_server_vector_corruption(
    settings: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client import models as qm

    from supportpilot.retrieval import index
    from supportpilot.retrieval.models import Recipe
    from supportpilot.retrieval.scale_recovery import verify_semantic_vectors
    from tests.unit.test_retrieval import FakeEncoder, chunk, spec, tokenizer

    monkeypatch.setattr(
        index,
        "load_kb",
        lambda *a, **kw: (
            "kb_demo_store-20261112-11111111",
            [chunk()],
            {},
        ),
    )
    artifact = index.build_artifact(
        settings, Recipe(encoder=spec()), encoder=FakeEncoder(), tokenizer=tokenizer()
    )
    client = QdrantClient(":memory:")
    try:
        result = index.sync_index(settings, artifact, client)
        assert verify_semantic_vectors(client, result["collection"], artifact)["points"] == 1
        client.update_vectors(
            result["collection"],
            points=[
                qm.PointVectors(
                    id=index.point_id(artifact.units[0].unit_id),
                    vector={"dense": (-artifact.vectors[0]).tolist()},
                )
            ],
            wait=True,
        )
        with pytest.raises(ValueError, match="dense vector differs"):
            verify_semantic_vectors(client, result["collection"], artifact)
    finally:
        client.close()


def test_freeze_rejects_changed_qrels_and_unexpected_files(tmp_path: Path) -> None:
    (tmp_path / "qrels.tsv").write_text("query\t0\ta\t3\n")
    write_json(
        tmp_path / "freeze.json",
        {
            "files": [entry.model_dump() for entry in scan_files(tmp_path)],
        },
    )
    assert verify_freeze(tmp_path)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        verify_freeze(tmp_path)
    (tmp_path / "extra.json").unlink()
    (tmp_path / "qrels.tsv").write_text("query\t0\tb\t3\n")
    with pytest.raises(ValueError, match="changed"):
        verify_freeze(tmp_path)


def test_scale_summary_keeps_unknown_cohorts_and_family_sensitivity() -> None:
    rows = [
        {
            "query_id": "q",
            "scenario_id": "s",
            "family_id": "f",
            "phase": "development",
            "slices": ["kk"],
            "answerable": True,
            "recall5": 1.0,
            "mrr10": 1.0,
            "ndcg10": 1.0,
            "answerable_fnr": 0.0,
            "abstention_accuracy": 1.0,
            "packed_coverage": 0.5,
            "packed_sufficiency": 0.0,
            "full_evidence5": 1.0,
            "full_evidence10": 1.0,
        }
    ]
    summary = summarise_scale(rows, seed=42)
    assert summary["kk"]["metrics"]["unanswerable_fpr"]["mean"] is None
    assert summary["kk"]["family_sensitivity"]["recall5"]["mean"] == 1
    assert paired_scale(rows, rows, seed=42)["kk"]["metrics"]["recall5"]["mean"] == 0
    with pytest.raises(ValueError, match="populations"):
        paired_scale(rows, [{**rows[0], "family_id": "different"}], seed=42)


def test_load_corpus_is_reproducible_isolated_and_detects_corruption(tmp_path: Path) -> None:
    args = {"count": 75, "dimensions": 16, "batch": 8, "seed": 42}
    first = workload(tmp_path / "one", **args)
    second = workload(tmp_path / "two", **args)
    assert first["identity"] == second["identity"]
    assert first["sha256"] == second["sha256"]
    assert first["semantic_quality"] == "NOT_EVALUATED"
    vectors = np.load(Path(first["path"]) / "vectors.npy")
    assert vectors.shape == (75, 16)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    records = [
        json.loads(line) for line in (Path(first["path"]) / "chunks.jsonl").read_text().splitlines()
    ]
    assert len(records) == 75
    assert all(row["synthetic_only"] for row in records)
    assert workload(tmp_path / "one", **args) == first
    with (Path(first["path"]) / "vectors.npy").open("ab") as stream:
        stream.write(b"damage")
    with pytest.raises(ValueError, match="checksum"):
        workload(tmp_path / "one", **args)


def test_selection_refuses_holdout_and_retains_control_when_ci_is_inconclusive() -> None:
    scopes = ["all", "ru", "kk", "en", "cross_lingual", "code_switch"]
    row = {
        "query_id": "q",
        "scenario_id": "s",
        "family_id": "f",
        "phase": "development",
        "slices": scopes,
        "answerable": True,
        "recall5": 0.8,
        "mrr10": 1.0,
        "ndcg10": 1.0,
    }
    rows = {"control": [row], "candidate": [row]}
    evaluations = {name: summarise_scale(batch, seed=42) for name, batch in rows.items()}
    protocol = {
        "seed": 42,
        "selection": {
            "control": "control",
            "maximum_recall_loss": 0.005,
            "maximum_cross_or_mixed_recall_loss": 0.01,
        },
    }
    assert select_development(evaluations, rows, protocol)["selected"] == "control"
    rows["candidate"] = [{**row, "phase": "holdout"}]
    with pytest.raises(ValueError, match="holdout"):
        select_development(evaluations, rows, protocol)


def test_index_namespace_preserves_baseline_identity_and_state(settings: Any) -> None:
    from supportpilot.retrieval.index import collection_name, index_root
    from supportpilot.retrieval.models import Recipe
    from tests.unit.test_retrieval import spec

    recipe = Recipe(encoder=spec())
    baseline = collection_name(settings, recipe)
    original = index_root(settings, recipe)
    scoped = settings.model_copy(
        update={"retrieval": settings.retrieval.model_copy(update={"namespace": "scale_test"})}
    )
    assert collection_name(scoped, recipe) == baseline + "_scale_test"
    assert index_root(scoped, recipe) != original
    assert index_root(scoped, recipe).name == original.name == recipe.identity
    assert collection_name(settings, recipe) == baseline
    with pytest.raises(ValidationError):
        type(settings.retrieval)(namespace="../escape")


def test_tracking_receipt_sees_finalized_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from collections.abc import Iterator
    from contextlib import contextmanager
    from types import SimpleNamespace

    from supportpilot.retrieval.stress import final_receipt

    logged: list[dict[str, Any]] = []
    monkeypatch.setitem(
        sys.modules,
        "mlflow",
        SimpleNamespace(
            log_artifact=lambda path: logged.append(json.loads(Path(path).read_text()))
        ),
    )
    result: dict[str, Any] = {"status": "FINISHED", "resources": {"seconds": 0}}

    @contextmanager
    def measured() -> Iterator[None]:
        try:
            yield
        finally:
            result["resources"]["seconds"] = 42

    with final_receipt(tmp_path / "receipt.json", result), measured():
        assert not logged
    assert logged[0]["resources"]["seconds"] == 42
    assert json.loads((tmp_path / "receipt.json").read_text()) == logged[0]
