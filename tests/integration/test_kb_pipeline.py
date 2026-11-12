"""Stage 5 integration tests: render -> build -> gates -> versions, on the real demo-store
sources copied to tmp: idempotency, incremental update of a changed fact, removal and
addition of documents, failed gates, rollback, validation from a stored version."""

from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import yaml

from supportpilot.common.io import read_json, write_json
from supportpilot.data.steps import StepError
from supportpilot.kb.build import (
    KBConfig,
    activate,
    build,
    current_version,
    render,
    status,
    validate,
)
from supportpilot.kb.register import EXPECTED

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "kb_sources" / "demo_store"
GOLD = ROOT / "authored" / "gold"


def count(texts: Sequence[str]) -> list[int]:
    return [max(1, len(t.split())) for t in texts]


def make_cfg(tmp: Path, gold: Path | None = None) -> KBConfig:
    src = tmp / "src"
    if not src.exists():
        shutil.copytree(SOURCES, src)
    return KBConfig(name="demo_store", source_root=src, locale=ROOT / "configs" / "locale_kz.yaml",
                    kb_dir=tmp / "kb", max_tokens=400, lid_min_letters=40, tokenizer="words",
                    gold_dir=gold)  # fmt: skip


def chunks_of(cfg: KBConfig) -> list[dict[str, Any]]:
    cur = current_version(cfg)
    assert cur is not None
    rows: list[dict[str, Any]] = pq.read_table(cur / "chunks.parquet").to_pylist()
    return rows


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[KBConfig, dict[str, Any]]:
    cfg = make_cfg(tmp_path_factory.mktemp("kb"), GOLD if GOLD.is_dir() else None)
    render(cfg)
    return cfg, build(cfg, count, code_commit="test")


def test_full_build_passes_every_gate(built: tuple[KBConfig, dict[str, Any]]) -> None:
    cfg, report = built
    assert report["status"] == "built"
    assert report["documents"] == {"added": 42, "changed": 0, "unchanged": 0, "removed": 0}
    assert all(report["gates"].values()), report["gates"]
    cur = current_version(cfg)
    assert cur is not None
    q = read_json(cur / "quality_report.json")["gates"]
    assert q["facts"]["recall"] == 1.0
    assert q["facts"]["placements"] > 800
    assert q["completeness"]["min_word_recall"] == 1.0
    assert q["locators"]["checked"] == q["facts"]["placements"]
    assert q["coverage"]["failure_count"] == 0
    assert q["language"]["agreement"] == 1.0
    if GOLD.is_dir():
        assert q["evidence"]["references"] > 0
        assert q["evidence"]["passed"]
    emap = read_json(cur / "evidence_map.json")
    assert set(emap["returns_policy.window"]) == {"en", "ru", "kk"}
    assert (cur / "sources" / "kk" / "returns_policy.pdf").is_file()


def test_rerun_is_unchanged_and_forced_rebuild_identical(
    built: tuple[KBConfig, dict[str, Any]],
) -> None:
    cfg, report = built
    again = render(cfg)
    assert again["written"] == []
    assert again["removed"] == []
    assert build(cfg, count, code_commit="test")["status"] == "unchanged"
    forced = build(cfg, count, code_commit="test", force=True)
    assert forced["status"] == "unchanged"
    assert forced["reparsed"] == 42
    assert forced["kb_version"] == report["kb_version"]
    assert status(cfg)["sources_match_current"] is True
    assert validate(cfg)["passed"]


def test_retrieval_build_binds_real_ids_and_preserves_previous_version_on_failure(
    built: tuple[KBConfig, dict[str, Any]],
    tmp_path: Path,
) -> None:
    import polars as pl

    from supportpilot.common.config import PathsSettings, Settings
    from supportpilot.common.manifest import scan_files, write_model_json
    from supportpilot.common.schemas import DatasetManifest
    from supportpilot.data.retrievalset import build_retrieval_set

    cfg, _ = built
    data = tmp_path / "data"
    shutil.copytree(cfg.kb_dir, data / "kb/demo_store")
    authored = tmp_path / "authored/retrieval"
    shutil.copytree(ROOT / "authored/retrieval", authored)
    sft = data / "processed/sft/sft-20261112-00000000"
    sft.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        pl.DataFrame(
            [
                {
                    "user_text": "Synthetic unrelated event for pipeline testing",
                    "scenario_id": f"{split}_scenario",
                    "family_id": f"{split}_family",
                    "messages_json": '[{"role":"user","content":"Unrelated fixture event"}]',
                }
            ]
        ).write_parquet(sft / f"{split}.parquet")
    from datetime import UTC, datetime

    write_model_json(
        sft / "manifest.json",
        DatasetManifest(
            dataset_version="sft-20261112-00000000",
            data_schema_version="sft/1",
            code_commit="test",
            inputs=["test"],
            row_counts={"examples": 3},
            files=scan_files(sft),
            created_at=datetime.now(UTC),
        ),
    )
    settings = Settings(root=tmp_path, paths=PathsSettings(data_dir=data))
    first = build_retrieval_set(settings, code_commit="test", semantic=False)
    assert first["passed"]
    out = Path(first["path"])
    contents = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    second = build_retrieval_set(settings, code_commit="test", semantic=False)
    assert second["benchmark_version"] == first["benchmark_version"]
    assert contents == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    rows = [json.loads(line) for line in contents["queries.jsonl"].decode().splitlines()]
    assert sum(not r["cross_lingual"] and not r["code_switch"] for r in rows) >= 450
    assert all(r["relevant"] == [] for r in rows if not r["answerable"])
    assert all(r["review_status"] == "draft" for r in rows)
    pointer = (out.parent / "current.json").read_bytes()
    path = authored / "queries.yaml"
    document = yaml.safe_load(path.read_text("utf8"))
    document["items"][0]["relevant"][0]["section"] = "missing.section"
    path.write_text(yaml.safe_dump(document, allow_unicode=True), encoding="utf8")
    with pytest.raises(StepError, match="missing KB section"):
        build_retrieval_set(settings, code_commit="test", semantic=False)
    assert (out.parent / "current.json").read_bytes() == pointer
    assert contents == {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}


def test_changed_fact_updates_only_its_documents(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    render(cfg)
    first = build(cfg, count, code_commit="test")
    parent = current_version(cfg)
    assert parent is not None
    old_chunks = {r["chunk_id"]: r for r in chunks_of(cfg)}

    facts_path = cfg.source_root / "facts.yaml"
    facts = facts_path.read_text(encoding="utf-8")
    assert "return_window_days: 14\n" in facts
    facts_path.write_text(facts.replace("return_window_days: 14\n", "return_window_days: 30\n"),
                          encoding="utf-8")  # fmt: skip
    rendered = render(cfg)
    assert sorted(rendered["written"]) == ["en/returns_policy.pdf", "kk/returns_policy.pdf",
                                           "ru/returns_policy.pdf"]  # fmt: skip
    second = build(cfg, count, code_commit="test")
    assert second["status"] == "built"
    assert second["parent"] == first["kb_version"]
    assert second["documents"] == {"added": 0, "changed": 3, "unchanged": 39, "removed": 0}
    cur = current_version(cfg)
    assert cur is not None
    assert cur != parent
    changes = read_json(cur / "changes.json")
    assert {i.split(":")[0] for i in changes["chunks_delete"]} == {"returns_policy"}
    assert {i.split(":")[0] for i in changes["chunks_upsert"]} == {"returns_policy"}
    new = chunks_of(cfg)
    window = [r for r in new if r["doc_id"] == "returns_policy" and r["section_id"] == "window"]
    assert {r["lang"] for r in window} == {"en", "ru", "kk"}
    assert all("30" in str(r["text"]) and "14" not in str(r["text"]) for r in window)
    # unchanged documents keep their chunk ids; stale ids are gone from the current set
    assert all(r["chunk_id"] in old_chunks for r in new if r["doc_id"] != "returns_policy")
    assert not {r["chunk_id"] for r in new} & set(changes["chunks_delete"])
    # the parent version stays intact and valid; rollback switches back
    assert validate(cfg, parent.name)["passed"]
    activate(cfg, parent.name)
    assert current_version(cfg) == parent
    assert {r["chunk_id"] for r in chunks_of(cfg)} == set(old_chunks)
    activate(cfg, str(second["kb_version"]))
    # reverting the edit reproduces the first version exactly: it is reused, not rewritten
    facts_path.write_text(facts, encoding="utf-8")
    render(cfg)
    again = build(cfg, count, code_commit="test")
    assert again["status"] == "activated_existing"
    assert again["kb_version"] == first["kb_version"]
    assert read_json(parent / "changes.json")["parent"] is None  # untouched


def test_removed_and_added_documents(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    render(cfg)
    build(cfg, count, code_commit="test")
    terms = cfg.source_root / "docs" / "terms.yaml"
    spec = yaml.safe_load(terms.read_text(encoding="utf-8"))
    terms.unlink()
    r = render(cfg)
    assert sorted(r["removed"]) == ["en/terms.docx", "kk/terms.docx", "ru/terms.docx"]
    rep = build(cfg, count, code_commit="test")
    assert rep["documents"]["removed"] == 3
    assert rep["documents"]["unchanged"] == 39
    assert not [c for c in chunks_of(cfg) if c["doc_id"] == "terms"]

    spec["doc_id"] = "sale_terms"
    spec["format"] = "pdf"  # the same content as a new PDF document
    (cfg.source_root / "docs" / "sale_terms.yaml").write_text(
        yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8"
    )
    render(cfg)
    rep = build(cfg, count, code_commit="test")
    assert rep["documents"] == {"added": 3, "changed": 0, "unchanged": 39, "removed": 0}
    assert all(rep["gates"].values())
    assert {c["lang"] for c in chunks_of(cfg) if c["doc_id"] == "sale_terms"} == {"en", "ru", "kk"}


def test_failed_gate_keeps_the_current_version(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    render(cfg)
    first = build(cfg, count, code_commit="test")
    # the source is edited after rendering: refused before parsing
    pdf = cfg.sources_dir / "en" / "returns_policy.pdf"
    original = pdf.read_bytes()
    pdf.write_bytes(original + b"\n%junk")
    with pytest.raises(StepError, match="changed after rendering"):
        build(cfg, count, code_commit="test")
    pdf.write_bytes(original)
    # ground truth that the document does not contain -> the facts gate fails
    exp_path = cfg.sources_dir / EXPECTED
    exp = read_json(exp_path)
    placement = exp["documents"]["ru/delivery_policy.docx"]["placements"][0]
    placement["surface"] = "999 999 тенге"
    write_json(exp_path, exp)
    with pytest.raises(StepError, match="quality gates failed"):
        build(cfg, count, code_commit="test", force=True)
    failed = list(cfg.versions_dir.glob("*.failed"))
    assert len(failed) == 1
    report = json.loads((failed[0] / "quality_report.json").read_text(encoding="utf-8"))
    assert not report["gates"]["facts"]["passed"]
    assert report["gates"]["facts"]["failures"][0].startswith("ru/delivery_policy.docx")
    cur = current_version(cfg)
    assert cur is not None
    assert cur.name == first["kb_version"]
    with pytest.raises(StepError, match="no KB version"):
        activate(cfg, "kb_demo_store-20000101-00000000")


def test_damaged_version_cannot_be_activated(tmp_path: Path) -> None:
    cfg = make_cfg(tmp_path)
    render(cfg)
    rep = build(cfg, count, code_commit="test")
    cur = current_version(cfg)
    assert cur is not None
    (cur / "chunks.parquet").write_bytes(b"broken")
    with pytest.raises(StepError, match="damaged"):
        activate(cfg, str(rep["kb_version"]))


def test_cli_build_status_validate_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    from supportpilot import cli
    from supportpilot.kb import cli as kb_cli

    shutil.copytree(SOURCES, tmp_path / "src")
    cfg = {
        "env": "test",
        "root": str(tmp_path),  # never the real project root
        "stage4": {"locale": str(ROOT / "configs" / "locale_kz.yaml")},
        "kb": {"source_root": str(tmp_path / "src"), "gold_dir": None, "tokenizer": "words"},
    }
    path = tmp_path / "test.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setattr(kb_cli, "token_counter", lambda settings: count)
    monkeypatch.setattr(
        kb_cli, "_tokenizer_spec", lambda s: SimpleNamespace(repo_id="words", revision="1")
    )

    def run(*args: str) -> tuple[int, dict[str, Any]]:
        code = cli.main(["--config", str(path), "kb", *args])
        return code, json.loads(capsys.readouterr().out)

    code, out = run("build")
    assert code == 0
    assert out["build"]["status"] == "built"
    first = out["build"]["kb_version"]
    assert (tmp_path / "data" / "kb" / "demo_store" / "current.json").is_file()
    code, out = run("build")
    assert code == 0
    assert out["build"]["status"] == "unchanged"
    assert out["render"]["written"] == []
    code, out = run("status")
    assert out["current"] == first
    assert out["sources_match_current"] is True
    code, out = run("validate")
    assert code == 0
    assert out["passed"] is True
    code, out = run("activate", "kb_demo_store-20000101-00000000")
    assert code == 2
    assert "no KB version" in out["error"]
