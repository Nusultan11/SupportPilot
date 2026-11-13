from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from supportpilot.common.config import Settings
from supportpilot.common.io import write_json
from supportpilot.common.schemas import LANGUAGES
from supportpilot.data.retrievalset import RetrievalQuery
from supportpilot.kb.spec import Tri
from supportpilot.retrieval.scale_diagnostics import (
    DiagnosticSource,
    diagnostic_queries,
    entity_typo,
)


def test_entity_typo_is_one_internal_unicode_deletion() -> None:
    name = "Ықшам пернетақта"
    changed = entity_typo(name)
    assert len(changed) == len(name) - 1
    assert changed.startswith("Ықшам ")
    assert entity_typo(name) == changed
    with pytest.raises(ValueError, match="long enough"):
        entity_typo("USB")


def test_diagnostic_preserves_needs_and_excludes_holdout(tmp_path: Path) -> None:
    source_dir = tmp_path / "kb_sources/demo_store"
    source_dir.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / "kb_sources/demo_store/technical_profiles.yaml",
        source_dir / "technical_profiles.yaml",
    )
    parent = tmp_path / "parent"
    parent.mkdir()
    metadata, queries = {}, []
    for index, slug in enumerate(
        (
            "compact_keyboard",
            "ergonomic_keyboard",
            "vertical_mouse",
            "trackball",
            "desk_usb_charger",
        )
    ):
        for lang in LANGUAGES:
            q = RetrievalQuery.model_validate(
                {
                    "query_id": f"fixture_{slug}_{lang}",
                    "scenario_id": slug,
                    "query": "Fixture?",
                    "lang": lang,
                    "corpus_lang": lang,
                    "cross_lingual": False,
                    "code_switch": False,
                    "category": "simple",
                    "answerable": True,
                    "relevant": [
                        {
                            "chunk_id": slug + lang,
                            "doc_id": "technical_" + slug,
                            "section_id": "mass_g",
                            "grade": 3,
                        }
                    ],
                    "rationale": "Fixture, not relevance/linguistic evidence",
                    "kb_version": "fixture",
                    "review_status": "draft",
                    "author": "fixture",
                }
            )
            queries.append(q)
            metadata[q.query_id] = {
                "phase": "development" if index < 4 else "holdout",
                "family_id": "fixture",
                "need_key": f"technical.{slug}.mass_g|current",
                "required_groups": [[slug + lang]],
                "as_of": "2026-11-12",
                "all_versions": False,
                "tags": [],
            }
    (parent / "queries.jsonl").write_text("".join(q.model_dump_json() + "\n" for q in queries))
    write_json(parent / "metadata.json", metadata)
    source = DiagnosticSource(
        schema_version="retrieval_diagnostic_source/1",
        origin="llm_generated",
        review_status="draft",
        author="fixture",
        allowed_use="eval_only",
        licence_note="fixture",
        scope="development only",
        templates={
            "mass_g": Tri(
                en="How heavy is {product}?",
                ru="Сколько весит {product}?",
                kk="{product} неше грамм тартады?",
            )
        },
    )
    result, meta = diagnostic_queries(Settings(root=tmp_path), parent, source, 42)
    assert len(result) == 31
    assert len({q.scenario_id for q in result}) == 4
    assert not any(q.scenario_id == "desk_usb_charger" for q in result)
    assert all(m["parent_phase"] == "development" for m in meta.values())
    assert diagnostic_queries(Settings(root=tmp_path), parent, source, 42) == (result, meta)
