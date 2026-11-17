"""CLI entry points for the human corpus (`supportpilot data human|human-verify`, ADR-042)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_text
from supportpilot.common.manifest import load_source_registry
from supportpilot.common.provenance import code_commit
from supportpilot.data.embed import Embedder, EmbedderSpec
from supportpilot.data.human import BuildInputs, historical_sources, load_recipe
from supportpilot.data.humangates import build_human, human_root, verify_human
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.taxonomy import load_taxonomy


def _lid(texts: Sequence[str]) -> list[tuple[str, float, bool, tuple[str, ...]]]:
    return [
        (r.lang, r.conf, r.code_switch, r.langs) for r in LanguageIdentifier().detect_many(texts)
    ]


def run_human(settings: Settings, *, semantic: bool = True) -> dict[str, Any]:
    data_dir = settings.resolve(settings.paths.data_dir)
    tax_path = settings.resolve(settings.stage4.taxonomy)
    recipe_path = settings.resolve(settings.human.recipe)
    tax = load_taxonomy(tax_path)
    recipe = load_recipe(recipe_path, tax)
    registry = load_source_registry(settings.resolve(settings.data.registry))
    encode = Embedder(EmbedderSpec(**settings.stage4.embedder)).encode if semantic else None
    inp = BuildInputs(
        entries=registry.sources,
        data_dir=data_dir,
        recipe=recipe,
        tax=tax,
        lid=_lid,
        encode=encode,
        history_paths=historical_sources(data_dir),
    )
    previous = settings.human.previous_corpus
    return build_human(
        inp,
        code_commit=code_commit(settings.root),
        recipe_sha256=sha256_text(recipe_path.read_text(encoding="utf-8")),
        taxonomy_sha256=sha256_text(tax_path.read_text(encoding="utf-8")),
        previous=data_dir / "processed" / "sft" / previous if previous else None,
    )


def run_human_verify(settings: Settings, version: str | None) -> dict[str, Any]:
    version = version or settings.human.active_corpus
    if version is None:
        raise ValueError("no version given and human.active_corpus is not set")
    root = human_root(settings.resolve(settings.paths.data_dir))
    return verify_human(root / version)
