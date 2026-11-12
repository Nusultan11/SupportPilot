"""Training KB variant (Stage 4, ADR-025; DATA_CONTRACTS §5).

Fictional stores (authored/kb_train/variants.yaml) x documents (authored/kb_train/docs.yaml)
x languages -> chunks with the DATA_CONTRACTS §1 chunk ids. Each variant also renders an
outdated version of its documents (earlier values, `effective_to` in the past), used by the
`conflicting_context` tasks. Policy values differ between variants and from the demo store
(`kb_variant_distinct` check), so the SFT model must read the context.

Output `data/processed/kb_train/chunks.parquet` (+ manifest with the KB version).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import yaml
from pydantic import BaseModel, ConfigDict, Field

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_chunk_id
from supportpilot.common.schemas import LANGUAGES, LanguageCode
from supportpilot.data.locale import LocaleCatalog
from supportpilot.data.steps import (
    StepError,
    commit,
    content_hash,
    counts,
    is_current,
    partial_dir,
    read_manifest,
    write_parquet,
)
from supportpilot.data.textfmt import render

KB_VERSION = 1
CHUNKS = "chunks.parquet"
OUTDATED_FROM = "2026-01-01"
WARRANTY_CATEGORIES = frozenset({"electronics", "appliances"})
CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("version", pa.string()),
        ("variant", pa.string()),
        ("status", pa.string()),
        ("lang", pa.string()),
        ("position", pa.int32()),
        ("section_id", pa.string()),
        ("section_path", pa.list_(pa.string())),
        ("doc_type", pa.string()),
        ("text", pa.string()),
        ("facts", pa.list_(pa.string())),
        ("fact_values_json", pa.string()),
        ("sku", pa.string()),
        ("effective_from", pa.string()),
        ("effective_to", pa.string()),
    ]
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Section(_Model):
    id: str
    facts: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    en: str
    ru: str
    kk: str

    def template(self, lang: str) -> str:
        return str(getattr(self, lang))


class Doc(_Model):
    doc_type: Literal["policy", "faq", "catalog", "terms"]
    title: dict[str, str]
    per_product: bool = False
    sections: list[Section]


class Docs(_Model):
    schema_version: Literal["kb_docs/1"] = "kb_docs/1"
    docs: dict[str, Doc]


class Variants(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["kb_variants/1"] = "kb_variants/1"
    distinct_facts: list[str]
    variants: list[dict[str, Any]] = Field(min_length=2)


def load_docs(path: Path) -> Docs:
    return Docs.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_variants(path: Path) -> Variants:
    v = Variants.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    ids = [x["id"] for x in v.variants]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate variant ids")
    return v


def _requires_ok(req: Sequence[str], values: Mapping[str, Any]) -> bool:
    for cond in req:
        kind, _, expr = cond.partition(":")
        if kind != "has":
            raise ValueError(f"unknown requirement {cond!r}")
        key, _, member = expr.partition("=")
        if member not in (values.get(key) or []):
            return False
    return True


def product_price(variant_id: str, sku: str, lo: int, hi: int) -> int:
    steps = (hi - lo) // 100 + 1
    return lo + (int(sha256_text(f"{variant_id}|{sku}")[:8], 16) % steps) * 100


def _norm(value: Any) -> Any:
    if isinstance(value, list):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in value.items()}
    return value


def render_variant(
    variant: Mapping[str, Any],
    docs: Docs,
    locale: LocaleCatalog,
    lang: LanguageCode,
    *,
    status: Literal["current", "outdated"] = "current",
) -> list[dict[str, Any]]:
    values = dict(variant)
    outdated = values.pop("outdated", {}) or {}
    eff_from, eff_to = str(values["effective_from"]), None
    if status == "outdated":
        values.update({k: v for k, v in outdated.items() if k != "effective_to"})
        eff_from, eff_to = OUTDATED_FROM, str(outdated["effective_to"])
    chunks: list[dict[str, Any]] = []
    for doc_id, doc in docs.docs.items():
        if status == "outdated" and doc.per_product:
            continue
        rendered: list[tuple[Section, str, dict[str, Any], str | None]] = []
        if doc.per_product:
            for p in locale.products:
                vals = {
                    **values,
                    "product": p.get(lang),
                    "sku": p.sku,
                    "price": product_price(str(values["id"]), p.sku, *p.price),
                }
                sec = doc.sections[1] if p.cat in WARRANTY_CATEGORIES else doc.sections[0]
                text = render(sec.template(lang), vals, lang, catalog=locale.lookup)
                text = text[:1].upper() + text[1:]
                facts = {k: vals[k] for k in sec.facts} | {"sku": p.sku}
                rendered.append((sec, text, facts, p.sku))
        else:
            for sec in doc.sections:
                if not _requires_ok(sec.requires, values):
                    continue
                text = render(sec.template(lang), values, lang, catalog=locale.lookup)
                rendered.append((sec, text, {k: values[k] for k in sec.facts}, None))
        body = "\n".join(t for _, t, _, _ in rendered)
        version = sha256_text(f"{values['id']}|{status}|{lang}|{doc_id}|{body}")
        for pos, (sec, text, facts, sku) in enumerate(rendered):
            chunks.append(
                {
                    "chunk_id": make_chunk_id(doc_id, version, pos),
                    "doc_id": doc_id,
                    "version": version,
                    "variant": str(values["id"]),
                    "status": status,
                    "lang": lang,
                    "position": pos,
                    "section_id": sec.id,
                    "section_path": [doc.title[lang], sec.id],
                    "doc_type": doc.doc_type,
                    "text": text,
                    "facts": sorted(facts),
                    "fact_values_json": json.dumps(
                        _norm(facts), ensure_ascii=False, sort_keys=True
                    ),
                    "sku": sku,
                    "effective_from": eff_from,
                    "effective_to": eff_to,
                }
            )
    return chunks


def render_all(variants: Variants, docs: Docs, locale: LocaleCatalog) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for v in variants.variants:
        for lang in LANGUAGES:
            out.extend(render_variant(v, docs, locale, lang))
            if v.get("outdated"):
                out.extend(render_variant(v, docs, locale, lang, status="outdated"))
    return out


def distinct_violations(variants: Variants, demo: Mapping[str, Any]) -> list[str]:
    """Facts where a training variant (current or outdated) reuses the demo-store value."""
    bad: list[str] = []
    for v in variants.variants:
        for key in variants.distinct_facts:
            for label, source in (("current", v), ("outdated", v.get("outdated") or {})):
                if key in source and key in demo and _norm(source[key]) == _norm(demo[key]):
                    bad.append(f"{v['id']}:{label}:{key}")
    return bad


def kb_dir(data_dir: Path) -> Path:
    return data_dir / "processed" / "kb_train"


def build_kb_train(
    authored: Path,
    demo_facts: Path,
    locale: LocaleCatalog,
    locale_path: Path,
    data_dir: Path,
    *,
    code_commit: str | None,
    force: bool = False,
) -> dict[str, Any]:
    vpath, dpath = authored / "kb_train" / "variants.yaml", authored / "kb_train" / "docs.yaml"
    params = {
        "kb_version": KB_VERSION,
        "variants_sha256": sha256_text(vpath.read_text(encoding="utf-8")),
        "docs_sha256": sha256_text(dpath.read_text(encoding="utf-8")),
        "locale_sha256": sha256_text(locale_path.read_text(encoding="utf-8")),
        "demo_facts_sha256": sha256_text(demo_facts.read_text(encoding="utf-8")),
    }
    out = kb_dir(data_dir)
    inputs = ["authored/kb_train", "configs/locale_kz.yaml", "kb_sources/demo_store/facts.yaml"]
    if not force and is_current(out, inputs, params):
        m = read_manifest(out)
        return {"status": "skipped", "kb_version": m.dataset_version, "rows": m.row_counts}
    variants, docs = load_variants(vpath), load_docs(dpath)
    demo = yaml.safe_load(demo_facts.read_text(encoding="utf-8"))
    if bad := distinct_violations(variants, demo):
        raise StepError(f"kb_variant_distinct: training KB reuses demo values: {bad}")
    chunks = render_all(variants, docs, locale)
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise StepError("duplicate chunk ids in the training KB")
    if any("{" in c["text"] or "}" in c["text"] for c in chunks):
        raise StepError("unresolved template braces in a training KB chunk")
    chunks.sort(key=lambda c: (c["variant"], c["status"], c["lang"], c["doc_id"], c["position"]))
    with partial_dir(out) as partial:
        write_parquet(partial / CHUNKS, chunks, CHUNK_SCHEMA)
        m = commit(
            partial,
            out,
            name="kbtrain",
            data_schema_version="kb_chunk_train/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts={"chunks": len(chunks), **counts(chunks, "lang", "status")},
            content_sha256=content_hash(chunks),
        )
    return {"status": "built", "kb_version": m.dataset_version, "rows": m.row_counts}
