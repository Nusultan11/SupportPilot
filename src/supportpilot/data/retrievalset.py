"""Independent, explicitly judged retrieval queries bound to an immutable demo KB.

Sources are authored separately from SFT and gold. Section selectors resolve to real
chunk IDs, never keyword-generated relevance labels. A corpus language is explicit so
cross-language retrieval cannot accidentally receive same-language credit.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.manifest import read_model_json, verify_files
from supportpilot.common.schemas import DatasetManifest, KBChunk, LanguageCode
from supportpilot.data.langid import LanguageIdentifier
from supportpilot.data.sftbuild import current_sft
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks, semantic_leaks
from supportpilot.data.steps import StepError, commit, content_hash, partial_dir
from supportpilot.data.textnorm import EMAIL_RE, PHONE_RE, sim_key

LANGS: tuple[LanguageCode, ...] = ("en", "ru", "kk")
Category = Literal["simple", "hard", "ambiguous", "multi_fragment", "unanswerable"]
_DIGITS = re.compile(r"\d+")
_PLACEHOLDER = re.compile(r"\{[a-z_]|\{\{")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Relevance(_Model):
    section: str = Field(pattern=r"^[a-z0-9_]+\.[a-z0-9_-]+$")
    grade: int = Field(ge=0, le=3)


class QuerySource(_Model):
    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: Category
    en: str = Field(min_length=5)
    ru: str = Field(min_length=5)
    kk: str = Field(min_length=5)
    relevant: list[Relevance] = Field(default_factory=list)
    cross_lingual: bool = False
    mix_ru: str | None = None
    mix_kk: str | None = None
    rationale: str = Field(min_length=8)

    @model_validator(mode="after")
    def _consistent(self) -> QuerySource:
        sections = [r.section for r in self.relevant]
        if len(sections) != len(set(sections)):
            raise ValueError(f"{self.id}: duplicate relevance selector")
        positive = [r for r in self.relevant if r.grade > 0]
        if (self.category == "unanswerable") == bool(positive):
            raise ValueError(f"{self.id}: answerability and relevance disagree")
        if self.category == "multi_fragment" and len(positive) < 2:
            raise ValueError(f"{self.id}: multi-fragment needs multiple relevant sections")
        return self


class QueryFile(_Model):
    schema_version: Literal["retrieval_source/1"]
    source_id: Literal["supportpilot_authored_retrieval"]
    origin: Literal["llm_generated"]
    review_status: Literal["draft", "llm_reviewed"]
    author: str = Field(min_length=3)
    allowed_use: Literal["eval_only"]
    licence_note: str = Field(min_length=10)
    items: list[QuerySource] = Field(min_length=1)


class Judgement(_Model):
    chunk_id: str
    doc_id: str
    section_id: str
    grade: int = Field(ge=0, le=3)


class RetrievalQuery(_Model):
    schema_version: Literal["retrieval_query/1"] = "retrieval_query/1"
    query_id: str
    scenario_id: str
    query: str
    lang: LanguageCode
    corpus_lang: LanguageCode
    code_switch: bool
    cross_lingual: bool
    category: Category
    answerable: bool
    relevant: list[Judgement]
    rationale: str
    kb_version: str
    source_id: Literal["supportpilot_authored_retrieval"] = "supportpilot_authored_retrieval"
    origin: Literal["llm_generated"] = "llm_generated"
    review_status: Literal["draft", "llm_reviewed"]
    author: str
    quality_flags: list[str] = Field(default_factory=list)


def load_queries(directory: Path) -> tuple[list[QueryFile], list[str]]:
    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise StepError(f"no retrieval query sources in {directory}")
    files = [QueryFile.model_validate(yaml.safe_load(p.read_text("utf-8"))) for p in paths]
    ids = [q.id for f in files for q in f.items]
    if len(ids) != len(set(ids)):
        raise StepError("duplicate retrieval source IDs")
    return files, [f"{p.name}@{sha256_file(p)}" for p in paths]


def resolve_queries(
    files: list[QueryFile], chunks: list[KBChunk], kb_version: str
) -> list[RetrievalQuery]:
    by_section: dict[tuple[str, str], list[KBChunk]] = {}
    for c in chunks:
        key = (f"{c.doc_id}.{c.section_id}", c.lang)
        by_section.setdefault(key, []).append(c)
    out: list[RetrievalQuery] = []
    for f in files:
        for q in f.items:
            variants = [(lang, getattr(q, lang), False) for lang in LANGS]
            variants += [
                (lang, text, True) for lang in ("ru", "kk") if (text := getattr(q, f"mix_{lang}"))
            ]
            for lang, query, mixed in variants:
                targets = LANGS if q.cross_lingual and not mixed else (lang,)
                for target in targets:
                    judged: dict[str, Judgement] = {}
                    for r in q.relevant:
                        found = by_section.get((r.section, target), [])
                        if not found:
                            raise StepError(f"{q.id}: missing KB section {r.section} ({target})")
                        for c in found:
                            judged[c.chunk_id] = Judgement(
                                chunk_id=c.chunk_id,
                                doc_id=c.doc_id,
                                section_id=c.section_id or "",
                                grade=r.grade,
                            )
                    out.append(
                        RetrievalQuery(
                            query_id=f"rq_{q.id}_{lang}_{target}{'_mix' if mixed else ''}",
                            scenario_id="rsc_" + sha256_text(f"retrieval|{q.id}")[:16],
                            query=query,
                            lang=lang,
                            corpus_lang=target,
                            code_switch=mixed,
                            cross_lingual=lang != target,
                            category=q.category,
                            answerable=q.category != "unanswerable",
                            relevant=sorted(judged.values(), key=lambda r: r.chunk_id),
                            rationale=q.rationale,
                            kb_version=kb_version,
                            review_status=f.review_status,
                            author=f.author,
                        )
                    )
    return sorted(out, key=lambda q: q.query_id)


def query_checks(
    items: list[RetrievalQuery], chunks: list[KBChunk], *, minimum: int = 150
) -> dict[str, Any]:
    failures: list[str] = []
    ids = [q.query_id for q in items]
    if len(ids) != len(set(ids)):
        failures.append("duplicate query IDs")
    mono = [q for q in items if not q.cross_lingual and not q.code_switch]
    texts = [(q.lang, sim_key(q.query)) for q in mono]
    if len(texts) != len(set(texts)):
        failures.append("duplicate monolingual query text")
    per_lang = Counter(q.lang for q in mono)
    for lang in LANGS:
        if per_lang[lang] < minimum:
            failures.append(f"{lang}: {per_lang[lang]} monolingual queries < {minimum}")
        categories = {q.category for q in mono if q.lang == lang}
        if categories != {"simple", "hard", "ambiguous", "multi_fragment", "unanswerable"}:
            failures.append(f"{lang}: incomplete category coverage")
    if not any(q.cross_lingual for q in items) or not any(q.code_switch for q in items):
        failures.append("missing cross-lingual or code-switched queries")
    existing = {c.chunk_id: c for c in chunks}
    lid = LanguageIdentifier()
    for q in items:
        if _PLACEHOLDER.search(q.query) or EMAIL_RE.search(q.query) or PHONE_RE.search(q.query):
            failures.append(f"{q.query_id}: placeholder or PII-like text")
        detected = lid.detect(q.query)
        if not q.code_switch and detected.lang not in (q.lang, "und"):
            failures.append(f"{q.query_id}: LID={detected.lang}, expected {q.lang}")
        for r in q.relevant:
            c = existing.get(r.chunk_id)
            if c is None or (c.doc_id, c.section_id, c.lang) != (
                r.doc_id,
                r.section_id,
                q.corpus_lang,
            ):
                failures.append(f"{q.query_id}: dangling or wrong-language judgement")
    return {
        "passed": not failures,
        "failures": failures,
        "monolingual_per_lang": dict(per_lang),
        "items": len(items),
    }


def leakage_checks(
    items: list[RetrievalQuery], sft: Path, settings: Settings, *, semantic: bool
) -> dict[str, Any]:
    """Check all SFT splits, including history; gold is deliberately not an input."""
    cfg = settings.stage4
    df = pl.concat([pl.read_parquet(sft / f"{s}.parquet") for s in ("train", "validation", "test")])
    texts = set(df["user_text"].to_list())
    for raw in df["messages_json"].to_list():
        texts.update(m["content"] for m in json.loads(raw) if m["role"] == "user")
    training = sorted(texts)
    queries = sorted({q.query for q in items})
    leaks: dict[str, list[str]] = {
        "exact": [queries[i] for i in sorted(exact_leaks(training, queries))],
        "lexical": [
            queries[i]
            for i in sorted(
                lexical_leaks(training, queries, threshold=cfg.leak_jaccard, seed=cfg.seed)
            )
        ],
        "ngram": [queries[i] for i in sorted(ngram_leaks(training, queries, cfg.leak_ngram))],
    }
    near: dict[int, float] = {}
    if semantic:
        from supportpilot.data.embed import Embedder, EmbedderSpec

        emb = Embedder(EmbedderSpec.model_validate(cfg.embedder))
        tv = emb.encode([_DIGITS.sub("0", t) for t in training])
        qv = emb.encode([_DIGITS.sub("0", q) for q in queries])
        near = semantic_leaks(tv, qv, cfg.leak_cosine)
    # Shared intent semantics is diagnostic, as for the separately authored gold set.
    # Exact / lexical / ngram copies always block. No evaluation query is auto-deleted.
    ids = set(df["scenario_id"].to_list()) | set(df["family_id"].to_list())
    id_overlap = sorted({q.scenario_id for q in items} & ids)
    return {
        "passed": not any(leaks.values()) and not id_overlap,
        "overlap": {k: len(v) for k, v in leaks.items()},
        "examples": leaks,
        "scenario_overlap": id_overlap,
        "sft_dataset": sft.name,
        "semantic_checked": semantic,
        "semantic_threshold": cfg.leak_cosine,
        "semantic_near_sft": [
            {"query": queries[i], "cosine": round(v, 6)} for i, v in sorted(near.items())
        ],
        "gold_used_as_source": False,
    }


def build_retrieval_set(
    settings: Settings, *, code_commit: str | None, semantic: bool = True
) -> dict[str, Any]:
    data = settings.resolve(settings.paths.data_dir)
    kb_root = data / "kb" / settings.kb.name
    kb_version = json.loads((kb_root / "current.json").read_text("utf-8"))["kb_version"]
    kb = kb_root / "versions" / kb_version
    km = read_model_json(kb / "manifest.json", DatasetManifest)
    if verify_files(kb, km.files, exclude=frozenset({"manifest.json"})):
        raise StepError("active KB manifest verification failed")
    chunks = [
        KBChunk.model_validate_json(r) for r in pl.read_parquet(kb / "chunks.parquet")["chunk_json"]
    ]
    sft = current_sft(data)
    if sft is None:
        raise StepError("build SFT before the retrieval set")
    sm = read_model_json(sft / "manifest.json", DatasetManifest)
    if verify_files(sft, sm.files, exclude=frozenset({"manifest.json"})):
        raise StepError("SFT manifest verification failed")
    files, source_hashes = load_queries(
        settings.resolve(settings.stage4.authored_dir) / "retrieval"
    )
    items = resolve_queries(files, chunks, kb_version)
    checked = query_checks(items, chunks)
    leaks = leakage_checks(items, sft, settings, semantic=semantic)
    near = {r["query"] for r in leaks["semantic_near_sft"]}
    items = [
        q.model_copy(update={"quality_flags": ["semantic_near_sft"]}) if q.query in near else q
        for q in items
    ]
    rows = [q.model_dump(mode="json") for q in items]
    report = {
        "passed": checked["passed"] and leaks["passed"],
        "integrity": checked,
        "leakage": leaks,
        "kb_version": kb_version,
        "categories": dict(Counter(q.category for q in items)),
        "cross_lingual": sum(q.cross_lingual for q in items),
        "code_switch": sum(q.code_switch for q in items),
        "independent_scenarios": len({q.scenario_id for q in items}),
        "review_status": dict(Counter(q.review_status for q in items)),
    }
    out = data / "benchmarks" / "retrieval" / "current"
    inputs = [f"kb@{kb_version}", f"sft@{sm.dataset_version}", *source_hashes]
    params = {"builder": 1, "semantic": semantic, "minimum_per_language": 150}
    with partial_dir(out) as partial:
        (partial / "queries.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), "utf-8"
        )
        (partial / "qrels.tsv").write_text(
            "".join(
                f"{q.query_id}\t0\t{r.chunk_id}\t{r.grade}\n" for q in items for r in q.relevant
            ),
            "utf-8",
        )
        write_json(partial / "quality_report.json", report)
        write_json(
            partial / "stats.json",
            {
                "queries": len(items),
                "monolingual_per_lang": checked["monolingual_per_lang"],
                "per_lang_category": dict(Counter(f"{q.lang}|{q.category}" for q in items)),
                "answerable": sum(q.answerable for q in items),
                "unanswerable": sum(not q.answerable for q in items),
                "judgements_by_grade": dict(
                    Counter(str(r.grade) for q in items for r in q.relevant)
                ),
                "evaluation": "Recall@5/MRR@10/nDCG@10: answerable only, by query/corpus language; "
                "report unanswerable abstention separately; bootstrap by scenario_id",
            },
        )
        m = commit(
            partial,
            out,
            name="retrieval",
            data_schema_version="retrieval_query/1",
            code_commit=code_commit,
            inputs=inputs,
            params=params,
            row_counts={"queries": len(items), **checked["monolingual_per_lang"]},
            content_sha256=sha256_text(
                content_hash(rows)
                + json.dumps({"inputs": inputs, "params": params}, sort_keys=True)
            ),
            quality_report="quality_report.json",
            gate=lambda _: bool(report["passed"]),
        )
    final = out.parent / m.dataset_version
    if final.exists():
        previous = read_model_json(final / "manifest.json", DatasetManifest)
        if verify_files(final, previous.files, exclude=frozenset({"manifest.json"})):
            raise StepError(f"stored retrieval version is corrupt: {final}")
        if (final / "queries.jsonl").read_bytes() != (out / "queries.jsonl").read_bytes():
            raise StepError("retrieval version collision")
        # Keep the first immutable snapshot, including its input provenance.
        import shutil

        shutil.rmtree(out)
    else:
        out.rename(final)
    write_json(
        final.parent / "current.json",
        {"benchmark_version": m.dataset_version, "kb_version": kb_version},
    )
    return {"path": str(final), "benchmark_version": m.dataset_version, **report}
