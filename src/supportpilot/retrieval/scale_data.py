"""Audited, family-disjoint Stage 6.1 inputs with a tamper-evident freeze."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.manifest import read_model_json, scan_files, verify_files
from supportpilot.common.schemas import DatasetManifest, KBChunk
from supportpilot.data.retrievalset import (
    QueryFile,
    QuerySource,
    leakage_checks,
    query_checks,
    resolve_queries,
)
from supportpilot.data.sftbuild import current_sft
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.steps import partial_dir
from supportpilot.data.textnorm import sim_key
from supportpilot.kb.register import Expected
from supportpilot.retrieval.corpus import load_kb
from supportpilot.retrieval.index import writer_lock


class EvidenceGroup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    fact: str
    sections: list[str] = Field(min_length=1)


class ScaleNeed(QuerySource):
    """Related wording shares a family even if fact keys or entity IDs differ."""

    family_id: str = Field(pattern=r"^[a-z0-9_]+$")
    need_key: str = Field(min_length=5)
    required: list[EvidenceGroup] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    as_of: date | None = None
    all_versions: bool = False
    negative_reason: (
        Literal[
            "live_api",
            "missing_fact",
            "missing_exception",
            "partial_evidence",
            "expired",
            "contradictory",
            "clarification",
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def valid_proof(self) -> Self:
        negative = self.category == "unanswerable"
        if negative != (self.negative_reason is not None) or negative == bool(self.required):
            raise ValueError("answerability requires proof, negatives require a reason")
        positive = {r.section for r in self.relevant if r.grade >= 2}
        if any(not set(group.sections) <= positive for group in self.required):
            raise ValueError("proof sections must have direct/partial relevance")
        if len({group.fact for group in self.required}) != len(self.required):
            raise ValueError("duplicate required fact")
        return self


class ScaleSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["retrieval_scale_source/1"]
    origin: Literal["llm_generated"]
    review_status: Literal["draft", "llm_reviewed"]
    author: str
    allowed_use: Literal["eval_only"]
    licence_note: str
    items: list[ScaleNeed] = Field(min_length=1)


def load_protocol(settings: Settings) -> dict[str, Any]:
    protocol = yaml.safe_load((settings.root / "configs/retrieval_scale.yaml").read_text("utf-8"))
    if protocol.get("schema_version") != "retrieval_scale_protocol/1":
        raise ValueError("unsupported scale protocol")
    return dict(protocol)


def load_needs(directory: Path) -> tuple[list[QueryFile], dict[str, ScaleNeed]]:
    files, needs = [], {}
    for path in sorted(directory.glob("*.yaml")):
        source = ScaleSource.model_validate(yaml.safe_load(path.read_text("utf-8")))
        for need in source.items:
            if need.id in needs:
                raise ValueError("duplicate scale need ID")
            needs[need.id] = need
        files.append(
            QueryFile(
                schema_version="retrieval_source/1",
                source_id="supportpilot_authored_retrieval",
                origin=source.origin,
                review_status=source.review_status,
                author=source.author,
                allowed_use=source.allowed_use,
                licence_note=source.licence_note,
                items=[
                    QuerySource.model_validate(
                        {
                            k: v
                            for k, v in need.model_dump().items()
                            if k in QuerySource.model_fields
                        }
                    )
                    for need in source.items
                ],
            )
        )
    if len({need.need_key for need in needs.values()}) != len(needs):
        raise ValueError("duplicate semantic need key; paraphrases must share a scenario")
    return files, needs


def family_phase(family: str, protocol: dict[str, Any]) -> str:
    split = protocol["split"]
    bucket = int(sha256_text(f"{protocol['seed']}|{family}")[:8], 16) % split["holdout_modulus"]
    return "holdout" if bucket == split["holdout_bucket"] else "development"


def resolve_proof(
    need: ScaleNeed,
    lang: str,
    chunks: list[KBChunk],
    expected: Expected,
    as_of: date | None = None,
) -> list[list[str]]:
    """Resolve a fact to chunks actually containing its recorded surface, never keywords."""
    result = []
    by_id = {chunk.chunk_id: chunk for chunk in chunks if chunk.lang == lang}
    for group in need.required:
        if group.fact not in expected.facts:
            raise ValueError(f"{need.id}: unknown canonical fact {group.fact}")
        matching: set[str] = set()
        for chunk in by_id.values():
            if as_of is not None and (
                chunk.effective_from > as_of
                or (chunk.effective_to is not None and chunk.effective_to < as_of)
            ):
                continue
            if f"{chunk.doc_id}.{chunk.section_id}" not in group.sections:
                continue
            path = chunk.source_path.removeprefix("sources/")
            document = expected.documents[path]
            for placement in document.placements:
                if (
                    placement.key == group.fact
                    and placement.section_id == chunk.section_id
                    and sim_key(placement.surface) in sim_key(chunk.text)
                ):
                    if placement.value != expected.facts[group.fact]:
                        raise ValueError("canonical fact value disagrees with placement")
                    matching.add(chunk.chunk_id)
        if not matching:
            raise ValueError(f"{need.id}: required fact has no supporting chunk ({lang})")
        result.append(sorted(matching))
    return result


def corpus_audit(chunks: list[KBChunk], expected: Expected) -> dict[str, Any]:
    identifiers = {"sku", "id", "name", "effective_from", "support_phone", "support_email"}
    facts = {
        key: value
        for key, value in expected.facts.items()
        if key.rsplit(".", 1)[-1] not in identifiers
    }
    # Same fact key may have many translated placements; it is still one fact.
    predicates = Counter(key.rsplit(".", 1)[-1] for key in facts)
    missing, conflicts = [], []
    for key, value in facts.items():
        found = set()
        for path, doc in expected.documents.items():
            for placement in doc.placements:
                if placement.key != key:
                    continue
                if placement.value != value:
                    conflicts.append(f"{path}:{key}")
                found.add(path.split("/")[0])
        if found != {"en", "ru", "kk"}:
            missing.append(key)
    normalised = Counter(sim_key(chunk.text) for chunk in chunks)
    return {
        "documents": len({chunk.doc_id for chunk in chunks}),
        "language_versions": len({(chunk.doc_id, chunk.lang) for chunk in chunks}),
        "chunks": len(chunks),
        "canonical_facts": len(facts),
        "registry_entries_including_identifiers": len(expected.facts),
        "atomic_profile_attributes": sum(
            key.startswith("technical.")
            and key.rsplit(".", 1)[-1] != "price"
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            for key, value in expected.facts.items()
        ),
        "distinct_predicates": len(predicates),
        "predicate_counts": dict(predicates),
        "per_language": dict(Counter(chunk.lang for chunk in chunks)),
        "exact_duplicate_chunk_bodies": sum(n - 1 for n in normalised.values() if n > 1),
        "missing_fact_languages": missing,
        "conflicting_placements": conflicts,
        "semantic_diversity": "NOT_ESTABLISHED_BY_COUNTS",
        "human_review": "NOT_PERFORMED",
    }


def prepare_scale(
    settings: Settings,
    *,
    freeze: bool = False,
    semantic: bool = False,
    kb_version: str | None = None,
) -> dict[str, Any]:
    """Audit first; volume/quality/leakage failures never publish a frozen benchmark."""
    protocol = load_protocol(settings)
    kb_version, chunks, _ = load_kb(settings, kb_version)
    data = settings.resolve(settings.paths.data_dir)
    kb = data / "kb" / settings.kb.name / "versions" / kb_version
    expected = read_model_json(kb / "sources/expected.json", Expected)
    sources = settings.resolve(Path(protocol["source"]))
    files, needs = load_needs(sources)
    queries = resolve_queries(files, chunks, kb_version)
    lookup = {"rsc_" + sha256_text(f"retrieval|{key}")[:16]: need for key, need in needs.items()}
    metadata = {}
    proof_cache: dict[tuple[str, str], list[list[str]]] = {}
    for query in queries:
        need = lookup[query.scenario_id]
        proof_key = (need.id, query.corpus_lang)
        if proof_key not in proof_cache:
            proof_cache[proof_key] = resolve_proof(
                need,
                query.corpus_lang,
                chunks,
                expected,
                None
                if need.all_versions
                else (need.as_of or date.fromisoformat(protocol["evaluation_date"])),
            )
        metadata[query.query_id] = {
            "family_id": need.family_id,
            "need_key": need.need_key,
            "phase": family_phase(need.family_id, protocol),
            "as_of": (need.as_of or date.fromisoformat(protocol["evaluation_date"])).isoformat(),
            "all_versions": need.all_versions,
            "tags": need.tags,
            "negative_reason": need.negative_reason,
            "required_groups": proof_cache[proof_key],
        }
    corpus = corpus_audit(chunks, expected)
    counts = {
        **{
            key: corpus[key]
            for key in ("documents", "language_versions", "chunks", "canonical_facts")
        },
        "needs": len(needs),
        "cross_lingual_variants": sum(q.cross_lingual for q in queries),
        "code_switch_needs": len({q.scenario_id for q in queries if q.code_switch}),
        "unanswerable_needs": sum(n.category == "unanswerable" for n in needs.values()),
    }
    failures = [
        f"{key}: {counts[key]} < {minimum}"
        for key, minimum in protocol["minimum"].items()
        if counts[key] < minimum
    ]
    integrity = (
        query_checks(queries, chunks) if queries else {"passed": False, "failures": ["no needs"]}
    )
    if not integrity["passed"]:
        failures.append("query integrity/coverage failed")
    if corpus["missing_fact_languages"] or corpus["conflicting_placements"]:
        failures.append("canonical fact coverage/consistency failed")
    review_path = settings.resolve(Path(protocol["review"]["diversity_gate"]))
    review = yaml.safe_load(review_path.read_text("utf-8")) if review_path.is_file() else {}
    need_hashes = {p.name: sha256_file(p) for p in sorted(sources.glob("*.yaml"))}
    if (
        review.get("schema_version") != "retrieval_scale_review/1"
        or not review.get("passed")
        or review.get("kb_version") != kb_version
        or review.get("source_hashes") != need_hashes
    ):
        failures.append("substantive diversity review missing, failed or stale")
    sft = current_sft(data)
    if sft is None:
        raise ValueError("SFT required for decontamination")
    sm = read_model_json(sft / "manifest.json", DatasetManifest)
    if verify_files(sft, sm.files, exclude=frozenset({"manifest.json"})):
        raise ValueError("SFT integrity failure")
    leaks: dict[str, Any] = (
        leakage_checks(queries, sft, settings, semantic=semantic) if queries else {"passed": False}
    )
    if not leaks["passed"]:
        failures.append("SFT leakage check failed")
    near = {row["query"] for row in leaks.get("semantic_near_sft", [])}
    queries = [
        q.model_copy(update={"quality_flags": ["semantic_near_sft"]}) if q.query in near else q
        for q in queries
    ]
    development = sorted(
        {q.query for q in queries if metadata[q.query_id]["phase"] == "development"}
    )
    holdout = sorted({q.query for q in queries if metadata[q.query_id]["phase"] == "holdout"})
    split_overlap = {
        "exact": len(exact_leaks(development, holdout)),
        "lexical": len(
            lexical_leaks(
                development, holdout, threshold=settings.stage4.leak_jaccard, seed=protocol["seed"]
            )
        ),
        "ngram": len(ngram_leaks(development, holdout, settings.stage4.leak_ngram)),
    }
    if any(split_overlap.values()) or not development or not holdout:
        failures.append("family split leakage check failed or a phase is empty")
    # Gold is an exclusion list only, never a source or a selection input.
    gold_files = sorted((data / "gold").rglob("gold.jsonl"))
    gold_versions = []
    gold_texts = set()
    for path in gold_files:
        gm = read_model_json(path.parent / "manifest.json", DatasetManifest)
        if verify_files(path.parent, gm.files, exclude=frozenset({"manifest.json"})):
            raise ValueError("gold exclusion input integrity failure")
        gold_versions.append(gm.dataset_version)
        for line in path.read_text("utf-8").splitlines():
            row = json.loads(line)
            for message in row.get("turns", row.get("messages", [])):
                if message.get("role") == "user":
                    gold_texts.add(message["content"])
            if isinstance(row.get("user_text"), str):
                gold_texts.add(row["user_text"])
    unique = sorted({q.query for q in queries})
    gold_leaks = (
        {
            "exact": len(exact_leaks(sorted(gold_texts), unique)),
            "lexical": len(
                lexical_leaks(
                    sorted(gold_texts),
                    unique,
                    threshold=settings.stage4.leak_jaccard,
                    seed=protocol["seed"],
                )
            ),
            "ngram": len(ngram_leaks(sorted(gold_texts), unique, settings.stage4.leak_ngram)),
        }
        if unique and gold_texts
        else {}
    )
    if not gold_texts or any(gold_leaks.values()):
        failures.append("gold exclusion check missing or failed")
    source_hashes = {
        p.relative_to(settings.root).as_posix(): sha256_file(p)
        for p in sorted(sources.glob("*.yaml"))
    }
    source_hashes["configs/retrieval_scale.yaml"] = sha256_file(
        settings.root / "configs/retrieval_scale.yaml"
    )
    if review_path.is_file():
        source_hashes[review_path.relative_to(settings.root).as_posix()] = sha256_file(review_path)
    source_hashes.update(
        {p.relative_to(settings.root).as_posix(): sha256_file(p) for p in gold_files}
    )
    code_paths = [
        settings.root / "src/supportpilot/retrieval" / name
        for name in (
            "scale_data.py",
            "scale_metrics.py",
            "scale_run.py",
            "evaluate.py",
            "search.py",
            "index.py",
            "models.py",
            "encode.py",
            "scale_author.py",
        )
    ]
    code_paths.extend(
        settings.root / name
        for name in (
            "src/supportpilot/kb/profiles.py",
            "src/supportpilot/kb/spec.py",
            "src/supportpilot/kb/build.py",
            "src/supportpilot/kb/checks.py",
            "configs/retrieval_models.yaml",
            "uv.lock",
            "kb_sources/demo_store/technical_profiles.yaml",
        )
    )
    source_hashes.update(
        {p.relative_to(settings.root).as_posix(): sha256_file(p) for p in code_paths if p.is_file()}
    )
    tokenizer = (
        settings.eda.tokenizers[0].model_dump(mode="json") if settings.eda.tokenizers else None
    )
    identity = sha256_text(
        json.dumps(
            {
                "sources": source_hashes,
                "kb": kb_version,
                "sft": sm.dataset_version,
                "semantic": semantic,
                "tokenizer": tokenizer,
            },
            sort_keys=True,
        )
    )
    report = {
        "passed": not failures,
        "failures": failures,
        "counts": counts,
        "corpus": corpus,
        "diversity_review": review,
        "query_integrity": integrity,
        "sft_leakage": leaks,
        "development_holdout_overlap": split_overlap,
        "gold_exclusion": gold_leaks,
        "gold_versions": gold_versions,
        "gold_files": len(gold_files),
        "families": len({need.family_id for need in needs.values()}),
        "phase_needs": dict(Counter(family_phase(n.family_id, protocol) for n in needs.values())),
        "kb_version": kb_version,
        "sft_version": sm.dataset_version,
        "source_hashes": source_hashes,
        "tokenizer": tokenizer,
        "input_identity": identity,
        "independent_human_review": False,
        "review_status": "draft",
        "semantic_checked": semantic,
    }
    root = settings.resolve(Path(protocol["output"]))
    write_json(root / "audits" / f"{identity}.json", report)
    if freeze:
        if failures:
            raise ValueError(f"scale freeze refused: {failures}")
        if not semantic:
            raise ValueError("scale freeze requires semantic leakage diagnostics")
        out = data / "benchmarks/retrieval_scale" / identity[:20]
        if out.exists():
            verify_freeze(out)
            return {**report, "freeze": str(out)}
        with writer_lock(out.parent / "freeze.lock"), partial_dir(out) as staging:
            if out.exists():
                verify_freeze(out)
                return {**report, "freeze": str(out)}
            write_json(staging / "protocol.json", protocol)
            write_json(staging / "metadata.json", metadata)
            write_json(staging / "audit.json", report)
            (staging / "queries.jsonl").write_text(
                "".join(q.model_dump_json() + "\n" for q in queries), "utf-8"
            )
            (staging / "qrels.tsv").write_text(
                "".join(
                    f"{q.query_id}\t0\t{r.chunk_id}\t{r.grade}\n"
                    for q in queries
                    for r in q.relevant
                ),
                "utf-8",
            )
            write_json(
                staging / "freeze.json",
                {
                    "schema_version": "retrieval_scale_freeze/1",
                    "input_identity": identity,
                    "created_at": datetime.now(UTC).isoformat(),
                    "files": [f.model_dump(mode="json") for f in scan_files(staging)],
                },
            )
            staging.rename(out)
        report["freeze"] = str(out)
    return report


def verify_freeze(path: Path) -> dict[str, Any]:
    from supportpilot.common.schemas import FileEntry

    stored = json.loads((path / "freeze.json").read_text("utf-8"))
    entries = [FileEntry.model_validate(row) for row in stored["files"]]
    if verify_files(path, entries, exclude=frozenset({"freeze.json"})):
        raise ValueError("frozen benchmark changed")
    return dict(stored)
