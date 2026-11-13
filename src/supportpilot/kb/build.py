"""Versioned, incremental KB build (Stage 5, ADR-027).

Layout under `<kb_dir>` (default `data/kb/<kb>/`):

    sources/                rendered documents <lang>/<doc_id>.<ext> + register.json + expected.json
    versions/<kb_version>/  immutable KB version:
        sources/            the exact source files of this version (locator checks, restore)
        documents.parquet   KBDocument rows (document_json + key columns)
        chunks.parquet      KBChunk rows (the current chunk set of this version)
        changes.json        documents added / changed / removed / unchanged and chunk ids
                            to delete / upsert versus the parent version (Stage 6 reindex)
        evidence_map.json   `doc_id.section_id` -> lang -> chunk ids
        quality_report.json gates
        manifest.json       dataset_manifest/1
    current.json            the active version (switched atomically after the gates pass)

`render` writes only files whose bytes changed and removes files of deleted documents.
`build` re-parses only documents whose sha256 (version) or processing parameters changed and
reuses the parent version's documents and chunks for the rest. Removed documents are not in
the new version. Identical content gives no new version. A version whose gates fail is kept
as `<kb_version>.failed` and never activated.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from supportpilot.common.hashing import sha256_bytes, sha256_file
from supportpilot.common.ids import make_dataset_version
from supportpilot.common.io import atomic_write_bytes, read_json, write_json
from supportpilot.common.manifest import verify_files
from supportpilot.common.schemas import LANGUAGES, DatasetManifest, DocFormat, KBChunk, KBDocument
from supportpilot.data.locale import load_locale
from supportpilot.data.steps import (
    MANIFEST,
    StepError,
    commit,
    params_sha,
    partial_dir,
    read_manifest,
)
from supportpilot.kb import chunk as chunker
from supportpilot.kb.blocks import ExtractionError, ParsedDoc
from supportpilot.kb.checks import evidence_map, run_gates
from supportpilot.kb.register import (
    EXPECTED,
    REGISTER,
    Expected,
    ExpectedDoc,
    ExpectedPlacement,
    Register,
    RegisterEntry,
    RegisterSection,
)
from supportpilot.kb.render import RenderedDoc, plain_heading, render_all
from supportpilot.kb.spec import fact_registry, load_sources
from supportpilot.kb.structure import build_document
from supportpilot.kb.writers import write, writer_params

DOCUMENTS = "documents.parquet"
CHUNKS = "chunks.parquet"
CHANGES = "changes.json"
EVIDENCE = "evidence_map.json"
QUALITY = "quality_report.json"
CURRENT = "current.json"
DOC_SCHEMA = pa.schema(
    [
        ("doc_id", pa.string()),
        ("lang", pa.string()),
        ("version", pa.string()),
        ("source_path", pa.string()),
        ("params_sha256", pa.string()),
        ("document_json", pa.string()),
    ]
)
CHUNK_SCHEMA = pa.schema(
    [
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("lang", pa.string()),
        ("version", pa.string()),
        ("position", pa.int32()),
        ("section_id", pa.string()),
        ("translation_group", pa.string()),
        ("doc_type", pa.string()),
        ("effective_from", pa.string()),
        ("effective_to", pa.string()),
        ("token_count", pa.int32()),
        ("text", pa.string()),
        ("chunk_json", pa.string()),
    ]
)

TokenCounter = Callable[[Sequence[str]], list[int]]


@dataclass(frozen=True)
class KBConfig:
    name: str  # KB slug, e.g. demo_store
    source_root: Path  # kb_sources/<name>: facts, catalog, docs specs
    locale: Path  # configs/locale_kz.yaml (city / carrier / payment names)
    kb_dir: Path  # data/kb/<name>
    max_tokens: int
    lid_min_letters: int
    tokenizer: str  # recorded in params; the counter itself is passed in
    gold_dir: Path | None = None

    @property
    def sources_dir(self) -> Path:
        return self.kb_dir / "sources"

    @property
    def versions_dir(self) -> Path:
        return self.kb_dir / "versions"


# ---------------------------------------------------------------- render


def _register_entry(doc: RenderedDoc, data: bytes) -> RegisterEntry:
    return RegisterEntry(
        path=doc.path,
        doc_id=doc.doc_id,
        lang=doc.lang,
        doc_type=doc.doc_type,
        format=doc.format,
        title=doc.title,
        effective_from=doc.effective_from,
        effective_to=doc.effective_to,
        sha256=sha256_bytes(data),
        bytes=len(data),
        key_column=0 if doc.records is not None else None,
        sections=[
            RegisterSection(id=s.id, heading=plain_heading(s.heading), level=s.level)
            for s in doc.sections
        ],
        records=[r.key for r in doc.records.rows] if doc.records else [],
    )


def render(cfg: KBConfig) -> dict[str, Any]:
    """Specs + facts -> source files, register and expected facts. Idempotent."""
    sources = load_sources(cfg.source_root)
    lookup = load_locale(cfg.locale).lookup
    docs = render_all(sources, lookup)
    out = cfg.sources_dir
    out.mkdir(parents=True, exist_ok=True)
    entries, expected, written = [], {}, []
    for d in docs:
        data = write(d)
        path = out / d.path
        if not path.is_file() or path.read_bytes() != data:
            atomic_write_bytes(path, data)
            written.append(d.path)
        entries.append(_register_entry(d, data))
        expected[d.path] = ExpectedDoc(
            title=d.title,
            effective_from=d.effective_from,
            effective_to=d.effective_to,
            placements=[ExpectedPlacement(**p.__dict__) for p in d.placements],
            content=d.content_strings(),
        )
    keep = {e.path for e in entries} | {REGISTER, EXPECTED}
    removed = []
    for p in sorted(out.rglob("*")):
        rel = p.relative_to(out).as_posix()
        if p.is_file() and rel not in keep:
            p.unlink()
            removed.append(rel)
    params = {"writers": writer_params(), "spec_files": _spec_hashes(cfg.source_root)}
    register = Register(kb=cfg.name, render_params=params, documents=entries)
    register_data = register.model_dump(mode="json")
    for entry in register_data["documents"]:
        if entry["effective_to"] is None:
            del entry["effective_to"]
    write_json(out / REGISTER, register_data)
    exp = Expected(facts=fact_registry(sources), documents=expected)
    expected_data = exp.model_dump(mode="json")
    for entry in expected_data["documents"].values():
        if entry["effective_to"] is None:
            del entry["effective_to"]
    write_json(out / EXPECTED, expected_data)
    return {"documents": len(entries), "written": written, "removed": removed,
            "unchanged": len(entries) - len(written)}  # fmt: skip


def _spec_hashes(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.yaml"))}


# ---------------------------------------------------------------- parse


def parse_file(data: bytes, entry: RegisterEntry) -> ParsedDoc:
    from supportpilot.kb.parse_office import parse_csv, parse_docx, parse_xlsx
    from supportpilot.kb.parse_pdf import parse_pdf
    from supportpilot.kb.parse_text import parse_json, parse_txt

    key = entry.key_column or 0
    match entry.format:
        case DocFormat.PDF:
            return parse_pdf(data)
        case DocFormat.DOCX:
            return parse_docx(data)
        case DocFormat.XLSX:
            return parse_xlsx(data, key)
        case DocFormat.CSV:
            return parse_csv(data, key)
        case DocFormat.JSON:
            return parse_json(data)
        case DocFormat.TXT:
            return parse_txt(data)


def parser_ids() -> dict[str, str]:
    import docx
    import openpyxl

    from supportpilot.kb import parse_office, parse_text
    from supportpilot.kb.parse_pdf import parser_id

    office = parse_office.PARSER_REVISION
    text = parse_text.PARSER_REVISION
    return {
        "pdf": parser_id(),
        "docx": f"python-docx/{getattr(docx, '__version__', 'unknown')}+kb{office}",
        "xlsx": f"openpyxl/{openpyxl.__version__}+kb{office}",
        "csv": f"csv/stdlib+kb{office}",
        "json": f"json/stdlib+kb{text}",
        "txt": f"txt/stdlib+kb{text}",
    }


def processing_params(cfg: KBConfig) -> dict[str, Any]:
    return {
        "parsers": parser_ids(),
        "chunker_revision": chunker.CHUNKER_REVISION,
        "max_tokens": cfg.max_tokens,
        "tokenizer": cfg.tokenizer,
    }


# ---------------------------------------------------------------- versions


def current_version(cfg: KBConfig) -> Path | None:
    p = cfg.kb_dir / CURRENT
    if not p.is_file():
        return None
    version = str(read_json(p)["kb_version"])
    d = cfg.versions_dir / version
    if not (d / MANIFEST).is_file():
        raise StepError(f"current.json points to a missing version {version}")
    return d


def load_version(vdir: Path) -> tuple[list[KBDocument], list[KBChunk], dict[str, str]]:
    dt = pq.read_table(vdir / DOCUMENTS).to_pylist()
    ct = pq.read_table(vdir / CHUNKS).to_pylist()
    docs = [KBDocument.model_validate_json(r["document_json"]) for r in dt]
    chunks = [KBChunk.model_validate_json(r["chunk_json"]) for r in ct]
    params = {f"{r['doc_id']}|{r['lang']}": r["params_sha256"] for r in dt}
    return docs, chunks, params


def _content_sha(docs: list[KBDocument], chunks: list[KBChunk]) -> str:
    rows = sorted(d.model_dump_json() for d in docs) + sorted(c.model_dump_json() for c in chunks)
    return sha256_bytes("\n".join(rows).encode("utf-8"))


def _write_version(vdir: Path, docs: list[KBDocument], chunks: list[KBChunk], psha: str) -> None:
    docs = sorted(docs, key=lambda d: (d.doc_id, LANGUAGES.index(d.lang)))
    chunks = sorted(chunks, key=lambda c: (c.doc_id, LANGUAGES.index(c.lang), c.position))
    drows = [
        {"doc_id": d.doc_id, "lang": d.lang, "version": d.version, "source_path": d.source_path,
         "params_sha256": psha, "document_json": d.model_dump_json()}
        for d in docs
    ]  # fmt: skip
    crows = [
        {
            "chunk_id": c.chunk_id, "doc_id": c.doc_id, "lang": c.lang, "version": c.version,
            "position": c.position, "section_id": c.section_id,
            "translation_group": c.translation_group, "doc_type": c.doc_type.value,
            "effective_from": c.effective_from.isoformat(),
            "effective_to": c.effective_to.isoformat() if c.effective_to else None,
            "token_count": c.token_count, "text": c.text, "chunk_json": c.model_dump_json(),
        }
        for c in chunks
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(drows, schema=DOC_SCHEMA), vdir / DOCUMENTS,
                   compression="zstd")  # fmt: skip
    pq.write_table(pa.Table.from_pylist(crows, schema=CHUNK_SCHEMA), vdir / CHUNKS,
                   compression="zstd")  # fmt: skip


def build(
    cfg: KBConfig,
    count_tokens: TokenCounter,
    *,
    code_commit: str | None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a new KB version from `sources/` (render first). Returns a JSON-able report."""
    src = cfg.sources_dir
    if not (src / REGISTER).is_file():
        raise StepError(f"no rendered sources in {src}: run `supportpilot kb render` first")
    register = Register.model_validate(read_json(src / REGISTER))
    expected = Expected.model_validate(read_json(src / EXPECTED))
    params = processing_params(cfg)
    psha = params_sha(params)

    parent = current_version(cfg)
    prev_docs: dict[tuple[str, str], KBDocument] = {}
    prev_chunks: dict[tuple[str, str], list[KBChunk]] = {}
    prev_params: dict[str, str] = {}
    if parent is not None:
        pdocs, pchunks, prev_params = load_version(parent)
        prev_docs = {(d.doc_id, d.lang): d for d in pdocs}
        for c in pchunks:
            prev_chunks.setdefault((c.doc_id, c.lang), []).append(c)

    docs: list[KBDocument] = []
    chunks: list[KBChunk] = []
    status: dict[str, list[str]] = {"added": [], "changed": [], "unchanged": [], "removed": []}
    errors: list[str] = []
    for e in register.documents:
        data = (src / e.path).read_bytes()
        if sha256_bytes(data) != e.sha256:
            raise StepError(f"{e.path} changed after rendering: run `supportpilot kb render`")
        key = (e.doc_id, e.lang)
        old = prev_docs.get(key)
        if (not force and old is not None and old.version == e.sha256
                and prev_params.get(f"{e.doc_id}|{e.lang}") == psha
                and (old.effective_from, old.effective_to, old.doc_type)
                == (e.effective_from, e.effective_to, e.doc_type)):  # fmt: skip
            docs.append(old)
            chunks.extend(prev_chunks.get(key, []))
            status["unchanged"].append(e.path)
            continue
        try:
            doc = build_document(parse_file(data, e), e)
            new_chunks = chunker.chunk_document(doc, max_tokens=cfg.max_tokens,
                                                count_tokens=count_tokens)  # fmt: skip
        except ExtractionError as err:
            errors.append(f"{e.path}: {err}")
            continue
        docs.append(doc)
        chunks.extend(new_chunks)
        status["added" if old is None else "changed"].append(e.path)
    if errors:
        raise StepError("extraction failed: " + "; ".join(errors))
    keys = {(e.doc_id, e.lang) for e in register.documents}
    status["removed"] = sorted(prev_docs[k].source_path for k in prev_docs if k not in keys)

    content_sha = _content_sha(docs, chunks)
    inputs = [f"register:{sha256_file(src / REGISTER)}", f"expected:{sha256_file(src / EXPECTED)}"]
    if parent is not None and not status["added"] and not status["removed"]:
        pm = read_manifest(parent)
        if _same_kb(pm, content_sha, inputs):
            return {"status": "unchanged", "kb_version": parent.name, "documents": len(docs),
                    "chunks": len(chunks), "reparsed": len(status["changed"])}  # fmt: skip

    old_ids = {c.chunk_id for cs in prev_chunks.values() for c in cs}
    new_ids = {c.chunk_id for c in chunks}
    changes: dict[str, Any] = {
        "parent": parent.name if parent else None,
        "documents": {k: sorted(v) for k, v in status.items()},
        "chunks_delete": sorted(old_ids - new_ids),
        "chunks_upsert": sorted(new_ids - old_ids),
        "chunks_kept": len(new_ids & old_ids),
    }
    name = f"kb_{cfg.name}"
    version = make_dataset_version(name, datetime.now(UTC).date(), content_sha)
    out = cfg.versions_dir / version
    if (out / MANIFEST).is_file():
        m = read_manifest(out)
        if _same_kb(m, content_sha, inputs):
            activate(cfg, version)  # e.g. a reverted edit: the stored version is reused
            return {"status": "activated_existing", "kb_version": version,
                    "parent": parent.name if parent else None, "chunks": len(chunks)}  # fmt: skip
    with partial_dir(out) as part:
        for e in register.documents:
            dst = part / "sources" / e.path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / e.path, dst)
        shutil.copyfile(src / REGISTER, part / "sources" / REGISTER)
        shutil.copyfile(src / EXPECTED, part / "sources" / EXPECTED)
        _write_version(part, docs, chunks, psha)
        report = run_gates(
            docs=docs, chunks=chunks, register=register, expected=expected,
            source_dir=part / "sources", max_tokens=cfg.max_tokens,
            lid_min_letters=cfg.lid_min_letters, gold_dir=cfg.gold_dir,
            facts=expected.facts,
        )  # fmt: skip
        write_json(part / QUALITY, report)
        write_json(part / CHANGES, changes)
        write_json(part / EVIDENCE, evidence_map(chunks))
        counts = _row_counts(docs, chunks)
        manifest = commit(
            part, out, name=name, data_schema_version="kb_chunk/1", code_commit=code_commit,
            inputs=[*inputs, f"parent:{parent.name}" if parent else "parent:none"],
            params={**params, "content_sha256": content_sha}, row_counts=counts,
            content_sha256=content_sha, quality_report=QUALITY,
            gate=lambda p: bool(read_json(p / QUALITY)["passed"]),
        )  # fmt: skip
    if manifest.dataset_version != version:
        raise StepError(f"version name changed during the build: {version}")
    activate(cfg, version)
    return {"status": "built", "kb_version": version, "parent": changes["parent"],
            "documents": {k: len(v) for k, v in status.items()},
            "chunks": len(chunks), "chunks_delete": len(changes["chunks_delete"]),
            "chunks_upsert": len(changes["chunks_upsert"]),
            "gates": {k: g["passed"] for k, g in report["gates"].items()}}  # fmt: skip


def _same_kb(m: DatasetManifest, content_sha: str, inputs: list[str]) -> bool:
    """Same chunks and documents checked against the same ground truth. The register's
    render parameters (spec file hashes) are provenance only: re-saving a spec with other
    line endings must not create a second version of identical content."""
    return m.params.get("content_sha256") == content_sha and inputs[1] in m.inputs


def _row_counts(docs: list[KBDocument], chunks: list[KBChunk]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in docs:
        k = f"documents|lang={d.lang}|format={d.format.value}"
        out[k] = out.get(k, 0) + 1
    for c in chunks:
        k = f"chunks|lang={c.lang}|doc_type={c.doc_type.value}"
        out[k] = out.get(k, 0) + 1
    return out


def activate(cfg: KBConfig, version: str) -> dict[str, Any]:
    """Make `version` the current KB (rollback = activate an older version)."""
    vdir = cfg.versions_dir / version
    if not (vdir / MANIFEST).is_file():
        raise StepError(f"no KB version {version}")
    m = read_manifest(vdir)
    if problems := verify_files(vdir, m.files, exclude=frozenset({MANIFEST})):
        raise StepError(f"{version} is damaged: {[p.model_dump() for p in problems[:5]]}")
    if not read_json(vdir / QUALITY)["passed"]:
        raise StepError(f"{version} did not pass its gates")
    write_json(cfg.kb_dir / CURRENT, {"kb_version": version,
                                      "activated_at": datetime.now(UTC).isoformat()})  # fmt: skip
    return {"kb_version": version}


def validate(cfg: KBConfig, version: str | None = None) -> dict[str, Any]:
    """Re-run the gates on a stored version from its own copy of the sources."""
    vdir = cfg.versions_dir / version if version else current_version(cfg)
    if vdir is None or not (vdir / MANIFEST).is_file():
        raise StepError("no KB version to validate")
    m = read_manifest(vdir)
    problems = verify_files(vdir, m.files, exclude=frozenset({MANIFEST}))
    docs, chunks, _ = load_version(vdir)
    register = Register.model_validate(read_json(vdir / "sources" / REGISTER))
    expected = Expected.model_validate(read_json(vdir / "sources" / EXPECTED))
    report = run_gates(
        docs=docs, chunks=chunks, register=register, expected=expected,
        source_dir=vdir / "sources", max_tokens=int(str(m.params["max_tokens"])),
        lid_min_letters=cfg.lid_min_letters, gold_dir=cfg.gold_dir, facts=expected.facts,
    )  # fmt: skip
    report["files"] = {"passed": not problems, "problems": [p.model_dump() for p in problems]}
    report["passed"] = report["passed"] and not problems
    report["kb_version"] = vdir.name
    return report


def status(cfg: KBConfig) -> dict[str, Any]:
    cur = current_version(cfg)
    versions = sorted(p.name for p in cfg.versions_dir.glob("*") if p.is_dir()) \
        if cfg.versions_dir.is_dir() else []  # fmt: skip
    out: dict[str, Any] = {"current": cur.name if cur else None, "versions": versions}
    if cur is not None:
        m = read_manifest(cur)
        out["row_counts"] = m.row_counts
        out["created_at"] = m.created_at.isoformat()
        reg = cfg.sources_dir / REGISTER
        if reg.is_file():
            stored = json.loads((cur / "sources" / REGISTER).read_text(encoding="utf-8"))
            out["sources_match_current"] = read_json(reg)["documents"] == stored["documents"]
    return out
