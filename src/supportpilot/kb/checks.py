"""KB quality gates (Stage 5). Every gate returns {"passed": bool, ...counts, "failures": [...]}.

| gate | rule |
|---|---|
| ids | chunk ids unique, recomputable; positions 0..n-1; one document per (doc_id, lang) |
| sources | every register file is in the version with sha256 = document version |
| metadata | title, type, format and effective date equal the register / rendering |
| sections_linked | every register section and record found exactly once; no heading without id |
| facts | every placed fact is in a chunk of its section (exact surface; integers parsed) |
| coverage | every canonical fact (facts.yaml, catalog.yaml) is placed in every language |
| locators | the source text at each fact chunk's locator contains the fact |
| completeness | every content word of a document is in its chunks, and chunks hold no other words |
| cross_lingual | each translation group has en/ru/kk with equal fact keys and equal numbers |
| language | LID of prose chunks = document language |
| tokens | every chunk within the token budget |
| pii | only the store's own e-mail and phone appear |
| evidence | every gold `{doc_id}.{section}` reference resolves to chunks in all languages |
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml

from supportpilot.common.hashing import sha256_bytes
from supportpilot.common.ids import make_chunk_id
from supportpilot.common.schemas import LANGUAGES, KBChunk, KBDocument, KBSection
from supportpilot.kb.locate import SourceReader
from supportpilot.kb.register import Expected, ExpectedPlacement, Register

MAX_FAILURES = 50
_WORD = re.compile(r"\w+")
_NUMBER = re.compile(r"\d{1,3}(?:[ ,]\d{3})+(?!\d)|\d+(?:\.\d+)?")
_SIGNED_NUMBER = re.compile(
    r"(?<![\w.+\-\u2212])[-+\u2212]?(?:\d{1,3}(?:[ ,]\d{3})+|\d+)(?:\.\d+)?(?!\d|\.\d)"
)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE = re.compile(r"\+?\d[\d ()-]{8,}\d")
_LETTER = re.compile(r"[^\W\d_]")


def norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    t = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in t)  # NBSP etc.
    return re.sub(r"\s+", " ", t).strip()


def words(text: str) -> Counter[str]:
    return Counter(w.casefold() for w in _WORD.findall(norm(text)))


def numbers(text: str) -> Counter[str]:
    return Counter(re.sub(r"[ ,]", "", m) for m in _NUMBER.findall(norm(text)))


def _gate(failures: list[str], **counts: Any) -> dict[str, Any]:
    return {"passed": not failures, **counts, "failures": failures[:MAX_FAILURES],
            "failure_count": len(failures)}  # fmt: skip


def fact_kind(key: str) -> str:
    leaf = key.rsplit(".", 1)[-1]
    if (
        leaf in ("price",)
        or leaf.endswith(("_cost", "_max"))
        or leaf in ("min_order", "free_delivery_from")
    ):
        return "money"
    if leaf in ("from", "to", "effective_from", "support_hours", "showroom_hours"):
        return "date_or_time"  # calendar dates and opening hours
    if leaf.endswith(("_days", "_months", "_hours", "_seconds", "_minutes", "_percent")):
        return "period_or_number"
    if leaf in ("sku", "id", "support_phone", "support_email"):
        return "identifier"
    if leaf in ("non_returnable", "carriers", "payment_methods"):
        return "condition_list"
    return "text"


def walk_sections(sections: Iterable[KBSection]) -> Iterable[KBSection]:
    for s in sections:
        yield s
        yield from walk_sections(s.children)


# ---------------------------------------------------------------- gates


def gate_ids(docs: list[KBDocument], chunks: list[KBChunk]) -> dict[str, Any]:
    fails: list[str] = []
    keys = Counter((d.doc_id, d.lang) for d in docs)
    fails += [f"document twice: {k}" for k, n in keys.items() if n > 1]
    ids = Counter(c.chunk_id for c in chunks)
    fails += [f"duplicate chunk_id {k}" for k, n in ids.items() if n > 1]
    version = {(d.doc_id, d.lang): d.version for d in docs}
    by_doc: dict[tuple[str, str], list[int]] = defaultdict(list)
    for c in chunks:
        if c.chunk_id != make_chunk_id(c.doc_id, c.version, c.position):
            fails.append(f"chunk_id not recomputable: {c.chunk_id}")
        if version.get((c.doc_id, c.lang)) != c.version:
            fails.append(f"chunk of a non-current version: {c.chunk_id}")
        by_doc[(c.doc_id, c.lang)].append(c.position)
    for k, pos in by_doc.items():
        if sorted(pos) != list(range(len(pos))):
            fails.append(f"positions not 0..n-1: {k}")
    fails += [f"document without chunks: {k}" for k in version if k not in by_doc]
    return _gate(fails, documents=len(docs), chunks=len(chunks))


def gate_sources(docs: list[KBDocument], register: Register, source_dir: Path) -> dict[str, Any]:
    fails = []
    by_path = {d.source_path: d for d in docs}
    for e in register.documents:
        p = source_dir / e.path
        if not p.is_file():
            fails.append(f"missing source {e.path}")
            continue
        sha = sha256_bytes(p.read_bytes())
        doc = by_path.get(e.path)
        if doc is None:
            fails.append(f"source without document {e.path}")
        elif sha != doc.version or sha != e.sha256:
            fails.append(f"sha256 mismatch {e.path}")
    fails += [f"document without register entry {p}" for p in set(by_path) -
              {e.path for e in register.documents}]  # fmt: skip
    return _gate(fails, files=len(register.documents))


def gate_metadata(docs: list[KBDocument], register: Register, expected: Expected) -> dict[str, Any]:
    fails = []
    by_path = {d.source_path: d for d in docs}
    for e in register.documents:
        d = by_path.get(e.path)
        x = expected.documents.get(e.path)
        if d is None or x is None:
            continue
        if d.title != x.title:
            fails.append(f"{e.path}: title {d.title!r} != {x.title!r}")
        if d.effective_from != x.effective_from:
            fails.append(f"{e.path}: effective_from {d.effective_from} != {x.effective_from}")
        if d.effective_to != x.effective_to or d.effective_to != e.effective_to:
            fails.append(f"{e.path}: effective_to differs from the source register")
        if (d.doc_id, d.lang, d.doc_type, d.format) != (e.doc_id, e.lang, e.doc_type, e.format):
            fails.append(f"{e.path}: identity differs from the register")
    return _gate(fails, documents=len(by_path))


def gate_sections_linked(docs: list[KBDocument], register: Register) -> dict[str, Any]:
    fails = []
    by_path = {d.source_path: d for d in docs}
    n = 0
    for e in register.documents:
        d = by_path.get(e.path)
        if d is None:
            continue
        found = [s for s in walk_sections(d.sections) if s.level > 0]
        ids = Counter(s.section_id for s in found)
        want = [s.id for s in e.sections] + [r.lower() for r in e.records]
        n += len(want)
        fails += [f"{e.path}: heading without section id: {s.heading!r}" for s in found
                  if s.section_id is None]  # fmt: skip
        fails += [f"{e.path}: section {w!r} not found" for w in want if ids[w] == 0]
        fails += [f"{e.path}: section {k!r} found {c} times" for k, c in ids.items()
                  if k is not None and c > 1]  # fmt: skip
        fails += [f"{e.path}: unexpected section {k!r}" for k in ids if k and k not in want]
    return _gate(fails, sections=n)


def _chunks_by_section(chunks: list[KBChunk]) -> dict[tuple[str, str, str | None], list[KBChunk]]:
    out: dict[tuple[str, str, str | None], list[KBChunk]] = defaultdict(list)
    for c in chunks:
        out[(c.source_path, c.lang, c.section_id)].append(c)
    return out


def _find(p: ExpectedPlacement, cands: list[KBChunk]) -> KBChunk | None:
    surface = norm(p.surface)
    for c in cands:
        text = norm(c.text)
        if surface in text and (
            not isinstance(p.value, int | float)
            or isinstance(p.value, bool)
            or any(
                Decimal(re.sub(r"[ ,]", "", token).replace("\u2212", "-")) == Decimal(str(p.value))
                for token in _SIGNED_NUMBER.findall(text)
            )
        ):
            return c
    return None


def gate_facts(
    chunks: list[KBChunk], expected: Expected
) -> tuple[dict[str, Any], list[tuple[ExpectedPlacement, KBChunk]]]:
    fails = []
    found: list[tuple[ExpectedPlacement, KBChunk]] = []
    per: Counter[str] = Counter()
    hit: Counter[str] = Counter()
    index = _chunks_by_section(chunks)
    for path, x in sorted(expected.documents.items()):
        lang = path.split("/", 1)[0]
        for p in x.placements:
            tag = f"{lang}|{fact_kind(p.key)}"
            per[tag] += 1
            c = _find(p, index.get((path, lang, p.section_id), []))
            if c is None:
                fails.append(f"{path}#{p.section_id}: {p.key} = {p.surface!r} not in its chunks")
                continue
            hit[tag] += 1
            found.append((p, c))
    by = {k: {"placed": per[k], "found": hit[k]} for k in sorted(per)}
    total = sum(per.values())
    return _gate(fails, placements=total, found=sum(hit.values()),
                 recall=round(sum(hit.values()) / total, 6) if total else 1.0,
                 by_lang_kind=by), found  # fmt: skip


def gate_coverage(expected: Expected) -> dict[str, Any]:
    placed: dict[str, set[str]] = {lang: set() for lang in LANGUAGES}
    for path, x in expected.documents.items():
        lang = path.split("/", 1)[0]
        placed[lang] |= {p.key for p in x.placements}
        placed[lang].add("effective_from")  # every document states it (metadata gate)
    fails = [f"{lang}: canonical fact {k!r} is in no document" for lang in LANGUAGES
             for k in sorted(expected.facts) if k not in placed[lang]]  # fmt: skip
    return _gate(fails, canonical_facts=len(expected.facts))


def gate_locators(
    found: list[tuple[ExpectedPlacement, KBChunk]], source_dir: Path
) -> dict[str, Any]:
    fails = []
    readers: dict[str, SourceReader] = {}
    for p, c in found:
        r = readers.get(c.source_path)
        if r is None:
            data = (source_dir / c.source_path).read_bytes()
            r = readers[c.source_path] = SourceReader(data, c.format, c.lang)
        if norm(p.surface) not in norm(r.text_at(c.locator)):
            fails.append(f"{c.chunk_id}: {p.key} not at {c.locator.model_dump(exclude_none=True)}")
    return _gate(fails, checked=len(found))


def gate_completeness(
    chunks: list[KBChunk], docs: list[KBDocument], expected: Expected
) -> dict[str, Any]:
    fails = []
    by_path: dict[str, list[KBChunk]] = defaultdict(list)
    for c in chunks:
        by_path[c.source_path].append(c)
    recall_min = 1.0
    for path, x in sorted(expected.documents.items()):
        want: Counter[str] = Counter()
        for s in x.content:
            want.update(words(s))
        got: Counter[str] = Counter()
        for c in by_path.get(path, []):
            got.update(words(c.text))
        missing = want - got
        allowed = set(want) | set(words(x.title))
        extra = sorted(set(got) - allowed)
        total = sum(want.values())
        recall = 1 - sum(missing.values()) / total if total else 1.0
        recall_min = min(recall_min, recall)
        if missing:
            fails.append(f"{path}: {sum(missing.values())} words lost, e.g. {sorted(missing)[:8]}")
        if extra:
            fails.append(f"{path}: words not in the document: {extra[:8]}")
    return _gate(fails, documents=len(expected.documents), min_word_recall=round(recall_min, 6))


def gate_cross_lingual(chunks: list[KBChunk], expected: Expected) -> dict[str, Any]:
    fails = []
    groups: dict[str, dict[str, list[KBChunk]]] = defaultdict(lambda: defaultdict(list))
    for c in chunks:
        groups[c.translation_group][c.lang].append(c)
    keys: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for path, x in expected.documents.items():
        lang, rest = path.split("/", 1)
        doc_id = rest.rsplit(".", 1)[0]
        for p in x.placements:
            keys[(doc_id, p.section_id or "intro")][lang].add(p.key)
    for g, by_lang in sorted(groups.items()):
        if set(by_lang) != set(LANGUAGES):
            fails.append(f"{g}: languages {sorted(by_lang)}")
            continue
        nums = {lang: sum((numbers(c.text) for c in cs), Counter()) for lang, cs in by_lang.items()}
        if len({tuple(sorted(n.items())) for n in nums.values()}) != 1:
            fails.append(f"{g}: numbers differ { ({k: dict(v) for k, v in nums.items()}) }")
    for (doc_id, sid), placed in sorted(keys.items()):
        if len({frozenset(v) for v in placed.values()}) != 1 or set(placed) != set(LANGUAGES):
            fails.append(f"{doc_id}#{sid}: placed fact keys differ across languages")
    return _gate(fails, groups=len(groups))


def gate_language(chunks: list[KBChunk], min_letters: int) -> dict[str, Any]:
    from supportpilot.data.langid import LanguageIdentifier

    prose = [c for c in chunks if c.doc_type.value != "catalog"
             and len(_LETTER.findall(c.text)) >= min_letters]  # fmt: skip
    res = LanguageIdentifier().detect_many([c.text for c in prose])
    fails = [f"{c.chunk_id}: {c.lang} detected as {r.lang} ({r.conf:.2f})"
             for c, r in zip(prose, res, strict=True) if r.lang != c.lang]  # fmt: skip
    agree = 1 - len(fails) / len(prose) if prose else 1.0
    return _gate(fails, checked=len(prose), agreement=round(agree, 6))


def gate_tokens(chunks: list[KBChunk], max_tokens: int) -> dict[str, Any]:
    fails = [f"{c.chunk_id}: {c.token_count} tokens" for c in chunks if c.token_count > max_tokens]
    counts = sorted(c.token_count for c in chunks)
    return _gate(fails, max_tokens=max_tokens, p50=counts[len(counts) // 2] if counts else 0,
                 max=counts[-1] if counts else 0)  # fmt: skip


def gate_pii(chunks: list[KBChunk], facts: Mapping[str, Any]) -> dict[str, Any]:
    allowed_mail = {str(facts.get("support_email", ""))}
    allowed_phone = {re.sub(r"\D", "", str(facts.get("support_phone", "")))}
    fails = []
    for c in chunks:
        fails += [
            f"{c.chunk_id}: e-mail {m}" for m in _EMAIL.findall(c.text) if m not in allowed_mail
        ]
        fails += [f"{c.chunk_id}: phone-like {m}" for m in _PHONE.findall(c.text)
                  if re.sub(r"\D", "", m) not in allowed_phone]  # fmt: skip
    return _gate(fails, chunks=len(chunks))


def gold_references(gold_dir: Path) -> set[str]:
    """`doc_id.section` strings cited by the gold sources (`ev:` lists)."""
    refs: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "ev" and isinstance(v, list):
                    refs.update(str(x) for x in v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    for p in sorted(gold_dir.glob("*.yaml")):
        walk(yaml.safe_load(p.read_text(encoding="utf-8")))
    return refs


def evidence_map(chunks: list[KBChunk]) -> dict[str, dict[str, list[str]]]:
    """`doc_id.section_id` -> lang -> chunk ids (the gold `accepted_evidence` resolution)."""
    out: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for c in sorted(chunks, key=lambda c: (c.doc_id, c.lang, c.position)):
        if c.section_id is not None:
            out[f"{c.doc_id}.{c.section_id}"][c.lang].append(c.chunk_id)
    return {k: dict(v) for k, v in sorted(out.items())}


def gate_evidence(chunks: list[KBChunk], gold_dir: Path | None) -> dict[str, Any]:
    if gold_dir is None or not gold_dir.is_dir():
        return {"passed": True, "status": "not_applicable", "reason": "no gold sources",
                "failures": [], "failure_count": 0}  # fmt: skip
    refs = gold_references(gold_dir)
    emap = evidence_map(chunks)
    fails = [f"{r}: no chunks in {sorted(set(LANGUAGES) - set(emap.get(r, {})))}"
             for r in sorted(refs) if set(emap.get(r, {})) != set(LANGUAGES)]  # fmt: skip
    return _gate(fails, references=len(refs))


def run_gates(
    *,
    docs: list[KBDocument],
    chunks: list[KBChunk],
    register: Register,
    expected: Expected,
    source_dir: Path,
    max_tokens: int,
    lid_min_letters: int,
    gold_dir: Path | None,
    facts: Mapping[str, Any],
) -> dict[str, Any]:
    facts_gate, found = gate_facts(chunks, expected)
    gates = {
        "ids": gate_ids(docs, chunks),
        "sources": gate_sources(docs, register, source_dir),
        "metadata": gate_metadata(docs, register, expected),
        "sections_linked": gate_sections_linked(docs, register),
        "facts": facts_gate,
        "coverage": gate_coverage(expected),
        "locators": gate_locators(found, source_dir),
        "completeness": gate_completeness(chunks, docs, expected),
        "cross_lingual": gate_cross_lingual(chunks, expected),
        "language": gate_language(chunks, lid_min_letters),
        "tokens": gate_tokens(chunks, max_tokens),
        "pii": gate_pii(chunks, facts),
        "evidence": gate_evidence(chunks, gold_dir),
    }
    return {"passed": all(g["passed"] for g in gates.values()), "gates": gates}
