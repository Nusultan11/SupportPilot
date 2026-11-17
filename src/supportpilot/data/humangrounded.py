"""Human-written, document-grounded responses (ADR-043, DATA_CONTRACTS §20.1).

The intent corpus has no human answer grounded in this store's KB, and none is synthesised.
This component instead keeps responses that people wrote *from a cited document* in another
domain (doc2dial: US government service pages), so the grounded-response behaviour (answer
from evidence, ask for a missing condition) can be trained with real human targets while the
facts stay in their own evidence namespace and never enter the SupportPilot KB.

One record per agent turn: the preceding user turn(s) are the query, earlier turns the context,
the writer-selected document spans the evidence and the agent utterance the target.
`grounding_status = source_annotated`: the writer saw the span while writing; support is not
independently verified. A lexical recall proxy (share of the target's content words found in
the evidence) routes weakly overlapping responses to `unverified`; it is not semantic proof.
Documents are the split unit (all dialogues on one document share a split).
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import make_example_id
from supportpilot.common.schemas import SourceEntry
from supportpilot.data.acquire import raw_dir, verify_raw
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.steps import StepError, content_hash, write_parquet
from supportpilot.data.textnorm import CARD_RE, EMAIL_RE, PHONE_RE, clean_text, garbled

GROUNDED_SCHEMA = "human_grounded/1"
GROUNDING_STATUS = "source_annotated"
REVIEW_STATUS = "unreviewed"
SPLITS = ("train", "validation", "test")
GROUNDED_FILES = (
    *(f"grounded_{s}.parquet" for s in SPLITS),
    "grounded_unverified.parquet",
    "grounded_excluded.parquet",
)
_TOKEN_RE = re.compile(r"[a-z0-9]{4,}")
_PLACEHOLDER_RE = re.compile(r"(?i)null|none|n/?a|undefined|-+|\.+")  # export artefacts
_NANP_RE = re.compile(r"(?<![\w+])(?:\(\d{3}\)\s?|\d{3}\s?[\s.-]\s?)\d{3}\s?[\s.-]\s?\d{4}(?!\w)")
# ADR-045 structural checks (lexical; they do not prove semantic support)
_MARKUP_RE = re.compile(r"\[\s*\d+\s*\]|(?i:\bon this page\b)")
_FUSED_RE = re.compile(r"[a-z]{2,}[A-Z][a-z]{2,}")
_POLARITY_RE = re.compile(
    r"(?i)^\W*(?:yes|no|yeah|yep|nope|correct|that'?s (?:right|correct)|i'?m afraid (?:so|not)"
    r"|of course|absolutely|not always|exactly)\b"
)
_EVIDENCE_POLARITY_RE = re.compile(r"(?i)\b(?:yes|no|not|never|cannot|can't|don't|won't)\b|n't\b")
_REFERRAL_RE = re.compile(r"(?i)^\W*(?:ok(?:ay)?\W*)?(?:great\W*)?(?:please\s+)?see\b")
_STOP_WORDS = (
    "that this with your have from will they them their there what when which would "
    "could should about been were into more also than then only other some such "
    "these those just here where while does doing make made like want need know very "
    "much many most each you're youre i'm okay sure please thank thanks great good "
    "well yeah right"
)
_STOP = frozenset(_STOP_WORDS.split())
_SUFFIXES = ("ings", "ing", "edly", "ed", "es", "s", "ly", "ions", "ion", "ment")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)]
            break
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def novel_tokens(target: str, evidence: Sequence[str], dialogue: Sequence[str]) -> list[str]:
    """Target content words (>= 4 chars, crude stem) found in neither evidence nor dialogue."""
    seen = {_stem(w) for w in _TOKEN_RE.findall(" ".join([*evidence, *dialogue]).lower())}
    return [w for w in _TOKEN_RE.findall(target.lower()) if w not in _STOP and _stem(w) not in seen]


def structural_problem(target: str, query: str, behaviour: str, evidence: str) -> str | None:
    """First structural defect of a grounded target (ADR-045), or None."""
    t = target.strip()
    if _MARKUP_RE.search(t) or _MARKUP_RE.search(query):
        return "markup_artifact"
    if _FUSED_RE.search(t):
        return "fused_words"
    if len(t.split()) < 4:
        return "fragment"
    if behaviour == "clarify" and "?" not in t:
        return "clarify_not_question"
    if behaviour == "answer" and t.endswith("?"):
        return "answer_is_question"
    if _POLARITY_RE.search(t) and not _EVIDENCE_POLARITY_RE.search(evidence):
        return "unsupported_polarity"
    if _REFERRAL_RE.search(t):
        return "referral_only"
    return None


Encoder = Callable[[Sequence[str]], Any]

GROUNDED_ARROW = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("source_split", pa.string()),
        ("origin", pa.string()),
        ("authorship_evidence", pa.string()),
        ("namespace", pa.string()),
        ("domain", pa.string()),
        ("doc_id", pa.string()),
        ("doc_title", pa.string()),
        ("lang", pa.string()),
        ("context_json", pa.string()),
        ("query", pa.string()),
        ("evidence_json", pa.string()),
        ("target", pa.string()),
        ("source_act", pa.string()),
        ("behaviour", pa.string()),
        ("grounding_status", pa.string()),
        ("review_status", pa.string()),
        ("evidence_recall", pa.float64()),
        ("family_id", pa.string()),
        ("split", pa.string()),
        ("reason", pa.string()),
    ]
)
GROUNDED_EXCLUDED_ARROW = pa.schema(
    [
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("text", pa.string()),
        ("reason", pa.string()),
        ("detail", pa.string()),
    ]
)


@dataclass
class GroundedRow:
    record_id: str
    source_split: str
    domain: str
    doc_id: str
    doc_title: str
    context: list[dict[str, str]]
    query: str
    evidence: list[dict[str, str]]
    target: str
    source_act: str
    behaviour: str
    evidence_recall: float
    family_id: str = ""
    split: str = ""


@dataclass
class GroundedResult:
    active: list[GroundedRow]
    unverified: list[tuple[GroundedRow, str]]
    excluded: list[dict[str, Any]]
    report: dict[str, Any] = field(default_factory=dict)


def evidence_recall(target: str, evidence: Sequence[str]) -> float:
    """Share of the target's content tokens (>= 4 alphanumerics) present in the evidence."""
    tgt = set(_TOKEN_RE.findall(target.lower()))
    if not tgt:
        return 0.0
    ev = set(_TOKEN_RE.findall(" ".join(evidence).lower()))
    return round(len(tgt & ev) / len(tgt), 4)


def pii_hits(text: str) -> int:
    return sum(len(p.findall(text)) for p in (EMAIL_RE, PHONE_RE, _NANP_RE, CARD_RE))


def _excl(rid: str, sid: str, text: str, reason: str, detail: str | None) -> dict[str, Any]:
    return {"record_id": rid, "source_id": sid, "text": text, "reason": reason, "detail": detail}


def read_doc2dial(
    entry: SourceEntry, data_dir: Path
) -> tuple[dict[str, dict[str, Any]], list[tuple[str, dict[str, Any]]]]:
    root = raw_dir(data_dir, entry)
    if problems := verify_raw(root):
        raise StepError(f"{entry.source_id}: raw revision fails verification: {problems[0]}")
    base = root / "data" / "doc2dial" / "v1.0.1"
    docs: dict[str, dict[str, Any]] = {}
    for domain, by_id in json.loads((base / "doc2dial_doc.json").read_text("utf-8"))[
        "doc_data"
    ].items():
        for doc_id, doc in by_id.items():
            docs[f"{domain}|{doc_id}"] = doc
    dials: list[tuple[str, dict[str, Any]]] = []
    for split in ("train", "validation"):
        data = json.loads((base / f"doc2dial_dial_{split}.json").read_text("utf-8"))
        for _domain, by_doc in sorted(data["dial_data"].items()):
            for _doc_id, items in sorted(by_doc.items()):
                dials.extend((split, d) for d in items)
    return docs, dials


def _evidence(doc: dict[str, Any], refs: Sequence[dict[str, Any]]) -> list[dict[str, str]] | None:
    """Referenced spans in reference order; None if a span is missing or misaligned."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    text = str(doc.get("doc_text", ""))
    for ref in refs:
        sp_id = str(ref.get("sp_id"))
        if sp_id in seen:
            continue
        seen.add(sp_id)
        sp = doc.get("spans", {}).get(sp_id)
        if sp is None or text[int(sp["start_sp"]) : int(sp["end_sp"])] != sp["text_sp"]:
            return None
        out.append(
            {
                "sp_id": sp_id,
                "label": str(ref.get("label", "")),
                "section": str(sp.get("title", "")),
                "text": str(sp["text_sp"]).strip(),
            }
        )
    return out


def candidate_rows(
    entry: SourceEntry,
    docs: dict[str, dict[str, Any]],
    dials: Sequence[tuple[str, dict[str, Any]]],
    acts: dict[str, str],
    min_recall: float,
    *,
    structural: bool = False,
    max_novel: int | None = None,
) -> tuple[list[GroundedRow], list[tuple[GroundedRow, str]], list[dict[str, Any]], int]:
    sid = entry.source_id
    rows: list[GroundedRow] = []
    unverified: list[tuple[GroundedRow, str]] = []
    excluded: list[dict[str, Any]] = []
    agent_turns = 0
    for split, dial in dials:
        key = f"{dial['domain']}|{dial['doc_id']}"
        doc = docs.get(key)
        history: list[dict[str, str]] = []
        pending: list[str] = []
        for turn in dial["turns"]:
            text = clean_text(str(turn["utterance"])).text
            if turn["role"] == "user":
                pending.append(text)
                continue
            agent_turns += 1
            rid = f"{sid}:{dial['dial_id']}:{turn['turn_id']}"
            context = list(history)
            history.extend({"role": "user", "content": t} for t in pending)
            history.append({"role": "assistant", "content": text})
            query = "\n".join(pending)
            pending = []
            act = str(turn["da"])
            if act not in acts:
                excluded.append(_excl(rid, sid, text, "act_not_mapped", act))
                continue
            if not query:
                excluded.append(_excl(rid, sid, text, "no_user_turn", None))
                continue
            if doc is None:
                excluded.append(_excl(rid, sid, text, "document_missing", key))
                continue
            ev = _evidence(doc, turn.get("references") or [])
            row = GroundedRow(
                rid,
                split,
                str(dial["domain"]),
                str(dial["doc_id"]),
                str(doc.get("title", "")),
                context,
                query,
                ev or [],
                text,
                act,
                acts[act],
                evidence_recall(text, [e["text"] for e in ev or []]),
                family_id="doc_" + sha256_text(key)[:16],
            )
            reason = None
            if not ev:
                reason = "evidence_missing"
            elif (
                not text.strip()
                or "�" in text
                or garbled(text)
                or garbled(query)
                or _PLACEHOLDER_RE.fullmatch(query.strip())
            ):
                reason = "corrupted_text"
            elif any(pii_hits(t) for t in (text, query, *(c["content"] for c in context))):
                reason = (
                    "contact_pattern"  # e-mail/phone/card pattern; often a public agency contact
                )
            elif structural and (
                problem := structural_problem(
                    text, query, acts[act], " ".join(e["text"] for e in ev)
                )
            ):
                reason = problem
            elif row.evidence_recall < min_recall:
                reason = "weak_evidence_overlap"
            elif max_novel is not None and (
                len(
                    novel_tokens(
                        text,
                        [e["text"] for e in ev],
                        [query, *(c["content"] for c in context)],
                    )
                )
                > max_novel
            ):
                reason = "novel_content"
            if reason is None:
                rows.append(row)
            else:
                unverified.append((row, reason))
    return rows, unverified, excluded, agent_turns


def assign_doc_splits(
    rows: Sequence[GroundedRow], ratios: dict[str, float], seed: int
) -> dict[str, str]:
    """document family -> split, stratified by domain in sha256 order."""
    by_domain: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_domain[r.domain].add(r.family_id)
    out: dict[str, str] = {}
    for _domain, fams in sorted(by_domain.items()):
        ordered = sorted(fams, key=lambda f: sha256_text(f"{seed}|grounded|{f}"))
        n = len(ordered)
        n_test = max(1, round(n * ratios["test"])) if n >= 3 else 0
        n_val = max(1, round(n * ratios["validation"])) if n >= 3 else 0
        for i, fam in enumerate(ordered):
            out[fam] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
    return out


def build_grounded(
    entry: SourceEntry,
    data_dir: Path,
    acts: dict[str, str],
    min_recall: float,
    ratios: dict[str, float],
    seed: int,
    *,
    history: Sequence[str],
    leak_jaccard: float,
    leak_ngram: int,
    leak_cosine: float,
    encode: Encoder | None,
    structural: bool = False,
    max_novel: int | None = None,
) -> GroundedResult:
    docs, dials = read_doc2dial(entry, data_dir)
    rows, unverified, excluded, agent_turns = candidate_rows(
        entry, docs, dials, acts, min_recall, structural=structural, max_novel=max_novel
    )
    pool = rows + [r for r, _ in unverified]
    hits: dict[int, str] = {}
    detail: dict[int, str] = {}
    for field_name in ("query", "target"):
        texts = [getattr(r, field_name) for r in pool]
        for i in exact_leaks(history, texts):
            hits.setdefault(i, "historical_exact")
        for i in lexical_leaks(history, texts, threshold=leak_jaccard, seed=seed):
            hits.setdefault(i, "historical_lexical")
        for i in ngram_leaks(history, texts, leak_ngram):
            hits.setdefault(i, "historical_ngram")
    semantic = encode is not None  # checked (vacuously when there is no history)
    if encode is not None and history and pool:
        from supportpilot.data.embed import max_cosine

        assert encode is not None
        best, arg = max_cosine(encode([r.query for r in pool]), encode(list(history)))
        for i, v in enumerate(np.asarray(best).tolist()):
            if v >= leak_cosine:
                hits.setdefault(i, "historical_semantic")
                detail[i] = f"cosine={v:.4f}; nearest={history[int(arg[i])]}"
    leaked = {id(pool[i]): reason for i, reason in hits.items()}
    for i, reason in hits.items():
        r = pool[i]
        excluded.append(
            _excl(
                r.record_id,
                entry.source_id,
                r.query,
                reason,
                detail.get(i, "overlaps an earlier SFT/gold/benchmark user text"),
            )
        )
    rows = [r for r in rows if id(r) not in leaked]
    unverified = [(r, why) for r, why in unverified if id(r) not in leaked]
    split_of = assign_doc_splits([*rows, *(r for r, _ in unverified)], ratios, seed)
    for r in rows:
        r.split = split_of[r.family_id]
    report = {
        "dialogues": len(dials),
        "documents_used": len({r.family_id for r in rows}),
        "agent_turns": agent_turns,
        "historical_overlap_removed": dict(sorted(Counter(hits.values()).items())),
        "semantic_checked": semantic,
        "min_evidence_recall": min_recall,
        "structural_checks": structural,
        "max_novel_tokens": max_novel,
    }
    return GroundedResult(rows, unverified, excluded, report)


def grounded_record(
    r: GroundedRow,
    *,
    source_id: str,
    origin: str,
    evidence_text: str,
    namespace: str,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "schema_version": GROUNDED_SCHEMA,
        "example_id": make_example_id(source_id, r.record_id, "en", origin, r.target),
        "record_id": r.record_id,
        "source_id": source_id,
        "source_split": r.source_split,
        "origin": origin,
        "authorship_evidence": evidence_text,
        "namespace": namespace,
        "domain": r.domain,
        "doc_id": r.doc_id,
        "doc_title": r.doc_title,
        "lang": "en",
        "context_json": json.dumps(r.context, ensure_ascii=False),
        "query": r.query,
        "evidence_json": json.dumps(r.evidence, ensure_ascii=False),
        "target": r.target,
        "source_act": r.source_act,
        "behaviour": r.behaviour,
        "grounding_status": GROUNDING_STATUS,
        "review_status": REVIEW_STATUS,
        "evidence_recall": r.evidence_recall,
        "family_id": r.family_id,
        "split": r.split if reason is None else "unverified",
        "reason": reason,
    }


def write_grounded(
    partial: Path,
    result: GroundedResult,
    *,
    source_id: str,
    origin: str,
    evidence_text: str,
    namespace: str,
) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    kw = {
        "source_id": source_id,
        "origin": origin,
        "evidence_text": evidence_text,
        "namespace": namespace,
    }
    for r in sorted(result.active, key=lambda r: r.record_id):
        by_split[r.split].append(grounded_record(r, reason=None, **kw))
    for s, recs in by_split.items():
        write_parquet(partial / f"grounded_{s}.parquet", recs, GROUNDED_ARROW)
    unv = [
        grounded_record(r, reason=why, **kw)
        for r, why in sorted(result.unverified, key=lambda x: x[0].record_id)
    ]
    write_parquet(partial / "grounded_unverified.parquet", unv, GROUNDED_ARROW)
    exc = sorted(result.excluded, key=lambda x: (x["record_id"], x["reason"]))
    write_parquet(partial / "grounded_excluded.parquet", exc, GROUNDED_EXCLUDED_ARROW)
    active = [r for recs in by_split.values() for r in recs]
    return {"content_sha256": content_hash([*active, *unv, *exc]), "active": active}


def grounded_stats(partial: Path, report: dict[str, Any]) -> dict[str, Any]:
    import polars as pl

    df = pl.concat([pl.read_parquet(partial / f"grounded_{s}.parquet") for s in SPLITS])
    unv = pl.read_parquet(partial / "grounded_unverified.parquet")
    exc = pl.read_parquet(partial / "grounded_excluded.parquet")
    rec = df["evidence_recall"].to_numpy() if df.height else np.zeros(0)
    return {
        "active": df.height,
        "unverified": unv.height,
        "excluded": exc.height,
        "by_split": {
            s: {
                "examples": int((df["split"] == s).sum()),
                "documents": df.filter(pl.col("split") == s)["family_id"].n_unique(),
            }
            for s in SPLITS
        },
        "behaviour": dict(sorted(Counter(df["behaviour"].to_list()).items())),
        "domain": dict(sorted(Counter(df["domain"].to_list()).items())),
        "lang": dict(Counter(df["lang"].to_list())),
        "with_context": int((df["context_json"] != "[]").sum()),
        "evidence_recall_quantiles": {
            q: round(float(np.quantile(rec, float(q))), 4) for q in ("0.1", "0.5", "0.9")
        }
        if rec.size
        else {},
        "unverified_reasons": dict(sorted(Counter(unv["reason"].to_list()).items())),
        "excluded_reasons": dict(sorted(Counter(exc["reason"].to_list()).items())),
        "grounding_status": dict(Counter(df["grounding_status"].to_list())),
        "review_status": dict(Counter(df["review_status"].to_list())),
        "namespace": dict(Counter(df["namespace"].to_list())),
        "build": report,
    }
