"""Stage 3 step 1: raw records -> cleaned records, per source (DATA_CONTRACTS §4b).

Output `data/interim/clean/{source_id}/{revision}/`: `clean.parquet` (one row per input
record, *including* rows rejected here, with `reject_reason` set) + `manifest.json`.

Per row: text repair (textnorm.clean_text), PII masking with synthetic values, template
placeholders -> synthetic slots, language ID, quality flags, dedup keys and group hints.
The raw records are only read; every output row keeps `record_id` and the raw text hash.

- Idempotent: skipped when the input records hash, `CLEAN_VERSION` and parameters match.
- Crash-safe: written under `.partial-*`, swapped in atomically (same as conversion).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from supportpilot.common.config import CleanSettings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.ids import make_example_id
from supportpilot.common.logging import get_logger
from supportpilot.common.manifest import read_model_json, scan_files, verify_files, write_model_json
from supportpilot.common.schemas import (
    LANGUAGES,
    CleanManifest,
    RecordsManifest,
    RecordType,
    SourceEntry,
)
from supportpilot.data.acquire import MANIFEST
from supportpilot.data.convert import RECORDS, records_dir, verify_records
from supportpilot.data.converters import ConversionError
from supportpilot.data.langid import LanguageIdentifier, LanguageResult
from supportpilot.data.placeholders import (
    PlaceholderCatalog,
    fill_text,
    key_form,
    malformed,
    placeholders,
    resolve_response,
)
from supportpilot.data.policy import Purpose, check_use
from supportpilot.data.textnorm import (
    PiiMasker,
    artefact_flags,
    clean_text,
    exact_key,
    garbled,
    has_letters,
    sim_key,
)

log = get_logger(__name__)
CLEAN = "clean.parquet"
CLEAN_VERSION = 1  # bump when output for the same input and parameters changes
BATCH = 5000
NL = chr(10)

CLEAN_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("example_id", pa.string()),
        ("record_id", pa.string()),
        ("source_id", pa.string()),
        ("revision", pa.string()),
        ("source_split", pa.string()),
        ("record_type", pa.string()),
        ("lang", pa.string()),
        ("lang_declared", pa.string()),
        ("lang_detected", pa.string()),
        ("lid_conf", pa.float64()),
        ("script", pa.string()),
        ("code_switch", pa.bool_()),
        ("langs", pa.list_(pa.string())),
        ("text", pa.string()),
        ("text_template", pa.string()),
        ("reference_response", pa.string()),
        ("slots", pa.map_(pa.string(), pa.string())),
        ("slot_signature", pa.string()),
        ("response_placeholders", pa.string()),
        ("source_intent", pa.string()),
        ("source_category", pa.string()),
        ("source_tags", pa.string()),
        ("context_json", pa.string()),
        ("n_turns", pa.int32()),
        ("exact_key", pa.string()),
        ("sim_text", pa.string()),
        ("family_hint", pa.string()),
        ("scenario_hint", pa.string()),
        ("flags", pa.list_(pa.string())),
        ("fixes", pa.list_(pa.string())),
        ("reject_reason", pa.string()),
        ("text_sha256", pa.string()),
    ]
)


class CleanResult(BaseModel):
    source_id: str
    revision: str
    status: Literal["cleaned", "skipped"]
    path: str
    rows: int
    rejected: int
    seconds: float


@dataclass
class _Content:
    """What identifies the example beyond its customer text, per record type."""

    key_text: str  # exact-duplicate / similarity content (template form)
    lid_text: str
    context: dict[str, Any]
    n_turns: int
    family_hint: str | None = None
    scenario_hint: str | None = None
    flags: tuple[str, ...] = ()


def clean_dir(data_dir: Path, entry: SourceEntry) -> Path:
    return data_dir / "interim" / "clean" / entry.source_id / entry.revision


def verify_clean(path: Path) -> list[Any]:
    if not (path / MANIFEST).is_file():
        return [{"path": MANIFEST, "problem": "missing"}]
    m = read_model_json(path / MANIFEST, CleanManifest)
    return verify_files(path, m.files, exclude=frozenset({MANIFEST}))


def clean_params(cfg: CleanSettings, catalog_path: Path) -> dict[str, object]:
    return {
        "lid_other_min_conf": cfg.lid_other_min_conf,
        "lid_low_conf": cfg.lid_low_conf,
        "lid_min_letters": cfg.lid_min_letters,
        "lid_relabel_min_conf": cfg.lid_relabel_min_conf,
        "sim_text_max_chars": cfg.sim_text_max_chars,
        "placeholders_sha256": sha256_file(catalog_path),
        "lingua": version("lingua-language-detector"),
    }


def _params_sha(params: dict[str, object]) -> str:
    return sha256_text(json.dumps(params, sort_keys=True))


class _Cleaner:
    def __init__(self, entry: SourceEntry, cfg: CleanSettings, catalog: PlaceholderCatalog):
        self.entry = entry
        self.cfg = cfg
        self.catalog = catalog
        self.lid = LanguageIdentifier(
            other_min_conf=cfg.lid_other_min_conf, min_letters=cfg.lid_min_letters
        )

    # ------------------------------------------------------------ per record type content
    def _clean_turn(self, raw: str, masker: PiiMasker) -> str:
        return masker.mask(clean_text(raw).text)

    def _content(self, rec: dict[str, Any], text_tpl: str, masker: PiiMasker) -> _Content:
        extra: dict[str, Any] = json.loads(rec["extra_json"])
        rtype = RecordType(rec["record_type"])
        if rtype is RecordType.DIALOGUE:
            turns = [
                {"role": str(m.get("role")), "content": self._clean_turn(str(m["content"]), masker)}
                for m in extra.get("messages", [])
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            ]
            key = NL.join(f"{t['role']}: {t['content']}" for t in turns)
            user = NL.join(t["content"] for t in turns if t["role"] == "user")
            flags = []
            if turns and turns[0]["role"] != "user":
                flags.append("starts_with_assistant")
            if not any(t["role"] == "assistant" for t in turns):
                flags.append("no_assistant_turn")
            if any(garbled(t["content"]) for t in turns):
                flags.append("garbled_turn")
            context: dict[str, Any] = {
                "messages": turns,
                "target": extra.get("target"),
                "label_meta": extra.get("label_meta"),
                "data_meta": extra.get("data_meta"),
                "conv_id": extra.get("conv_id"),
            }
            return _Content(key, user[:2000], context, len(turns), flags=tuple(flags))
        if rtype is RecordType.POLICY_DECISION:
            meta = json.loads(extra.get("metadata_json") or "{}")
            context = {
                k: extra.get(k)
                for k in (
                    "action",
                    "recommended_action",
                    "priority",
                    "sentiment",
                    "human_escalation",
                    "missing_info",
                    "supported_rule_ids",
                )
            }
            context |= {
                "policies": json.loads(extra.get("policies_json") or "[]"),
                "facts": json.loads(extra.get("facts_json") or "[]"),
                "history": json.loads(extra.get("history_json") or "[]"),
                "capabilities": json.loads(extra.get("capabilities_json") or "{}"),
                "scenario_family_id": meta.get("scenario_family_id"),
                "scenario_group_id": meta.get("scenario_group_id"),
                "state": meta.get("state"),
                "source": meta.get("source"),
                "human_reviewed": meta.get("human_reviewed"),
            }
            fact_list: list[dict[str, Any]] = list(context["facts"])
            policy_list: list[dict[str, Any]] = list(context["policies"])
            facts = " ".join(f"{f.get('key')}={f.get('value')}" for f in fact_list)
            pol = " ".join(str(p.get("text")) for p in policy_list)
            hist = json.dumps(context["history"], ensure_ascii=False)
            key = NL.join((text_tpl, facts, pol, hist))
            fam = meta.get("scenario_group_id")
            scn = meta.get("scenario_family_id")
            return _Content(
                key,
                text_tpl,
                context,
                1,
                family_hint=f"sjr_group:{fam}" if fam else None,
                scenario_hint=f"sjr_family:{scn}" if scn else None,
            )
        if rtype is RecordType.READING_COMPREHENSION:
            options = [str(extra.get(f"mc_answer{i}", "")) for i in range(1, 5)]
            passage = clean_text(str(extra.get("flores_passage", ""))).text
            cleaned_options = [clean_text(o).text for o in options]
            context = {
                "passage": passage,
                "options": cleaned_options,
                "correct_answer_num": extra.get("correct_answer_num"),
                "link": extra.get("link"),
                "question_number": extra.get("question_number"),
                "dialect": extra.get("dialect"),
            }
            group = f"belebele:{extra.get('link')}#{extra.get('question_number')}"
            key = NL.join((passage, text_tpl, *cleaned_options))
            return _Content(key, text_tpl, context, 1, family_hint=group, scenario_hint=group)
        return _Content(text_tpl, text_tpl, {}, 1)

    # ------------------------------------------------------------ one row
    def row(self, rec: dict[str, Any], lid: LanguageResult | None) -> dict[str, Any]:
        e = self.entry
        masker = PiiMasker()
        cleaned = clean_text(rec["text"])
        text_tpl = masker.mask(cleaned.text)
        flags: list[str] = ["repaired"] if cleaned.fixes else []
        reject: str | None = None
        names = placeholders(text_tpl)
        if not text_tpl:
            reject = "empty_text"
        elif malformed(text_tpl):
            reject = "malformed_placeholder"
        elif any(n not in self.catalog.slots for n in names):
            reject = "unknown_placeholder"
        elif "�" in text_tpl:
            reject = "encoding_damage"
        filled = fill_text(text_tpl, self.catalog, rec["record_id"])
        content = self._content(rec, key_form(text_tpl, self.catalog), masker)
        rtype = RecordType(rec["record_type"])
        if reject is None and (garbled(text_tpl) or "garbled_turn" in content.flags):
            reject = "garbled_text"
        if reject is None and rtype is not RecordType.DIALOGUE and not has_letters(text_tpl):
            reject = "no_letters"
        if names:
            flags.append("placeholder_filled")
        flags += artefact_flags(text_tpl)
        flags += content.flags
        if rtype is RecordType.DIALOGUE and "short_text" in flags:
            flags.append("weak_first_turn")

        response: str | None = None
        kept: dict[str, list[str]] = {}
        resp_masker = PiiMasker()
        if rec["response"]:
            resp = resp_masker.mask(clean_text(rec["response"]).text)
            if malformed(resp):
                flags.append("response_malformed_placeholder")
            resolved = resolve_response(resp, filled.shown, self.catalog)
            response, kept = resolved.text, resolved.kept
            flags += [f"response_{cls}" for cls in kept]
        elif rtype in (RecordType.SINGLE_TURN, RecordType.DIALOGUE):
            flags.append("response_missing")
        flags += sorted(set(masker.found))
        flags += sorted({f"response_{f}" for f in resp_masker.found})

        declared = rec["lang"]
        lang = declared
        detected = lid.lang if lid else "und"
        conf = lid.conf if lid else 0.0
        if lid is None or lid.lang == "und":
            flags.append("lid_undetermined")
        elif lid.lang.startswith("other:"):
            reject = reject or "unsupported_language"
        else:
            if lid.lang != declared:
                if conf >= self.cfg.lid_relabel_min_conf:
                    lang = lid.lang
                    flags.append("lang_relabelled")
                else:
                    flags.append("lid_disagrees")
            if conf < self.cfg.lid_low_conf:
                flags.append("lid_low_conf")
        if lid is not None and lid.code_switch:
            flags.append("code_switch")
        if lang not in LANGUAGES:  # pragma: no cover - declared langs are validated upstream
            raise ConversionError(f"{rec['record_id']}: unsupported declared language {lang}")

        sim_text = sim_key(content.key_text)[: self.cfg.sim_text_max_chars]
        return {
            "schema_version": "clean_record/1",
            "example_id": make_example_id(
                e.source_id, rec["record_id"], lang, "original", filled.text or "-"
            ),
            "record_id": rec["record_id"],
            "source_id": e.source_id,
            "revision": e.revision,
            "source_split": rec["source_split"],
            "record_type": rec["record_type"],
            "lang": lang,
            "lang_declared": declared,
            "lang_detected": detected,
            "lid_conf": conf,
            "script": lid.script if lid else "None",
            "code_switch": bool(lid and lid.code_switch),
            "langs": list(lid.langs) if lid and lid.langs else [lang],
            "text": filled.text,
            "text_template": text_tpl,
            "reference_response": response,
            "slots": sorted(filled.slots.items()),
            "slot_signature": "|".join(sorted(filled.slots)),
            "response_placeholders": json.dumps(kept, sort_keys=True) if kept else None,
            "source_intent": rec["source_intent"],
            "source_category": rec["source_category"],
            "source_tags": rec["source_tags"],
            "context_json": json.dumps(content.context, ensure_ascii=False, sort_keys=True)
            if content.context
            else None,
            "n_turns": content.n_turns,
            "exact_key": exact_key(content.key_text),
            "sim_text": sim_text,
            "family_hint": content.family_hint,
            "scenario_hint": content.scenario_hint,
            "flags": sorted(set(flags)),
            "fixes": list(cleaned.fixes),
            "reject_reason": reject,
            "text_sha256": rec["text_sha256"],
        }

    def lid_texts(self, recs: list[dict[str, Any]]) -> list[str]:
        out = []
        for rec in recs:
            if rec["record_type"] == RecordType.DIALOGUE.value:
                extra = json.loads(rec["extra_json"])
                user = [
                    str(m.get("content", ""))
                    for m in extra.get("messages", [])
                    if isinstance(m, dict) and m.get("role") == "user"
                ]
                out.append(clean_text(NL.join(user)[:2000]).text)
            else:
                out.append(clean_text(rec["text"]).text)
        return out


def _is_current(out: Path, records_sha: str, params_sha: str) -> bool:
    if not (out / MANIFEST).is_file():
        return False
    m = read_model_json(out / MANIFEST, CleanManifest)
    return (
        m.records_content_sha256 == records_sha
        and m.params_sha256 == params_sha
        and m.clean_version == CLEAN_VERSION
        and not verify_clean(out)
    )


def clean_source(
    entry: SourceEntry,
    data_dir: Path,
    cfg: CleanSettings,
    catalog: PlaceholderCatalog,
    params: dict[str, object],
    *,
    code_commit: str | None = None,
    force: bool = False,
) -> CleanResult:
    check_use(entry, Purpose.ANALYSIS)
    start = time.perf_counter()
    rec_dir = records_dir(data_dir, entry)
    if not (rec_dir / MANIFEST).is_file():
        raise ConversionError(f"{entry.source_id}: no converted records at {rec_dir}")
    if problems := verify_records(rec_dir):
        raise ConversionError(f"{entry.source_id}: records fail verification: {problems[:3]}")
    rmf = read_model_json(rec_dir / MANIFEST, RecordsManifest)
    params_sha = _params_sha(params)
    out = clean_dir(data_dir, entry)
    for stale in out.parent.glob(".partial-*"):  # left by a killed run
        shutil.rmtree(stale, ignore_errors=True)
    if not force and _is_current(out, rmf.content_sha256, params_sha):
        m = read_model_json(out / MANIFEST, CleanManifest)
        return CleanResult(
            source_id=entry.source_id,
            revision=entry.revision,
            status="skipped",
            path=str(out),
            rows=m.row_count,
            rejected=m.rejected_count,
            seconds=round(time.perf_counter() - start, 2),
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.parent / f".partial-{entry.revision}-{uuid.uuid4().hex[:8]}"
    partial.mkdir()
    try:
        cleaner = _Cleaner(entry, cfg, catalog)
        content = hashlib.sha256()
        reasons: Counter[str] = Counter()
        flag_counts: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        n = 0
        with pq.ParquetWriter(partial / CLEAN, CLEAN_SCHEMA, compression="zstd") as writer:
            for batch in pq.ParquetFile(rec_dir / RECORDS).iter_batches(batch_size=BATCH):
                recs = batch.to_pylist()
                lids = cleaner.lid.detect_many(cleaner.lid_texts(recs))
                rows = [cleaner.row(r, lid) for r, lid in zip(recs, lids, strict=True)]
                for row in rows:
                    content.update(
                        (json.dumps(row, ensure_ascii=False, sort_keys=True) + NL).encode()
                    )
                    if row["reject_reason"]:
                        reasons[row["reject_reason"]] += 1
                    flag_counts.update(row["flags"])
                    counts[f"lang={row['lang']}"] += 1
                writer.write_table(pa.Table.from_pylist(rows, schema=CLEAN_SCHEMA))
                n += len(rows)
        if n != rmf.row_count:
            raise ConversionError(f"{entry.source_id}: {n} clean rows != {rmf.row_count} records")
        manifest = CleanManifest(
            source_id=entry.source_id,
            revision=entry.revision,
            allowed_use=entry.allowed_use,
            records_content_sha256=rmf.content_sha256,
            clean_version=CLEAN_VERSION,
            params=params,
            params_sha256=params_sha,
            code_commit=code_commit,
            created_at=datetime.now(UTC),
            row_count=n,
            rejected_count=sum(reasons.values()),
            reject_reasons=dict(sorted(reasons.items())),
            flag_counts=dict(sorted(flag_counts.items())),
            counts=dict(sorted(counts.items())),
            content_sha256=content.hexdigest(),
            files=scan_files(partial),
        )
        write_model_json(partial / MANIFEST, manifest)
        old = out.parent / f".old-{uuid.uuid4().hex[:8]}"
        if out.exists():
            out.rename(old)
        partial.rename(out)
        shutil.rmtree(old, ignore_errors=True)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    log.info("source cleaned", extra={"source_id": entry.source_id, "rows": n})
    return CleanResult(
        source_id=entry.source_id,
        revision=entry.revision,
        status="cleaned",
        path=str(out),
        rows=n,
        rejected=manifest.rejected_count,
        seconds=round(time.perf_counter() - start, 2),
    )
